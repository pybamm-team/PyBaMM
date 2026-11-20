use numpy::ndarray::Dimension;
use numpy::{
    PyArray, PyArray1, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyReadonlyArray,
    PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyFloat, PyInt, PyMapping};

use pybamm_core::ParamRegistry;

/// What a compiled function takes and returns, used to check and pack each call's arguments.
#[derive(Debug, Clone)]
pub struct FunctionMetadata {
    /// The graph's input registry when compiled: names, registration indices and packed slots.
    pub(crate) params: ParamRegistry,
    /// Length of the packed `p`: the registry's packed width, or a compiled model's wider one.
    pub(crate) input_width: usize,
    pub(crate) n_states: usize,
    pub(crate) uses_y_dot: bool,
    pub(crate) output_len: usize,
    pub(crate) name: Option<String>,
}

impl FunctionMetadata {
    pub(crate) fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or("<anonymous>")
    }

    /// Length of a parameter-direction vector: one entry per registration index, unregistered
    /// gaps included, so not the packed width.
    pub(crate) const fn vp_len(&self) -> usize {
        self.params.index_extent()
    }

    /// Write input `name`'s values into its packed `slot`, from a real number or any array-like
    /// of exactly `slot.len()` real values.
    fn write_input(&self, value: &Bound<'_, PyAny>, name: &str, slot: &mut [f64]) -> PyResult<()> {
        let refuse = || {
            PyValueError::new_err(format!(
                "{}: input '{name}' must be a real number or an array of real numbers, got {}",
                self.display_name(),
                describe(value)
            ))
        };
        let got = if value.is_instance_of::<PyFloat>() || value.is_instance_of::<PyInt>() {
            if let [only] = slot {
                *only = value.extract::<f64>()?;
            }
            1
        } else {
            // Anything else goes through NumPy unconverted: a float conversion would drop a
            // complex value's imaginary part, and NumPy deprecates it for a one-element array.
            let array = match value.cast::<PyUntypedArray>() {
                Ok(array) => array.clone(),
                Err(_) => as_array(value).map_err(|_| refuse())?,
            };
            if !matches!(array.dtype().kind(), b'b' | b'i' | b'u' | b'f') {
                return Err(refuse());
            }
            self.write_array(&array, name, slot)?
        };
        if got != slot.len() {
            return Err(PyValueError::new_err(if slot.len() == 1 {
                format!(
                    "{}: input '{name}' must be a real scalar or length-1 array, got {got} values",
                    self.display_name()
                )
            } else {
                format!(
                    "{}: input '{name}' must have {} value(s), got {got}",
                    self.display_name(),
                    slot.len()
                )
            }));
        }
        Ok(())
    }

    /// Copy a real `array` into `slot` in C order when the counts agree; returns its count.
    fn write_array(
        &self,
        array: &Bound<'_, PyUntypedArray>,
        name: &str,
        slot: &mut [f64],
    ) -> PyResult<usize> {
        let count = array.len();
        if count == slot.len() {
            let values = match array.cast::<PyArrayDyn<f64>>() {
                Ok(values) => values.clone(),
                Err(_) => array
                    .call_method1("astype", (numpy::dtype::<f64>(array.py()),))?
                    .cast_into::<PyArrayDyn<f64>>()?,
            };
            let values = self.read(&values, &format!("input '{name}'"))?;
            for (dst, src) in slot.iter_mut().zip(values.as_array().iter()) {
                *dst = *src;
            }
        }
        Ok(count)
    }

    /// Pack a {name: value} mapping into the stacked layout, rejecting missing
    /// or unknown names and wrong widths.
    pub(crate) fn pack(&self, inputs: &Bound<'_, PyMapping>) -> PyResult<Vec<f64>> {
        let py = inputs.py();
        // A dict is read without `__missing__`, so a defaultdict cannot invent a missing input.
        let dict = inputs.cast::<PyDict>().ok();
        let mut packed = vec![0.0; self.input_width];
        for (_, name, slot) in self.params.registered() {
            let value = match dict {
                Some(dict) => dict.get_item(name)?,
                None => match inputs.get_item(name) {
                    Ok(value) => Some(value),
                    Err(err) if err.is_instance_of::<PyKeyError>(py) => None,
                    Err(err) => return Err(err),
                },
            };
            let value = value.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "{}: missing input '{name}'; expected inputs {:?}",
                    self.display_name(),
                    self.params.names()
                ))
            })?;
            let (offset, width) = (slot.offset.index(), slot.width.index());
            self.write_input(&value, name, &mut packed[offset..offset + width])?;
        }
        if inputs.len()? != self.params.n_registered() {
            for key in inputs.keys()?.iter() {
                let known = key
                    .extract::<String>()
                    .is_ok_and(|key| self.params.index_of(&key).is_some());
                if !known {
                    return Err(PyValueError::new_err(format!(
                        "{}: unknown input '{key}'; expected inputs {:?}",
                        self.display_name(),
                        self.params.names()
                    )));
                }
            }
        }
        Ok(packed)
    }

    /// `p` as a packed copy, from a 1-D float64 array or a {name: value} mapping to pack.
    pub(crate) fn extract_p(&self, p: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        if let Ok(array) = p.cast::<PyArray1<f64>>() {
            let array = self.read(array, "p")?;
            let values = array.as_array().to_vec();
            self.check_p(values.len())?;
            return Ok(values);
        }
        if let Ok(inputs) = p.cast::<PyMapping>() {
            return self.pack(inputs);
        }
        Err(PyValueError::new_err(format!(
            "{}: p must be a 1-D float64 array or a {{name: value}} mapping, got {}",
            self.display_name(),
            describe(p)
        )))
    }

    /// Borrow `array` for reading, refusing one already borrowed for writing.
    fn read<'py, D: Dimension>(
        &self,
        array: &Bound<'py, PyArray<f64, D>>,
        what: &str,
    ) -> PyResult<PyReadonlyArray<'py, f64, D>> {
        array.try_readonly().map_err(|err| {
            PyValueError::new_err(format!(
                "{}: {what} shares memory with out ({err})",
                self.display_name()
            ))
        })
    }

    /// Reject a `what` buffer whose length is not `expected`.
    pub(crate) fn check_len(&self, what: &str, got: usize, expected: usize) -> PyResult<()> {
        if got != expected {
            return Err(PyValueError::new_err(format!(
                "{}: expected {what} of length {expected}, got {got}",
                self.display_name()
            )));
        }
        Ok(())
    }

    pub(crate) fn check_p(&self, len: usize) -> PyResult<()> {
        if len != self.input_width {
            return Err(PyValueError::new_err(format!(
                "{}: expected {} input values ({} parameters), got {}",
                self.display_name(),
                self.input_width,
                self.params.n_registered(),
                len
            )));
        }
        Ok(())
    }

    /// Reject a parameter-direction vector that is not [`vp_len`](Self::vp_len) long.
    pub(crate) fn check_vp(&self, len: usize) -> PyResult<()> {
        let vp_len = self.vp_len();
        if len != vp_len {
            return Err(PyValueError::new_err(format!(
                "{}: expected vp of length {vp_len} (one direction per input index), got {len}",
                self.display_name()
            )));
        }
        Ok(())
    }

    /// Validate the trajectory `what`, of shape `(n_rows, n_cols)`, against `n_states` and
    /// `len(ts)`.
    pub(crate) fn check_trajectory(
        &self,
        what: &str,
        n_rows: usize,
        n_cols: usize,
        n_ts: usize,
    ) -> PyResult<()> {
        if n_rows != self.n_states {
            return Err(PyValueError::new_err(format!(
                "{}: {what}.shape[0] must equal n_states ({}), got {n_rows}",
                self.display_name(),
                self.n_states
            )));
        }
        if n_ts != n_cols {
            return Err(PyValueError::new_err(format!(
                "{}: len(ts) ({n_ts}) must equal {what}.shape[1] ({n_cols})",
                self.display_name()
            )));
        }
        Ok(())
    }

    /// Reject knot times too few, not finite, or not strictly increasing, to
    /// Hermite-interpolate within.
    pub(crate) fn check_knots(&self, ts: &[f64]) -> PyResult<()> {
        if ts.len() < 2 {
            return Err(PyValueError::new_err(format!(
                "{}: need >= 2 knots for Hermite interpolation, got {}",
                self.display_name(),
                ts.len()
            )));
        }
        if !ts.iter().all(|t| t.is_finite()) || !ts.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(PyValueError::new_err(format!(
                "{}: knot times must be finite and strictly increasing",
                self.display_name()
            )));
        }
        Ok(())
    }

    /// Reject `y_dot`-using expressions where `y_dot` is empty, which would panic.
    pub(crate) fn reject_y_dot(&self, op: &str) -> PyResult<()> {
        if self.uses_y_dot {
            return Err(PyValueError::new_err(format!(
                "{}: {} is not defined for expressions that use y_dot",
                self.display_name(),
                op
            )));
        }
        Ok(())
    }
}

/// What `value` is, for error messages: an array's dimensions and dtype, otherwise its type.
fn describe(value: &Bound<'_, PyAny>) -> String {
    if let Ok(array) = value.cast::<PyUntypedArray>() {
        return format!("a {}-D array of dtype {}", array.ndim(), array.dtype());
    }
    value
        .get_type()
        .name()
        .map_or_else(|_| "an unnamed type".to_owned(), |name| name.to_string())
}

/// `numpy.asarray(value)`, with no dtype forced on it.
fn as_array<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyUntypedArray>> {
    static ASARRAY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = value.py();
    let asarray = ASARRAY.get_or_try_init(py, || {
        py.import("numpy")?.getattr("asarray").map(Bound::unbind)
    })?;
    Ok(asarray.bind(py).call1((value,))?.cast_into()?)
}
