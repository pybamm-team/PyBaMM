use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pybamm_core::ParamRegistry;

#[derive(Debug, Clone)]
pub struct FunctionSignature {
    /// The graph's input registry: names, registration indices and packed slots.
    pub params: ParamRegistry,
    pub n_states: usize,
    pub uses_y_dot: bool,
    pub output_len: usize,
    pub name: Option<String>,
}

impl FunctionSignature {
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or("<anonymous>")
    }

    const fn total_width(&self) -> usize {
        self.params.packed_width().get()
    }

    /// Coerce a real number or length-1 numeric array to `f64`.
    fn extract_scalar(&self, v: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
        if let Ok(x) = v.extract::<f64>() {
            return Ok(x);
        }
        let bad = || {
            PyValueError::new_err(format!(
                "{}: input '{}' must be a real scalar or length-1 numeric array",
                self.display_name(),
                name
            ))
        };
        let size: usize = v
            .getattr("size")
            .and_then(|s| s.extract::<usize>())
            .map_err(|_| bad())?;
        if size != 1 {
            return Err(PyValueError::new_err(format!(
                "{}: input '{}' must be a scalar or length-1 array, got size {}",
                self.display_name(),
                name,
                size
            )));
        }
        v.call_method0("item")?.extract::<f64>().map_err(|_| bad())
    }

    /// Coerce an array of any shape with `width` values to a flat `Vec<f64>`.
    fn extract_vector(&self, v: &Bound<'_, PyAny>, name: &str, width: usize) -> PyResult<Vec<f64>> {
        let bad_shape = |size: usize| {
            PyValueError::new_err(format!(
                "{}: input '{}' must have {} value(s), got {}",
                self.display_name(),
                name,
                width,
                size
            ))
        };
        let bad_type = || {
            PyValueError::new_err(format!(
                "{}: input '{}' must be a numeric array of length {}",
                self.display_name(),
                name,
                width
            ))
        };
        let raveled = v.call_method0("ravel").map_err(|_| bad_type())?;
        let arr: PyReadonlyArray1<'_, f64> = raveled.extract().map_err(|_| bad_type())?;
        let slice = arr.as_slice()?;
        if slice.len() != width {
            return Err(bad_shape(slice.len()));
        }
        Ok(slice.to_vec())
    }

    /// Pack a {name: value} mapping into the stacked layout, rejecting missing
    /// or unknown names and wrong widths.
    pub fn pack(&self, dict: &Bound<'_, PyDict>) -> PyResult<Vec<f64>> {
        let mut packed = vec![0.0; self.total_width()];
        for (_, name, slot) in self.params.registered() {
            let (offset, width) = (slot.offset.index(), slot.width.index());
            match dict.get_item(name)? {
                Some(v) => {
                    if width == 1 {
                        packed[offset] = self.extract_scalar(&v, name)?;
                    } else {
                        let values = self.extract_vector(&v, name, width)?;
                        packed[offset..offset + width].copy_from_slice(&values);
                    }
                },
                None => {
                    return Err(PyValueError::new_err(format!(
                        "{}: missing input '{}'; expected inputs {:?}",
                        self.display_name(),
                        name,
                        self.params.names()
                    )));
                },
            }
        }
        if dict.len() != self.params.n_registered() {
            for key in dict.keys() {
                let k: String = key.extract()?;
                if self.params.index_of(&k).is_none() {
                    return Err(PyValueError::new_err(format!(
                        "{}: unknown input '{}'; expected inputs {:?}",
                        self.display_name(),
                        k,
                        self.params.names()
                    )));
                }
            }
        }
        Ok(packed)
    }

    /// `p` as a stacked slice copy, packing a {name: value} mapping when given one.
    pub fn extract_p(&self, p: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        if let Ok(dict) = p.cast::<PyDict>() {
            return self.pack(dict);
        }
        let arr: PyReadonlyArray1<'_, f64> = p.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "{}: p must be a 1-D float64 array or a {{name: value}} mapping",
                self.display_name()
            ))
        })?;
        let slice = arr.as_slice()?;
        self.check_p(slice.len())?;
        Ok(slice.to_vec())
    }

    /// Reject a `what` buffer whose length is not `expected`.
    pub fn check_len(&self, what: &str, got: usize, expected: usize) -> PyResult<()> {
        if got != expected {
            return Err(PyValueError::new_err(format!(
                "{}: expected {what} of length {expected}, got {got}",
                self.display_name()
            )));
        }
        Ok(())
    }

    pub fn check_p(&self, len: usize) -> PyResult<()> {
        let total_width = self.total_width();
        if len != total_width {
            return Err(PyValueError::new_err(format!(
                "{}: expected {} input values ({} parameters), got {}",
                self.display_name(),
                total_width,
                self.params.n_registered(),
                len
            )));
        }
        Ok(())
    }

    /// Validate a parameter-direction vector: one entry per registered name, not
    /// the packed width.
    pub fn check_vp(&self, len: usize) -> PyResult<()> {
        let n_params = self.params.index_extent();
        if len != n_params {
            return Err(PyValueError::new_err(format!(
                "{}: expected vp of length {} (one direction per parameter), got {}",
                self.display_name(),
                n_params,
                len
            )));
        }
        Ok(())
    }

    /// Validate an `(n_states, n_t)` trajectory shape against `len(ts)`.
    pub fn check_trajectory(&self, n_rows: usize, n_cols: usize, n_ts: usize) -> PyResult<()> {
        if n_rows != self.n_states {
            return Err(PyValueError::new_err(format!(
                "{}: Y.shape[0] must equal n_states ({}), got {}",
                self.display_name(),
                self.n_states,
                n_rows
            )));
        }
        if n_ts != n_cols {
            return Err(PyValueError::new_err(format!(
                "{}: len(ts) ({}) must equal Y.shape[1] ({})",
                self.display_name(),
                n_ts,
                n_cols
            )));
        }
        Ok(())
    }

    /// Reject a knot grid too short to Hermite-interpolate within.
    pub fn check_knots(&self, n_knots: usize) -> PyResult<()> {
        if n_knots < 2 {
            return Err(PyValueError::new_err(format!(
                "{}: need >= 2 knots for Hermite interpolation, got {}",
                self.display_name(),
                n_knots
            )));
        }
        Ok(())
    }

    /// Reject `y_dot`-using expressions where `y_dot` is empty, which would panic.
    pub fn reject_y_dot(&self, op: &str) -> PyResult<()> {
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
