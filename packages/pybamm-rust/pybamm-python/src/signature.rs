use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Debug, Clone)]
pub struct FunctionSignature {
    /// Input names ordered by registration index (== stacked layout).
    pub input_names: Vec<String>,
    /// Packed width per input (>1 for vector-valued inputs), parallel to `input_names`.
    pub input_widths: Vec<usize>,
    pub n_states: usize,
    pub uses_y_dot: bool,
    pub output_len: usize,
    pub name: Option<String>,
}

impl FunctionSignature {
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or("<anonymous>")
    }

    /// Total packed width across all inputs (sum of `input_widths`).
    fn total_width(&self) -> usize {
        self.input_widths.iter().sum()
    }

    /// Coerce a single input value to a scalar `f64`. Accepts a real number
    /// or a length-1 numeric array.
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

    /// Coerce a single input value to a flat `width`-length `Vec<f64>`.
    /// Accepts any array shape whose flattened size matches `width`
    /// (e.g. the column-vector convention `np.array([[..]])` used elsewhere).
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

    /// Pack a {name: value} mapping into the stacked layout. Validates
    /// keys both ways, names offenders, and checks each value's length
    /// against the parameter's declared width.
    pub fn pack(&self, dict: &Bound<'_, PyDict>) -> PyResult<Vec<f64>> {
        let mut packed = vec![0.0; self.total_width()];
        let mut offset = 0usize;
        for (i, name) in self.input_names.iter().enumerate() {
            let width = self.input_widths[i];
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
                        self.input_names
                    )));
                },
            }
            offset += width;
        }
        if dict.len() != self.input_names.len() {
            for key in dict.keys() {
                let k: String = key.extract()?;
                if !self.input_names.iter().any(|n| n == &k) {
                    return Err(PyValueError::new_err(format!(
                        "{}: unknown input '{}'; expected inputs {:?}",
                        self.display_name(),
                        k,
                        self.input_names
                    )));
                }
            }
        }
        Ok(packed)
    }

    /// `p` as a stacked slice copy, packing a {name: value} mapping when given
    /// one. The single home for the dict-vs-array branch on every call path.
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

    pub fn check_y(&self, len: usize) -> PyResult<()> {
        if len != self.n_states {
            return Err(PyValueError::new_err(format!(
                "{}: expected y of length {}, got {}",
                self.display_name(),
                self.n_states,
                len
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
                self.input_names.len(),
                len
            )));
        }
        Ok(())
    }

    /// Validate a parameter-direction vector (`vp`/`dp` seed): one scalar
    /// direction per registered parameter name (registration-index space,
    /// what `TangentParameter` reads), NOT the packed total width.
    pub fn check_vp(&self, len: usize) -> PyResult<()> {
        if len != self.input_names.len() {
            return Err(PyValueError::new_err(format!(
                "{}: expected vp of length {} (one direction per parameter), got {}",
                self.display_name(),
                self.input_names.len(),
                len
            )));
        }
        Ok(())
    }

    pub fn check_y_dot(&self, len: usize) -> PyResult<()> {
        if len != self.n_states {
            return Err(PyValueError::new_err(format!(
                "{}: expected y_dot of length {}, got {}",
                self.display_name(),
                self.n_states,
                len
            )));
        }
        Ok(())
    }

    /// Reject `y_dot`-using expressions for operations that evaluate with
    /// an empty one, the tape would slice it and panic across `PyO3`.
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
