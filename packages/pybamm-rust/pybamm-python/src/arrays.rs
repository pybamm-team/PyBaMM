//! Array arguments as slices: borrowed when their layout allows, else copied once.

use std::borrow::Cow;

use numpy::PyReadonlyArray1;
use numpy::ndarray::ArrayView2;

/// `array`'s values in order: borrowed when contiguous, else one copy.
pub fn values<'a>(array: &'a PyReadonlyArray1<'_, f64>) -> Cow<'a, [f64]> {
    array
        .as_slice()
        .map_or_else(|_| Cow::Owned(array.as_array().to_vec()), Cow::Borrowed)
}

/// `view`'s columns end to end: borrowed when F-contiguous, else one gathering copy.
pub fn columns<'a>(view: &'a ArrayView2<'a, f64>) -> Cow<'a, [f64]> {
    match view.as_slice_memory_order() {
        Some(values) if view.strides()[0] == 1 && view.strides().iter().all(|&st| st >= 0) => {
            Cow::Borrowed(values)
        },
        _ => {
            let mut owned = Vec::with_capacity(view.len());
            for column in view.columns() {
                owned.extend(column);
            }
            Cow::Owned(owned)
        },
    }
}
