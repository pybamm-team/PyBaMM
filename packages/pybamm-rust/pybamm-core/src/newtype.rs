//! Shared definition of the crate's index newtypes, so distinct integer roles cannot be swapped.

/// Define a `#[repr(transparent)]` newtype over `$repr` with `new` and `get` accessors.
///
/// `Default` is deliberately not derived: it would hand back a plausible index 0.
macro_rules! index_newtype {
    ($(#[$attr:meta])* $name:ident($repr:ty)) => {
        $(#[$attr])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name($repr);

        impl $name {
            /// Wrap a raw value.
            pub const fn new(raw: $repr) -> Self {
                Self(raw)
            }

            /// The raw value.
            pub const fn get(self) -> $repr {
                self.0
            }
        }

        impl ::std::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

pub(crate) use index_newtype;
