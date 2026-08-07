# Vendored CasADi internal headers

**These files are CasADi, not PyBaMM, and are licensed LGPL-3.0-or-later.**
The rest of `pybammsolvers` is BSD-3-Clause. Keep that distinction in mind before
copying anything out of this directory.

## Why they are here

Subclassing `casadi::Rootfinder` to register the `brent` plugin
(`src/pybammsolvers/idaklu_source/brent.cpp`) requires CasADi's internal headers.
CasADi does not install them: `casadi/core/CMakeLists.txt` gates `${CASADI_INTERNAL}`
behind `option(INSTALL_INTERNAL_HEADERS ... OFF)`, so neither the pip wheel nor the
vcpkg/conda-forge builds ship them. The compiled plugin links the ordinary shared
`libcasadi`, so this is a compile-time dependency only.

## Provenance

Extracted verbatim from CasADi at git revision `0e672301aa9046162d311f69421f6375805f4fca`,
which is the `CASADI_GIT_REVISION` recorded in the casadi 3.7.2 wheel's `config.h`:

```bash
for h in rootfinder_impl plugin_interface oracle_function function_internal casadi_os; do
  git -C <casadi> show 0e672301:casadi/core/$h.hpp > vendor/casadi/core/$h.hpp
done
```

## Updating

Bumping the pinned `casadi` version in `pyproject.toml` requires re-extracting these
five headers from the matching revision and re-running
`tests/test_brent_rootfinder.py`. They are compiled against non-ABI-stable internals,
so a mismatch is a runtime failure, not a compile error.
