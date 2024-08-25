@echo off

echo Setting environment variables...

setx PYBAMM_USE_VCPKG ON
setx VCPKG_DEFAULT_TRIPLET x64-windows-static-md
setx VCPKG_FEATURE_FLAGS manifests,registries

echo Environment variables have been set. Please open a new Command Prompt to see the changes.
pause
