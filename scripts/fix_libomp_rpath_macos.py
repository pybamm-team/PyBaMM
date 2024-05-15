# A helper script to fix the rpath of the OpenMP dynamic library. This
# is to be used when building the wheels for PyBaMM on macOS (on both
# amd64 and arm64 architectures).

import os
import subprocess

homedir = os.path.expanduser("~")
libomp_path = os.path.join(homedir, ".local/lib/libomp.dylib")

subprocess.run(
    [
        "install_name_tool",
        "-change",
        "/usr/local/lib/libomp.dylib",
        libomp_path,
        libomp_path,
    ]
)
