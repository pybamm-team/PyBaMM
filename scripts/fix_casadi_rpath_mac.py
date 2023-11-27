"""
Removes the rpath from libcasadi.dylib and libcasadi.3.7.dylib in the casadi python
install and uses a fixed path

Used when building the wheels for macOS
"""
import casadi
import os
import subprocess

casadi_dir = casadi.__path__[0]
print("Removing rpath references in python casadi install at", casadi_dir)

libcpp_name = "libc++.1.0.dylib"
libcppabi_name = "libc++abi.dylib"
libcasadi_name = "libcasadi.dylib"
libcasadi_37_name = "libcasadi.3.7.dylib"

install_name_tool_args_for_libcasadi_name = [
    "-change",
    os.path.join("@rpath", libcpp_name),
    os.path.join(casadi_dir, libcpp_name),
    os.path.join(casadi_dir, libcasadi_name),
]

install_name_tool_args_for_libcasadi_37_name = [
    "-change",
    os.path.join("@rpath", libcpp_name),
    os.path.join(casadi_dir, libcpp_name),
    os.path.join(casadi_dir, libcasadi_37_name),
]

subprocess.run(["otool", "-L", os.path.join(casadi_dir, libcasadi_name)])

print(" ".join(["install_name_tool", *install_name_tool_args_for_libcasadi_name]))
subprocess.run(["install_name_tool", *install_name_tool_args_for_libcasadi_name])

print(" ".join(["install_name_tool", *install_name_tool_args_for_libcasadi_37_name]))
subprocess.run(["install_name_tool", *install_name_tool_args_for_libcasadi_37_name])

subprocess.run(["otool", "-L", os.path.join(casadi_dir, libcasadi_name)])

install_name_tool_args = [
    "-change",
    os.path.join("@rpath", libcppabi_name),
    os.path.join(casadi_dir, libcppabi_name),
    os.path.join(casadi_dir, libcpp_name),
]
subprocess.run(["otool", "-L", os.path.join(casadi_dir, libcpp_name)])

print(" ".join(["install_name_tool", *install_name_tool_args]))
subprocess.run(["install_name_tool", *install_name_tool_args])

subprocess.run(["otool", "-L", os.path.join(casadi_dir, libcpp_name)])

# Copy libcasadi.3.7.dylib and libc++.1.0.dylib to LD_LIBRARY_PATH
# This is needed for the casadi python bindings to work while repairing the wheel

subprocess.run(
    ["cp",
    os.path.join(casadi_dir, libcasadi_37_name),
    os.path.join(os.getenv("HOME"),".local/lib")
    ]
)

subprocess.run(
    ["cp",
    os.path.join(casadi_dir, libcpp_name),
    os.path.join(os.getenv("HOME"),".local/lib")
    ]
)
