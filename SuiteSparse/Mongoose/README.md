# Mongoose

[![Build Status](https://travis-ci.com/ScottKolo/Mongoose.svg?token=EK93uAGLjknx2p216TUE&branch=edgesep)](https://travis-ci.com/ScottKolo/Mongoose) [![codecov](https://codecov.io/gh/ScottKolo/Mongoose/branch/edgesep/graph/badge.svg?token=s3KMuP6lOp)](https://codecov.io/gh/ScottKolo/Mongoose)


Mongoose is a graph partitioning library. Currently, Mongoose only supports 
edge partitioning, but in the future a vertex separator extension will be added.

## Prerequisites and Dependencies

Mongoose requires CMake 2.8 and any ISO/IEC 14882:1998 compliant C++ compiler. Mongoose has been tested to work with GNU GCC 4.4+ and LLVM Clang 3.5+ on Linux, and Apple Xcode 6.4+ on macOS.

## Installation

Mongoose uses CMake. To build Mongoose, follow the commands below:

```shell
git clone https://github.com/ScottKolo/Mongoose
cd Mongoose
make         # Builds Mongoose (uses CMake) and runs the demo
```

Then to install, do

```shell
sudo make install 
```

After compilation, the Mongoose demo can be run from the `build` directory using `./bin/demo`.

## Usage

You can use Mongoose in one of three ways:

1. **The `mongoose` executable.** Once built, the `mongoose` executable will be located in `build/bin/mongoose`. This executable can read a Matrix Market file containing an adjacency matrix and output timing and partitioning information to a plain-text file. Simply call it with the following syntax: `mongoose <MM-input-file.mtx> [output-file]`
2. **The C++ API.** A static library is built at `Lib/libmongoose.a`. Include the header file `Include/Mongoose.hpp` and link to this library to access the C++ API.
3. **The MATLAB API.** From MATLAB, navigate to the `Mongoose/MATLAB/` directory and build the Mongoose MEX functions by calling `mongoose_make`. This will build Mongoose, run a demo, and allows access to the MATLAB API.

For more details about the specific APIs and their available functionality, see the Mongoose user guide located at [`Doc/Mongoose_UserGuide.pdf`](Doc/Mongoose_UserGuide.pdf).


## Credits

The following people have made significant contributions to Mongoose:

* Nuri Yeralan, Microsoft Research
* Scott Kolodziej, Texas A&M University
* Tim Davis, Texas A&M University
* William Hager, University of Florida

## License

Mongoose is licensed under the GNU Public License, version 3. For commercial license inquiries, please contact Dr. Tim Davis at davis@tamu.edu. The specific text of the license can be found in [`Doc/License.txt`](Doc/License.txt).
