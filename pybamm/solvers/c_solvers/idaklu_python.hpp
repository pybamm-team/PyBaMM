#ifndef PYBAMM_IDAKLU_PYTHON_HPP
#define PYBAMM_IDAKLU_PYTHON_HPP

#include <idas/idas.h>                 /* prototypes for IDAS fcts., consts.    */
#include <nvector/nvector_serial.h>  /* access to serial N_Vector            */
#include <sundials/sundials_math.h>  /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sundials/sundials_types.h> /* defs. of realtype, sunindextype      */
#include <sunlinsol/sunlinsol_klu.h> /* access to KLU linear solver          */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense linear solver          */
#include <sunmatrix/sunmatrix_sparse.h> /* access to sparse SUNMatrix           */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix           */

#include <pybind11/numpy.h>

namespace py = pybind11;
using np_array = py::array_t<realtype>;
using np_array_dense = py::array_t<realtype, py::array::c_style | py::array::forcecast>;
using np_array_int = py::array_t<int64_t>;


#endif // PYBAMM_IDAKLU_PYTHON_HPP
