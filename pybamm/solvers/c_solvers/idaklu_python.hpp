#ifndef PYBAMM_IDAKLU_PYTHON_HPP
#define PYBAMM_IDAKLU_PYTHON_HPP

#include <idas/idas.h>                 /* prototypes for IDAS fcts., consts.    */
#include <nvector/nvector_serial.h>  /* access to serial N_Vector            */
#include <sundials/sundials_math.h>  /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sundials/sundials_types.h> /* defs. of realtype, sunindextype      */
#include <sunlinsol/sunlinsol_klu.h> /* access to KLU linear solver          */
#include <sunmatrix/sunmatrix_sparse.h> /* access to sparse SUNMatrix           */

#include <pybind11/numpy.h>

namespace py = pybind11;
using np_array = py::array_t<realtype>;

#endif // PYBAMM_IDAKLU_PYTHON_HPP
