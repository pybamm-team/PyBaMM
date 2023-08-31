#ifndef PYBAMM_IDAKLU_COMMON_HPP
#define PYBAMM_IDAKLU_COMMON_HPP

#include <idas/idas.h>                 /* prototypes for IDAS fcts., consts.    */
#include <idas/idas_bbdpre.h>         /* access to IDABBDPRE preconditioner          */

#include <nvector/nvector_serial.h>  /* access to serial N_Vector            */
#include <nvector/nvector_openmp.h>  /* access to openmp N_Vector            */
#include <sundials/sundials_math.h>  /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sundials/sundials_config.h>  /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sundials/sundials_types.h> /* defs. of realtype, sunindextype      */


#if SUNDIALS_VERSION_MAJOR >= 6
  #include <sundials/sundials_context.h>
#endif

#include <sunlinsol/sunlinsol_klu.h> /* access to KLU linear solver          */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense linear solver          */
#include <sunlinsol/sunlinsol_band.h> /* access to dense linear solver          */
#include <sunlinsol/sunlinsol_spbcgs.h> /* access to spbcgs iterative linear solver          */
#include <sunlinsol/sunlinsol_spfgmr.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunlinsol/sunlinsol_sptfqmr.h>

#include <sunmatrix/sunmatrix_sparse.h> /* access to sparse SUNMatrix           */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix           */



#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using np_array = py::array_t<realtype>;
using np_array_dense = py::array_t<realtype, py::array::c_style | py::array::forcecast>;
using np_array_int = py::array_t<int64_t>;

#ifdef NDEBUG
#define DEBUG(x)
#else
#define DEBUG(x) do { std::cerr << __FILE__ << ':' << __LINE__ << ' ' << x << std::endl; } while (0)
#endif

#ifdef NDEBUG
#define DEBUG_VECTOR(vector)
#define DEBUG_VECTORn(vector)
#else

#define DEBUG_VECTORn(vector, N) {\
  std::cout << #vector << "[n=" << N << "] = ["; \
  auto array_ptr = N_VGetArrayPointer(vector); \
  for (int i = 0; i < N; i++) { \
    std::cout << array_ptr[i]; \
    if (i < N-1) { \
      std::cout << ", "; \
    } \
  } \
  std::cout << "]" << std::endl;  }

#define DEBUG_VECTOR(vector) {\
  std::cout << #vector << " = ["; \
  auto array_ptr = N_VGetArrayPointer(vector); \
  auto N = N_VGetLength(vector); \
  for (int i = 0; i < N; i++) { \
    std::cout << array_ptr[i]; \
    if (i < N-1) { \
      std::cout << ", "; \
    } \
  } \
  std::cout << "]" << std::endl;  }

#define DEBUG_v(v, N) {\
  std::cout << #v << "[n=" << N << "] = ["; \
  for (int i = 0; i < N; i++) { \
    std::cout << v[i]; \
    if (i < N-1) { \
      std::cout << ", "; \
    } \
  } \
  std::cout << "]" << std::endl;  }

#endif

#endif // PYBAMM_IDAKLU_COMMON_HPP
