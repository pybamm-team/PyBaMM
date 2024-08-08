#ifndef PYBAMM_IDAKLU_COMMON_HPP
#define PYBAMM_IDAKLU_COMMON_HPP

#include <iostream>

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

/**
 * Utility function to convert compressed-sparse-column (CSC) to/from
 * compressed-sparse-row (CSR) matrix representation. Conversion is symmetric /
 * invertible using this function.
 * @brief Utility function to convert to/from CSC/CSR matrix representations.
 * @param f Data vector containing the sparse matrix elements
 * @param c Index pointer to column starts
 * @param r Array of row indices
 * @param nf New data vector that will contain the transformed sparse matrix
 * @param nc New array of column indices
 * @param nr New index pointer to row starts
 */
template<typename T1, typename T2>
void csc_csr(const realtype f[], const T1 c[], const T1 r[], realtype nf[], T2 nc[], T2 nr[], int N, int cols) {
  std::vector<int> nn(cols+1);
  std::vector<int> rr(N);
  for (int i=0; i<cols+1; i++)
    nc[i] = 0;

  for (int k = 0, i = 0; i < cols+1; i++) {
    for (int j = 0; j < r[i+1] - r[i]; j++) {
      if (k == N)  // SUNDIALS indexing does not include the count element
        break;
      rr[k++] = i;
    }
  }
  for (int i = 0; i < N; i++)
    nc[c[i]+1]++;
  for (int i = 1; i <= cols; i++)
    nc[i] += nc[i-1];
  for (int i = 0; i < cols+1; i++)
    nn[i] = nc[i];
  for (int i = 0; i < N; i++) {
    int x = nn[c[i]]++;
    nf[x] = f[i];
    nr[x] = rr[i];
  }
}

#ifdef NDEBUG
#define DEBUG_VECTOR(vector)
#define DEBUG_VECTORn(vector, N)
#define DEBUG_v(v, N)
#define DEBUG(x)
#define DEBUG_n(x)
#define ASSERT(x)
#else

#define DEBUG_VECTORn(vector, L) {\
  auto M = N_VGetLength(vector); \
  auto N = (M < L) ? M : L; \
  std::cout << #vector << "[" << N << " of " << M << "] = ["; \
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

#define DEBUG_v(v, M) {\
  int N = 2; \
  std::cout << #v << "[n=" << N << "] = ["; \
  for (int i = 0; i < N; i++) { \
    std::cout << v[i]; \
    if (i < N-1) { \
      std::cout << ", "; \
    } \
  } \
  std::cout << "]" << std::endl;  }

#define DEBUG(x) { \
    std::cerr << __FILE__ << ":" << __LINE__ << " " << x << std::endl; \
  }

#define DEBUG_n(x) { \
    std::cerr << __FILE__ << ":" << __LINE__ << "," << #x << " = " << x << std::endl; \
  }

#define ASSERT(x) { \
    if (!(x)) { \
      std::cerr << __FILE__ << ":" << __LINE__ << " Assertion failed: " << #x << std::endl; \
      throw std::runtime_error("Assertion failed: " #x); \
    } \
  }

#endif

#endif // PYBAMM_IDAKLU_COMMON_HPP
