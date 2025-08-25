#ifndef PYBAMM_IDAKLU_COMMON_HPP
#define PYBAMM_IDAKLU_COMMON_HPP

#include <iostream>
#include <limits>

#include <idas/idas.h>                 /* prototypes for IDAS fcts., consts.    */
#include <idas/idas_bbdpre.h>         /* access to IDABBDPRE preconditioner          */

#include <nvector/nvector_serial.h>  /* access to serial N_Vector            */
#include <nvector/nvector_openmp.h>  /* access to openmp N_Vector            */
#include <sundials/sundials_math.h>  /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sundials/sundials_config.h>  /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sundials/sundials_types.h> /* defs. of sunrealtype, sunindextype      */

#include <sundials/sundials_context.h>

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
// note: we rely on c_style ordering for numpy arrays so don't change this!
using np_array = py::array_t<sunrealtype, py::array::c_style | py::array::forcecast>;
using np_array_realtype = py::array_t<sunrealtype>;
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
void csc_csr(const sunrealtype f[], const T1 c[], const T1 r[], sunrealtype nf[], T2 nc[], T2 nr[], int N, int cols) {
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


/**
 * @brief Utility function to convert numpy array to std::vector<sunrealtype>
 */
std::vector<sunrealtype> numpy2sunrealtype(const np_array& input_np);

/**
 * @brief Utility function to compute the set difference of two vectors
 */
template <typename T1, typename T2>
std::vector<sunrealtype> setDiff(const T1 a_begin, const T1 a_end, const T2 b_begin, const T2 b_end) {
    std::vector<sunrealtype> result;
    if (std::distance(a_begin, a_end) > 0) {
      std::set_difference(a_begin, a_end, b_begin, b_end, std::back_inserter(result));
    }
    return result;
}

/**
 * @brief Utility function to make a sorted and unique vector
 */
template <typename T>
std::vector<sunrealtype> makeSortedUnique(const T input_begin, const T input_end) {
    std::unordered_set<sunrealtype> uniqueSet(input_begin, input_end); // Remove duplicates
    std::vector<sunrealtype> uniqueVector(uniqueSet.begin(), uniqueSet.end()); // Convert to vector
    std::sort(uniqueVector.begin(), uniqueVector.end()); // Sort the vector
    return uniqueVector;
}

std::vector<sunrealtype> makeSortedUnique(const np_array& input_np);

/**
 * @brief Apply a small perturbation to a time value to avoid roundoff errors
 */
inline sunrealtype perturb_time(const sunrealtype t, bool increasing) {
  const sunrealtype eps = std::numeric_limits<sunrealtype>::epsilon();
  const sunrealtype delta = SUNRsqrt(eps);
  const sunrealtype sign = increasing ? SUN_RCONST(1.0) : -SUN_RCONST(1.0);
  // Relative nudge ensures progress away from t, absolute nudge covers t == 0
  return (SUN_RCONST(1.0) + delta) * t + delta * sign;
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

#define DEBUG_v(v, N) {\
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
