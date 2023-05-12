#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_HPP

#include <casadi/casadi.hpp>
using Function = casadi::Function;

#include "casadi_functions.hpp"
#include "common.hpp"
#include "options.hpp"
#include "solution.hpp"

class CasadiSolver
{
public:
  CasadiSolver(np_array atol_np, double rel_tol, np_array rhs_alg_id,
               int number_of_parameters, int number_of_events,
               int jac_times_cjmass_nnz, int jac_bandwidth_lower, int jac_bandwidth_upper,
               std::unique_ptr<CasadiFunctions> functions, const Options& options);
  ~CasadiSolver();

  void *ida_mem; // pointer to memory

#if SUNDIALS_VERSION_MAJOR >= 6
  SUNContext sunctx;
#else
  // Wrap older sundials function calls in new-style wrappers
  void* IDACreate(SUNContext sunctx)
  {
    return IDACreate();
  }

  N_Vector N_VNew_Serial(sunindextype vec_length, SUNContext sunctx)
  {
    return N_VNew_Serial(vec_length);
  }

  SUNMatrix SUNSparseMatrix(sunindextype M, sunindextype N, sunindextype NNZ, int sparsetype, SUNContext sunctx)
  {
    return SUNMatrix SUNSparseMatrix(M, N, NNZ, sparsetype);
  }

  SUNMatrix SUNBandMatrix(sunindextype N, sunindextype mu, sunindextype ml, SUNContext sunctx)
  {
    return SUNMatrix SUNBandMatrix(N, mu, ml);
  }

  SUNMatrix SUNDenseMatrix(sunindextype M, sunindextype N, SUNContext sunctx)
  {
    return SUNDenseMatrix(M, N, sunctx);
  }

  SUNLinearSolver SUNLinSol_Dense(N_Vector y, SUNMatrix A, SUNContext sunctx)
  {
    return SUNLinSol_Dense(y, A, sunctx);
  }

  SUNLinearSolver SUNLinSol_KLU(N_Vector y, SUNMatrix A, SUNContext sunctx)
  {
    return SUNLinSol_KLU(y, A, sunctx);
  }

  SUNLinearSolver SUNLinSol_Band(N_Vector y, SUNMatrix A, SUNContext sunctx)
  {
    return SUNLinSol_Band(y, A, sunctx);
  }

  SUNLinearSolver SUNLinSol_SPBCGS(N_Vector y, int pretype, int maxl, SUNContext sunctx)
  {
    return SUNLinSol_SPBCGS(y, pretype, maxl);
  }

  SUNLinearSolver SUNLinSol_SPFGMR(N_Vector y, int pretype, int maxl, SUNContext sunctx)
  {
    return SUNLinSol_SPFGMR(y, pretype, maxl);
  }

  SUNLinearSolver SUNLinSol_SPGMR(N_Vector y, int pretype, int maxl, SUNContext sunctx)
  {
    return SUNLinSol_SPGMR(y, pretype, maxl);
  }

  SUNLinearSolver SUNLinSol_SPTFQMR(N_Vector y, int pretype, int maxl, SUNContext sunctx)
  {
    return SUNLinSol_SPTFQMR(y, pretype, maxl);
  }
#endif

  int number_of_states;
  int number_of_parameters;
  int number_of_events;
  N_Vector yy, yp, avtol; // y, y', and absolute tolerance
  N_Vector *yyS, *ypS;    // y, y' for sensitivities
  N_Vector id;            // rhs_alg_id
  realtype rtol;
  const int jac_times_cjmass_nnz;

  SUNMatrix J;
  SUNLinearSolver LS;

  std::unique_ptr<CasadiFunctions> functions;
  Options options;

  Solution solve(np_array t_np, np_array y0_np, np_array yp0_np,
                 np_array_dense inputs);
};

CasadiSolver *
create_casadi_solver(int number_of_states, int number_of_parameters,
                     const Function &rhs_alg, const Function &jac_times_cjmass,
                     const np_array_int &jac_times_cjmass_colptrs,
                     const np_array_int &jac_times_cjmass_rowvals,
                     const int jac_times_cjmass_nnz, 
                     const int jac_bandwidth_lower, const int jac_bandwidth_upper, 
                     const Function &jac_action,
                     const Function &mass_action, const Function &sens,
                     const Function &event, const int number_of_events,
                     np_array rhs_alg_id, np_array atol_np,
                     double rel_tol, int inputs_length, py::dict options);

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_HPP
