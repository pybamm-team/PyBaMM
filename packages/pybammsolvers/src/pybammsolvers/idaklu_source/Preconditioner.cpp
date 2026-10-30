#include "Preconditioner.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <nvector/nvector_serial.h>

namespace {

struct PreconditionerSolverContent {
  Preconditioner *preconditioner;
  sunindextype last_flag;
};

PreconditionerSolverContent *content(SUNLinearSolver S) {
  return static_cast<PreconditionerSolverContent *>(S->content);
}

SUNLinearSolver_Type get_type(SUNLinearSolver) { return SUNLINEARSOLVER_DIRECT; }

SUNLinearSolver_ID get_id(SUNLinearSolver) { return SUNLINEARSOLVER_CUSTOM; }

SUNErrCode initialize(SUNLinearSolver S) {
  content(S)->last_flag = 0;
  return SUN_SUCCESS;
}

int setup(SUNLinearSolver S, SUNMatrix A) {
  const int flag = content(S)->preconditioner->setup(SUNSparseMatrix_Data(A));
  content(S)->last_flag = flag;
  return flag == 0 ? SUN_SUCCESS : SUNLS_PACKAGE_FAIL_REC;
}

int solve(SUNLinearSolver S, SUNMatrix, N_Vector x, N_Vector b, sunrealtype) {
  content(S)->preconditioner->solve(N_VGetArrayPointer(b), N_VGetArrayPointer(x));
  content(S)->last_flag = 0;
  return SUN_SUCCESS;
}

sunindextype last_flag(SUNLinearSolver S) { return content(S)->last_flag; }

SUNErrCode free_solver(SUNLinearSolver S) {
  if (S == nullptr) return SUN_SUCCESS;
  delete content(S);
  S->content = nullptr;
  SUNLinSolFreeEmpty(S);
  return SUN_SUCCESS;
}

}  // namespace

SUNLinearSolver SUNLinSol_Preconditioner(Preconditioner *preconditioner, SUNContext sunctx) {
  SUNLinearSolver S = SUNLinSolNewEmpty(sunctx);
  if (S == nullptr) return nullptr;
  S->ops->gettype = get_type;
  S->ops->getid = get_id;
  S->ops->initialize = initialize;
  S->ops->setup = setup;
  S->ops->solve = solve;
  S->ops->lastflag = last_flag;
  S->ops->free = free_solver;
  S->content = new PreconditionerSolverContent{preconditioner, 0};
  return S;
}
