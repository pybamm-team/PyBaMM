#include "casadi_sundials_functions.hpp"
#include "casadi_functions.hpp"
#include "common.hpp"

int residual_casadi(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr,
                    void *user_data)
{
  DEBUG("residual_casadi");
  CasadiFunctions *p_python_functions =
      static_cast<CasadiFunctions *>(user_data);

  p_python_functions->rhs_alg.m_arg[0] = &tres;
  p_python_functions->rhs_alg.m_arg[1] = NV_DATA_OMP(yy);
  p_python_functions->rhs_alg.m_arg[2] = p_python_functions->inputs.data();
  p_python_functions->rhs_alg.m_res[0] = NV_DATA_OMP(rr);
  p_python_functions->rhs_alg();

  realtype *tmp = p_python_functions->get_tmp_state_vector();
  p_python_functions->mass_action.m_arg[0] = NV_DATA_OMP(yp);
  p_python_functions->mass_action.m_res[0] = tmp;
  p_python_functions->mass_action();

  // AXPY: y <- a*x + y
  const int ns = p_python_functions->number_of_states;
  casadi::casadi_axpy(ns, -1., tmp, NV_DATA_OMP(rr));

  //DEBUG_VECTOR(yy);
  //DEBUG_VECTOR(yp);
  //DEBUG_VECTOR(rr);

  // now rr has rhs_alg(t, y) - mass_matrix * yp
  return 0;
}

// This Gres function computes G(t, y, yp). It loads the vector gval as a
// function of tt, yy, and yp.
//
// Arguments:
// Nlocal – is the local vector length.
//
// tt – is the value of the independent variable.
//
// yy – is the dependent variable.
//
// yp – is the derivative of the dependent variable.
//
// gval – is the output vector.
//
// user_data – is a pointer to user data, the same as the user_data parameter
// passed to IDASetUserData().
//
// Return value:
//
// An IDABBDLocalFn function type should return 0 to indicate success, 1 for a
// recoverable error, or -1 for a non-recoverable error.
//
// Notes:
//
// This function must assume that all inter-processor communication of data
// needed to calculate gval has already been done, and this data is accessible
// within user_data.
//
// The case where G is mathematically identical to F is allowed.
int residual_casadi_approx(sunindextype Nlocal, realtype tt, N_Vector yy,
                           N_Vector yp, N_Vector gval, void *user_data)
{
  DEBUG("residual_casadi_approx");

  // Just use true residual for now
  int result = residual_casadi(tt, yy, yp, gval, user_data);
  return result;
}

// Purpose This function computes the product Jv of the DAE system Jacobian J
// (or an approximation to it) and a given vector v, where J is defined by Eq.
// (2.6).
//    J = ∂F/∂y + cj ∂F/∂y˙
// Arguments tt is the current value of the independent variable.
//     yy is the current value of the dependent variable vector, y(t).
//     yp is the current value of ˙y(t).
//     rr is the current value of the residual vector F(t, y, y˙).
//     v is the vector by which the Jacobian must be multiplied to the right.
//     Jv is the computed output vector.
//     cj is the scalar in the system Jacobian, proportional to the inverse of
//     the step
//        size (α in Eq. (2.6) ).
//     user data is a pointer to user data, the same as the user data parameter
//     passed to
//        IDASetUserData.
//     tmp1
//     tmp2 are pointers to memory allocated for variables of type N Vector
//     which can
//        be used by IDALsJacTimesVecFn as temporary storage or work space.
int jtimes_casadi(realtype tt, N_Vector yy, N_Vector yp, N_Vector rr,
                  N_Vector v, N_Vector Jv, realtype cj, void *user_data,
                  N_Vector tmp1, N_Vector tmp2)
{
  DEBUG("jtimes_casadi");
  CasadiFunctions *p_python_functions =
      static_cast<CasadiFunctions *>(user_data);

  // Jv has ∂F/∂y v
  p_python_functions->jac_action.m_arg[0] = &tt;
  p_python_functions->jac_action.m_arg[1] = NV_DATA_OMP(yy);
  p_python_functions->jac_action.m_arg[2] = p_python_functions->inputs.data();
  p_python_functions->jac_action.m_arg[3] = NV_DATA_OMP(v);
  p_python_functions->jac_action.m_res[0] = NV_DATA_OMP(Jv);
  p_python_functions->jac_action();

  // tmp has -∂F/∂y˙ v
  realtype *tmp = p_python_functions->get_tmp_state_vector();
  p_python_functions->mass_action.m_arg[0] = NV_DATA_OMP(v);
  p_python_functions->mass_action.m_res[0] = tmp;
  p_python_functions->mass_action();

  // AXPY: y <- a*x + y
  // Jv has ∂F/∂y v + cj ∂F/∂y˙ v
  const int ns = p_python_functions->number_of_states;
  casadi::casadi_axpy(ns, -cj, tmp, NV_DATA_OMP(Jv));

  return 0;
}

// Arguments tt is the current value of the independent variable t.
//   cj is the scalar in the system Jacobian, proportional to the inverse of the
//   step
//     size (α in Eq. (2.6) ).
//   yy is the current value of the dependent variable vector, y(t).
//   yp is the current value of ˙y(t).
//   rr is the current value of the residual vector F(t, y, y˙).
//   Jac is the output (approximate) Jacobian matrix (of type SUNMatrix), J =
//     ∂F/∂y + cj ∂F/∂y˙.
//   user data is a pointer to user data, the same as the user data parameter
//   passed to
//     IDASetUserData.
//   tmp1
//   tmp2
//   tmp3 are pointers to memory allocated for variables of type N Vector which
//   can
//     be used by IDALsJacFn function as temporary storage or work space.
int jacobian_casadi(realtype tt, realtype cj, N_Vector yy, N_Vector yp,
                    N_Vector resvec, SUNMatrix JJ, void *user_data,
                    N_Vector tempv1, N_Vector tempv2, N_Vector tempv3)
{
  DEBUG("jacobian_casadi");

  CasadiFunctions *p_python_functions =
      static_cast<CasadiFunctions *>(user_data);

  // create pointer to jac data, column pointers, and row values
  realtype *jac_data;
  if (p_python_functions->options.using_sparse_matrix)
  {
    jac_data = SUNSparseMatrix_Data(JJ);
  }
  else if (p_python_functions->options.using_banded_matrix) {
    jac_data = p_python_functions->get_tmp_sparse_jacobian_data();
  }
  else
  {
    jac_data = SUNDenseMatrix_Data(JJ);
  }

  // args are t, y, cj, put result in jacobian data matrix
  p_python_functions->jac_times_cjmass.m_arg[0] = &tt;
  p_python_functions->jac_times_cjmass.m_arg[1] = NV_DATA_OMP(yy);
  p_python_functions->jac_times_cjmass.m_arg[2] =
      p_python_functions->inputs.data();
  p_python_functions->jac_times_cjmass.m_arg[3] = &cj;
  p_python_functions->jac_times_cjmass.m_res[0] = jac_data;

  p_python_functions->jac_times_cjmass();


  if (p_python_functions->options.using_banded_matrix) 
  {
    // copy data from temporary matrix to the banded matrix
    auto jac_colptrs = p_python_functions->jac_times_cjmass_colptrs.data();
    auto jac_rowvals = p_python_functions->jac_times_cjmass_rowvals.data();
    int ncols = p_python_functions->number_of_states;
    for (int col_ij = 0; col_ij < ncols; col_ij++) {
      realtype *banded_col = SM_COLUMN_B(JJ, col_ij);
      for (auto data_i = jac_colptrs[col_ij]; data_i < jac_colptrs[col_ij+1]; data_i++) {
        auto row_ij = jac_rowvals[data_i];
        const realtype value_ij = jac_data[data_i];
        DEBUG("(" << row_ij << ", " << col_ij << ") = " << value_ij);
        SM_COLUMN_ELEMENT_B(banded_col, row_ij, col_ij) = value_ij;
      }
    }
  } 
  else if (p_python_functions->options.using_sparse_matrix)
  {

    sunindextype *jac_colptrs = SUNSparseMatrix_IndexPointers(JJ);
    sunindextype *jac_rowvals = SUNSparseMatrix_IndexValues(JJ);
    // row vals and col ptrs
    const int n_row_vals = p_python_functions->jac_times_cjmass_rowvals.size();
    auto p_jac_times_cjmass_rowvals =
        p_python_functions->jac_times_cjmass_rowvals.data();

    // just copy across row vals (do I need to do this every time?)
    // (or just in the setup?)
    for (int i = 0; i < n_row_vals; i++)
    {
      jac_rowvals[i] = p_jac_times_cjmass_rowvals[i];
    }

    const int n_col_ptrs = p_python_functions->jac_times_cjmass_colptrs.size();
    auto p_jac_times_cjmass_colptrs =
        p_python_functions->jac_times_cjmass_colptrs.data();

    // just copy across col ptrs (do I need to do this every time?)
    for (int i = 0; i < n_col_ptrs; i++)
    {
      jac_colptrs[i] = p_jac_times_cjmass_colptrs[i];
    }
  }

  return (0);
}

int events_casadi(realtype t, N_Vector yy, N_Vector yp, realtype *events_ptr,
                  void *user_data)
{
  CasadiFunctions *p_python_functions =
      static_cast<CasadiFunctions *>(user_data);

  // args are t, y, put result in events_ptr
  p_python_functions->events.m_arg[0] = &t;
  p_python_functions->events.m_arg[1] = NV_DATA_OMP(yy);
  p_python_functions->events.m_arg[2] = p_python_functions->inputs.data();
  p_python_functions->events.m_res[0] = events_ptr;
  p_python_functions->events();

  return (0);
}

// This function computes the sensitivity residual for all sensitivity
// equations. It must compute the vectors
// (∂F/∂y)s i (t)+(∂F/∂ ẏ) ṡ i (t)+(∂F/∂p i ) and store them in resvalS[i].
// Ns is the number of sensitivities.
// t is the current value of the independent variable.
// yy is the current value of the state vector, y(t).
// yp is the current value of ẏ(t).
// resval contains the current value F of the original DAE residual.
// yS contains the current values of the sensitivities s i .
// ypS contains the current values of the sensitivity derivatives ṡ i .
// resvalS contains the output sensitivity residual vectors.
// Memory allocation for resvalS is handled within idas.
// user data is a pointer to user data.
// tmp1, tmp2, tmp3 are N Vectors of length N which can be used as
// temporary storage.
//
// Return value An IDASensResFn should return 0 if successful,
// a positive value if a recoverable error
// occurred (in which case idas will attempt to correct),
// or a negative value if it failed unrecoverably (in which case the integration
// is halted and IDA SRES FAIL is returned)
//
int sensitivities_casadi(int Ns, realtype t, N_Vector yy, N_Vector yp,
                         N_Vector resval, N_Vector *yS, N_Vector *ypS,
                         N_Vector *resvalS, void *user_data, N_Vector tmp1,
                         N_Vector tmp2, N_Vector tmp3)
{

  DEBUG("sensitivities_casadi");
  CasadiFunctions *p_python_functions =
      static_cast<CasadiFunctions *>(user_data);

  const int np = p_python_functions->number_of_parameters;

  // args are t, y put result in rr
  p_python_functions->sens.m_arg[0] = &t;
  p_python_functions->sens.m_arg[1] = NV_DATA_OMP(yy);
  p_python_functions->sens.m_arg[2] = p_python_functions->inputs.data();
  for (int i = 0; i < np; i++)
  {
    p_python_functions->sens.m_res[i] = NV_DATA_OMP(resvalS[i]);
  }
  // resvalsS now has (∂F/∂p i )
  p_python_functions->sens();
  
  for (int i = 0; i < np; i++)
  {
    // put (∂F/∂y)s i (t) in tmp
    realtype *tmp = p_python_functions->get_tmp_state_vector();
    p_python_functions->jac_action.m_arg[0] = &t;
    p_python_functions->jac_action.m_arg[1] = NV_DATA_OMP(yy);
    p_python_functions->jac_action.m_arg[2] = p_python_functions->inputs.data();
    p_python_functions->jac_action.m_arg[3] = NV_DATA_OMP(yS[i]);
    p_python_functions->jac_action.m_res[0] = tmp;
    p_python_functions->jac_action();

    const int ns = p_python_functions->number_of_states;
    casadi::casadi_axpy(ns, 1., tmp, NV_DATA_OMP(resvalS[i]));

    // put -(∂F/∂ ẏ) ṡ i (t) in tmp2
    p_python_functions->mass_action.m_arg[0] = NV_DATA_OMP(ypS[i]);
    p_python_functions->mass_action.m_res[0] = tmp;
    p_python_functions->mass_action();

    // (∂F/∂y)s i (t)+(∂F/∂ ẏ) ṡ i (t)+(∂F/∂p i )
    // AXPY: y <- a*x + y
    casadi::casadi_axpy(ns, -1., tmp, NV_DATA_OMP(resvalS[i]));
  }

  return 0;
}
