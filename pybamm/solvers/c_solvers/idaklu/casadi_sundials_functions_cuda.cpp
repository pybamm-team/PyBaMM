#include "casadi_sundials_functions_cuda.hpp"
#include "casadi_functions.hpp"
#include "common.hpp"

//#define NV_DATA(x) NV_DATA_S(x)
#define NV_DATA(x) N_VGetArrayPointer(x)
//#define NV_DATA_HOST(x) N_VGetHostArrayPointer_Cuda(x)
//#define NV_DATA_DEVICE(x) N_VGetDeviceArrayPointer_Cuda(x)
//#define NV_DATA(x) NV_DATA_DEVICE(x)

int residual_casadi_cuda(
  realtype tres,
  N_Vector yy,
  N_Vector yp,
  N_Vector rr,
  void *user_data)
{
  DEBUG("residual_casadi");

  CasadiFunctions *p_python_functions =
    static_cast<CasadiFunctions *>(user_data);

  realtype *yyd = N_VGetArrayPointer(yy);
  realtype *ypd = N_VGetArrayPointer(yp);
  realtype *rrd = N_VGetArrayPointer(rr);

  p_python_functions->rhs_alg.m_arg[0] = &tres;
  p_python_functions->rhs_alg.m_arg[1] = yyd;
  p_python_functions->rhs_alg.m_arg[2] = p_python_functions->inputs.data();
  p_python_functions->rhs_alg.m_res[0] = rrd;
  p_python_functions->rhs_alg();

  realtype *tmp = p_python_functions->get_tmp_state_vector();
  p_python_functions->mass_action.m_arg[0] = ypd;
  p_python_functions->mass_action.m_res[0] = tmp;
  p_python_functions->mass_action();

  // AXPY: y <- a*x + y
  const int ns = p_python_functions->number_of_states;
  casadi::casadi_axpy(ns, -1., tmp, rrd); 
  
  DEBUG_VECTORn(yy, 5);
  DEBUG_VECTORn(yp, 5);
  DEBUG_VECTORn(rr, 5);
  
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
int residual_casadi_approx_cuda(
  sunindextype Nlocal,
  realtype tt,
  N_Vector yy,
  N_Vector yp,
  N_Vector gval,
  void *user_data)
{
  DEBUG("residual_casadi_approx");

  // Just use true residual for now
  int result = residual_casadi_cuda(tt, yy, yp, gval, user_data);
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
int jtimes_casadi_cuda(
  realtype tt,
  N_Vector yy,
  N_Vector yp,
  N_Vector rr,
  N_Vector v,
  N_Vector Jv,
  realtype cj,
  void *user_data,
  N_Vector tmp1,
  N_Vector tmp2)
{
  DEBUG("jtimes_casadi");
  CasadiFunctions *p_python_functions =
    static_cast<CasadiFunctions *>(user_data);

  // Jv has ∂F/∂y v
  p_python_functions->jac_action.m_arg[0] = &tt;
  p_python_functions->jac_action.m_arg[1] = NV_DATA(yy);
  p_python_functions->jac_action.m_arg[2] = p_python_functions->inputs.data();
  p_python_functions->jac_action.m_arg[3] = NV_DATA(v);
  p_python_functions->jac_action.m_res[0] = NV_DATA(Jv);
  p_python_functions->jac_action();

  // tmp has -∂F/∂y˙ v
  realtype *tmp = p_python_functions->get_tmp_state_vector();
  p_python_functions->mass_action.m_arg[0] = NV_DATA(v);
  p_python_functions->mass_action.m_res[0] = tmp;
  p_python_functions->mass_action();

  // AXPY: y <- a*x + y
  // Jv has ∂F/∂y v + cj ∂F/∂y˙ v
  const int ns = p_python_functions->number_of_states;
  casadi::casadi_axpy(ns, -cj, tmp, NV_DATA(Jv));

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
int jacobian_casadi_cuda(
  realtype tt,
  realtype cj,
  N_Vector yy,
  N_Vector yp,
  N_Vector resvec,
  SUNMatrix JJ,
  void *user_data,
  N_Vector tempv1,
  N_Vector tempv2,
  N_Vector tempv3)
{
  DEBUG("jacobian_casadi");
  return 0;

  CasadiFunctions *p_python_functions =
    static_cast<CasadiFunctions *>(user_data);
  
  // create pointer to jac data, column pointers, and row values
  realtype jac_data[SUNMatrix_cuSparse_NNZ(JJ)];
  sunindextype jac_ptrs[SUNMatrix_cuSparse_BlockRows(JJ)+1];
  sunindextype jac_vals[SUNMatrix_cuSparse_BlockNNZ(JJ)];
//  if (SUNMatrix_cuSparse_CopyFromDevice(JJ, jac_data, jac_colptrs, jac_rowvals))
//    throw std::runtime_error("SUNMatrix_cuSparse_CopyFromDevice: Failed");

  if (p_python_functions->options.using_sparse_matrix)
  {
    // CSR
    DEBUG("CSR");

    realtype newjac[SUNMatrix_cuSparse_NNZ(JJ)];
    //sunindextype *jac_ptrs = SUNSparseMatrix_IndexPointers(JJ);
    //sunindextype *jac_vals = SUNSparseMatrix_IndexValues(JJ);
    
    // args are t, y, cj, put result in jacobian data matrix
    p_python_functions->jac_times_cjmass.m_arg[0] = &tt;
    p_python_functions->jac_times_cjmass.m_arg[1] = NV_DATA(yy);
    p_python_functions->jac_times_cjmass.m_arg[2] =
      p_python_functions->inputs.data();
    p_python_functions->jac_times_cjmass.m_arg[3] = &cj;
    p_python_functions->jac_times_cjmass.m_res[0] = newjac;
    p_python_functions->jac_times_cjmass();

    // convert (casadi's) CSC format to CSR
    csc_csr<long, int>(
      newjac,
      p_python_functions->jac_times_cjmass_rowvals.data(),
      p_python_functions->jac_times_cjmass_colptrs.data(),
      jac_data,
      jac_ptrs,
      jac_vals,
      SUNMatrix_cuSparse_NNZ(JJ),
      SUNMatrix_cuSparse_BlockRows(JJ)
    );
  }
  else
    throw std::runtime_error("Invalid matrix type provided.");
  
  // Copy to device
  SUNMatrix_cuSparse_CopyToDevice(
    JJ,
    jac_data,
    jac_ptrs,
    jac_vals
  );
  cudaDeviceSynchronize();

  return 0;
}

int events_casadi_cuda(
  realtype t,
  N_Vector yy,
  N_Vector yp,
  realtype *events_ptr,
  void *user_data)
{
  DEBUG("events_casadi");
  CasadiFunctions *p_python_functions =
    static_cast<CasadiFunctions *>(user_data);

  // args are t, y, put result in events_ptr
  p_python_functions->events.m_arg[0] = &t;
  p_python_functions->events.m_arg[1] = NV_DATA(yy);
  p_python_functions->events.m_arg[2] = p_python_functions->inputs.data();
  p_python_functions->events.m_res[0] = events_ptr;
  p_python_functions->events();

  return 0;
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
int sensitivities_casadi_cuda(
  int Ns,
  realtype t,
  N_Vector yy,
  N_Vector yp,
  N_Vector resval,
  N_Vector *yS,
  N_Vector *ypS,
  N_Vector *resvalS,
  void *user_data,
  N_Vector tmp1,
  N_Vector tmp2,
  N_Vector tmp3)
{
  DEBUG("sensitivities_casadi");
  CasadiFunctions *p_python_functions =
    static_cast<CasadiFunctions *>(user_data);

  const int np = p_python_functions->number_of_parameters;

  // args are t, y put result in rr
  p_python_functions->sens.m_arg[0] = &t;
  p_python_functions->sens.m_arg[1] = NV_DATA(yy);
  p_python_functions->sens.m_arg[2] = p_python_functions->inputs.data();
  for (int i = 0; i < np; i++)
  {
    p_python_functions->sens.m_res[i] = NV_DATA(resvalS[i]);
  }
  // resvalsS now has (∂F/∂p i )
  p_python_functions->sens();

  for (int i = 0; i < np; i++)
  {
    // put (∂F/∂y)s i (t) in tmp
    realtype *tmp = p_python_functions->get_tmp_state_vector();
    p_python_functions->jac_action.m_arg[0] = &t;
    p_python_functions->jac_action.m_arg[1] = NV_DATA(yy);
    p_python_functions->jac_action.m_arg[2] = p_python_functions->inputs.data();
    p_python_functions->jac_action.m_arg[3] = NV_DATA(yS[i]);
    p_python_functions->jac_action.m_res[0] = tmp;
    p_python_functions->jac_action();

    const int ns = p_python_functions->number_of_states;
    casadi::casadi_axpy(ns, 1., tmp, NV_DATA(resvalS[i]));

    // put -(∂F/∂ ẏ) ṡ i (t) in tmp2
    p_python_functions->mass_action.m_arg[0] = NV_DATA(ypS[i]);
    p_python_functions->mass_action.m_res[0] = tmp;
    p_python_functions->mass_action();

    // (∂F/∂y)s i (t)+(∂F/∂ ẏ) ṡ i (t)+(∂F/∂p i )
    // AXPY: y <- a*x + y
    casadi::casadi_axpy(ns, -1., tmp, NV_DATA(resvalS[i]));
  }

  return 0;
}
