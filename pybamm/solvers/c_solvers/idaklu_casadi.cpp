#include "idaklu_casadi.hpp"
#include "idaklu_python.hpp"

#include <vector>

#include <iostream>
using casadi::casadi_axpy;

class CasadiFunction {
public:
  CasadiFunction(const Function &f):m_func(f) {
    size_t sz_arg;
    size_t sz_res;
    size_t sz_iw;
    size_t sz_w;
    m_func.sz_work(sz_arg, sz_res, sz_iw, sz_w);
    // std::cout << "name = "<< m_func.name() << " arg = " << sz_arg << " res = " << sz_res << " iw = " << sz_iw << " w = " << sz_w << std::endl;
    m_arg.resize(sz_arg);
    m_res.resize(sz_res);
    m_iw.resize(sz_iw);
    m_w.resize(sz_w);
  }

  // only call this once m_arg and m_res have been set appropriatelly
  void operator()() {
    int mem = m_func.checkout();
    m_func(m_arg.data(), m_res.data(), m_iw.data(), m_w.data(), mem);
    m_func.release(mem);
  }

public:
  std::vector<const double *> m_arg;
  std::vector<double *> m_res;

private:
  const Function &m_func;
  std::vector<casadi_int> m_iw;
  std::vector<double> m_w;
};


class PybammFunctions {
public:
  int number_of_states;
  int number_of_parameters;
  int number_of_events;
  CasadiFunction rhs_alg;
  CasadiFunction sens;
  CasadiFunction jac_times_cjmass;
  const np_array &jac_times_cjmass_rowvals;
  const np_array &jac_times_cjmass_colptrs;
  CasadiFunction jac_action;
  CasadiFunction mass_action;
  CasadiFunction events;

  PybammFunctions(const Function &rhs_alg, 
                  const Function &jac_times_cjmass,
                  const np_array &jac_times_cjmass_rowvals,
                  const np_array &jac_times_cjmass_colptrs,
                  const Function &jac_action,
                  const Function &mass_action,
                  const Function &sens,
                  const Function &events, 
                  const int n_s, int n_e, const int n_p)
      : number_of_states(n_s), number_of_events(n_e), 
        number_of_parameters(n_p),
        rhs_alg(rhs_alg), 
        jac_times_cjmass(jac_times_cjmass), 
        jac_times_cjmass_rowvals(jac_times_cjmass_rowvals), 
        jac_times_cjmass_colptrs(jac_times_cjmass_colptrs), 
        jac_action(jac_action),
        mass_action(mass_action),
        sens(sens),
        events(events),
        tmp(number_of_states)
  {}

  realtype *get_tmp() {
    return tmp.data();
  }

private:
  std::vector<realtype> tmp;
};

int residual_casadi(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr,
             void *user_data)
{
  PybammFunctions *p_python_functions =
      static_cast<PybammFunctions *>(user_data);

  //std::cout << "RESIDUAL t = " << tres << " y = [";
  //for (int i = 0; i < p_python_functions->number_of_states; i++) {
  //  std::cout << NV_DATA_S(yy)[i] << " ";
  //}
  //std::cout << "] yp = [";
  //for (int i = 0; i < p_python_functions->number_of_states; i++) {
  //  std::cout << NV_DATA_S(yp)[i] << " ";
  //}
  //std::cout << "]" << std::endl;
  // args are t, y, put result in rr
  p_python_functions->rhs_alg.m_arg[0] = &tres;
  p_python_functions->rhs_alg.m_arg[1] = NV_DATA_S(yy);
  p_python_functions->rhs_alg.m_res[0] = NV_DATA_S(rr);
  p_python_functions->rhs_alg();

  //std::cout << "rhs_alg = [";
  //for (int i = 0; i < p_python_functions->number_of_states; i++) {
  //  std::cout << NV_DATA_S(rr)[i] << " ";
  //}
  //std::cout << "]" << std::endl;

  realtype *tmp = p_python_functions->get_tmp();
  // args is yp, put result in tmp
  p_python_functions->mass_action.m_arg[0] = NV_DATA_S(yp);
  p_python_functions->mass_action.m_res[0] = tmp;
  p_python_functions->mass_action();

  //std::cout << "tmp = [";
  //for (int i = 0; i < p_python_functions->number_of_states; i++) {
  //  std::cout << tmp[i] << " ";
  //}
  //std::cout << "]" << std::endl;

  // AXPY: y <- a*x + y
  const int ns = p_python_functions->number_of_states;
  casadi_axpy(ns, -1., tmp, NV_DATA_S(rr));

  //std::cout << "residual = [";
  //for (int i = 0; i < p_python_functions->number_of_states; i++) {
  //  std::cout << NV_DATA_S(rr)[i] << " ";
  //}
  //std::cout << "]" << std::endl;

  // now rr has rhs_alg(t, y) - mass_matrix * yp

  return 0;
}

// Purpose This function computes the product Jv of the DAE system Jacobian J 
// (or an approximation to it) and a given vector v, where J is defined by Eq. (2.6).
//    J = ∂F/∂y + cj ∂F/∂y˙
// Arguments tt is the current value of the independent variable.
//     yy is the current value of the dependent variable vector, y(t).
//     yp is the current value of ˙y(t).
//     rr is the current value of the residual vector F(t, y, y˙).
//     v is the vector by which the Jacobian must be multiplied to the right.
//     Jv is the computed output vector.
//     cj is the scalar in the system Jacobian, proportional to the inverse of the step
//        size (α in Eq. (2.6) ).
//     user data is a pointer to user data, the same as the user data parameter passed to
//        IDASetUserData.
//     tmp1
//     tmp2 are pointers to memory allocated for variables of type N Vector which can
//        be used by IDALsJacTimesVecFn as temporary storage or work space.
int jtimes_casadi(realtype tt, N_Vector yy, N_Vector yp, N_Vector rr,
           N_Vector v, N_Vector Jv, realtype cj, void *user_data,
           N_Vector tmp1, N_Vector tmp2) {
  PybammFunctions *p_python_functions =
      static_cast<PybammFunctions *>(user_data);

  // rr has ∂F/∂y v
  p_python_functions->jac_action.m_arg[0] = &tt;
  p_python_functions->jac_action.m_arg[1] = NV_DATA_S(yy);
  p_python_functions->jac_action.m_arg[2] = &cj;
  p_python_functions->jac_action.m_arg[3] = NV_DATA_S(v);
  p_python_functions->jac_action.m_res[0] = NV_DATA_S(rr);
  p_python_functions->jac_action();

  // tmp1 has -∂F/∂y˙ v
  p_python_functions->mass_action.m_arg[0] = NV_DATA_S(v);
  p_python_functions->mass_action.m_res[0] = NV_DATA_S(tmp1);
  p_python_functions->mass_action();

  // AXPY: y <- a*x + y
  // rr has ∂F/∂y v + cj ∂F/∂y˙ v
  const int ns = p_python_functions->number_of_states;
  casadi_axpy(ns, -cj, NV_DATA_S(tmp1), NV_DATA_S(rr));

  return 0;
}


// Arguments tt is the current value of the independent variable t.
//   cj is the scalar in the system Jacobian, proportional to the inverse of the step
//     size (α in Eq. (2.6) ).
//   yy is the current value of the dependent variable vector, y(t).
//   yp is the current value of ˙y(t).
//   rr is the current value of the residual vector F(t, y, y˙).
//   Jac is the output (approximate) Jacobian matrix (of type SUNMatrix), J =
//     ∂F/∂y + cj ∂F/∂y˙.
//   user data is a pointer to user data, the same as the user data parameter passed to
//     IDASetUserData.
//   tmp1
//   tmp2
//   tmp3 are pointers to memory allocated for variables of type N Vector which can
//     be used by IDALsJacFn function as temporary storage or work space.
int jacobian_casadi(realtype tt, realtype cj, N_Vector yy, N_Vector yp,
             N_Vector resvec, SUNMatrix JJ, void *user_data, N_Vector tempv1,
             N_Vector tempv2, N_Vector tempv3) {

  PybammFunctions *p_python_functions =
      static_cast<PybammFunctions *>(user_data);

  // create pointer to jac data, column pointers, and row values
  sunindextype *jac_colptrs = SUNSparseMatrix_IndexPointers(JJ);
  sunindextype *jac_rowvals = SUNSparseMatrix_IndexValues(JJ);
  realtype *jac_data = SUNSparseMatrix_Data(JJ);

  // args are t, y, cj, put result in jacobian data matrix
  p_python_functions->jac_times_cjmass.m_arg[0] = &tt;
  p_python_functions->jac_times_cjmass.m_arg[1] = NV_DATA_S(yy);
  p_python_functions->jac_times_cjmass.m_arg[2] = &cj;
  p_python_functions->jac_times_cjmass.m_res[0] = jac_data; 
  p_python_functions->jac_times_cjmass();



  // row vals and col ptrs
  const np_array &jac_times_cjmass_rowvals = p_python_functions->jac_times_cjmass_rowvals;
  const int n_row_vals = jac_times_cjmass_rowvals.request().size;
  auto p_jac_times_cjmass_rowvals = jac_times_cjmass_rowvals.unchecked<1>();

  //std::cout << "jac_data = [";
  //for (int i = 0; i < n_row_vals; i++) {
  //  std::cout << jac_data[i] << " ";
  //}
  //std::cout << "]" << std::endl;

  // just copy across row vals (do I need to do this every time?)
  // (or just in the setup?)
  for (int i = 0; i < n_row_vals; i++) {
    //std::cout << "check row vals " << jac_rowvals[i] << " " << p_jac_times_cjmass_rowvals[i] << std::endl;
    jac_rowvals[i] = p_jac_times_cjmass_rowvals[i];
  }

  const np_array &jac_times_cjmass_colptrs = p_python_functions->jac_times_cjmass_colptrs;
  const int n_col_ptrs = jac_times_cjmass_colptrs.request().size;
  auto p_jac_times_cjmass_colptrs = jac_times_cjmass_colptrs.unchecked<1>();

  // just copy across col ptrs (do I need to do this every time?)
  for (int i = 0; i < n_col_ptrs; i++) {
    //std::cout << "check col ptrs " << jac_colptrs[i] << " " << p_jac_times_cjmass_colptrs[i] << std::endl;
    jac_colptrs[i] = p_jac_times_cjmass_colptrs[i];
  }

  return (0);
}

int events_casadi(realtype t, N_Vector yy, N_Vector yp, realtype *events_ptr,
           void *user_data)
{
  PybammFunctions *p_python_functions =
      static_cast<PybammFunctions *>(user_data);

  //std::cout << "EVENTS" << std::endl;
  //std::cout << "t = " << t << " y = [";
  //for (int i = 0; i < p_python_functions->number_of_states; i++) {
  //  std::cout << NV_DATA_S(yy)[i] << " ";
  //}
  //std::cout << "] yp = [";
  //for (int i = 0; i < p_python_functions->number_of_states; i++) {
  //  std::cout << NV_DATA_S(yp)[i] << " ";
  //}
  //std::cout << "]" << std::endl;

  // args are t, y, put result in events_ptr
  p_python_functions->events.m_arg[0] = &t;
  p_python_functions->events.m_arg[1] = NV_DATA_S(yy);
  p_python_functions->events.m_res[0] = events_ptr; 
  p_python_functions->events();

  //std::cout << "events = [";
  //for (int i = 0; i < p_python_functions->number_of_events; i++) {
  //  std::cout << events_ptr[i] << " ";
  //}
  //std::cout << "]" << std::endl;

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
// or a negative value if it failed unrecoverably (in which case the integration is halted and IDA SRES FAIL is returned)
//
int sensitivities_casadi(int Ns, realtype t, N_Vector yy, N_Vector yp, 
    N_Vector resval, N_Vector *yS, N_Vector *ypS, N_Vector *resvalS, 
    void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {

  PybammFunctions *p_python_functions =
      static_cast<PybammFunctions *>(user_data);

  const int np = p_python_functions->number_of_parameters;

  //std::cout << "SENS t = " << t << " y = [";
  //for (int i = 0; i < p_python_functions->number_of_states; i++) {
  //  std::cout << NV_DATA_S(yy)[i] << " ";
  //}
  //std::cout << "] yp = [";
  //for (int i = 0; i < p_python_functions->number_of_states; i++) {
  //  std::cout << NV_DATA_S(yp)[i] << " ";
  //}
  //std::cout << "] yS = [";
  //for (int i = 0; i < p_python_functions->number_of_states; i++) {
  //  std::cout << NV_DATA_S(yS[0])[i] << " ";
  //}
  //std::cout << "] ypS = [";
  //for (int i = 0; i < p_python_functions->number_of_states; i++) {
  //  std::cout << NV_DATA_S(ypS[0])[i] << " ";
  //}
  //std::cout << "]" << std::endl;

  // args are t, y put result in rr
  p_python_functions->sens.m_arg[0] = &t;
  p_python_functions->sens.m_arg[1] = NV_DATA_S(yy);
  for (int i = 0; i < np; i++) {
    p_python_functions->sens.m_res[i] = NV_DATA_S(resvalS[i]);
  }
  // resvalsS now has (∂F/∂p i )
  p_python_functions->sens();

  for (int i = 0; i < np; i++) {
    //std::cout << "dF/dp = [" << i << "] = [";
    //for (int j = 0; j < p_python_functions->number_of_states; j++) {
    //  std::cout << NV_DATA_S(resvalS[i])[j] << " ";
    //}
    //std::cout << "]" << std::endl;


    // put (∂F/∂y)s i (t) in tmp
    realtype *tmp = p_python_functions->get_tmp();
    p_python_functions->jac_action.m_arg[0] = &t;
    p_python_functions->jac_action.m_arg[1] = NV_DATA_S(yy);
    p_python_functions->jac_action.m_arg[2] = NV_DATA_S(yS[i]);
    p_python_functions->jac_action.m_res[0] = tmp;
    p_python_functions->jac_action();

    //std::cout << "jac_action = [" << i << "] = [";
    //for (int j = 0; j < p_python_functions->number_of_states; j++) {
    //  std::cout << tmp[j] << " ";
    //}
    //std::cout << "]" << std::endl;

    const int ns = p_python_functions->number_of_states;
    casadi_axpy(ns, 1., tmp, NV_DATA_S(resvalS[i]));

    // put -(∂F/∂ ẏ) ṡ i (t) in tmp2
    p_python_functions->mass_action.m_arg[0] = NV_DATA_S(ypS[i]);
    p_python_functions->mass_action.m_res[0] = tmp;
    p_python_functions->mass_action();

    //std::cout << "mass_Action = [" << i << "] = [";
    //for (int j = 0; j < p_python_functions->number_of_states; j++) {
    //  std::cout << tmp[j] << " ";
    //}
    //std::cout << "]" << std::endl;


    // (∂F/∂y)s i (t)+(∂F/∂ ẏ) ṡ i (t)+(∂F/∂p i ) 
    // AXPY: y <- a*x + y
    casadi_axpy(ns, -1., tmp, NV_DATA_S(resvalS[i]));

    //std::cout << "resvalS[" << i << "] = [";
    //for (int j = 0; j < p_python_functions->number_of_states; j++) {
    //  std::cout << NV_DATA_S(resvalS[i])[j] << " ";
    //}
    //std::cout << "]" << std::endl;
  }

  return 0;
}



/* main program */
  
Solution solve_casadi(np_array t_np, np_array y0_np, np_array yp0_np,
               const Function &rhs_alg, 
               const Function &jac_times_cjmass, 
               const np_array &jac_times_cjmass_rowvals, 
               const np_array &jac_times_cjmass_colptrs, 
               const int jac_times_cjmass_nnz,
               const Function &jac_action, 
               const Function &mass_action, 
               const Function &sens, 
               const Function &events, 
               const int number_of_events, 
               int use_jacobian, 
               np_array rhs_alg_id,
               np_array atol_np, double rel_tol, int number_of_parameters)
{
  
  auto t = t_np.unchecked<1>();
  auto y0 = y0_np.unchecked<1>();
  auto yp0 = yp0_np.unchecked<1>();
  auto atol = atol_np.unchecked<1>();

  int number_of_states = y0_np.request().size;
  int number_of_timesteps = t_np.request().size;
  void *ida_mem;          // pointer to memory
  N_Vector yy, yp, avtol; // y, y', and absolute tolerance
  N_Vector *yyS, *ypS;      // y, y' for sensitivities
  realtype rtol, *yval, *ypval, *atval, *ySval;
  int retval;
  SUNMatrix J;
  SUNLinearSolver LS;

  // allocate vectors
  yy = N_VNew_Serial(number_of_states);
  yp = N_VNew_Serial(number_of_states);
  avtol = N_VNew_Serial(number_of_states);

  if (number_of_parameters > 0) {
    yyS = N_VCloneVectorArray(number_of_parameters, yy);
    ypS = N_VCloneVectorArray(number_of_parameters, yp);
  }

  // set initial value
  yval = N_VGetArrayPointer(yy);
  if (number_of_parameters > 0) {
    ySval = N_VGetArrayPointer(yyS[0]);
  }
  ypval = N_VGetArrayPointer(yp);
  atval = N_VGetArrayPointer(avtol);
  int i;
  for (i = 0; i < number_of_states; i++)
  {
    yval[i] = y0[i];
    ypval[i] = yp0[i];
    atval[i] = atol[i];
  }

  for (int is = 0 ; is < number_of_parameters; is++) {
    N_VConst(RCONST(0.0), yyS[is]);
    N_VConst(RCONST(0.0), ypS[is]);
  }

  // allocate memory for solver
  ida_mem = IDACreate();

  // initialise solver
  realtype t0 = RCONST(t(0));
  IDAInit(ida_mem, residual_casadi, t0, yy, yp);

  // set tolerances
  rtol = RCONST(rel_tol);

  IDASVtolerances(ida_mem, rtol, avtol);

  // set events
  IDARootInit(ida_mem, number_of_events, events_casadi);

  // set pybamm functions by passing pointer to it
  PybammFunctions* p_pybamm_functions = new PybammFunctions(
      rhs_alg, 
      jac_times_cjmass, 
      jac_times_cjmass_rowvals,
      jac_times_cjmass_colptrs, 
      jac_action, mass_action, 
      sens, events,
      number_of_states, number_of_events,
      number_of_parameters);
  PybammFunctions &pybamm_functions = *p_pybamm_functions;

  void *user_data = &pybamm_functions;
  IDASetUserData(ida_mem, user_data);

  // set linear solver
  J = SUNSparseMatrix(number_of_states, number_of_states, jac_times_cjmass_nnz, CSR_MAT);

  // copy across row vals and col ptrs
  const int n_row_vals = jac_times_cjmass_rowvals.request().size;
  auto p_jac_times_cjmass_rowvals = jac_times_cjmass_rowvals.unchecked<1>();

  sunindextype *jac_rowvals = SUNSparseMatrix_IndexValues(J);
  for (i = 0; i < n_row_vals; i++) {
    jac_rowvals[i] = p_jac_times_cjmass_rowvals[i];
  }

  const int n_col_ptrs = jac_times_cjmass_colptrs.request().size;
  auto p_jac_times_cjmass_colptrs = jac_times_cjmass_colptrs.unchecked<1>();

  sunindextype *jac_colptrs = SUNSparseMatrix_IndexPointers(J);
  for (i = 0; i < n_col_ptrs; i++) {
    jac_colptrs[i] = p_jac_times_cjmass_colptrs[i];
  }

  //std::cout << "setting up jacobian nnz = "<< jac_times_cjmass_nnz 
  //  << " rowvals = " << n_row_vals << " colptrs = " << n_col_ptrs << std::endl; 

  LS = SUNLinSol_KLU(yy, J);
  IDASetLinearSolver(ida_mem, LS, J);

  if (use_jacobian == 1)
  {
    IDASetJacFn(ida_mem, jacobian_casadi);
  }

  if (number_of_parameters > 0)
  {
    IDASensInit(ida_mem, number_of_parameters, 
                IDA_SIMULTANEOUS, sensitivities_casadi, yyS, ypS);
    IDASensEEtolerances(ida_mem);
  }

  int t_i = 1;
  realtype tret;
  realtype t_next;
  realtype t_final = t(number_of_timesteps - 1);

  // set return vectors
  std::vector<double> t_return(number_of_timesteps);
  std::vector<double> y_return(number_of_timesteps * number_of_states);
  std::vector<double> yS_return(number_of_parameters * number_of_timesteps * number_of_states);

  t_return[0] = t(0);
  for (int j = 0; j < number_of_states; j++)
  {
    y_return[j] = yval[j];
  }
  for (int j = 0; j < number_of_parameters; j++) {
    const int base_index = j * number_of_timesteps * number_of_states;
    for (int k = 0; k < number_of_states; k++) {
      yS_return[base_index + k] = ySval[j * number_of_states + k];
    }
  }

  // calculate consistent initial conditions
  N_Vector id;
  auto id_np_val = rhs_alg_id.unchecked<1>();
  id = N_VNew_Serial(number_of_states);
  realtype *id_val;
  id_val = N_VGetArrayPointer(id);

  int ii;
  for (ii = 0; ii < number_of_states; ii++)
  {
    id_val[ii] = id_np_val[ii];
  }

  IDASetId(ida_mem, id);
  IDACalcIC(ida_mem, IDA_YA_YDP_INIT, t(1));

  while (true)
  {
    t_next = t(t_i);
    IDASetStopTime(ida_mem, t_next);
    retval = IDASolve(ida_mem, t_final, &tret, yy, yp, IDA_NORMAL);

    if (retval == IDA_TSTOP_RETURN || retval == IDA_SUCCESS || retval == IDA_ROOT_RETURN) {
      if (number_of_parameters > 0) {
        IDAGetSens(ida_mem, &tret, yyS);
      }

      t_return[t_i] = tret;
      for (int j = 0; j < number_of_states; j++)
      {
        y_return[t_i * number_of_states + j] = yval[j];
      }
      for (int j = 0; j < number_of_parameters; j++) {
        const int base_index = j * number_of_timesteps * number_of_states 
                               + t_i * number_of_states;
        for (int k = 0; k < number_of_states; k++) {
          yS_return[base_index + k] = ySval[j * number_of_states + k];
        }
      }
      t_i += 1;
      if (retval == IDA_SUCCESS || retval == IDA_ROOT_RETURN) {
        break;
      }

    } else {
      // failed
      break;
    }
  }


  /* Free memory */
  if (number_of_parameters > 0) {
    IDASensFree(ida_mem);
  }
  IDAFree(&ida_mem);
  SUNLinSolFree(LS);
  SUNMatDestroy(J);
  N_VDestroy(avtol);
  N_VDestroy(yy);
  N_VDestroy(yp);
  N_VDestroy(id);
  if (number_of_parameters > 0) {
    N_VDestroyVectorArray(yyS, number_of_parameters);
    N_VDestroyVectorArray(ypS, number_of_parameters);
  }

  np_array t_ret = np_array(t_i, &t_return[0]);
  np_array y_ret = np_array(t_i * number_of_states, &y_return[0]);
  np_array yS_ret = np_array(
      std::vector<ptrdiff_t>{number_of_parameters, t_i, number_of_states},
      &yS_return[0] 
      );

  Solution sol(retval, t_ret, y_ret, yS_ret);

  //std::cout << "finished solving 9" << std::endl;
  // Why does this bus error?
  // delete p_pybamm_functions;

  //std::cout << "finished solving 10" << std::endl;

  return sol;
}

