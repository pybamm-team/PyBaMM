#include <math.h>
#include <stdio.h>

#include <idas/idas.h>                 /* prototypes for IDAS fcts., consts.    */
#include <nvector/nvector_serial.h>  /* access to serial N_Vector            */
#include <sundials/sundials_math.h>  /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sundials/sundials_types.h> /* defs. of realtype, sunindextype      */
#include <sunlinsol/sunlinsol_klu.h> /* access to KLU linear solver          */
#include <sunmatrix/sunmatrix_sparse.h> /* access to sparse SUNMatrix           */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <casadi.hpp>
#include <vector>

//#include <iostream>
namespace py = pybind11;

using Function = casadi::Function
using casadi_int= casadi::casadi_int
using casadi_axpy = casadi::casadi_axpy

class CasadiFunction {
  CasadiFunction(const Function &f):func(f) {
    size_t sz_arg;
    size_t sz_res;
    size_t sz_iw;
    size_t sz_w;
    func.sz_work(&sz_arg, &sz_res, &sz_iw, &sz_w);
    m_arg.resize(sz_arg);
    m_res.resize(sz_res);
    m_iw.resize(sz_iw);
    m_w.resize(sz_w);
  }

  // only call this once m_arg and m_res have been set appropriatelly
  void operator()() {
    int mem = func.checkout();
    eval(m_arg.data(), m_res.data(), m_iw.data(), m_w.data(), mem);
    func.release(mem)
  }

  public:
    std::vector<double *> m_arg;
    std::vector<double *> m_res;

  private:
    Function func;
    std::vector<casadi_int> m_iw;
    std::vector<double> m_iw;
}


class PybammFunctions {
public:
  int number_of_states;
  int number_of_parameters;
  int number_of_events;
  CasadiFunction rhs_alg;
  CasadiFunction sens;
  CasadiFunction jac;
  CasadiFunction mass_action;
  CasadiFunction event;

  PybammFunctions(const Function &rhs_alg, const Function &jac,
                  const Function &mass_action,
                  const Function &sens,
                  const Function &event, 
                  const int n_s, int n_e, const int n_p)
      : number_of_states(n_s), number_of_events(n_e), 
        number_of_parameters(n_p),
        res(res), jac(jac),
        mass_action(mass_action),
        sens(sens),
        event(event)
  {}
};

int residual(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr,
             void *user_data)
{
  PybammFunctions *p_python_functions =
      static_cast<PybammFunctions *>(user_data);
  // args are t, y, put result in rr
  p_python_functions->rhs_alg.m_arg[0] = &tres;
  p_python_functions->rhs_alg.m_arg[1] = NV_DATA_S(yy);
  p_python_functions->rhs_alg.m_res[0] = NV_DATA_S(rr);
  p_python_functions->rhs_alg();

  realtype *tmp = p_python_functions->get_tmp()
  // args is yp, put result in tmp
  p_python_functions->mass_action.m_arg[0] = NV_DATA_S(yp);
  p_python_functions->res.m_res[0] = tmp;
  p_python_functions->res();

  // AXPY: y <- a*x + y
  const ns = p_python_functions->number_of_states;
  casadi_axpy(ns, -1., tmp, NV_DATA_S(rr));

  // now rr has rhs_alg(t, y) - mass_matrix * yp

  return 0;
}

int jacobian(realtype tt, realtype cj, N_Vector yy, N_Vector yp,
             N_Vector resvec, SUNMatrix JJ, void *user_data, N_Vector tempv1,
             N_Vector tempv2, N_Vector tempv3)
{

  PybammFunctions *p_python_functions =
      static_cast<PybammFunctions *>(user_data);

  // create pointer to jac data, column pointers, and row values
  sunindextype *jac_colptrs = SUNSparseMatrix_IndexPointers(JJ);
  sunindextype *jac_rowvals = SUNSparseMatrix_IndexValues(JJ);
  realtype *jac_data = SUNSparseMatrix_Data(JJ);

  // args are t, y, cj, put result in jacobian data matrix
  p_python_functions->jac.m_arg[0] = &tres ;
  p_python_functions->jac.m_arg[1] = NV_DATA_S(yy);
  p_python_functions->jac.m_arg[2] = &cj;
  p_python_functions->jac.m_res[0] = jac_data; 
  p_python_functions->jac();

  return (0);
}

int events(realtype t, N_Vector yy, N_Vector yp, realtype *events_ptr,
           void *user_data)
{
  PybammFunctions *p_python_functions =
      static_cast<PybammFunctions *>(user_data);

  // args are t, y, put result in events_ptr
  p_python_functions->jac.m_arg[0] = &tres ;
  p_python_functions->jac.m_arg[1] = NV_DATA_S(yy);
  p_python_functions->jac.m_res[0] = events_ptr; 
  p_python_functions->jac();

  return (0);
}

int sensitivities(int Ns, realtype t, N_Vector yy, N_Vector yp, 
    N_Vector resval, N_Vector *yS, N_Vector *ypS, N_Vector *resvalS, 
    void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
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
  PybammFunctions *p_python_functions =
      static_cast<PybammFunctions *>(user_data);

  // args are t, y put result in rr
  p_python_functions->sens.m_arg[0] = &tres;
  p_python_functions->sens.m_arg[1] = NV_DATA_S(yy);
  const int np = p_python_functions->number_of_parameters;
  for (int i = 0; i < np; i++) {
    p_python_functions->sens.m_res[i] = NV_DATA_S(resvalS[i]);
  }
  p_python_functions->sens();

  // memory managed by sundials, so pass a destructor that does nothing
  auto state_vector_shape = std::vector<ptrdiff_t>{n, 1};
  np_array y_np = np_array(state_vector_shape, N_VGetArrayPointer(yy), 
                           py::capsule(&yy, [](void* p) {}));
  np_array yp_np = np_array(state_vector_shape, N_VGetArrayPointer(yp),
                           py::capsule(&yp, [](void* p) {}));

  std::vector<np_array> yS_np(np);
  for (int i = 0; i < np; i++) {
    auto capsule = py::capsule(yS + i, [](void* p) {});
    yS_np[i] = np_array(state_vector_shape, N_VGetArrayPointer(yS[i]), capsule);
  }

  std::vector<np_array> ypS_np(np);
  for (int i = 0; i < np; i++) {
    auto capsule = py::capsule(ypS + i, [](void* p) {});
    ypS_np[i] = np_array(state_vector_shape, N_VGetArrayPointer(ypS[i]), capsule);
  }

  std::vector<np_array> resvalS_np(np);
  for (int i = 0; i < np; i++) {
    auto capsule = py::capsule(resvalS + i, [](void* p) {});
    resvalS_np[i] = np_array(state_vector_shape, 
                             N_VGetArrayPointer(resvalS[i]), capsule);
  }

  realtype *ptr1 = static_cast<realtype *>(resvalS_np[0].request().ptr);
  const realtype* resvalSval = N_VGetArrayPointer(resvalS[0]);

  python_functions.sensitivities(resvalS_np, t, y_np, yp_np, yS_np, ypS_np);

  return 0;
}

class Solution
{
public:
  Solution(int retval, np_array t_np, np_array y_np, np_array yS_np)
      : flag(retval), t(t_np), y(y_np), yS(yS_np)
  {
  }

  int flag;
  np_array t;
  np_array y;
  np_array yS;
};

/* main program */
Solution solve(np_array t_np, np_array y0_np, np_array yp0_np,
               residual_type res, jacobian_type jac, 
               sensitivities_type sens,
               jac_get_type gjd, jac_get_type gjrv, jac_get_type gjcp, 
               int nnz, event_type event,
               int number_of_events, int use_jacobian, np_array rhs_alg_id,
               np_array atol_np, double rel_tol, int number_of_parameters)
{
  IdasInterface interface("pybamm_idaklu_casadi", dae);
  
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
  IDAInit(ida_mem, residual, t0, yy, yp);

  // set tolerances
  rtol = RCONST(rel_tol);

  IDASVtolerances(ida_mem, rtol, avtol);

  // set events
  IDARootInit(ida_mem, number_of_events, events);

  // set pybamm functions by passing pointer to it
  PybammFunctions pybamm_functions(res, jac, sens, gjd, gjrv, gjcp, event,
                                   number_of_states, number_of_events,
                                   number_of_parameters);
  void *user_data = &pybamm_functions;
  IDASetUserData(ida_mem, user_data);

  // set linear solver
  J = SUNSparseMatrix(number_of_states, number_of_states, nnz, CSR_MAT);

  LS = SUNLinSol_KLU(yy, J);
  IDASetLinearSolver(ida_mem, LS, J);

  if (use_jacobian == 1)
  {
    IDASetJacFn(ida_mem, jacobian);
  }

  if (number_of_parameters > 0)
  {
    IDASensInit(ida_mem, number_of_parameters, 
                IDA_SIMULTANEOUS, sensitivities, yyS, ypS);
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

    if (retval == IDA_TSTOP_RETURN || retval == IDA_SUCCESS || retval == IDA_ROOT_RETURN)
    {
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
  N_VDestroy(yp);
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

  return sol;
}

PYBIND11_MODULE(idaklu, m)
{
  m.doc() = "sundials solvers"; // optional module docstring

  py::bind_vector<std::vector<np_array>>(m, "VectorNdArray");

  m.def("solve", &solve, "The solve function", py::arg("t"), py::arg("y0"),
        py::arg("yp0"), py::arg("res"), py::arg("jac"), py::arg("sens"), 
        py::arg("get_jac_data"),
        py::arg("get_jac_row_vals"), py::arg("get_jac_col_ptr"), py::arg("nnz"),
        py::arg("events"), py::arg("number_of_events"), py::arg("use_jacobian"),
        py::arg("rhs_alg_id"), py::arg("atol"), py::arg("rtol"),
        py::arg("number_of_sensitivity_parameters"),
        py::return_value_policy::take_ownership);

  py::class_<Solution>(m, "solution")
      .def_readwrite("t", &Solution::t)
      .def_readwrite("y", &Solution::y)
      .def_readwrite("yS", &Solution::yS)
      .def_readwrite("flag", &Solution::flag);
}
