#include <math.h>
#include <stdio.h>

#include <ida/ida.h>                 /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h>  /* access to serial N_Vector            */
#include <sundials/sundials_math.h>  /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sundials/sundials_types.h> /* defs. of realtype, sunindextype      */
#include <sunlinsol/sunlinsol_klu.h> /* access to KLU linear solver          */
#include <sunmatrix/sunmatrix_sparse.h> /* access to sparse SUNMatrix           */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

using residual_type = std::function<py::array_t<double>(
    double, py::array_t<double>, py::array_t<double>)>;
using jacobian_type =
    std::function<py::array_t<double>(double, py::array_t<double>, double)>;
using event_type =
    std::function<py::array_t<double>(double, py::array_t<double>)>;
using np_array = py::array_t<double>;

using jac_get_type = std::function<np_array()>;

class PybammFunctions
{
public:
  int number_of_states;
  int number_of_events;

  PybammFunctions(const residual_type &res, const jacobian_type &jac,
                  const jac_get_type &get_jac_data_in,
                  const jac_get_type &get_jac_row_vals_in,
                  const jac_get_type &get_jac_col_ptrs_in,
                  const event_type &event, const int n_s, int n_e)
      : number_of_states(n_s), number_of_events(n_e), py_res(res), py_jac(jac),
        py_event(event), py_get_jac_data(get_jac_data_in),
        py_get_jac_row_vals(get_jac_row_vals_in),
        py_get_jac_col_ptrs(get_jac_col_ptrs_in)
  {
  }

  py::array_t<double> operator()(double t, py::array_t<double> y,
                                 py::array_t<double> yp)
  {
    return py_res(t, y, yp);
  }

  py::array_t<double> res(double t, py::array_t<double> y,
                          py::array_t<double> yp)
  {
    return py_res(t, y, yp);
  }

  void jac(double t, py::array_t<double> y, double cj)
  {
    // this function evaluates the jacobian and sets it to be the attribute
    // of a python class which can then be called by get_jac_data,
    // get_jac_col_ptr, etc
    py_jac(t, y, cj);
  }

  np_array get_jac_data() { return py_get_jac_data(); }

  np_array get_jac_row_vals() { return py_get_jac_row_vals(); }

  np_array get_jac_col_ptrs() { return py_get_jac_col_ptrs(); }

  np_array events(double t, np_array y) { return py_event(t, y); }

private:
  residual_type py_res;
  jacobian_type py_jac;
  event_type py_event;
  jac_get_type py_get_jac_data;
  jac_get_type py_get_jac_row_vals;
  jac_get_type py_get_jac_col_ptrs;
};

int residual(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr,
             void *user_data)
{
  PybammFunctions *python_functions_ptr =
      static_cast<PybammFunctions *>(user_data);
  PybammFunctions python_functions = *python_functions_ptr;

  realtype *yval, *ypval, *rval;
  yval = N_VGetArrayPointer(yy);
  ypval = N_VGetArrayPointer(yp);
  rval = N_VGetArrayPointer(rr);

  int n = python_functions.number_of_states;
  py::array_t<double> y_np = py::array_t<double>(n, yval);
  py::array_t<double> yp_np = py::array_t<double>(n, ypval);

  py::array_t<double> r_np;

  r_np = python_functions.res(tres, y_np, yp_np);

  double *r_np_ptr = (double *)r_np.request().ptr;

  // just copying data
  int i;
  for (i = 0; i < n; i++)
  {
    rval[i] = r_np_ptr[i];
  }
  return 0;
}

int jacobian(realtype tt, realtype cj, N_Vector yy, N_Vector yp,
             N_Vector resvec, SUNMatrix JJ, void *user_data, N_Vector tempv1,
             N_Vector tempv2, N_Vector tempv3)
{
  realtype *yval, *ypval;
  yval = N_VGetArrayPointer(yy);
  ypval = N_VGetArrayPointer(yp);

  PybammFunctions *python_functions_ptr =
      static_cast<PybammFunctions *>(user_data);
  PybammFunctions python_functions = *python_functions_ptr;

  int n = python_functions.number_of_states;
  py::array_t<double> y_np = py::array_t<double>(n, yval);

  // create pointer to jac data, column pointers, and row values
  sunindextype *jac_colptrs = SUNSparseMatrix_IndexPointers(JJ);
  sunindextype *jac_rowvals = SUNSparseMatrix_IndexValues(JJ);
  realtype *jac_data = SUNSparseMatrix_Data(JJ);

  py::array_t<double> jac_np_array;

  python_functions.jac(tt, y_np, cj);

  np_array jac_np_data = python_functions.get_jac_data();
  int n_data = jac_np_data.request().size;
  double *jac_np_data_ptr = (double *)jac_np_data.request().ptr;

  // just copy across data
  int i;
  for (i = 0; i < n_data; i++)
  {
    jac_data[i] = jac_np_data_ptr[i];
  }

  np_array jac_np_row_vals = python_functions.get_jac_row_vals();
  int n_row_vals = jac_np_row_vals.request().size;
  double *jac_np_row_vals_ptr = (double *)jac_np_row_vals.request().ptr;

  // just copy across row vals (this might be unneeded)
  for (i = 0; i < n_row_vals; i++)
  {
    jac_rowvals[i] = jac_np_row_vals_ptr[i];
  }

  np_array jac_np_col_ptrs = python_functions.get_jac_col_ptrs();
  int n_col_ptrs = jac_np_col_ptrs.request().size;
  double *jac_np_col_ptrs_ptr = (double *)jac_np_col_ptrs.request().ptr;

  // just copy across col ptrs (this might be unneeded)
  for (i = 0; i < n_col_ptrs; i++)
  {
    jac_colptrs[i] = jac_np_col_ptrs_ptr[i];
  }

  return (0);
}

int events(realtype t, N_Vector yy, N_Vector yp, realtype *events_ptr,
           void *user_data)
{
  realtype *yval, *ypval;
  yval = N_VGetArrayPointer(yy);
  ypval = N_VGetArrayPointer(yp);

  PybammFunctions *python_functions_ptr =
      static_cast<PybammFunctions *>(user_data);
  PybammFunctions python_functions = *python_functions_ptr;

  int number_of_events = python_functions.number_of_events;
  int number_of_states = python_functions.number_of_states;
  py::array_t<double> y_np = py::array_t<double>(number_of_states, yval);

  py::array_t<double> events_np_array;

  events_np_array = python_functions.events(t, y_np);

  double *events_np_data_ptr = (double *)events_np_array.request().ptr;

  // just copying data (figure out how to pass pointers later)
  int i;
  for (i = 0; i < number_of_events; i++)
  {
    events_ptr[i] = events_np_data_ptr[i];
  }

  return (0);
}

class Solution
{
public:
  Solution(int retval, np_array t_np, np_array y_np)
      : flag(retval), t(t_np), y(y_np)
  {
  }

  int flag;
  np_array t;
  np_array y;
};

/* main program */
Solution solve(np_array t_np, np_array y0_np, np_array yp0_np,
               residual_type res, jacobian_type jac, jac_get_type gjd,
               jac_get_type gjrv, jac_get_type gjcp, int nnz, event_type event,
               int number_of_events, int use_jacobian, np_array rhs_alg_id,
               double abs_tol, double rel_tol)
{
  auto t = t_np.unchecked<1>();
  auto y0 = y0_np.unchecked<1>();
  auto yp0 = yp0_np.unchecked<1>();

  int number_of_states;
  number_of_states = y0_np.request().size;
  int number_of_timesteps;
  number_of_timesteps = t_np.request().size;

  void *ida_mem;          // pointer to memory
  N_Vector yy, yp, avtol; // y, y', and absolute tolerance
  realtype rtol, *yval, *ypval, *atval;
  int retval;
  SUNMatrix J;
  SUNLinearSolver LS;

  // allocate vectors
  yy = N_VNew_Serial(number_of_states);
  yp = N_VNew_Serial(number_of_states);
  avtol = N_VNew_Serial(number_of_states);

  // set initial value
  yval = N_VGetArrayPointer(yy);
  ypval = N_VGetArrayPointer(yp);
  int i;
  for (i = 0; i < number_of_states; i++)
  {
    yval[i] = y0[i];
    ypval[i] = yp0[i];
  }

  // allocate memory for solver
  ida_mem = IDACreate();

  // initialise solver
  realtype t0 = RCONST(t(0));
  IDAInit(ida_mem, residual, t0, yy, yp);

  // set tolerances
  rtol = RCONST(rel_tol);
  atval = N_VGetArrayPointer(avtol);

  for (i = 0; i < number_of_states; i++)
  {
    atval[i] =
        RCONST(abs_tol); // nb: this can be set differently for each state
  }

  IDASVtolerances(ida_mem, rtol, avtol);

  // set events
  IDARootInit(ida_mem, number_of_events, events);

  // set pybamm functions by passing pointer to it
  PybammFunctions pybamm_functions(res, jac, gjd, gjrv, gjcp, event,
                                   number_of_states, number_of_events);
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

  int t_i = 1;
  realtype tret;
  realtype t_next;
  realtype t_final = t(number_of_timesteps - 1);

  // set return vectors
  // double t_return[number_of_timesteps] = {0};
  // double y_return[number_of_timesteps * number_of_states] = {0};

  std::vector<double> t_return(number_of_timesteps);
  std::vector<double> y_return(number_of_timesteps * number_of_states);

  t_return[0] = t(0);
  int j;
  for (j = 0; j < number_of_states; j++)
  {
    y_return[j] = yval[j];
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

    if (retval == IDA_TSTOP_RETURN)
    {
      t_return[t_i] = tret;
      for (j = 0; j < number_of_states; j++)
      {
        y_return[t_i * number_of_states + j] = yval[j];
      }
      t_i += 1;
    }

    if (retval == IDA_SUCCESS || retval == IDA_ROOT_RETURN)
    {
      t_return[t_i] = tret;
      for (j = 0; j < number_of_states; j++)
      {
        y_return[t_i * number_of_states + j] = yval[j];
      }
      break;
    }
  }

  /* Free memory */
  IDAFree(&ida_mem);
  SUNLinSolFree(LS);
  SUNMatDestroy(J);
  N_VDestroy(avtol);
  N_VDestroy(yp);

  py::array_t<double> t_ret = py::array_t<double>((t_i + 1), &t_return[0]);
  py::array_t<double> y_ret =
      py::array_t<double>((t_i + 1) * number_of_states, &y_return[0]);

  Solution sol(retval, t_ret, y_ret);

  return sol;
}

PYBIND11_MODULE(klu, m)
{
  m.doc() = "sundials solvers"; // optional module docstring

  m.def("solve", &solve, "The solve function", py::arg("t"), py::arg("y0"),
        py::arg("yp0"), py::arg("res"), py::arg("jac"), py::arg("get_jac_data"),
        py::arg("get_jac_row_vals"), py::arg("get_jac_col_ptr"), py::arg("nnz"),
        py::arg("events"), py::arg("number_of_events"), py::arg("use_jacobian"),
        py::arg("rhs_alg_id"), py::arg("rtol"), py::arg("atol"),
        py::return_value_policy::take_ownership);

  py::class_<Solution>(m, "solution")
      .def_readwrite("t", &Solution::t)
      .def_readwrite("y", &Solution::y)
      .def_readwrite("flag", &Solution::flag);
}
