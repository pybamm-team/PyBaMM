// pybind wrapping of sundials with the KLU linear solver
// currently just in a quick and dirty form to get something working

#include <stdio.h>
#include <math.h>

#include <ida/ida.h>                    /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h>     /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_sparse.h> /* access to sparse SUNMatrix           */
#include <sunlinsol/sunlinsol_klu.h>    /* access to KLU linear solver          */
#include <sundials/sundials_types.h>    /* defs. of realtype, sunindextype      */
#include <sundials/sundials_math.h>     /* defs. of SUNRabs, SUNRexp, etc.      */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

using residual_type = std::function<py::array_t<double>(double, py::array_t<double>, py::array_t<double>)>;
using jacobian_type = std::function<py::array_t<double>(double, py::array_t<double>, double)>;
using event_type = std::function<py::array_t<double>(double, py::array_t<double>)>;
using np_array = py::array_t<double>;

using jac_get_type = std::function<np_array()>;

class PybammFunctions
{
public:
    PybammFunctions(const residual_type &res,
                    const jacobian_type &jac,
                    const jac_get_type &get_jac_data_in,
                    const jac_get_type &get_jac_row_vals_in,
                    const jac_get_type &get_jac_col_ptrs_in,
                    const event_type &event,
                    const int n_s,
                    int n_e)
        : py_res(res),
          py_jac(jac),
          py_event(event),
          py_get_jac_data(get_jac_data_in),
          py_get_jac_row_vals(get_jac_row_vals_in),
          py_get_jac_col_ptrs(get_jac_col_ptrs_in),
          number_of_states(n_s),
          number_of_events(n_e)
    {
    }

    int number_of_states;
    int number_of_events;

    py::array_t<double> operator()(double t, py::array_t<double> y, py::array_t<double> yp)
    {
        return py_res(t, y, yp);
    }

    py::array_t<double> res(double t, py::array_t<double> y, py::array_t<double> yp)
    {
        return py_res(t, y, yp);
    }

    void jac(double t, py::array_t<double> y, double cj)
    {
        // this function evaluates the jacobian and sets it to be the attribute of a python class
        // which can then be called by get_jac_data, get_jac_col_ptr, etc
        py_jac(t, y, cj);
    }

    np_array get_jac_data()
    {
        return py_get_jac_data();
    }

    np_array get_jac_row_vals()
    {
        return py_get_jac_row_vals();
    }

    np_array get_jac_col_ptrs()
    {
        return py_get_jac_col_ptrs();
    }

    np_array events(double t, np_array y)
    {
        return py_event(t, y);
    }

private:
    residual_type py_res;
    jacobian_type py_jac;
    jac_get_type py_get_jac_data;
    jac_get_type py_get_jac_row_vals;
    jac_get_type py_get_jac_col_ptrs;
    event_type py_event;
};

int residual(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void *user_data)
{
    PybammFunctions *python_functions_ptr = static_cast<PybammFunctions *>(user_data);
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

int jacobian(realtype tt, realtype cj,
             N_Vector yy, N_Vector yp, N_Vector resvec,
             SUNMatrix JJ, void *user_data,
             N_Vector tempv1, N_Vector tempv2, N_Vector tempv3)
{

    realtype *yval, *ypval;
    yval = N_VGetArrayPointer(yy);
    ypval = N_VGetArrayPointer(yp);

    yval = N_VGetArrayPointer(yy);

    PybammFunctions *python_functions_ptr = static_cast<PybammFunctions *>(user_data);
    PybammFunctions python_functions = *python_functions_ptr;

    int n = python_functions.number_of_states;
    py::array_t<double> y_np = py::array_t<double>(n, yval);
    py::array_t<double> yp_np = py::array_t<double>(n, ypval);

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

    py::array_t<double> y_np = py::array_t<double>(2, yval);
    py::array_t<double> yp_np = py::array_t<double>(2, ypval);

    PybammFunctions *python_functions_ptr = static_cast<PybammFunctions *>(user_data);
    PybammFunctions python_functions = *python_functions_ptr;

    py::array_t<double> events_np_array;

    events_np_array = python_functions.events(t, y_np);

    double *events_np_data_ptr = (double *)events_np_array.request().ptr;

    // just copying data into Sunmatrix (figure out how to pass pointers later)
    int number_of_events = 2;
    int i;
    for (i = 0; i < number_of_events; i++)
    {
        events_ptr[i] = events_np_data_ptr[i];
    }

    return (0);
}

/* main program */
np_array solve(np_array t_np,
               np_array y0_np,
               np_array yp0_np,
               residual_type res,
               jacobian_type jac,
               jac_get_type gjd,
               jac_get_type gjrv,
               jac_get_type gjcp,
               int nnz,
               event_type event,
               int number_of_events, int use_jacobian)
{
    auto t = t_np.unchecked<1>();
    auto y0 = y0_np.unchecked<1>();
    auto yp0 = yp0_np.unchecked<1>();

    int number_of_states;
    number_of_states = y0_np.request().size;

    void *ida_mem;          // pointer to memory
    N_Vector yy, yp, avtol; // y, y', and absolute tolerance
    realtype rtol, *yval, *ypval, *atval;
    int iout, retval, retvalr;
    int rootsfound[number_of_events];
    SUNMatrix J;
    SUNLinearSolver LS;
    SUNNonlinearSolver NLS;

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
    realtype t0 = RCONST(0.0);
    retval = IDAInit(ida_mem, residual, t0, yy, yp);

    // set tolerances
    rtol = RCONST(1.0e-4);
    atval = N_VGetArrayPointer(avtol);

    for (i = 0; i < number_of_states; i++)
    {
        atval[i] = RCONST(1.0e-8); // nb: this can be set differently for each state
    }

    retval = IDASVtolerances(ida_mem, rtol, avtol);

    // set events
    retval = IDARootInit(ida_mem, number_of_events, events);

    // set pybamm functions by passing pointer to it
    PybammFunctions pybamm_functions(res, jac, gjd, gjrv, gjcp, event, number_of_states, number_of_events);
    void *user_data = &pybamm_functions;
    IDASetUserData(ida_mem, user_data);

    // set linear solver
    J = SUNSparseMatrix(number_of_states, number_of_states, nnz, CSR_MAT); // jacobian type (must be dense for dense solvers, p183 of ida_guide.pdf)
    LS = SUNLinSol_KLU(yy, J);
    retval = IDASetLinearSolver(ida_mem, LS, J);

    // sparse stuff  (must use sparse solvers e.g. KLU or SuperLUMT, p183 of ida_guide.pdf)
    // J = SUNSparseMatrix(2, 2, 2, CSR_MAT); // template jacobian
    // // set the indices values and pts of the jacobian (leaving the values unset)
    // realtype *csr_index_vals_ptr = SM_INDEXVALS_S(JJ);
    // realtype *csr_index_ptrs_ptr = SM_INDEXPTRS_S(JJ);

    if (use_jacobian == 1)
    {
        printf("\nSetting jacobian \n");
        retval = IDASetJacFn(ida_mem, jacobian);
    }

    realtype tout, tret;
    realtype t_final = t(99);

    while (tret < t_final)
    {
        // IDA_ONE_STEP_TSTOP
        // IDA_NORMAL
        retval = IDASolve(ida_mem, t_final, &tret, yy, yp, IDA_ONE_STEP);

        if (retval == IDA_ROOT_RETURN)
        {
            break;
        }

        printf("t=%f, y=%f, a=%f \n", tret, yval[0], yval[1]);
    }

    /* Free memory */
    IDAFree(&ida_mem);
    SUNLinSolFree(LS);
    SUNMatDestroy(J);
    N_VDestroy(avtol);
    N_VDestroy(yp);

    printf("t=%f, y=%f, a=%f \n", tret, yval[0], yval[1]);

    return py::array_t<double>(number_of_states, yval);
}

PYBIND11_MODULE(klu, m)
{
    m.doc() = "sundials solvers"; // optional module docstring

    m.def("solve", &solve, "The solve function",
          py::arg("t"), py::arg("y0"), py::arg("yp0"),
          py::arg("res"),
          py::arg("jac"),
          py::arg("get_jac_data"),
          py::arg("get_jac_row_vals"),
          py::arg("get_jac_col_ptr"),
          py::arg("nnz"),
          py::arg("events"),
          py::arg("number_of_events"),
          py::arg("use_jacobian"),
          py::return_value_policy::take_ownership);
}
