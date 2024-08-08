#include "common.hpp"
#include "python.hpp"
#include <iostream>

class PybammFunctions
{
public:
    int number_of_states;
    int number_of_parameters;
    int number_of_events;

    PybammFunctions(const residual_type &res, const jacobian_type &jac,
                    const sensitivities_type &sens,
                    const jac_get_type &get_jac_data_in,
                    const jac_get_type &get_jac_row_vals_in,
                    const jac_get_type &get_jac_col_ptrs_in,
                    const event_type &event,
                    const int n_s, int n_e, const int n_p,
                    const np_array &inputs)
        : number_of_states(n_s), number_of_events(n_e),
          number_of_parameters(n_p),
          py_res(res), py_jac(jac),
          py_sens(sens),
          py_event(event), py_get_jac_data(get_jac_data_in),
          py_get_jac_row_vals(get_jac_row_vals_in),
          py_get_jac_col_ptrs(get_jac_col_ptrs_in),
          inputs(inputs)
    {
    }

    np_array operator()(double t, np_array y, np_array yp)
    {
        return py_res(t, y, inputs, yp);
    }

    np_array res(double t, np_array y, np_array yp)
    {
        return py_res(t, y, inputs, yp);
    }

    void jac(double t, np_array y, double cj)
    {
        // this function evaluates the jacobian and sets it to be the attribute
        // of a python class which can then be called by get_jac_data,
        // get_jac_col_ptr, etc
        py_jac(t, y, inputs, cj);
    }

    void sensitivities(
        std::vector<np_array>& resvalS,
        const double t, const np_array& y, const np_array& yp,
        const std::vector<np_array>& yS, const std::vector<np_array>& ypS)
    {
        // this function evaluates the sensitivity equations required by IDAS,
        // returning them in resvalS, which is preallocated as a numpy array
        // of size (np, n), where n is the number of states and np is the number
        // of parameters
        //
        // yS and ypS are also shape (np, n), y and yp are shape (n)
        //
        // dF/dy * s_i + dF/dyd * sd + dFdp_i for i in range(np)
        py_sens(resvalS, t, y, inputs, yp, yS, ypS);
    }

    np_array get_jac_data() {
        return py_get_jac_data();
    }

    np_array get_jac_row_vals() {
        return py_get_jac_row_vals();
    }

    np_array get_jac_col_ptrs() {
        return py_get_jac_col_ptrs();
    }

    np_array events(double t, np_array y) {
        return py_event(t, y, inputs);
    }

private:
    residual_type py_res;
    sensitivities_type py_sens;
    jacobian_type py_jac;
    event_type py_event;
    jac_get_type py_get_jac_data;
    jac_get_type py_get_jac_row_vals;
    jac_get_type py_get_jac_col_ptrs;
    const np_array &inputs;
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

    auto r_np_ptr = r_np.unchecked<1>();

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
    realtype *yval;
    yval = N_VGetArrayPointer(yy);

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
    auto jac_np_data_ptr = jac_np_data.unchecked<1>();

    // just copy across data
    int i;
    for (i = 0; i < n_data; i++)
    {
        jac_data[i] = jac_np_data_ptr[i];
    }

    np_array jac_np_row_vals = python_functions.get_jac_row_vals();
    int n_row_vals = jac_np_row_vals.request().size;

    auto jac_np_row_vals_ptr = jac_np_row_vals.unchecked<1>();
    // just copy across row vals (this might be unneeded)
    for (i = 0; i < n_row_vals; i++)
    {
        jac_rowvals[i] = jac_np_row_vals_ptr[i];
    }

    np_array jac_np_col_ptrs = python_functions.get_jac_col_ptrs();
    int n_col_ptrs = jac_np_col_ptrs.request().size;
    auto jac_np_col_ptrs_ptr = jac_np_col_ptrs.unchecked<1>();

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
    realtype *yval;
    yval = N_VGetArrayPointer(yy);

    PybammFunctions *python_functions_ptr =
        static_cast<PybammFunctions *>(user_data);
    PybammFunctions python_functions = *python_functions_ptr;

    int number_of_events = python_functions.number_of_events;
    int number_of_states = python_functions.number_of_states;
    py::array_t<double> y_np = py::array_t<double>(number_of_states, yval);

    py::array_t<double> events_np_array;

    events_np_array = python_functions.events(t, y_np);

    auto events_np_data_ptr = events_np_array.unchecked<1>();

    // just copying data (figure out how to pass pointers later)
    int i;
    for (i = 0; i < number_of_events; i++)
    {
        events_ptr[i] = events_np_data_ptr[i];
    }

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
    PybammFunctions *python_functions_ptr =
        static_cast<PybammFunctions *>(user_data);
    PybammFunctions python_functions = *python_functions_ptr;

    int n = python_functions.number_of_states;
    int np = python_functions.number_of_parameters;

    // memory managed by sundials, so pass a destructor that does nothing
    auto state_vector_shape = std::vector<ptrdiff_t> {n, 1};
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

/* main program */
Solution solve_python(np_array t_np, np_array y0_np, np_array yp0_np,
                      residual_type res, jacobian_type jac,
                      sensitivities_type sens,
                      jac_get_type gjd, jac_get_type gjrv, jac_get_type gjcp,
                      int nnz, event_type event,
                      int number_of_events, int use_jacobian, np_array rhs_alg_id,
                      np_array atol_np, double rel_tol, np_array inputs,
                      int number_of_parameters)
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
    N_Vector id;
    realtype rtol, *yval, *ypval, *atval;
    std::vector<realtype *> ySval(number_of_parameters);
    int retval;
    SUNMatrix J;
    SUNLinearSolver LS;

#if SUNDIALS_VERSION_MAJOR >= 6
    SUNContext sunctx;
    SUNContext_Create(NULL, &sunctx);

    // allocate memory for solver
    ida_mem = IDACreate(sunctx);

    // allocate vectors
    yy = N_VNew_Serial(number_of_states, sunctx);
    yp = N_VNew_Serial(number_of_states, sunctx);
    avtol = N_VNew_Serial(number_of_states, sunctx);
    id = N_VNew_Serial(number_of_states, sunctx);
#else
    // allocate memory for solver
    ida_mem = IDACreate();

    // allocate vectors
    yy = N_VNew_Serial(number_of_states);
    yp = N_VNew_Serial(number_of_states);
    avtol = N_VNew_Serial(number_of_states);
    id = N_VNew_Serial(number_of_states);
#endif

    if (number_of_parameters > 0) {
        yyS = N_VCloneVectorArray(number_of_parameters, yy);
        ypS = N_VCloneVectorArray(number_of_parameters, yp);
    }

    // set initial value
    yval = N_VGetArrayPointer(yy);
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
        ySval[is] = N_VGetArrayPointer(yyS[is]);
        N_VConst(RCONST(0.0), yyS[is]);
        N_VConst(RCONST(0.0), ypS[is]);
    }

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
                                     number_of_parameters, inputs);
    void *user_data = &pybamm_functions;
    IDASetUserData(ida_mem, user_data);

    // set linear solver
#if SUNDIALS_VERSION_MAJOR >= 6
    J = SUNSparseMatrix(number_of_states, number_of_states, nnz, CSR_MAT, sunctx);
    LS = SUNLinSol_KLU(yy, J, sunctx);
#else
    J = SUNSparseMatrix(number_of_states, number_of_states, nnz, CSR_MAT);
    LS = SUNLinSol_KLU(yy, J);
#endif

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
            yS_return[base_index + k] = ySval[j][k];
        }
    }

    // calculate consistent initial conditions
    auto id_np_val = rhs_alg_id.unchecked<1>();
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
                    yS_return[base_index + k] = ySval[j][k];
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
#if SUNDIALS_VERSION_MAJOR >= 6
    SUNContext_Free(&sunctx);
#endif

    np_array t_ret = np_array(t_i, &t_return[0]);
    np_array y_ret = np_array(t_i * number_of_states, &y_return[0]);
    np_array yS_ret = np_array(
                          std::vector<ptrdiff_t> {number_of_parameters, number_of_timesteps, number_of_states},
                          &yS_return[0]
                      );
    np_array yterm_ret = np_array(0);

    Solution sol(retval, t_ret, y_ret, yS_ret, yterm_ret);

    return sol;
}
