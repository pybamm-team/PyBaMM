#include "Expressions/Expressions.hpp"
#include "sundials_functions.hpp"
#include <vector>
#include "common.hpp"
#include "SolutionData.hpp"

template <class ExprSet>
IDAKLUSolverOpenMP<ExprSet>::IDAKLUSolverOpenMP(
  np_array atol_np_input,
  double rel_tol,
  np_array rhs_alg_id_input,
  int number_of_parameters_input,
  int number_of_events_input,
  int jac_times_cjmass_nnz_input,
  int jac_bandwidth_lower_input,
  int jac_bandwidth_upper_input,
  std::unique_ptr<ExprSet> functions_arg,
  const SetupOptions &setup_input,
  const SolverOptions &solver_input
) :
  atol_np(atol_np_input),
  rhs_alg_id(rhs_alg_id_input),
  number_of_states(atol_np_input.request().size),
  number_of_parameters(number_of_parameters_input),
  number_of_events(number_of_events_input),
  jac_times_cjmass_nnz(jac_times_cjmass_nnz_input),
  jac_bandwidth_lower(jac_bandwidth_lower_input),
  jac_bandwidth_upper(jac_bandwidth_upper_input),
  functions(std::move(functions_arg)),
  sensitivity(number_of_parameters > 0),
  save_outputs_only(functions->var_fcns.size() > 0),
  setup_opts(setup_input),
  solver_opts(solver_input)
{
  // Construction code moved to Initialize() which is called from the
  // (child) IDAKLUSolver_* class constructors.
  DEBUG("IDAKLUSolverOpenMP:IDAKLUSolverOpenMP");
  auto atol = atol_np.unchecked<1>();

  // create SUNDIALS context object
  SUNContext_Create(NULL, &sunctx);  // calls null-wrapper if Sundials Ver<6

  // allocate memory for solver
  ida_mem = IDACreate(sunctx);

  // create the vector of initial values
  AllocateVectors();
  if (sensitivity) {
    yyS = N_VCloneVectorArray(number_of_parameters, yy);
    yypS = N_VCloneVectorArray(number_of_parameters, yyp);
  }
  // set initial values
  realtype *atval = N_VGetArrayPointer(avtol);
  for (int i = 0; i < number_of_states; i++) {
    atval[i] = atol[i];
  }

  for (int is = 0; is < number_of_parameters; is++) {
    N_VConst(RCONST(0.0), yyS[is]);
    N_VConst(RCONST(0.0), yypS[is]);
  }

  // create Matrix objects
  SetMatrix();

  // initialise solver
  IDAInit(ida_mem, residual_eval<ExprSet>, 0, yy, yyp);

  // set tolerances
  rtol = RCONST(rel_tol);
  IDASVtolerances(ida_mem, rtol, avtol);

  // Set events
  IDARootInit(ida_mem, number_of_events, events_eval<ExprSet>);

  // Set user data
  void *user_data = functions.get();
  IDASetUserData(ida_mem, user_data);

  // Specify preconditioner type
  precon_type = SUN_PREC_NONE;
  if (this->setup_opts.preconditioner != "none") {
    precon_type = SUN_PREC_LEFT;
  }

  // The default is to solve a DAE for generality. This may be changed
  // to an ODE during the Initialize() call
  is_ODE = false;

  // Will be overwritten during the solve() call
  save_hermite = solver_opts.hermite_interpolation;
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::AllocateVectors() {
  DEBUG("IDAKLUSolverOpenMP::AllocateVectors (num_threads = " << setup_opts.num_threads << ")");
  // Create vectors
  if (setup_opts.num_threads == 1) {
    yy = N_VNew_Serial(number_of_states, sunctx);
    yyp = N_VNew_Serial(number_of_states, sunctx);
    y_cache = N_VNew_Serial(number_of_states, sunctx);
    avtol = N_VNew_Serial(number_of_states, sunctx);
    id = N_VNew_Serial(number_of_states, sunctx);
  } else {
    DEBUG("IDAKLUSolverOpenMP::AllocateVectors OpenMP");
    yy = N_VNew_OpenMP(number_of_states, setup_opts.num_threads, sunctx);
    yyp = N_VNew_OpenMP(number_of_states, setup_opts.num_threads, sunctx);
    y_cache = N_VNew_OpenMP(number_of_states, setup_opts.num_threads, sunctx);
    avtol = N_VNew_OpenMP(number_of_states, setup_opts.num_threads, sunctx);
    id = N_VNew_OpenMP(number_of_states, setup_opts.num_threads, sunctx);
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::InitializeStorage(int const N) {
  length_of_return_vector = ReturnVectorLength();

  t = vector<realtype>(N, 0.0);

  y = vector<vector<realtype>>(
      N,
      vector<realtype>(length_of_return_vector, 0.0)
  );

  yS = vector<vector<vector<realtype>>>(
      N,
      vector<vector<realtype>>(
          number_of_parameters,
          vector<realtype>(length_of_return_vector, 0.0)
      )
  );

  if (save_hermite) {
    InitializeHermiteStorage(N);
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::InitializeHermiteStorage(int const N) {
  yp = vector<vector<realtype>>(
      N,
      vector<realtype>(number_of_states, 0.0)
  );

  ypS = vector<vector<vector<realtype>>>(
      N,
      vector<vector<realtype>>(
          number_of_parameters,
          vector<realtype>(number_of_states, 0.0)
      )
  );
}

template <class ExprSet>
int IDAKLUSolverOpenMP<ExprSet>::ReturnVectorLength() {
  if (!save_outputs_only) {
    return number_of_states;
  }

  // set return vectors
  int length_of_return_vector = 0;
  size_t max_res_size = 0;  // maximum result size (for common result buffer)
  size_t max_res_dvar_dy = 0, max_res_dvar_dp = 0;
  // return only the requested variables list after computation
  for (auto& var_fcn : functions->var_fcns) {
    max_res_size = std::max(max_res_size, size_t(var_fcn->out_shape(0)));
    length_of_return_vector += var_fcn->nnz_out();
    for (auto& dvar_fcn : functions->dvar_dy_fcns) {
      max_res_dvar_dy = std::max(max_res_dvar_dy, size_t(dvar_fcn->out_shape(0)));
    }
    for (auto& dvar_fcn : functions->dvar_dp_fcns) {
      max_res_dvar_dp = std::max(max_res_dvar_dp, size_t(dvar_fcn->out_shape(0)));
    }

    res.resize(max_res_size);
    res_dvar_dy.resize(max_res_dvar_dy);
    res_dvar_dp.resize(max_res_dvar_dp);
  }

  return length_of_return_vector;
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetSolverOptions() {
  // Maximum order of the linear multistep method
  CheckErrors(IDASetMaxOrd(ida_mem, solver_opts.max_order_bdf));

  // Maximum number of steps to be taken by the solver in its attempt to reach
  // the next output time
  CheckErrors(IDASetMaxNumSteps(ida_mem, solver_opts.max_num_steps));

  // Initial step size
  CheckErrors(IDASetInitStep(ida_mem, solver_opts.dt_init));

  // Minimum absolute step size
  CheckErrors(IDASetMinStep(ida_mem, solver_opts.dt_min));

  // Maximum absolute step size
  CheckErrors(IDASetMaxStep(ida_mem, solver_opts.dt_max));

  // Maximum number of error test failures in attempting one step
  CheckErrors(IDASetMaxErrTestFails(ida_mem, solver_opts.max_error_test_failures));

  // Maximum number of nonlinear solver iterations at one step
  CheckErrors(IDASetMaxNonlinIters(ida_mem, solver_opts.max_nonlinear_iterations));

  // Maximum number of nonlinear solver convergence failures at one step
  CheckErrors(IDASetMaxConvFails(ida_mem, solver_opts.max_convergence_failures));

  // Safety factor in the nonlinear convergence test
  CheckErrors(IDASetNonlinConvCoef(ida_mem, solver_opts.nonlinear_convergence_coefficient));

  // Suppress algebraic variables from error test
  CheckErrors(IDASetSuppressAlg(ida_mem, solver_opts.suppress_algebraic_error));

  // Positive constant in the Newton iteration convergence test within the initial
  // condition calculation
  CheckErrors(IDASetNonlinConvCoefIC(ida_mem, solver_opts.nonlinear_convergence_coefficient_ic));

  // Maximum number of steps allowed when icopt=IDA_YA_YDP_INIT in IDACalcIC
  CheckErrors(IDASetMaxNumStepsIC(ida_mem, solver_opts.max_num_steps_ic));

  // Maximum number of the approximate Jacobian or preconditioner evaluations
  // allowed when the Newton iteration appears to be slowly converging
  CheckErrors(IDASetMaxNumJacsIC(ida_mem, solver_opts.max_num_jacobians_ic));

  // Maximum number of Newton iterations allowed in any one attempt to solve
  // the initial conditions calculation problem
  CheckErrors(IDASetMaxNumItersIC(ida_mem, solver_opts.max_num_iterations_ic));

  // Maximum number of linesearch backtracks allowed in any Newton iteration,
  // when solving the initial conditions calculation problem
  CheckErrors(IDASetMaxBacksIC(ida_mem, solver_opts.max_linesearch_backtracks_ic));

  // Turn off linesearch
  CheckErrors(IDASetLineSearchOffIC(ida_mem, solver_opts.linesearch_off_ic));

  // Ratio between linear and nonlinear tolerances
  CheckErrors(IDASetEpsLin(ida_mem, solver_opts.epsilon_linear_tolerance));

  // Increment factor used in DQ Jv approximation
  CheckErrors(IDASetIncrementFactor(ida_mem, solver_opts.increment_factor));

  int LS_type = SUNLinSolGetType(LS);
  if (LS_type == SUNLINEARSOLVER_DIRECT || LS_type == SUNLINEARSOLVER_MATRIX_ITERATIVE) {
    // Enable or disable linear solution scaling
    CheckErrors(IDASetLinearSolutionScaling(ida_mem, solver_opts.linear_solution_scaling));
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetMatrix() {
  // Create Matrix object
  if (setup_opts.jacobian == "sparse") {
    DEBUG("\tsetting sparse matrix");
    J = SUNSparseMatrix(
      number_of_states,
      number_of_states,
      jac_times_cjmass_nnz,
      CSC_MAT,
      sunctx
    );
  } else if (setup_opts.jacobian == "banded") {
    DEBUG("\tsetting banded matrix");
    J = SUNBandMatrix(
      number_of_states,
      jac_bandwidth_upper,
      jac_bandwidth_lower,
      sunctx
    );
  } else if (setup_opts.jacobian == "dense" || setup_opts.jacobian == "none") {
    DEBUG("\tsetting dense matrix");
    J = SUNDenseMatrix(
      number_of_states,
      number_of_states,
      sunctx
    );
  } else if (setup_opts.jacobian == "matrix-free") {
    DEBUG("\tsetting matrix-free");
    J = NULL;
  } else {
    throw std::invalid_argument("Unsupported matrix requested");
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::Initialize() {
  // Call after setting the solver

  // attach the linear solver
  if (LS == nullptr) {
    throw std::invalid_argument("Linear solver not set");
  }
  CheckErrors(IDASetLinearSolver(ida_mem, LS, J));

  if (setup_opts.preconditioner != "none") {
    DEBUG("\tsetting IDADDB preconditioner");
    // setup preconditioner
    CheckErrors(IDABBDPrecInit(
      ida_mem, number_of_states, setup_opts.precon_half_bandwidth,
      setup_opts.precon_half_bandwidth, setup_opts.precon_half_bandwidth_keep,
      setup_opts.precon_half_bandwidth_keep, 0.0, residual_eval_approx<ExprSet>, NULL));
  }

  if (setup_opts.jacobian == "matrix-free") {
    CheckErrors(IDASetJacTimes(ida_mem, NULL, jtimes_eval<ExprSet>));
  } else if (setup_opts.jacobian != "none") {
    CheckErrors(IDASetJacFn(ida_mem, jacobian_eval<ExprSet>));
  }

  if (sensitivity) {
    CheckErrors(IDASensInit(ida_mem, number_of_parameters, IDA_SIMULTANEOUS,
      sensitivities_eval<ExprSet>, yyS, yypS));
    CheckErrors(IDASensEEtolerances(ida_mem));
  }

  CheckErrors(SUNLinSolInitialize(LS));

  auto id_np_val = rhs_alg_id.unchecked<1>();
  realtype *id_val;
  id_val = N_VGetArrayPointer(id);

  // Determine if the system is an ODE
  is_ODE = number_of_states > 0;
  for (int ii = 0; ii < number_of_states; ii++) {
    id_val[ii] = id_np_val[ii];
    // check if id_val[ii] approximately equals 1 (>0.999) handles
    // cases where id_val[ii] is not exactly 1 due to numerical errors
    is_ODE &= id_val[ii] > 0.999;
  }

  // Variable types: differential (1) and algebraic (0)
  CheckErrors(IDASetId(ida_mem, id));
}

template <class ExprSet>
IDAKLUSolverOpenMP<ExprSet>::~IDAKLUSolverOpenMP() {
  DEBUG("IDAKLUSolverOpenMP::~IDAKLUSolverOpenMP");
  // Free memory
  if (sensitivity) {
      IDASensFree(ida_mem);
  }

  CheckErrors(SUNLinSolFree(LS));

  SUNMatDestroy(J);
  N_VDestroy(avtol);
  N_VDestroy(yy);
  N_VDestroy(yyp);
  N_VDestroy(y_cache);
  N_VDestroy(id);

  if (sensitivity) {
    N_VDestroyVectorArray(yyS, number_of_parameters);
    N_VDestroyVectorArray(yypS, number_of_parameters);
  }

  IDAFree(&ida_mem);
  SUNContext_Free(&sunctx);
}

template <class ExprSet>
SolutionData IDAKLUSolverOpenMP<ExprSet>::solve(
  const std::vector<realtype> &t_eval,
  const std::vector<realtype> &t_interp,
  const realtype *y0,
  const realtype *yp0,
  const realtype *inputs,
  bool save_adaptive_steps,
  bool save_interp_steps
)
{
  DEBUG("IDAKLUSolver::solve");
  const int number_of_evals = t_eval.size();
  const int number_of_interps = t_interp.size();

  // Hermite interpolation is only available when saving
  // 1. adaptive steps and 2. the full solution
  save_hermite = (
    solver_opts.hermite_interpolation &&
    save_adaptive_steps &&
    !save_outputs_only
  );

  InitializeStorage(number_of_evals + number_of_interps);

  int i_save = 0;

  realtype t0 = t_eval.front();
  realtype tf = t_eval.back();

  realtype t_val = t0;
  realtype t_prev = t0;
  int i_eval = 0;

  realtype t_interp_next;
  int i_interp = 0;
  // If t_interp is empty, save all adaptive steps
  if (save_interp_steps) {
    t_interp_next = t_interp[0];
  }

  auto n_coeffs = number_of_states + number_of_parameters * number_of_states;

  // set inputs
  for (int i = 0; i < functions->inputs.size(); i++) {
    functions->inputs[i] = inputs[i];
  }

  // Setup consistent initialization
  realtype *y_val = N_VGetArrayPointer(yy);
  realtype *yp_val = N_VGetArrayPointer(yyp);
  vector<realtype *> yS_val(number_of_parameters);
  vector<realtype *> ypS_val(number_of_parameters);
  for (int p = 0 ; p < number_of_parameters; p++) {
    yS_val[p] = N_VGetArrayPointer(yyS[p]);
    ypS_val[p] = N_VGetArrayPointer(yypS[p]);
    for (int i = 0; i < number_of_states; i++) {
      yS_val[p][i] = y0[i + (p + 1) * number_of_states];
      ypS_val[p][i] = yp0[i + (p + 1) * number_of_states];
    }
  }

  for (int i = 0; i < number_of_states; i++) {
    y_val[i] = y0[i];
    yp_val[i] = yp0[i];
  }

  SetSolverOptions();

  // Prepare first time step
  i_eval = 1;
  realtype t_eval_next = t_eval[i_eval];


  // Consistent initialization
  ReinitializeIntegrator(t0);
  int const init_type = solver_opts.init_all_y_ic ? IDA_Y_INIT : IDA_YA_YDP_INIT;
  if (solver_opts.calc_ic) {
    ConsistentInitialization(t0, t_eval_next, init_type);
  }

  // Set the initial stop time
  IDASetStopTime(ida_mem, t_eval_next);

  // Progress one step. This must be done before the while loop to ensure
  // that we can run IDAGetDky at t0 for dky = 1
  int retval = IDASolve(ida_mem, tf, &t_val, yy, yyp, IDA_ONE_STEP);

  // Store consistent initialization
  CheckErrors(IDAGetDky(ida_mem, t0, 0, yy));
  if (sensitivity) {
    CheckErrors(IDAGetSensDky(ida_mem, t0, 0, yyS));
  }

  SetStep(t0, y_val, yp_val, yS_val, ypS_val, i_save);

  // Reset the states at t = t_val. Sensitivities are handled in the while-loop
  CheckErrors(IDAGetDky(ida_mem, t_val, 0, yy));

  // Solve the system
  DEBUG("IDASolve");
  while (true) {
    if (retval < 0) {
      // failed
      break;
    } else if (t_prev == t_val) {
      // IDA sometimes returns an identical time point twice
      // instead of erroring. Assign a retval and break
      retval = IDA_ERR_FAIL;
      break;
    }

    bool hit_tinterp = save_interp_steps && t_interp_next >= t_prev;
    bool hit_teval = retval == IDA_TSTOP_RETURN;
    bool hit_final_time = t_val >= tf || (hit_teval && i_eval == number_of_evals);
    bool hit_event = retval == IDA_ROOT_RETURN;
    bool hit_adaptive = save_adaptive_steps && retval == IDA_SUCCESS;

    if (sensitivity) {
      CheckErrors(IDAGetSensDky(ida_mem, t_val, 0, yyS));
    }

    if (hit_tinterp) {
      // Save the interpolated state at t_prev < t < t_val, for all t in t_interp
      SetStepInterp(
        i_interp,
        t_interp_next,
        t_interp,
        t_val,
        t_prev,
        t_eval_next,
        y_val,
        yp_val,
        yS_val,
        ypS_val,
        i_save);
    }

    if (hit_adaptive || hit_teval || hit_event || hit_final_time) {
      if (hit_tinterp) {
        // Reset the states and sensitivities at t = t_val
        CheckErrors(IDAGetDky(ida_mem, t_val, 0, yy));
        if (sensitivity) {
          CheckErrors(IDAGetSensDky(ida_mem, t_val, 0, yyS));
        }
      }

      // Save the current state at t_val
      // First, check to make sure that the t_val is not equal to the current t value
      // If it is, we don't want to save the current state twice
      if (!hit_tinterp || t_val != t.back()) {
        if (hit_adaptive) {
          // Dynamically allocate memory for the adaptive step
          ExtendAdaptiveArrays();
        }

        SetStep(t_val, y_val, yp_val, yS_val, ypS_val, i_save);
      }
    }

    if (hit_final_time || hit_event) {
      // Successful simulation. Exit the while loop
      break;
    } else if (hit_teval) {
      // Set the next stop time
      i_eval++;
      t_eval_next = t_eval[i_eval];
      CheckErrors(IDASetStopTime(ida_mem, t_eval_next));

      // Reinitialize the solver to deal with the discontinuity at t = t_val.
      ReinitializeIntegrator(t_val);
      ConsistentInitialization(t_val, t_eval_next, IDA_YA_YDP_INIT);
    }

    t_prev = t_val;

    // Progress one step
    retval = IDASolve(ida_mem, tf, &t_val, yy, yyp, IDA_ONE_STEP);
  }

  int const length_of_final_sv_slice = save_outputs_only ? number_of_states : 0;
  realtype *yterm_return = new realtype[length_of_final_sv_slice];
  if (save_outputs_only) {
    // store final state slice if outout variables are specified
    std::memcpy(yterm_return, y_val, length_of_final_sv_slice * sizeof(realtype*));
  }

  if (solver_opts.print_stats) {
    PrintStats();
  }

  // store number of timesteps so we can generate the solution later
  number_of_timesteps = i_save;

  // Copy the data to return as numpy arrays

  // Time, t
  realtype *t_return = new realtype[number_of_timesteps];
  for (size_t i = 0; i < number_of_timesteps; i++) {
    t_return[i] = t[i];
  }

  // States, y
  realtype *y_return = new realtype[number_of_timesteps * length_of_return_vector];
  int count = 0;
  for (size_t i = 0; i < number_of_timesteps; i++) {
    for (size_t j = 0; j < length_of_return_vector; j++) {
      y_return[count] = y[i][j];
      count++;
    }
  }

  // Sensitivity states, yS
  // Note: Ordering of vector is different if computing outputs vs returning
  // the complete state vector
  auto const arg_sens0 = (save_outputs_only ? number_of_timesteps : number_of_parameters);
  auto const arg_sens1 = (save_outputs_only ? length_of_return_vector : number_of_timesteps);
  auto const arg_sens2 = (save_outputs_only ? number_of_parameters : length_of_return_vector);

  realtype *yS_return = new realtype[arg_sens0 * arg_sens1 * arg_sens2];
  count = 0;
  for (size_t idx0 = 0; idx0 < arg_sens0; idx0++) {
    for (size_t idx1 = 0; idx1 < arg_sens1; idx1++) {
      for (size_t idx2 = 0; idx2 < arg_sens2; idx2++) {
        auto i = (save_outputs_only ? idx0 : idx1);
        auto j = (save_outputs_only ? idx1 : idx2);
        auto k = (save_outputs_only ? idx2 : idx0);

        yS_return[count] = yS[i][k][j];
        count++;
      }
    }
  }

  realtype *yp_return = new realtype[(save_hermite ? 1 : 0) * (number_of_timesteps * number_of_states)];
  realtype *ypS_return = new realtype[(save_hermite ? 1 : 0) * (arg_sens0 * arg_sens1 * arg_sens2)];
  if (save_hermite) {
    count = 0;
    for (size_t i = 0; i < number_of_timesteps; i++) {
      for (size_t j = 0; j < number_of_states; j++) {
        yp_return[count] = yp[i][j];
        count++;
      }
    }

    // Sensitivity states, ypS
    // Note: Ordering of vector is different if computing outputs vs returning
    // the complete state vector
    count = 0;
    for (size_t idx0 = 0; idx0 < arg_sens0; idx0++) {
      for (size_t idx1 = 0; idx1 < arg_sens1; idx1++) {
        for (size_t idx2 = 0; idx2 < arg_sens2; idx2++) {
          auto i = (save_outputs_only ? idx0 : idx1);
          auto j = (save_outputs_only ? idx1 : idx2);
          auto k = (save_outputs_only ? idx2 : idx0);

          ypS_return[count] = ypS[i][k][j];
          count++;
        }
      }
    }
  }

  return SolutionData(
    retval,
    number_of_timesteps,
    length_of_return_vector,
    arg_sens0,
    arg_sens1,
    arg_sens2,
    length_of_final_sv_slice,
    save_hermite,
    t_return,
    y_return,
    yp_return,
    yS_return,
    ypS_return,
    yterm_return);
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::ExtendAdaptiveArrays() {
  DEBUG("IDAKLUSolver::ExtendAdaptiveArrays");
  // Time
  t.emplace_back(0.0);

  // States
  y.emplace_back(length_of_return_vector, 0.0);

  // Sensitivity
  if (sensitivity) {
    yS.emplace_back(number_of_parameters, vector<realtype>(length_of_return_vector, 0.0));
  }

  if (save_hermite) {
    ExtendHermiteArrays();
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::ExtendHermiteArrays() {
  DEBUG("IDAKLUSolver::ExtendHermiteArrays");
  // States
  yp.emplace_back(number_of_states, 0.0);

  // Sensitivity
  if (sensitivity) {
    ypS.emplace_back(number_of_parameters, vector<realtype>(number_of_states, 0.0));
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::ReinitializeIntegrator(const realtype& t_val) {
  DEBUG("IDAKLUSolver::ReinitializeIntegrator");
  CheckErrors(IDAReInit(ida_mem, t_val, yy, yyp));
  if (sensitivity) {
    CheckErrors(IDASensReInit(ida_mem, IDA_SIMULTANEOUS, yyS, yypS));
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::ConsistentInitialization(
  const realtype& t_val,
  const realtype& t_next,
  const int& icopt) {
  DEBUG("IDAKLUSolver::ConsistentInitialization");

  if (is_ODE && icopt == IDA_YA_YDP_INIT) {
    ConsistentInitializationODE(t_val);
  } else {
    ConsistentInitializationDAE(t_val, t_next, icopt);
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::ConsistentInitializationDAE(
  const realtype& t_val,
  const realtype& t_next,
  const int& icopt) {
  DEBUG("IDAKLUSolver::ConsistentInitializationDAE");
  IDACalcIC(ida_mem, icopt, t_next);
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::ConsistentInitializationODE(
  const realtype& t_val) {
  DEBUG("IDAKLUSolver::ConsistentInitializationODE");

  // For ODEs where the mass matrix M = I, we can simplify the problem
  // by analytically computing the yp values. If we take our implicit
  // DAE system res(t,y,yp) = f(t,y) - I*yp, then yp = res(t,y,0). This
  // avoids an expensive call to IDACalcIC.
  realtype *y_cache_val = N_VGetArrayPointer(y_cache);
  std::memset(y_cache_val, 0, number_of_states * sizeof(realtype));
  // Overwrite yp
  residual_eval<ExprSet>(t_val, yy, y_cache, yyp, functions.get());
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStep(
  realtype &tval,
  realtype *y_val,
  realtype *yp_val,
  vector<realtype *> const &yS_val,
  vector<realtype *> const &ypS_val,
  int &i_save
) {
  // Set adaptive step results for y and yS
  DEBUG("IDAKLUSolver::SetStep");

  // Time
  t[i_save] = tval;

  if (save_outputs_only) {
    SetStepOutput(tval, y_val, yS_val, i_save);
  } else {
    SetStepFull(tval, y_val, yS_val, i_save);

    if (save_hermite) {
      SetStepHermite(tval, yp_val, ypS_val, i_save);
    }
  }

  i_save++;
}


template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStepInterp(
  int &i_interp,
  realtype &t_interp_next,
  vector<realtype> const &t_interp,
  realtype &t_val,
  realtype &t_prev,
  realtype const &t_eval_next,
  realtype *y_val,
  realtype *yp_val,
  vector<realtype *> const &yS_val,
  vector<realtype *> const &ypS_val,
  int &i_save
  ) {
  // Save the state at the requested time
  DEBUG("IDAKLUSolver::SetStepInterp");

  while (i_interp <= (t_interp.size()-1) && t_interp_next <= t_val) {
    CheckErrors(IDAGetDky(ida_mem, t_interp_next, 0, yy));
    if (sensitivity) {
      CheckErrors(IDAGetSensDky(ida_mem, t_interp_next, 0, yyS));
    }

    // Memory is already allocated for the interpolated values
    SetStep(t_interp_next, y_val, yp_val, yS_val, ypS_val, i_save);

    i_interp++;
    if (i_interp == (t_interp.size())) {
      // Reached the final t_interp value
      break;
    }
    t_interp_next = t_interp[i_interp];
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStepFull(
  realtype &tval,
  realtype *y_val,
  vector<realtype *> const &yS_val,
  int &i_save
) {
  // Set adaptive step results for y and yS
  DEBUG("IDAKLUSolver::SetStepFull");

  // States
  auto &y_back = y[i_save];
  for (size_t j = 0; j < number_of_states; ++j) {
    y_back[j] = y_val[j];
  }

  // Sensitivity
  if (sensitivity) {
    SetStepFullSensitivities(tval, y_val, yS_val, i_save);
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStepFullSensitivities(
  realtype &tval,
  realtype *y_val,
  vector<realtype *> const &yS_val,
  int &i_save
) {
  DEBUG("IDAKLUSolver::SetStepFullSensitivities");

  // Calculate sensitivities for the full yS array
  for (size_t j = 0; j < number_of_parameters; ++j) {
    auto &yS_back_j = yS[i_save][j];
    auto &ySval_j = yS_val[j];
    for (size_t k = 0; k < number_of_states; ++k) {
      yS_back_j[k] = ySval_j[k];
    }
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStepOutput(
    realtype &tval,
    realtype *y_val,
    const vector<realtype*>& yS_val,
    int &i_save
) {
  DEBUG("IDAKLUSolver::SetStepOutput");
  // Evaluate functions for each requested variable and store

  size_t j = 0;
  for (auto& var_fcn : functions->var_fcns) {
    (*var_fcn)({&tval, y_val, functions->inputs.data()}, {&res[0]});
    // store in return vector
    for (size_t jj=0; jj<var_fcn->nnz_out(); jj++) {
      y[i_save][j++] = res[jj];
    }
  }
  // calculate sensitivities
  if (sensitivity) {
    SetStepOutputSensitivities(tval, y_val, yS_val, i_save);
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStepOutputSensitivities(
  realtype &tval,
  realtype *y_val,
  const vector<realtype*>& yS_val,
    int &i_save
  ) {
  DEBUG("IDAKLUSolver::SetStepOutputSensitivities");
  // Calculate sensitivities
  vector<realtype> dens_dvar_dp = vector<realtype>(number_of_parameters, 0);
  for (size_t dvar_k=0; dvar_k<functions->dvar_dy_fcns.size(); dvar_k++) {
    // Isolate functions
    Expression* dvar_dy = functions->dvar_dy_fcns[dvar_k];
    Expression* dvar_dp = functions->dvar_dp_fcns[dvar_k];
    // Calculate dvar/dy
    (*dvar_dy)({&tval, y_val, functions->inputs.data()}, {&res_dvar_dy[0]});
    // Calculate dvar/dp and convert to dense array for indexing
    (*dvar_dp)({&tval, y_val, functions->inputs.data()}, {&res_dvar_dp[0]});
    for (int k=0; k<number_of_parameters; k++) {
      dens_dvar_dp[k]=0;
    }
    for (int k=0; k<dvar_dp->nnz_out(); k++) {
      dens_dvar_dp[dvar_dp->get_row()[k]] = res_dvar_dp[k];
    }
    // Calculate sensitivities
    for (int paramk=0; paramk<number_of_parameters; paramk++) {
      auto &yS_back_paramk = yS[i_save][paramk];
      yS_back_paramk[dvar_k] = dens_dvar_dp[paramk];

      for (int spk=0; spk<dvar_dy->nnz_out(); spk++) {
        yS_back_paramk[dvar_k] += res_dvar_dy[spk] * yS_val[paramk][dvar_dy->get_col()[spk]];
      }
    }
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStepHermite(
  realtype &tval,
  realtype *yp_val,
  vector<realtype *> const &ypS_val,
  int &i_save
) {
  // Set adaptive step results for yp and ypS
  DEBUG("IDAKLUSolver::SetStepHermite");

  // States
  CheckErrors(IDAGetDky(ida_mem, tval, 1, yyp));
  auto &yp_back = yp[i_save];
    for (size_t j = 0; j < length_of_return_vector; ++j) {
    yp_back[j] = yp_val[j];

  }

  // Sensitivity
  if (sensitivity) {
    SetStepHermiteSensitivities(tval, yp_val, ypS_val, i_save);
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStepHermiteSensitivities(
  realtype &tval,
  realtype *yp_val,
  vector<realtype *> const &ypS_val,
  int &i_save
) {
  DEBUG("IDAKLUSolver::SetStepHermiteSensitivities");

  // Calculate sensitivities for the full ypS array
  CheckErrors(IDAGetSensDky(ida_mem, tval, 1, yypS));
  for (size_t j = 0; j < number_of_parameters; ++j) {
    auto &ypS_back_j = ypS[i_save][j];
    auto &ypSval_j = ypS_val[j];
    for (size_t k = 0; k < number_of_states; ++k) {
      ypS_back_j[k] = ypSval_j[k];
    }
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::CheckErrors(int const & flag) {
  if (flag < 0) {
    auto message = std::string("IDA failed with flag ") + std::to_string(flag);
    throw std::runtime_error(message.c_str());
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::PrintStats() {
  long nsteps, nrevals, nlinsetups, netfails;
  int klast, kcur;
  realtype hinused, hlast, hcur, tcur;

  CheckErrors(IDAGetIntegratorStats(
    ida_mem,
    &nsteps,
    &nrevals,
    &nlinsetups,
    &netfails,
    &klast,
    &kcur,
    &hinused,
    &hlast,
    &hcur,
    &tcur
  ));

  long nniters, nncfails;
  CheckErrors(IDAGetNonlinSolvStats(ida_mem, &nniters, &nncfails));

  long int ngevalsBBDP = 0;
  if (setup_opts.using_iterative_solver) {
    CheckErrors(IDABBDPrecGetNumGfnEvals(ida_mem, &ngevalsBBDP));
  }

  py::print("Solver Stats:");
  py::print("\tNumber of steps =", nsteps);
  py::print("\tNumber of calls to residual function =", nrevals);
  py::print("\tNumber of calls to residual function in preconditioner =",
            ngevalsBBDP);
  py::print("\tNumber of linear solver setup calls =", nlinsetups);
  py::print("\tNumber of error test failures =", netfails);
  py::print("\tMethod order used on last step =", klast);
  py::print("\tMethod order used on next step =", kcur);
  py::print("\tInitial step size =", hinused);
  py::print("\tStep size on last step =", hlast);
  py::print("\tStep size on next step =", hcur);
  py::print("\tCurrent internal time reached =", tcur);
  py::print("\tNumber of nonlinear iterations performed =", nniters);
  py::print("\tNumber of nonlinear convergence failures =", nncfails);
}
