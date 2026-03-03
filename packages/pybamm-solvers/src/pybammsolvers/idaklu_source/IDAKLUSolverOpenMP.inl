#pragma once

#include "Expressions/Expressions.hpp"
#include "sundials_functions.hpp"
#include <vector>
#include "common.hpp"
#include "SolutionData.hpp"
#include "sundials_error_handler.hpp"

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

  // Optionally silence SUNDIALS error messages (handled in PyBaMM)
  #if SUNDIALS_VERSION_MAJOR >= 7
    if (solver_input.silence_sundials_errors) {
      SUNContext_ClearErrHandlers(sunctx);
    }
  #endif

  // allocate memory for solver
  ida_mem = IDACreate(sunctx);

  // create the vector of initial values
  AllocateVectors();
  if (sensitivity) {
    yyS = N_VCloneVectorArray(number_of_parameters, yy);
    yypS = N_VCloneVectorArray(number_of_parameters, yyp);
  }
  // set initial values
  sunrealtype *atval = N_VGetArrayPointer(avtol);
  for (int i = 0; i < number_of_states; i++) {
    atval[i] = atol[i];
  }

  for (int is = 0; is < number_of_parameters; is++) {
    N_VConst(SUN_RCONST(0.0), yyS[is]);
    N_VConst(SUN_RCONST(0.0), yypS[is]);
  }

  // create Matrix objects
  SetMatrix();

  // initialise solver
  IDAInit(ida_mem, residual_eval<ExprSet>, 0, yy, yyp);

  // set tolerances
  rtol = SUN_RCONST(rel_tol);
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
int IDAKLUSolverOpenMP<ExprSet>::ReturnVectorLength() {
  if (!save_outputs_only) {
    return number_of_states;
  }

  // Compute the total length of the output variable vector
  int length_of_return_vector = 0;
  for (auto& var_fcn : functions->var_fcns) {
    length_of_return_vector += var_fcn->nnz_out();
  }
  return length_of_return_vector;
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetSolverOptions() {
  // Maximum order of the linear multistep method
  CheckErrors(IDASetMaxOrd(ida_mem, solver_opts.max_order_bdf), "IDASetMaxOrd");

  // Maximum number of steps to be taken by the solver in its attempt to reach
  // the next output time
  CheckErrors(IDASetMaxNumSteps(ida_mem, solver_opts.max_num_steps), "IDASetMaxNumSteps");

  // Initial step size
  CheckErrors(IDASetInitStep(ida_mem, solver_opts.dt_init), "IDASetInitStep");

  // Minimum absolute step size
  CheckErrors(IDASetMinStep(ida_mem, solver_opts.dt_min), "IDASetMinStep");

  // Maximum absolute step size
  CheckErrors(IDASetMaxStep(ida_mem, solver_opts.dt_max), "IDASetMaxStep");

  // Maximum number of error test failures in attempting one step
  CheckErrors(IDASetMaxErrTestFails(ida_mem, solver_opts.max_error_test_failures), "IDASetMaxErrTestFails");

  // Maximum number of nonlinear solver iterations at one step
  CheckErrors(IDASetMaxNonlinIters(ida_mem, solver_opts.max_nonlinear_iterations), "IDASetMaxNonlinIters");

  // Maximum number of nonlinear solver convergence failures at one step
  CheckErrors(IDASetMaxConvFails(ida_mem, solver_opts.max_convergence_failures), "IDASetMaxConvFails");

  // Safety factor in the nonlinear convergence test
  CheckErrors(IDASetNonlinConvCoef(ida_mem, solver_opts.nonlinear_convergence_coefficient), "IDASetNonlinConvCoef");

  // Suppress algebraic variables from error test
  CheckErrors(IDASetSuppressAlg(ida_mem, solver_opts.suppress_algebraic_error), "IDASetSuppressAlg");

  // Positive constant in the Newton iteration convergence test within the initial
  // condition calculation
  CheckErrors(IDASetNonlinConvCoefIC(ida_mem, solver_opts.nonlinear_convergence_coefficient_ic), "IDASetNonlinConvCoefIC");

  // Maximum number of steps allowed when icopt=IDA_YA_YDP_INIT in IDACalcIC
  CheckErrors(IDASetMaxNumStepsIC(ida_mem, solver_opts.max_num_steps_ic), "IDASetMaxNumStepsIC");

  // Maximum number of the approximate Jacobian or preconditioner evaluations
  // allowed when the Newton iteration appears to be slowly converging
  CheckErrors(IDASetMaxNumJacsIC(ida_mem, solver_opts.max_num_jacobians_ic), "IDASetMaxNumJacsIC");

  // Maximum number of Newton iterations allowed in any one attempt to solve
  // the initial conditions calculation problem
  CheckErrors(IDASetMaxNumItersIC(ida_mem, solver_opts.max_num_iterations_ic), "IDASetMaxNumItersIC");

  // Maximum number of linesearch backtracks allowed in any Newton iteration,
  // when solving the initial conditions calculation problem
  CheckErrors(IDASetMaxBacksIC(ida_mem, solver_opts.max_linesearch_backtracks_ic), "IDASetMaxBacksIC");

  // Turn off linesearch
  CheckErrors(IDASetLineSearchOffIC(ida_mem, solver_opts.linesearch_off_ic), "IDASetLineSearchOffIC");

  // Ratio between linear and nonlinear tolerances
  CheckErrors(IDASetEpsLin(ida_mem, solver_opts.epsilon_linear_tolerance), "IDASetEpsLin");

  // Increment factor used in DQ Jv approximation
  CheckErrors(IDASetIncrementFactor(ida_mem, solver_opts.increment_factor), "IDASetIncrementFactor");

  int LS_type = SUNLinSolGetType(LS);
  if (LS_type == SUNLINEARSOLVER_DIRECT || LS_type == SUNLINEARSOLVER_MATRIX_ITERATIVE) {
    // Enable or disable linear solution scaling
    CheckErrors(IDASetLinearSolutionScaling(ida_mem, solver_opts.linear_solution_scaling), "IDASetLinearSolutionScaling");
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
  CheckErrors(IDASetLinearSolver(ida_mem, LS, J), "IDASetLinearSolver");

  if (setup_opts.preconditioner != "none") {
    DEBUG("\tsetting IDADDB preconditioner");
    // setup preconditioner
    CheckErrors(IDABBDPrecInit(
      ida_mem, number_of_states, setup_opts.precon_half_bandwidth,
      setup_opts.precon_half_bandwidth, setup_opts.precon_half_bandwidth_keep,
      setup_opts.precon_half_bandwidth_keep, 0.0, residual_eval_approx<ExprSet>, NULL), "IDABBDPrecInit");
  }

  if (setup_opts.jacobian == "matrix-free") {
    CheckErrors(IDASetJacTimes(ida_mem, NULL, jtimes_eval<ExprSet>), "IDASetJacTimes");
  } else if (setup_opts.jacobian != "none") {
    CheckErrors(IDASetJacFn(ida_mem, jacobian_eval<ExprSet>), "IDASetJacFn");
  }

  if (sensitivity) {
    CheckErrors(IDASensInit(ida_mem, number_of_parameters, IDA_SIMULTANEOUS,
      sensitivities_eval<ExprSet>, yyS, yypS), "IDASensInit");
    CheckErrors(IDASensEEtolerances(ida_mem), "IDASensEEtolerances");
  }

  CheckErrors(SUNLinSolInitialize(LS), "SUNLinSolInitialize");

  auto id_np_val = rhs_alg_id.unchecked<1>();
  sunrealtype *id_val;
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
  CheckErrors(IDASetId(ida_mem, id), "IDASetId");
}

template <class ExprSet>
IDAKLUSolverOpenMP<ExprSet>::~IDAKLUSolverOpenMP() {
  DEBUG("IDAKLUSolverOpenMP::~IDAKLUSolverOpenMP");
  // Free memory
  if (sensitivity) {
      IDASensFree(ida_mem);
  }

  CheckErrors(SUNLinSolFree(LS), "SUNLinSolFree");

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
  const std::vector<sunrealtype> &t_eval,
  const std::vector<sunrealtype> &t_interp,
  const sunrealtype *y0,
  const sunrealtype *yp0,
  const sunrealtype *inputs,
  bool save_adaptive_steps,
  bool save_interp_steps,
  py::object logger
)
{
  DEBUG("IDAKLUSolver::solve");

  log_ = SolverLog(std::move(logger));

  // Store solve parameters as member state
  save_adaptive_steps_ = save_adaptive_steps;
  save_interp_steps_ = save_interp_steps;

  const int number_of_evals = t_eval.size();

  // setup
  InitializeSolveStorage(number_of_evals, t_interp.size());
  SetupInitialState(t_eval, y0, yp0, inputs);

  sunrealtype t0 = t_eval.front();
  sunrealtype tf = t_eval.back();
  const bool increasing = (tf > t0);
  sunrealtype tf_perturbed = perturb_time(tf, increasing);

  sunrealtype t_val = t0;
  sunrealtype t_prev = t0;
  int i_eval = 1;
  sunrealtype t_eval_next = t_eval[i_eval];
  int i_interp = 0;
  sunrealtype t_interp_next = save_interp_steps_ ? t_interp[0] : 0;

  log_.log_start(t0, tf);

  // first step
  // Progress one step before the loop to ensure IDAGetDky works at t0 for dky = 1
  int n_steps = 0;
  int retval = IDASolve(ida_mem, tf_perturbed, &t_val, yy, yyp, IDA_ONE_STEP);
  log_.log_step(++n_steps, t_val);
  CheckErrors(retval, "IDASolve at t0");

  NoProgressGuard no_progression(solver_opts.num_steps_no_progress, solver_opts.t_no_progress);
  no_progression.Initialize();
  no_progression.AddDt(t_val - t0);

  StoreInitialPoint(t0);

  // Reset the states at t = t_val. Sensitivities are handled in the while-loop
  CheckErrors(IDAGetDky(ida_mem, t_val, 0, yy), "IDAGetDky at t_val");

  // main integration loop
  DEBUG("IDASolve");
  while (true) {
    if (retval < 0) {
      break;
    } else if (t_prev == t_val || no_progression.Violated()) {
      retval = IDA_ERR_FAIL;
      break;
    }

    bool hit_teval = retval == IDA_TSTOP_RETURN;
    bool hit_final_time = t_val >= tf || (hit_teval && i_eval == number_of_evals);
    bool hit_event = retval == IDA_ROOT_RETURN;
    bool hit_adaptive = save_adaptive_steps_ && retval == IDA_SUCCESS;

    if (sensitivity) {
      CheckErrors(IDAGetSensDky(ida_mem, t_val, 0, yyS), "IDAGetSensDky during solve");
    }

    if (save_interp_steps_ && !use_knot_reduction_ && t_interp_next >= t_prev) {
      SaveInterpPoints(i_interp, t_interp_next, t_interp,
                       t_val, t_prev, t_eval_next);
    }

    if (hit_adaptive || hit_teval || hit_event || hit_final_time) {
      bool will_exit_loop = hit_final_time || hit_event;
      bool is_breakpoint = (hit_teval || hit_event) && !will_exit_loop;
      SavePoint(t_val, /*extend_arrays=*/hit_adaptive, is_breakpoint);
    }

    if (hit_final_time || hit_event) {
      break;
    } else if (hit_teval) {
      HandleBreakpoint(t_val, t_eval, i_eval, t_eval_next, no_progression);
    }

    t_prev = t_val;
    retval = IDASolve(ida_mem, tf_perturbed, &t_val, yy, yyp, IDA_ONE_STEP);
    log_.log_step(++n_steps, t_val);
    no_progression.AddDt(t_val - t_prev);
  }

  log_.log_integration_complete(n_steps, t_val);

  return BuildSolutionData(retval);
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::InitializeSolveStorage(
  int n_evals, int n_interps
) {
  DEBUG("IDAKLUSolver::InitializeSolveStorage");

  // Determine mode
  save_hermite = (
    solver_opts.hermite_interpolation &&
    save_adaptive_steps_ &&
    !save_outputs_only
  );

  use_knot_reduction_ = (
    solver_opts.hermite_reduction_factor > 1.0 &&
    save_hermite &&
    !sensitivity
  );

  length_of_return_vector = ReturnVectorLength();
  i_save_ = 0;

  // Allocate output arrays. Pre-allocate 64 elements for initial storage.
  int est = std::max(n_evals + n_interps, 64);
  auto init_vec = [&](auto& v, size_t n) {
    v.clear();
    use_knot_reduction_ ? v.reserve(n) : v.resize(n, 0.0);
  };

  init_vec(t, est);
  init_vec(y, est * length_of_return_vector);
  if (save_hermite)  init_vec(yp, est * number_of_states);
  if (sensitivity) {
    init_vec(yS, est * number_of_parameters * length_of_return_vector);
    if (save_hermite)  init_vec(ypS, est * number_of_parameters * number_of_states);
  }

  // Allocate scratch buffers for save_outputs_only mode
  if (save_outputs_only) {
    size_t max_res_size = 0, max_res_dvar_dy = 0, max_res_dvar_dp = 0;
    for (auto& var_fcn : functions->var_fcns) {
      max_res_size = std::max(max_res_size, size_t(var_fcn->out_shape(0)));
    }
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

  // Create knot reducer if active
  if (use_knot_reduction_) {
    knot_reducer = std::make_unique<HermiteKnotReducer>(
      number_of_states, rtol, N_VGetArrayPointer(avtol),
      solver_opts.hermite_reduction_factor,
      t, y, yp
    );
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetupInitialState(
  const std::vector<sunrealtype> &t_eval,
  const sunrealtype *y0,
  const sunrealtype *yp0,
  const sunrealtype *inputs
) {
  DEBUG("IDAKLUSolver::SetupInitialState");

  // Set inputs
  for (size_t i = 0; i < functions->inputs.size(); i++) {
    functions->inputs[i] = inputs[i];
  }

  // Setup SUNDIALS vector pointers (member state)
  y_val_ = N_VGetArrayPointer(yy);
  yp_val_ = N_VGetArrayPointer(yyp);
  yS_val_.resize(number_of_parameters);
  ypS_val_.resize(number_of_parameters);
  for (int p = 0; p < number_of_parameters; p++) {
    yS_val_[p] = N_VGetArrayPointer(yyS[p]);
    ypS_val_[p] = N_VGetArrayPointer(yypS[p]);
    for (int i = 0; i < number_of_states; i++) {
      yS_val_[p][i] = y0[i + (p + 1) * number_of_states];
      ypS_val_[p][i] = yp0[i + (p + 1) * number_of_states];
    }
  }

  for (int i = 0; i < number_of_states; i++) {
    y_val_[i] = y0[i];
    yp_val_[i] = yp0[i];
  }

  SetSolverOptions();

  // Reset accumulated stats for this solve
  accumulated_stats.reset();

  // Consistent initialization
  sunrealtype t0 = t_eval.front();
  sunrealtype t_eval_next = t_eval[1];
  ReinitializeIntegrator(t0);
  int const init_type = solver_opts.init_all_y_ic ? IDA_Y_INIT : IDA_YA_YDP_INIT;
  if (solver_opts.calc_ic) {
    ConsistentInitialization(t0, t_eval_next, init_type);
    log_.log_consistent_init(t0);
  }

  // Set the initial stop time
  IDASetStopTime(ida_mem, t_eval_next);
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::StoreInitialPoint(sunrealtype t0) {
  DEBUG("IDAKLUSolver::StoreInitialPoint");

  // Retrieve consistent initial values
  CheckErrors(IDAGetDky(ida_mem, t0, 0, yy), "IDAGetDky at t0");
  if (sensitivity) {
    CheckErrors(IDAGetSensDky(ida_mem, t0, 0, yyS), "IDAGetSensDky at t0");
  }

  // First point: always a breakpoint (must be kept)
  if (use_knot_reduction_) {
    CheckErrors(IDAGetDky(ida_mem, t0, 1, yyp), "IDAGetDky derivative at t0");
    knot_reducer->ProcessPoint(t0, y_val_, yp_val_, /*is_breakpoint=*/true);
  } else {
    SetStep(t0);
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SavePoint(
  sunrealtype t_val, bool extend_arrays, bool is_breakpoint
) {
  DEBUG("IDAKLUSolver::SavePoint");

  if (use_knot_reduction_) {
    // TRUE Hermite knot reducer: Process point inline, reducer decides whether to keep
    CheckErrors(IDAGetDky(ida_mem, t_val, 1, yyp), "IDAGetDky derivative for Hermite knot reducer");
    knot_reducer->ProcessPoint(t_val, y_val_, yp_val_, is_breakpoint);
  } else {
    // Non-Hermite knot reducer: check for duplicates and save
    if (t_val != t[i_save_ - 1]) {
      if (extend_arrays) {
        ExtendAdaptiveArrays();
      }
      SetStep(t_val);
    }
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::HandleBreakpoint(
  sunrealtype t_val,
  const std::vector<sunrealtype> &t_eval,
  int &i_eval,
  sunrealtype &t_eval_next,
  NoProgressGuard &no_progression
) {
  DEBUG("IDAKLUSolver::HandleBreakpoint");

  log_.log_breakpoint(t_val);

  // Advance to the next evaluation point
  i_eval++;
  t_eval_next = t_eval[i_eval];
  CheckErrors(IDASetStopTime(ida_mem, t_eval_next), "IDASetStopTime");
  if (solver_opts.print_stats) {
    // Save stats before reinitializing (reinit resets IDA counters)
    SaveStats();
  }

  // Reinitialize the solver to deal with the discontinuity at t = t_val
  ReinitializeIntegrator(t_val);
  ConsistentInitialization(t_val, t_eval_next, IDA_YA_YDP_INIT);
  log_.log_consistent_init(t_val);

  // Reset the no-progress guard
  no_progression.Initialize();
}

template <class ExprSet>
SolutionData IDAKLUSolverOpenMP<ExprSet>::BuildSolutionData(int retval) {
  DEBUG("IDAKLUSolver::BuildSolutionData");

  if (solver_opts.print_stats) {
    SaveStats();
    PrintStats(accumulated_stats);
  }

  // Finalize output arrays
  if (use_knot_reduction_) {
    knot_reducer->Finalize();
    number_of_timesteps = knot_reducer->GetOutputCount();
  } else {
    number_of_timesteps = i_save_;
    t.resize(number_of_timesteps);
    y.resize(number_of_timesteps * length_of_return_vector);
    if (save_hermite) {
      yp.resize(number_of_timesteps * number_of_states);
    }
  }

  // Sensitivity dimensions for numpy layout
  auto const arg_sens0 = (save_outputs_only ? number_of_timesteps : number_of_parameters);
  auto const arg_sens1 = (save_outputs_only ? length_of_return_vector : number_of_timesteps);
  auto const arg_sens2 = (save_outputs_only ? number_of_parameters : length_of_return_vector);

  // Reorder sensitivities from [i][p][j] to expected numpy layout
  std::vector<sunrealtype> yS_reordered, ypS_reordered;
  if (sensitivity) {
    ReorderSensitivities(yS_reordered, ypS_reordered);
  }

  // Final state slice (for outputs_only mode)
  std::vector<sunrealtype> yterm_vec;
  if (save_outputs_only) {
    yterm_vec.assign(y_val_, y_val_ + number_of_states);
  }

  return SolutionData(
    retval,
    std::move(t),
    std::move(y),
    save_hermite ? std::move(yp) : std::vector<sunrealtype>(),
    std::move(yS_reordered),
    std::move(ypS_reordered),
    std::move(yterm_vec),
    arg_sens0,
    arg_sens1,
    arg_sens2,
    save_hermite
  );
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::ReorderSensitivities(
  std::vector<sunrealtype> &yS_out,
  std::vector<sunrealtype> &ypS_out
) {
  DEBUG("IDAKLUSolver::ReorderSensitivities");

  // Sensitivities are stored during solve as yS[(i * n_params + p) * stride + j].
  // Python expects (n_params, n_timesteps, stride) for !save_outputs_only
  // or (n_timesteps, stride, n_params) for save_outputs_only.
  size_t const nt = number_of_timesteps;
  size_t const np = number_of_parameters;
  size_t const stride = length_of_return_vector;

  yS.resize(nt * np * stride);
  yS_out.resize(nt * np * stride);

  for (size_t i = 0; i < nt; ++i) {
    for (size_t p = 0; p < np; ++p) {
      for (size_t j = 0; j < stride; ++j) {
        size_t src = (i * np + p) * stride + j;
        size_t dst = save_outputs_only
          ? (i * stride + j) * np + p        // (i, j, p) layout
          : (p * nt + i) * stride + j;       // (p, i, j) layout
        yS_out[dst] = yS[src];
      }
    }
  }

  if (save_hermite) {
    size_t const ns = number_of_states;
    ypS.resize(nt * np * ns);
    ypS_out.resize(nt * np * ns);

    for (size_t i = 0; i < nt; ++i) {
      for (size_t p = 0; p < np; ++p) {
        for (size_t j = 0; j < ns; ++j) {
          size_t src = (i * np + p) * ns + j;
          size_t dst = save_outputs_only
            ? (i * ns + j) * np + p           // (i, j, p) layout
            : (p * nt + i) * ns + j;          // (p, i, j) layout
          ypS_out[dst] = ypS[src];
        }
      }
    }
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::ExtendAdaptiveArrays() {
  DEBUG("IDAKLUSolver::ExtendAdaptiveArrays");
  // Extend flat arrays by one timestep worth of elements
  t.resize(t.size() + 1, 0.0);
  y.resize(y.size() + length_of_return_vector, 0.0);
  if (sensitivity) {
    yS.resize(yS.size() + number_of_parameters * length_of_return_vector, 0.0);
  }
  if (save_hermite) {
    yp.resize(yp.size() + number_of_states, 0.0);
    if (sensitivity) {
      ypS.resize(ypS.size() + number_of_parameters * number_of_states, 0.0);
    }
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::ReinitializeIntegrator(const sunrealtype& t_val) {
  DEBUG("IDAKLUSolver::ReinitializeIntegrator");
  CheckErrors(IDAReInit(ida_mem, t_val, yy, yyp), "IDAReInit");
  if (sensitivity) {
    CheckErrors(IDASensReInit(ida_mem, IDA_SIMULTANEOUS, yyS, yypS), "IDASensReInit");
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::ConsistentInitialization(
  const sunrealtype& t_val,
  const sunrealtype& t_next,
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
  const sunrealtype& t_val,
  const sunrealtype& t_next,
  const int& icopt) {
  DEBUG("IDAKLUSolver::ConsistentInitializationDAE");
  // The solver requires a future time point to calculate the direction
  // of the initial step and its order of magnitude estimate. Use a
  // small perturbation that is consistent with the intended direction.
  const bool increasing = (t_next > t_val);
  sunrealtype t_next_perturbed = perturb_time(t_next, increasing);
  IDACalcIC(ida_mem, icopt, t_next_perturbed);
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::ConsistentInitializationODE(
  const sunrealtype& t_val) {
  DEBUG("IDAKLUSolver::ConsistentInitializationODE");

  // For ODEs where the mass matrix M = I, we can simplify the problem
  // by analytically computing the yp values. If we take our implicit
  // DAE system res(t,y,yp) = f(t,y) - I*yp, then yp = res(t,y,0). This
  // avoids an expensive call to IDACalcIC.
  sunrealtype *y_cache_val = N_VGetArrayPointer(y_cache);
  std::memset(y_cache_val, 0, number_of_states * sizeof(sunrealtype));
  // Overwrite yp
  residual_eval<ExprSet>(t_val, yy, y_cache, yyp, functions.get());
}

// Step storage methods (use member state: y_val_, yp_val_, yS_val_, ypS_val_, i_save_)
template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStep(
  sunrealtype &tval
) {
  // Set adaptive step results for y and yS
  DEBUG("IDAKLUSolver::SetStep");

  // Time
  t[i_save_] = tval;

  if (save_outputs_only) {
    SetStepOutput(tval);
  } else {
    SetStepFull(tval);

    if (save_hermite) {
      SetStepHermite(tval);
    }
  }

  i_save_++;
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SaveInterpPoints(
  int &i_interp,
  sunrealtype &t_interp_next,
  const std::vector<sunrealtype> &t_interp,
  sunrealtype t_val,
  sunrealtype t_prev,
  sunrealtype t_eval_next
) {
  // Save the state at the requested interpolation times
  DEBUG("IDAKLUSolver::SaveInterpPoints");

  while (i_interp <= (t_interp.size()-1) && t_interp_next <= t_val) {
    CheckErrors(IDAGetDky(ida_mem, t_interp_next, 0, yy), "IDAGetDky for interpolation");
    if (sensitivity) {
      CheckErrors(IDAGetSensDky(ida_mem, t_interp_next, 0, yyS), "IDAGetSensDky for interpolation");
    }

    // Memory is already allocated for the interpolated values
    SetStep(t_interp_next);

    i_interp++;
    if (i_interp == (t_interp.size())) {
      // Reached the final t_interp value
      break;
    }
    t_interp_next = t_interp[i_interp];
  }

  // Reset the states and sensitivities to t = t_val
  CheckErrors(IDAGetDky(ida_mem, t_val, 0, yy), "IDAGetDky reset after interpolation");
  if (sensitivity) {
    CheckErrors(IDAGetSensDky(ida_mem, t_val, 0, yyS), "IDAGetSensDky reset after interpolation");
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStepFull(
  sunrealtype &tval
) {
  // FLAT STORAGE: Copy states to y[i_save_ * stride_y + j]
  DEBUG("IDAKLUSolver::SetStepFull");

  sunrealtype* y_dest = &y[i_save_ * length_of_return_vector];
  std::copy(y_val_, y_val_ + number_of_states, y_dest);

  if (sensitivity) {
    SetStepFullSensitivities(tval);
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStepFullSensitivities(
  sunrealtype &tval
) {
  DEBUG("IDAKLUSolver::SetStepFullSensitivities");

  // FLAT STORAGE: yS[(i * n_params + p) * stride + j]
  // stride_yS_per_timestep = number_of_parameters * length_of_return_vector
  size_t base = i_save_ * number_of_parameters * length_of_return_vector;
  for (size_t p = 0; p < number_of_parameters; ++p) {
    sunrealtype* dest = &yS[base + p * length_of_return_vector];
    std::copy(yS_val_[p], yS_val_[p] + number_of_states, dest);
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStepOutput(
    sunrealtype &tval
) {
  DEBUG("IDAKLUSolver::SetStepOutput");
  // FLAT STORAGE: Write output variables to y[i_save_ * stride + j]
  
  sunrealtype* y_dest = &y[i_save_ * length_of_return_vector];
  size_t j = 0;
  for (auto& var_fcn : functions->var_fcns) {
    (*var_fcn)({&tval, y_val_, functions->inputs.data()}, {&res[0]});
    for (size_t jj = 0; jj < var_fcn->nnz_out(); jj++) {
      y_dest[j++] = res[jj];
    }
  }
  
  if (sensitivity) {
    SetStepOutputSensitivities(tval);
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStepOutputSensitivities(
  sunrealtype &tval
) {
  DEBUG("IDAKLUSolver::SetStepOutputSensitivities");

  // FLAT STORAGE: yS[(i * n_params + p) * stride + j]
  // Base offset for this timestep
  size_t yS_base = i_save_ * number_of_parameters * length_of_return_vector;

  // Running index over the flattened outputs
  size_t global_out_idx = 0;

  // Loop over each variable
  for (size_t dvar_k = 0; dvar_k < functions->var_fcns.size(); ++dvar_k) {
    Expression* dvar_dy = functions->dvar_dy_fcns[dvar_k];
    Expression* dvar_dp = functions->dvar_dp_fcns[dvar_k];

    // Calculate dvar/dy
    (*dvar_dy)({&tval, y_val_, functions->inputs.data()}, {&res_dvar_dy[0]});
    // Calculate dvar/dp
    (*dvar_dp)({&tval, y_val_, functions->inputs.data()}, {&res_dvar_dp[0]});

    // Get number of output components for this function (e.g., scalar -> 1; vector -> >1)
    const size_t n_rows = functions->var_fcns[dvar_k]->nnz_out();

    // Number of nonzeros in the sparse Jacobians (for dvar/dy and dvar/dp)
    const size_t dvar_dy_nnz = dvar_dy->nnz_out();
    const size_t dvar_dp_nnz = dvar_dp->nnz_out();

    // Row/column indices of nonzero entries (compressed sparse row format)
    const auto& dvar_dy_row = dvar_dy->get_row();
    const auto& dvar_dy_col = dvar_dy->get_col();
    const auto& dvar_dp_row = dvar_dp->get_row();
    const auto& dvar_dp_col = dvar_dp->get_col();

    // Temporary dense vector to hold doutput_row/dp_k for each parameter
    vector<sunrealtype> dvar_dp_dense(number_of_parameters, 0.0);

    // Loop over each scalar component (row) of the output function
    for (size_t row = 0; row < n_rows; ++row, ++global_out_idx) {
      // Dense dvar_row/dp_k vector (reset to zero)
      std::fill(dvar_dp_dense.begin(), dvar_dp_dense.end(), 0.0);

      // Fill in dvar_row/dp_k from sparse structure
      for (size_t nz = 0; nz < dvar_dp_nnz; ++nz) {
        if (dvar_dp_row[nz] == static_cast<int>(row)) {
          dvar_dp_dense[dvar_dp_col[nz]] = res_dvar_dp[nz];
        }
      }

      // For each parameter p_k, compute total d(output_row)/d(p_k)
      for (int paramk = 0; paramk < number_of_parameters; paramk++) {
        // Start with direct contribution doutput/dp_k
        sunrealtype sens = dvar_dp_dense[paramk];

        // Add chain rule term
        for (size_t nz = 0; nz < dvar_dy_nnz; ++nz) {
          if (dvar_dy_row[nz] == static_cast<int>(row)) {
            sens += res_dvar_dy[nz] * yS_val_[paramk][dvar_dy_col[nz]];
          }
        }

        // FLAT STORAGE: yS[yS_base + paramk * stride + global_out_idx]
        yS[yS_base + paramk * length_of_return_vector + global_out_idx] = sens;
      }
    }
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStepHermite(
  sunrealtype &tval
) {
  // FLAT STORAGE: Copy derivatives to yp[i_save_ * stride_yp + j]
  DEBUG("IDAKLUSolver::SetStepHermite");

  CheckErrors(IDAGetDky(ida_mem, tval, 1, yyp), "IDAGetDky for Hermite (derivative 1)");
  
  sunrealtype* yp_dest = &yp[i_save_ * number_of_states];
  std::copy(yp_val_, yp_val_ + number_of_states, yp_dest);

  if (sensitivity) {
    SetStepHermiteSensitivities(tval);
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SetStepHermiteSensitivities(
  sunrealtype &tval
) {
  DEBUG("IDAKLUSolver::SetStepHermiteSensitivities");

  // FLAT STORAGE: ypS[(i * n_params + p) * stride + j]
  CheckErrors(IDAGetSensDky(ida_mem, tval, 1, yypS), "IDAGetSensDky for Hermite (derivative 1)");
  
  size_t base = i_save_ * number_of_parameters * number_of_states;
  for (size_t p = 0; p < number_of_parameters; ++p) {
    sunrealtype* dest = &ypS[base + p * number_of_states];
    std::copy(ypS_val_[p], ypS_val_[p] + number_of_states, dest);
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::CheckErrors(int const & flag) {
  if (flag < 0) {
    throw_sundials_error(flag, "SUNDIALS operation");
  }
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::CheckErrors(int const & flag, const char* context) {
  if (flag < 0) {
    throw_sundials_error(flag, context);
  }
}

template <class ExprSet>
IDAKLUStats IDAKLUSolverOpenMP<ExprSet>::GetStats() {
  IDAKLUStats stats;
  int klast, kcur;
  sunrealtype hinused, hlast, hcur, tcur;

  CheckErrors(IDAGetIntegratorStats(
    ida_mem,
    &stats.nsteps,
    &stats.nrevals,
    &stats.nlinsetups,
    &stats.netfails,
    &klast,
    &kcur,
    &hinused,
    &hlast,
    &hcur,
    &tcur
  ), "IDAGetIntegratorStats");

  CheckErrors(IDAGetNonlinSolvStats(ida_mem, &stats.nniters, &stats.nncfails), "IDAGetNonlinSolvStats");

  CheckErrors(IDAGetNumJacEvals(ida_mem, &stats.njevals), "IDAGetNumJacEvals");
  if (setup_opts.using_iterative_solver) {
    CheckErrors(IDAGetNumLinIters(ida_mem, &stats.nliters), "IDAGetNumLinIters");
    CheckErrors(IDAGetNumLinConvFails(ida_mem, &stats.nlcfails), "IDAGetNumLinConvFails");
    CheckErrors(IDABBDPrecGetNumGfnEvals(ida_mem, &stats.ngevalsBBDP), "IDABBDPrecGetNumGfnEvals");
  }

  return stats;
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::SaveStats() {
  accumulated_stats += GetStats();
}

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::PrintStats(const IDAKLUStats& stats) {
  // Get current point-in-time values from IDA (these are not accumulated)
  long nsteps_unused, nrevals_unused, nlinsetups_unused, netfails_unused;
  int klast, kcur;
  sunrealtype hinused, hlast, hcur, tcur;

  CheckErrors(IDAGetIntegratorStats(
    ida_mem,
    &nsteps_unused,
    &nrevals_unused,
    &nlinsetups_unused,
    &netfails_unused,
    &klast,
    &kcur,
    &hinused,
    &hlast,
    &hcur,
    &tcur
  ), "IDAGetIntegratorStats");

  py::print("Solver Stats:");
  py::print("\tNumber of steps =", stats.nsteps);
  py::print("\tNumber of calls to residual function =", stats.nrevals);
  py::print("\tNumber of calls to residual function in preconditioner =",
            stats.ngevalsBBDP);
  py::print("\tNumber of linear solver setup calls =", stats.nlinsetups);
  py::print("\tNumber of error test failures =", stats.netfails);
  py::print("\tMethod order used on last step =", klast);
  py::print("\tMethod order used on next step =", kcur);
  py::print("\tInitial step size =", hinused);
  py::print("\tStep size on last step =", hlast);
  py::print("\tStep size on next step =", hcur);
  py::print("\tCurrent internal time reached =", tcur);
  py::print("\tNumber of nonlinear iterations performed =", stats.nniters);
  py::print("\tNumber of nonlinear convergence failures =", stats.nncfails);
  py::print("\tNumber of Jacobian evaluations =", stats.njevals);
  if (setup_opts.using_iterative_solver) {
    py::print("\tNumber of linear iterations =", stats.nliters);
    py::print("\tNumber of linear convergence failures =", stats.nlcfails);
  }
}
