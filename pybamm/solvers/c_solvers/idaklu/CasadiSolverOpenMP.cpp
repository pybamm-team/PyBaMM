#include "CasadiSolverOpenMP.hpp"
#include "casadi_sundials_functions.hpp"
#include <casadi/casadi.hpp>
#include <casadi/core/function.hpp>
#include <casadi/core/sparsity.hpp>

CasadiSolverOpenMP::CasadiSolverOpenMP(
  np_array atol_np,
  double rel_tol,
  np_array rhs_alg_id,
  int number_of_parameters,
  int number_of_events,
  int jac_times_cjmass_nnz,
  int jac_bandwidth_lower,
  int jac_bandwidth_upper,
  std::unique_ptr<CasadiFunctions> functions_arg,
  const Options &options
) :
  atol_np(atol_np),
  rhs_alg_id(rhs_alg_id),
  number_of_states(atol_np.request().size),
  number_of_parameters(number_of_parameters),
  number_of_events(number_of_events),
  jac_times_cjmass_nnz(jac_times_cjmass_nnz),
  jac_bandwidth_lower(jac_bandwidth_lower),
  jac_bandwidth_upper(jac_bandwidth_upper),
  functions(std::move(functions_arg)),
  options(options)
{
  // Construction code moved to Initialize() which is called from the
  // (child) CasadiSolver_XXX class constructors.
  DEBUG("CasadiSolverOpenMP::CasadiSolverOpenMP");
  auto atol = atol_np.unchecked<1>();

  // create SUNDIALS context object
  SUNContext_Create(NULL, &sunctx);  // calls null-wrapper if Sundials Ver<6

  // allocate memory for solver
  ida_mem = IDACreate(sunctx);

  // create the vector of initial values
  AllocateVectors();
  if (number_of_parameters > 0)
  {
    yyS = N_VCloneVectorArray(number_of_parameters, yy);
    ypS = N_VCloneVectorArray(number_of_parameters, yp);
  }
  // set initial values
  realtype *atval = N_VGetArrayPointer(avtol);
  for (int i = 0; i < number_of_states; i++)
    atval[i] = atol[i];
  for (int is = 0; is < number_of_parameters; is++)
  {
    N_VConst(RCONST(0.0), yyS[is]);
    N_VConst(RCONST(0.0), ypS[is]);
  }

  // create Matrix objects
  SetMatrix();

  // initialise solver
  IDAInit(ida_mem, residual_casadi, 0, yy, yp);

  // set tolerances
  rtol = RCONST(rel_tol);
  IDASVtolerances(ida_mem, rtol, avtol);

  // set events
  IDARootInit(ida_mem, number_of_events, events_casadi);
  void *user_data = functions.get();
  IDASetUserData(ida_mem, user_data);

  // specify preconditioner type
  precon_type = SUN_PREC_NONE;
  if (options.preconditioner != "none") {
    precon_type = SUN_PREC_LEFT;
  }
}

void CasadiSolverOpenMP::AllocateVectors() {
  // Create vectors
  yy = N_VNew_OpenMP(number_of_states, options.num_threads, sunctx);
  yp = N_VNew_OpenMP(number_of_states, options.num_threads, sunctx);
  avtol = N_VNew_OpenMP(number_of_states, options.num_threads, sunctx);
  id = N_VNew_OpenMP(number_of_states, options.num_threads, sunctx);
}

void CasadiSolverOpenMP::SetMatrix() {
  // Create Matrix object
  if (options.jacobian == "sparse")
  {
    DEBUG("\tsetting sparse matrix");
    J = SUNSparseMatrix(
      number_of_states,
      number_of_states,
      jac_times_cjmass_nnz,
      CSC_MAT,  // CSC is used by casadi; CSR requires a conversion step
      sunctx
    );
  }
  else if (options.jacobian == "banded") {
    DEBUG("\tsetting banded matrix");
    J = SUNBandMatrix(
      number_of_states,
      jac_bandwidth_upper,
      jac_bandwidth_lower,
      sunctx
    );
  } else if (options.jacobian == "dense" || options.jacobian == "none")
  {
    DEBUG("\tsetting dense matrix");
    J = SUNDenseMatrix(
      number_of_states,
      number_of_states,
      sunctx
    );
  }
  else if (options.jacobian == "matrix-free")
  {
    DEBUG("\tsetting matrix-free");
    J = NULL;
  }
  else
    throw std::invalid_argument("Unsupported matrix requested");
}

void CasadiSolverOpenMP::Initialize() {
  // Call after setting the solver

  // attach the linear solver
  if (LS == nullptr)
    throw std::invalid_argument("Linear solver not set");
  IDASetLinearSolver(ida_mem, LS, J);

  if (options.preconditioner != "none")
  {
    DEBUG("\tsetting IDADDB preconditioner");
    // setup preconditioner
    IDABBDPrecInit(
      ida_mem, number_of_states, options.precon_half_bandwidth,
      options.precon_half_bandwidth, options.precon_half_bandwidth_keep,
      options.precon_half_bandwidth_keep, 0.0, residual_casadi_approx, NULL);
  }

  if (options.jacobian == "matrix-free")
    IDASetJacTimes(ida_mem, NULL, jtimes_casadi);
  else if (options.jacobian != "none")
    IDASetJacFn(ida_mem, jacobian_casadi);

  if (number_of_parameters > 0)
  {
    IDASensInit(ida_mem, number_of_parameters, IDA_SIMULTANEOUS,
                sensitivities_casadi, yyS, ypS);
    IDASensEEtolerances(ida_mem);
  }

  SUNLinSolInitialize(LS);

  auto id_np_val = rhs_alg_id.unchecked<1>();
  realtype *id_val;
  id_val = N_VGetArrayPointer(id);

  int ii;
  for (ii = 0; ii < number_of_states; ii++)
    id_val[ii] = id_np_val[ii];

  IDASetId(ida_mem, id);
}

CasadiSolverOpenMP::~CasadiSolverOpenMP()
{
  // Free memory
  if (number_of_parameters > 0)
    IDASensFree(ida_mem);

  SUNLinSolFree(LS);
  SUNMatDestroy(J);
  N_VDestroy(avtol);
  N_VDestroy(yy);
  N_VDestroy(yp);
  N_VDestroy(id);

  if (number_of_parameters > 0)
  {
    N_VDestroyVectorArray(yyS, number_of_parameters);
    N_VDestroyVectorArray(ypS, number_of_parameters);
  }

  IDAFree(&ida_mem);
  SUNContext_Free(&sunctx);
}

void CasadiSolverOpenMP::CalcVars(
    realtype *y_return,
    size_t length_of_return_vector,
    size_t t_i,
    realtype *tret,
    realtype *yval,
    const std::vector<realtype*>& ySval,
    realtype *yS_return,
    size_t *ySk
) {
  // Evaluate casadi functions for each requested variable and store
  size_t j = 0;
  for (auto& var_fcn : functions->var_casadi_fcns) {
    var_fcn({tret, yval, functions->inputs.data()}, {res});
    // store in return vector
    for (size_t jj=0; jj<var_fcn.nnz_out(); jj++)
      y_return[t_i*length_of_return_vector + j++] = res[jj];
  }
  // calculate sensitivities
  CalcVarsSensitivities(tret, yval, ySval, yS_return, ySk);
}

void CasadiSolverOpenMP::CalcVarsSensitivities(
    realtype *tret,
    realtype *yval,
    const std::vector<realtype*>& ySval,
    realtype *yS_return,
    size_t *ySk
) {
  // Calculate sensitivities

  // Loop over variables
  realtype* dens_dvar_dp = new realtype[number_of_parameters];
  for (size_t dvar_k=0; dvar_k<functions->dvar_dy_fcns.size(); dvar_k++) {
    // Isolate functions
    CasadiFunction dvar_dy = functions->dvar_dy_fcns[dvar_k];
    CasadiFunction dvar_dp = functions->dvar_dp_fcns[dvar_k];
    // Calculate dvar/dy
    dvar_dy({tret, yval, functions->inputs.data()}, {res_dvar_dy});
    casadi::Sparsity spdy = dvar_dy.sparsity_out(0);
    // Calculate dvar/dp and convert to dense array for indexing
    dvar_dp({tret, yval, functions->inputs.data()}, {res_dvar_dp});
    casadi::Sparsity spdp = dvar_dp.sparsity_out(0);
    for(int k=0; k<number_of_parameters; k++)
      dens_dvar_dp[k]=0;
    for(int k=0; k<spdp.nnz(); k++)
      dens_dvar_dp[spdp.get_row()[k]] = res_dvar_dp[k];
    // Calculate sensitivities
    for(int paramk=0; paramk<number_of_parameters; paramk++) {
      yS_return[*ySk] = dens_dvar_dp[paramk];
      for(int spk=0; spk<dvar_dy.nnz_out(); spk++)
        yS_return[*ySk] += res_dvar_dy[spk] * ySval[paramk][spdy.get_col()[spk]];
      (*ySk)++;
    }
  }
}

Solution CasadiSolverOpenMP::solve(
    np_array t_np,
    np_array y0_np,
    np_array yp0_np,
    np_array_dense inputs
)
{
  DEBUG("CasadiSolver::solve");

  int number_of_timesteps = t_np.request().size;
  auto t = t_np.unchecked<1>();
  realtype t0 = RCONST(t(0));
  auto y0 = y0_np.unchecked<1>();
  auto yp0 = yp0_np.unchecked<1>();
  auto n_coeffs = number_of_states + number_of_parameters * number_of_states;

  if (y0.size() != n_coeffs)
    throw std::domain_error(
      "y0 has wrong size. Expected " + std::to_string(n_coeffs) +
      " but got " + std::to_string(y0.size()));

  if (yp0.size() != n_coeffs)
    throw std::domain_error(
      "yp0 has wrong size. Expected " + std::to_string(n_coeffs) +
      " but got " + std::to_string(yp0.size()));

  // set inputs
  auto p_inputs = inputs.unchecked<2>();
  for (int i = 0; i < functions->inputs.size(); i++)
    functions->inputs[i] = p_inputs(i, 0);

  // set initial conditions
  realtype *yval = N_VGetArrayPointer(yy);
  realtype *ypval = N_VGetArrayPointer(yp);
  std::vector<realtype *> ySval(number_of_parameters);
  std::vector<realtype *> ypSval(number_of_parameters);
  for (int p = 0 ; p < number_of_parameters; p++) {
    ySval[p] = N_VGetArrayPointer(yyS[p]);
    ypSval[p] = N_VGetArrayPointer(ypS[p]);
    for (int i = 0; i < number_of_states; i++) {
      ySval[p][i] = y0[i + (p + 1) * number_of_states];
      ypSval[p][i] = yp0[i + (p + 1) * number_of_states];
    }
  }

  for (int i = 0; i < number_of_states; i++)
  {
    yval[i] = y0[i];
    ypval[i] = yp0[i];
  }

  IDAReInit(ida_mem, t0, yy, yp);
  if (number_of_parameters > 0)
    IDASensReInit(ida_mem, IDA_SIMULTANEOUS, yyS, ypS);

  // correct initial values
  DEBUG("IDACalcIC");
  IDACalcIC(ida_mem, IDA_YA_YDP_INIT, t(1));
  if (number_of_parameters > 0)
    IDAGetSens(ida_mem, &t0, yyS);

  realtype tret;
  realtype t_final = t(number_of_timesteps - 1);

  // set return vectors
  int length_of_return_vector = 0;
  size_t max_res_size = 0;  // maximum result size (for common result buffer)
  size_t max_res_dvar_dy = 0, max_res_dvar_dp = 0;
  if (functions->var_casadi_fcns.size() > 0) {
    // return only the requested variables list after computation
    for (auto& var_fcn : functions->var_casadi_fcns) {
      max_res_size = std::max(max_res_size, size_t(var_fcn.nnz_out()));
      length_of_return_vector += var_fcn.nnz_out();
      for (auto& dvar_fcn : functions->dvar_dy_fcns)
        max_res_dvar_dy = std::max(max_res_dvar_dy, size_t(dvar_fcn.nnz_out()));
      for (auto& dvar_fcn : functions->dvar_dp_fcns)
        max_res_dvar_dp = std::max(max_res_dvar_dp, size_t(dvar_fcn.nnz_out()));
    }
  } else {
    // Return full y state-vector
    length_of_return_vector = number_of_states;
  }
  realtype *t_return = new realtype[number_of_timesteps];
  realtype *y_return = new realtype[number_of_timesteps *
                                    length_of_return_vector];
  realtype *yS_return = new realtype[number_of_parameters *
                                     number_of_timesteps *
                                     length_of_return_vector];

  res = new realtype[max_res_size];
  res_dvar_dy = new realtype[max_res_dvar_dy];
  res_dvar_dp = new realtype[max_res_dvar_dp];

  py::capsule free_t_when_done(
    t_return,
    [](void *f) {
      realtype *vect = reinterpret_cast<realtype *>(f);
      delete[] vect;
    }
  );
  py::capsule free_y_when_done(
    y_return,
    [](void *f) {
      realtype *vect = reinterpret_cast<realtype *>(f);
      delete[] vect;
    }
  );
  py::capsule free_yS_when_done(
    yS_return,
    [](void *f) {
      realtype *vect = reinterpret_cast<realtype *>(f);
      delete[] vect;
    }
  );

  // Initial state (t_i=0)
  int t_i = 0;
  size_t ySk = 0;
  t_return[t_i] = t(t_i);
  if (functions->var_casadi_fcns.size() > 0) {
    // Evaluate casadi functions for each requested variable and store
    CalcVars(y_return, length_of_return_vector, t_i,
             &tret, yval, ySval, yS_return, &ySk);
  } else {
    // Retain complete copy of the state vector
    for (int j = 0; j < number_of_states; j++)
      y_return[j] = yval[j];
    for (int j = 0; j < number_of_parameters; j++)
    {
      const int base_index = j * number_of_timesteps * number_of_states;
      for (int k = 0; k < number_of_states; k++)
        yS_return[base_index + k] = ySval[j][k];
    }
  }

  // Subsequent states (t_i>0)
  int retval;
  t_i = 1;
  while (true)
  {
    realtype t_next = t(t_i);
    IDASetStopTime(ida_mem, t_next);
    DEBUG("IDASolve");
    retval = IDASolve(ida_mem, t_final, &tret, yy, yp, IDA_NORMAL);

    if (retval == IDA_TSTOP_RETURN ||
        retval == IDA_SUCCESS ||
        retval == IDA_ROOT_RETURN)
    {
      if (number_of_parameters > 0)
        IDAGetSens(ida_mem, &tret, yyS);

      // Evaluate and store results for the time step
      t_return[t_i] = tret;
      if (functions->var_casadi_fcns.size() > 0) {
        // Evaluate casadi functions for each requested variable and store
        // NOTE: Indexing of yS_return is (time:var:param)
        CalcVars(y_return, length_of_return_vector, t_i,
                 &tret, yval, ySval, yS_return, &ySk);
      } else {
        // Retain complete copy of the state vector
        for (int j = 0; j < number_of_states; j++)
          y_return[t_i * number_of_states + j] = yval[j];
        for (int j = 0; j < number_of_parameters; j++)
        {
          const int base_index =
            j * number_of_timesteps * number_of_states +
            t_i * number_of_states;
          for (int k = 0; k < number_of_states; k++)
            // NOTE: Indexing of yS_return is (time:param:yvec)
            yS_return[base_index + k] = ySval[j][k];
        }
      }
      t_i += 1;

      if (retval == IDA_SUCCESS ||
          retval == IDA_ROOT_RETURN)
        break;
    }
    else
    {
      // failed
      break;
    }
  }

  np_array t_ret = np_array(
    t_i,
    &t_return[0],
    free_t_when_done
  );
  np_array y_ret = np_array(
    t_i * length_of_return_vector,
    &y_return[0],
    free_y_when_done
  );
  // Note: Ordering of vector is differnet if computing variables vs returning
  // the complete state vector
  np_array yS_ret;
  if (functions->var_casadi_fcns.size() > 0) {
    yS_ret = np_array(
      std::vector<ptrdiff_t> {
        number_of_timesteps,
        length_of_return_vector,
        number_of_parameters
      },
      &yS_return[0],
      free_yS_when_done
    );
  } else {
    yS_ret = np_array(
      std::vector<ptrdiff_t> {
        number_of_parameters,
        number_of_timesteps,
        length_of_return_vector
      },
      &yS_return[0],
      free_yS_when_done
    );
  }

  Solution sol(retval, t_ret, y_ret, yS_ret);

  if (options.print_stats)
  {
    long nsteps, nrevals, nlinsetups, netfails;
    int klast, kcur;
    realtype hinused, hlast, hcur, tcur;

    IDAGetIntegratorStats(
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
    );

    long nniters, nncfails;
    IDAGetNonlinSolvStats(ida_mem, &nniters, &nncfails);

    long int ngevalsBBDP = 0;
    if (options.using_iterative_solver)
      IDABBDPrecGetNumGfnEvals(ida_mem, &ngevalsBBDP);

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

  return sol;
}
