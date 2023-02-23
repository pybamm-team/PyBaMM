#include "casadi_solver.hpp"
#include "casadi_sundials_functions.hpp"
#include "common.hpp"
#include <memory>

CasadiSolver *
create_casadi_solver(int number_of_states, int number_of_parameters,
                     const Function &rhs_alg, const Function &jac_times_cjmass,
                     const np_array_int &jac_times_cjmass_colptrs,
                     const np_array_int &jac_times_cjmass_rowvals,
                     const int jac_times_cjmass_nnz, 
                     const int jac_bandwidth_lower, const int jac_bandwidth_upper, 
                     const Function &jac_action,
                     const Function &mass_action, const Function &sens,
                     const Function &events, const int number_of_events,
                     np_array rhs_alg_id, np_array atol_np, double rel_tol,
                     int inputs_length, py::dict options)
{
  auto options_cpp = Options(options);
  auto functions = std::make_unique<CasadiFunctions>(
      rhs_alg, jac_times_cjmass, jac_times_cjmass_nnz, jac_bandwidth_lower, jac_bandwidth_upper,  jac_times_cjmass_rowvals,
      jac_times_cjmass_colptrs, inputs_length, jac_action, mass_action, sens,
      events, number_of_states, number_of_events, number_of_parameters,
      options_cpp);

  return new CasadiSolver(atol_np, rel_tol, rhs_alg_id, number_of_parameters,
                          number_of_events, jac_times_cjmass_nnz, 
                          jac_bandwidth_lower, jac_bandwidth_upper,
                          std::move(functions), options_cpp);
}

CasadiSolver::CasadiSolver(np_array atol_np, double rel_tol,
                           np_array rhs_alg_id, int number_of_parameters,
                           int number_of_events, int jac_times_cjmass_nnz,
                           int jac_bandwidth_lower, int jac_bandwidth_upper,
                           std::unique_ptr<CasadiFunctions> functions_arg,
                           const Options &options)
    : number_of_states(atol_np.request().size),
      number_of_parameters(number_of_parameters),
      number_of_events(number_of_events),
      jac_times_cjmass_nnz(jac_times_cjmass_nnz),
      functions(std::move(functions_arg)), options(options)
{
  DEBUG("CasadiSolver::CasadiSolver");
  auto atol = atol_np.unchecked<1>();

  // allocate memory for solver
#if SUNDIALS_VERSION_MAJOR >= 6
  SUNContext_Create(NULL, &sunctx);
  ida_mem = IDACreate(sunctx);
#else
  ida_mem = IDACreate();
#endif

  // allocate vectors
#if SUNDIALS_VERSION_MAJOR >= 6
  yy = N_VNew_Serial(number_of_states, sunctx);
  yp = N_VNew_Serial(number_of_states, sunctx);
  avtol = N_VNew_Serial(number_of_states, sunctx);
  id = N_VNew_Serial(number_of_states, sunctx);
#else
  yy = N_VNew_Serial(number_of_states);
  yp = N_VNew_Serial(number_of_states);
  avtol = N_VNew_Serial(number_of_states);
  id = N_VNew_Serial(number_of_states);
#endif

  if (number_of_parameters > 0)
  {
    yyS = N_VCloneVectorArray(number_of_parameters, yy);
    ypS = N_VCloneVectorArray(number_of_parameters, yp);
  }

  // set initial value
  realtype *atval = N_VGetArrayPointer(avtol);
  for (int i = 0; i < number_of_states; i++)
  {
    atval[i] = atol[i];
  }

  for (int is = 0; is < number_of_parameters; is++)
  {
    N_VConst(RCONST(0.0), yyS[is]);
    N_VConst(RCONST(0.0), ypS[is]);
  }

  // initialise solver

  IDAInit(ida_mem, residual_casadi, 0, yy, yp);

  // set tolerances
  rtol = RCONST(rel_tol);

  IDASVtolerances(ida_mem, rtol, avtol);

  // set events
  IDARootInit(ida_mem, number_of_events, events_casadi);

  void *user_data = functions.get();
  IDASetUserData(ida_mem, user_data);

  // set matrix
  if (options.jacobian == "sparse")
  {
    DEBUG("\tsetting sparse matrix");
#if SUNDIALS_VERSION_MAJOR >= 6
    J = SUNSparseMatrix(number_of_states, number_of_states,
                        jac_times_cjmass_nnz, CSC_MAT, sunctx);
#else
    J = SUNSparseMatrix(number_of_states, number_of_states,
                        jac_times_cjmass_nnz, CSC_MAT);
#endif
  }
  else if (options.jacobian == "banded") {
    DEBUG("\tsetting banded matrix");
    #if SUNDIALS_VERSION_MAJOR >= 6
        J = SUNBandMatrix(number_of_states, jac_bandwidth_upper, jac_bandwidth_lower, sunctx);
    #else
        J = SUNBandMatrix(number_of_states, jac_bandwidth_upper, jac_bandwidth_lower);
    #endif
  } else if (options.jacobian == "dense" || options.jacobian == "none")
  {
    DEBUG("\tsetting dense matrix");
#if SUNDIALS_VERSION_MAJOR >= 6
    J = SUNDenseMatrix(number_of_states, number_of_states, sunctx);
#else
    J = SUNDenseMatrix(number_of_states, number_of_states);
#endif
  }
  else if (options.jacobian == "matrix-free")
  {
    DEBUG("\tsetting matrix-free");
    J = NULL;
  }

  #if SUNDIALS_VERSION_MAJOR >= 6
  int precon_type = SUN_PREC_NONE;
  if (options.preconditioner != "none") {
    precon_type = SUN_PREC_LEFT;
  }
  #else
  int precon_type = PREC_NONE;
  if (options.preconditioner != "none") {
    precon_type = PREC_LEFT;
  }
  #endif

  // set linear solver
  if (options.linear_solver == "SUNLinSol_Dense")
  {
    DEBUG("\tsetting SUNLinSol_Dense linear solver");
#if SUNDIALS_VERSION_MAJOR >= 6
    LS = SUNLinSol_Dense(yy, J, sunctx);
#else
    LS = SUNLinSol_Dense(yy, J);
#endif
  }
  else if (options.linear_solver == "SUNLinSol_KLU")
  {
    DEBUG("\tsetting SUNLinSol_KLU linear solver");
#if SUNDIALS_VERSION_MAJOR >= 6
    LS = SUNLinSol_KLU(yy, J, sunctx);
#else
    LS = SUNLinSol_KLU(yy, J);
#endif
  }
  else if (options.linear_solver == "SUNLinSol_Band")
  {
    DEBUG("\tsetting SUNLinSol_Band linear solver");  
#if SUNDIALS_VERSION_MAJOR >= 6
    LS = SUNLinSol_Band(yy, J, sunctx);
#else
    LS = SUNLinSol_Band(yy, J);
#endif
  }
  else if (options.linear_solver == "SUNLinSol_SPBCGS")
  {
    DEBUG("\tsetting SUNLinSol_SPBCGS_linear solver");
#if SUNDIALS_VERSION_MAJOR >= 6
    LS = SUNLinSol_SPBCGS(yy, precon_type, options.linsol_max_iterations,
                          sunctx);
#else
    LS = SUNLinSol_SPBCGS(yy, precon_type, options.linsol_max_iterations);
#endif
  }
  else if (options.linear_solver == "SUNLinSol_SPFGMR")
  {
    DEBUG("\tsetting SUNLinSol_SPFGMR_linear solver");
#if SUNDIALS_VERSION_MAJOR >= 6
    LS = SUNLinSol_SPFGMR(yy, precon_type, options.linsol_max_iterations,
                          sunctx);
#else
    LS = SUNLinSol_SPFGMR(yy, precon_type, options.linsol_max_iterations);
#endif
  }
  else if (options.linear_solver == "SUNLinSol_SPGMR")
  {
    DEBUG("\tsetting SUNLinSol_SPGMR solver");
#if SUNDIALS_VERSION_MAJOR >= 6
    LS = SUNLinSol_SPGMR(yy, precon_type, options.linsol_max_iterations,
                          sunctx);
#else
    LS = SUNLinSol_SPGMR(yy, precon_type, options.linsol_max_iterations);
#endif
  }
  else if (options.linear_solver == "SUNLinSol_SPTFQMR")
  {
    DEBUG("\tsetting SUNLinSol_SPGMR solver");
#if SUNDIALS_VERSION_MAJOR >= 6
    LS = SUNLinSol_SPTFQMR(yy, precon_type, options.linsol_max_iterations,
                          sunctx);
#else
    LS = SUNLinSol_SPTFQMR(yy, precon_type, options.linsol_max_iterations);
#endif
  }



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
  {
    IDASetJacTimes(ida_mem, NULL, jtimes_casadi);
  }
  else if (options.jacobian != "none")
  {
    IDASetJacFn(ida_mem, jacobian_casadi);
  }

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
  {
    id_val[ii] = id_np_val[ii];
  }

  IDASetId(ida_mem, id);
}

CasadiSolver::~CasadiSolver()
{

  /* Free memory */
  if (number_of_parameters > 0)
  {
    IDASensFree(ida_mem);
  }
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
#if SUNDIALS_VERSION_MAJOR >= 6
  SUNContext_Free(&sunctx);
#endif
}

Solution CasadiSolver::solve(np_array t_np, np_array y0_np, np_array yp0_np,
                             np_array_dense inputs)
{
  DEBUG("CasadiSolver::solve");
  int number_of_timesteps = t_np.request().size;

  // set inputs
  auto p_inputs = inputs.unchecked<2>();
  for (int i = 0; i < functions->inputs.size(); i++)
  {
    functions->inputs[i] = p_inputs(i, 0);
  }

  realtype *yval = N_VGetArrayPointer(yy);
  realtype *ypval = N_VGetArrayPointer(yp);
  std::vector<realtype *> ySval(number_of_parameters);
  for (int is = 0 ; is < number_of_parameters; is++) {
    ySval[is] = N_VGetArrayPointer(yyS[is]);
    N_VConst(RCONST(0.0), yyS[is]);
    N_VConst(RCONST(0.0), ypS[is]);
  }

  auto t = t_np.unchecked<1>();
  auto y0 = y0_np.unchecked<1>();
  auto yp0 = yp0_np.unchecked<1>();
  for (int i = 0; i < number_of_states; i++)
  {
    yval[i] = y0[i];
    ypval[i] = yp0[i];
  }

  realtype t0 = RCONST(t(0));
  IDAReInit(ida_mem, t0, yy, yp);

  int t_i = 1;
  realtype tret;
  realtype t_next;
  realtype t_final = t(number_of_timesteps - 1);

  // set return vectors
  realtype *t_return = new realtype[number_of_timesteps];
  realtype *y_return = new realtype[number_of_timesteps * number_of_states];
  realtype *yS_return = new realtype[number_of_parameters *
                                     number_of_timesteps * number_of_states];

  py::capsule free_t_when_done(t_return,
                               [](void *f)
                               {
                                 realtype *vect =
                                     reinterpret_cast<realtype *>(f);
                                 delete[] vect;
                               });
  py::capsule free_y_when_done(y_return,
                               [](void *f)
                               {
                                 realtype *vect =
                                     reinterpret_cast<realtype *>(f);
                                 delete[] vect;
                               });
  py::capsule free_yS_when_done(yS_return,
                                [](void *f)
                                {
                                  realtype *vect =
                                      reinterpret_cast<realtype *>(f);
                                  delete[] vect;
                                });

  t_return[0] = t(0);
  for (int j = 0; j < number_of_states; j++)
  {
    y_return[j] = yval[j];
  }
  for (int j = 0; j < number_of_parameters; j++)
  {
    const int base_index = j * number_of_timesteps * number_of_states;
    for (int k = 0; k < number_of_states; k++)
    {
      yS_return[base_index + k] = ySval[j][k];
    }
  }

  // calculate consistent initial conditions
  DEBUG("IDACalcIC");
  IDACalcIC(ida_mem, IDA_YA_YDP_INIT, t(1));

  int retval;
  while (true)
  {
    t_next = t(t_i);
    IDASetStopTime(ida_mem, t_next);
    DEBUG("IDASolve");
    retval = IDASolve(ida_mem, t_final, &tret, yy, yp, IDA_NORMAL);

    if (retval == IDA_TSTOP_RETURN || retval == IDA_SUCCESS ||
        retval == IDA_ROOT_RETURN)
    {
      if (number_of_parameters > 0)
      {
        IDAGetSens(ida_mem, &tret, yyS);
      }

      t_return[t_i] = tret;
      for (int j = 0; j < number_of_states; j++)
      {
        y_return[t_i * number_of_states + j] = yval[j];
      }
      for (int j = 0; j < number_of_parameters; j++)
      {
        const int base_index =
            j * number_of_timesteps * number_of_states + t_i * number_of_states;
        for (int k = 0; k < number_of_states; k++)
        {
          yS_return[base_index + k] = ySval[j][k];
        }
      }
      t_i += 1;
      if (retval == IDA_SUCCESS || retval == IDA_ROOT_RETURN)
      {
        break;
      }
    }
    else
    {
      // failed
      break;
    }
  }

  np_array t_ret = np_array(t_i, &t_return[0], free_t_when_done);
  np_array y_ret =
      np_array(t_i * number_of_states, &y_return[0], free_y_when_done);
  np_array yS_ret = np_array(
      std::vector<ptrdiff_t>{number_of_parameters, number_of_timesteps, number_of_states},
      &yS_return[0], free_yS_when_done);

  Solution sol(retval, t_ret, y_ret, yS_ret);

  if (options.print_stats)
  {
    long nsteps, nrevals, nlinsetups, netfails;
    int klast, kcur;
    realtype hinused, hlast, hcur, tcur;

    IDAGetIntegratorStats(ida_mem, &nsteps, &nrevals, &nlinsetups, &netfails,
                          &klast, &kcur, &hinused, &hlast, &hcur, &tcur);

    long nniters, nncfails;
    IDAGetNonlinSolvStats(ida_mem, &nniters, &nncfails);

    long int ngevalsBBDP = 0;
    if (options.using_iterative_solver)
    {
      IDABBDPrecGetNumGfnEvals(ida_mem, &ngevalsBBDP);
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

  return sol;
}
