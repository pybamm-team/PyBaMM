#include "casadi_solver.hpp"
#include "casadi_sundials_functions.hpp"
#include <memory>

CasadiSolver *
create_casadi_solver(int number_of_states, int number_of_parameters,
                     const Function &rhs_alg, const Function &jac_times_cjmass,
                     const np_array_int &jac_times_cjmass_colptrs,
                     const np_array_int &jac_times_cjmass_rowvals,
                     const int jac_times_cjmass_nnz, const Function &jac_action,
                     const Function &mass_action, const Function &sens,
                     const Function &events, const int number_of_events,
                     int use_jacobian, np_array rhs_alg_id, np_array atol_np,
                     double rel_tol, np_array_dense inputs)
{
  std::cout << "create_casadi_solver" << std::endl;

  std::cout << "create CAsadiFunctions" << std::endl;
  auto functions = std::make_unique<CasadiFunctions>(
      rhs_alg, jac_times_cjmass, jac_times_cjmass_nnz, jac_times_cjmass_rowvals,
      jac_times_cjmass_colptrs, inputs, jac_action, mass_action, sens, events,
      number_of_states, number_of_events, number_of_parameters);

  std::cout << "create CAsadiSolver" << std::endl;
  return new CasadiSolver(atol_np, rel_tol, rhs_alg_id, number_of_parameters,
                      use_jacobian, jac_times_cjmass_nnz, std::move(functions));
}

CasadiSolver::CasadiSolver(np_array atol_np, double rel_tol,
                           np_array rhs_alg_id, int number_of_parameters,
                           bool use_jacobian, int jac_times_cjmass_nnz,
                           std::unique_ptr<CasadiFunctions> functions_arg)
    : number_of_states(atol_np.request().size),
      number_of_parameters(number_of_parameters),
      jac_times_cjmass_nnz(jac_times_cjmass_nnz), functions(std::move(functions_arg))
{
  std::cout << "CasadiSolver construct start" << std::endl;
  auto atol = atol_np.unchecked<1>();

  // allocate memory for solver
  std::cout << "\t allocate memory for solver" << std::endl;
#if SUNDIALS_VERSION_MAJOR >= 6
  SUNContext_Create(NULL, &sunctx);
  ida_mem = IDACreate(sunctx);
#else
  ida_mem = IDACreate();
#endif

  // allocate vectors
  std::cout << "\t allocate vectors" << std::endl;
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
  if (number_of_parameters > 0)
  {
  }
  realtype *ypval = N_VGetArrayPointer(yp);
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

  std::cout << "\t initialise solver" << std::endl;
  // initialise solver
  IDAInit(ida_mem, residual_casadi, 0, yy, yp);

  // set tolerances
  rtol = RCONST(rel_tol);

  std::cout << "\t set tolerances" << std::endl;
  IDASVtolerances(ida_mem, rtol, avtol);

  // set events
  std::cout << "\t set events" << std::endl;
  IDARootInit(ida_mem, number_of_events, events_casadi);

  void *user_data = functions.get();
  std::cout << "\t set user_data " << user_data << std::endl;
  IDASetUserData(ida_mem, user_data);

  // set linear solver
  std::cout << "\t set linear solver" << std::endl;
#if SUNDIALS_VERSION_MAJOR >= 6
  if (use_jacobian == 1)
  {
    J = SUNSparseMatrix(number_of_states, number_of_states,
                        jac_times_cjmass_nnz, CSC_MAT, sunctx);
    LS = SUNLinSol_KLU(yy, J, sunctx);
  }
  else
  {
    J = SUNDenseMatrix(number_of_states, number_of_states, sunctx);
    LS = SUNLinSol_Dense(yy, J, sunctx);
  }
#else
  if (use_jacobian == 1)
  {
    J = SUNSparseMatrix(number_of_states, number_of_states,
                        jac_times_cjmass_nnz, CSC_MAT);
    LS = SUNLinSol_KLU(yy, J);
  }
  else
  {
    J = SUNDenseMatrix(number_of_states, number_of_states);
    LS = SUNLinSol_Dense(yy, J);
  }
#endif

  IDASetLinearSolver(ida_mem, LS, J);

  if (use_jacobian == 1)
  {
    IDASetJacFn(ida_mem, jacobian_casadi);
  }

  std::cout << "\t set sens" << std::endl;
  if (number_of_parameters > 0)
  {
    IDASensInit(ida_mem, number_of_parameters, IDA_SIMULTANEOUS,
                sensitivities_casadi, yyS, ypS);
    IDASensEEtolerances(ida_mem);
  }

  SUNLinSolInitialize(LS);

  std::cout << "\t set id" << std::endl;
  auto id_np_val = rhs_alg_id.unchecked<1>();
  realtype *id_val;
  id_val = N_VGetArrayPointer(id);

  int ii;
  for (ii = 0; ii < number_of_states; ii++)
  {
    id_val[ii] = id_np_val[ii];
  }

  IDASetId(ida_mem, id);
  std::cout << "CasadiSolver construct end" << std::endl;
}

CasadiSolver::~CasadiSolver()
{

  std::cout << "CasadiSolver deconstruct start" << std::endl;
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
  std::cout << "CasadiSolver deconstruct end" << std::endl;
}

Solution CasadiSolver::solve(np_array t_np, np_array y0_np, np_array yp0_np,
                             np_array_dense inputs)
{
  std::cout << "CasadiSolver solve start" << std::endl;
  int number_of_timesteps = t_np.request().size;

  realtype *yval = N_VGetArrayPointer(yy);
  realtype *ypval = N_VGetArrayPointer(yp);
  realtype *ySval;
  if (number_of_parameters > 0)
  {
    ySval = N_VGetArrayPointer(yyS[0]);
  }

  auto t = t_np.unchecked<1>();
  auto y0 = y0_np.unchecked<1>();
  auto yp0 = yp0_np.unchecked<1>();
  for (int i = 0; i < number_of_states; i++)
  {
    yval[i] = y0[i];
    ypval[i] = yp0[i];
  }

  std::cout << "\t IDAReIinit" << std::endl;
  realtype t0 = RCONST(t(0));
  IDAReInit(ida_mem, t0, yy, yp);

  std::cout << "\t set return vectors" << std::endl;
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
      yS_return[base_index + k] = ySval[j * number_of_states + k];
    }
  }

  // calculate consistent initial conditions
  std::cout << "\t calcIC " << t(1) << std::endl;
  IDACalcIC(ida_mem, IDA_YA_YDP_INIT, t(1));

  std::cout << "\t main loop" << std::endl;
  int retval;
  while (true)
  {
    t_next = t(t_i);
    IDASetStopTime(ida_mem, t_next);
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
          yS_return[base_index + k] = ySval[j * number_of_states + k];
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

  std::cout << "\t construting return solution" << std::endl;
  np_array t_ret = np_array(t_i, &t_return[0], free_t_when_done);
  np_array y_ret =
      np_array(t_i * number_of_states, &y_return[0], free_y_when_done);
  np_array yS_ret = np_array(
      std::vector<ptrdiff_t>{number_of_parameters, t_i, number_of_states},
      &yS_return[0], free_yS_when_done);

  Solution sol(retval, t_ret, y_ret, yS_ret);

  // TODO config input to choose stuff like this
  const bool print_stats = false;
  if (print_stats)
  {
    long nsteps, nrevals, nlinsetups, netfails;
    int klast, kcur;
    realtype hinused, hlast, hcur, tcur;

    IDAGetIntegratorStats(ida_mem, &nsteps, &nrevals, &nlinsetups, &netfails,
                          &klast, &kcur, &hinused, &hlast, &hcur, &tcur);

    long nniters, nncfails;
    IDAGetNonlinSolvStats(ida_mem, &nniters, &nncfails);

    std::cout << "Solver Stats: \n"
              << "  Number of steps = " << nsteps << "\n"
              << "  Number of calls to residual function = " << nrevals << "\n"
              << "  Number of linear solver setup calls = " << nlinsetups
              << "\n"
              << "  Number of error test failures = " << netfails << "\n"
              << "  Method order used on last step = " << klast << "\n"
              << "  Method order used on next step = " << kcur << "\n"
              << "  Initial step size = " << hinused << "\n"
              << "  Step size on last step = " << hlast << "\n"
              << "  Step size on next step = " << hcur << "\n"
              << "  Current internal time reached = " << tcur << "\n"
              << "  Number of nonlinear iterations performed = " << nniters
              << "\n"
              << "  Number of nonlinear convergence failures = " << nncfails
              << "\n"
              << std::endl;
  }

  std::cout << "CasadiSolver solve end" << std::endl;
  // std::cout << "finished solving 9" << std::endl;

  // std::cout << "finished solving 10" << std::endl;

  return sol;
}
