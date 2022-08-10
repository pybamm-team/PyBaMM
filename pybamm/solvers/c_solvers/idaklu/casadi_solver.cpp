#include "casadi_solver.hpp"
#include "casadi_sundials_functions.hpp"


CasadiSolver
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

  CasadiFunctions functions(
      rhs_alg, jac_times_cjmass, jac_times_cjmass_nnz, jac_times_cjmass_rowvals,
      jac_times_cjmass_colptrs, inputs, jac_action, mass_action, sens, events,
      number_of_states, number_of_events, number_of_parameters);

  CasadiSolver solver(atol_np, number_of_parameters, use_jacobian,
                      jac_times_cjmass_nnz, functions);
  return solver;
}

CasadiSolver::CasadiSolver(np_array atol_np, int number_of_parameters,
                           bool use_jacobian, int jac_times_cjmass_nnz,
                           CasadiFunctions &functions)
    : number_of_states(atol_np.request().size),
      number_of_parameters(number_of_parameters),
      jac_times_cjmass_nnz(jac_times_cjmass_nnz), functions(functions)
{
  auto atol = atol_np.unchecked<1>();

  // allocate vectors
  yy = N_VNew_Serial(number_of_states);
  yp = N_VNew_Serial(number_of_states);
  avtol = N_VNew_Serial(number_of_states);

  if (number_of_parameters > 0)
  {
    yyS = N_VCloneVectorArray(number_of_parameters, yy);
    ypS = N_VCloneVectorArray(number_of_parameters, yp);
  }

  // set initial value
  yval = N_VGetArrayPointer(yy);
  if (number_of_parameters > 0)
  {
    ySval = N_VGetArrayPointer(yyS[0]);
  }
  ypval = N_VGetArrayPointer(yp);
  atval = N_VGetArrayPointer(avtol);
  for (int i = 0; i < number_of_states; i++)
  {
    atval[i] = atol[i];
  }

  for (int is = 0; is < number_of_parameters; is++)
  {
    N_VConst(RCONST(0.0), yyS[is]);
    N_VConst(RCONST(0.0), ypS[is]);
  }

  // allocate memory for solver
  ida_mem = IDACreate();

  // initialise solver
  IDAInit(ida_mem, residual_casadi, 0, yy, yp);

  // set tolerances
  rtol = RCONST(rel_tol);

  IDASVtolerances(ida_mem, rtol, avtol);

  // set events
  IDARootInit(ida_mem, number_of_events, events_casadi);

  void *user_data = &functions;
  IDASetUserData(ida_mem, user_data);

  // set linear solver
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

  IDASetLinearSolver(ida_mem, LS, J);

  if (use_jacobian == 1)
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
}

Solution CasadiSolver::solve(np_array t_np, np_array y0_np, np_array yp0_np,
                             np_array_dense inputs)
{
  int number_of_timesteps = t_np.request().size;
  auto t = t_np.unchecked<1>();

  for (int i = 0; i < number_of_states; i++)
  {
    yval[i] = y0[i];
    ypval[i] = yp0[i];
  }

  IDAReInit(ida_mem, residual_casadi, t0, yy, yp);

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

  // std::cout << "finished solving 9" << std::endl;

  // std::cout << "finished solving 10" << std::endl;

  return sol;
}
