#include "CasadiSolverOpenMP.hpp"
#include "casadi_sundials_functions.hpp"

/*
 * This is an abstract class that implements an OpenMP solution but
 * requires a linear solver to create a concrete class. Hook functions
 * are intended to be overriden to support alternative solver
 * approaches, as needed.
 */
    
/* Skeleton workflow:
   https://sundials.readthedocs.io/en/latest/ida/Usage/index.html
      1. (N/A) Initialize parallel or multi-threaded environment
      2. Create the SUNDIALS context object
      3. Create the vector of initial values
      4. Create matrix object (if appropriate)
      5. Create linear solver object
      6. (N/A) Create nonlinear solver object
      7. Create IDA object
      8. Initialize IDA solver
      9. Specify integration tolerances
     10. Attach the linear solver
     11. Set linear solver optional inputs
     12. (N/A) Attach nonlinear solver module
     13. (N/A) Set nonlinear solver optional inputs
     14. Specify rootfinding problem (optional)
     15. Set optional inputs
     16. Correct initial values (optional)
     17. Advance solution in time
     18. Get optional outputs
     19. Destroy objects
     20. (N/A) Finalize MPI
*/

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
      CSR_MAT,
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
  DEBUG("CasadiSolverOpenMP::Initialize");
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

  // create linear solver object
  SetLinearSolver();

  // attach the linear solver
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
        " but got " + std::to_string(y0.size())
      );

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

    int t_i = 1;
    realtype tret;
    realtype t_next;
    realtype t_final = t(number_of_timesteps - 1);

    // set return vectors
    realtype *t_return = new realtype[number_of_timesteps];
    realtype *y_return = new realtype[number_of_timesteps *
                                      number_of_states];
    realtype *yS_return = new realtype[number_of_parameters *
                                       number_of_timesteps *
                                       number_of_states];

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

    t_return[0] = t(0);
    for (int j = 0; j < number_of_states; j++)
      y_return[j] = yval[j];
    for (int j = 0; j < number_of_parameters; j++)
    {
      const int base_index = j * number_of_timesteps * number_of_states;
      for (int k = 0; k < number_of_states; k++)
        yS_return[base_index + k] = ySval[j][k];
    }

    int retval;
    while (true)
    {
      t_next = t(t_i);
      IDASetStopTime(ida_mem, t_next);
      DEBUG("IDASolve");
      retval = IDASolve(ida_mem, t_final, &tret, yy, yp, IDA_NORMAL);

      if (retval == IDA_TSTOP_RETURN ||
          retval == IDA_SUCCESS ||
          retval == IDA_ROOT_RETURN)
      {
        if (number_of_parameters > 0)
          IDAGetSens(ida_mem, &tret, yyS);

        t_return[t_i] = tret;
        for (int j = 0; j < number_of_states; j++)
          y_return[t_i * number_of_states + j] = yval[j];
        for (int j = 0; j < number_of_parameters; j++)
        {
          const int base_index =
            j * number_of_timesteps * number_of_states +
            t_i * number_of_states;
          for (int k = 0; k < number_of_states; k++)
            yS_return[base_index + k] = ySval[j][k];
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
      t_i * number_of_states,
      &y_return[0],
      free_y_when_done
    );
    np_array yS_ret = np_array(
      std::vector<ptrdiff_t> {
        number_of_parameters,
        number_of_timesteps,
        number_of_states
      },
      &yS_return[0],
      free_yS_when_done
    );

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
