#include "CasadiSolverCuda_solvers.hpp"
#include "casadi_sundials_functions_cuda.hpp"

/*
 * Experimental CUDA implementation
 */

CasadiSolverCuda_cuSolverSp_batchQR::CasadiSolverCuda_cuSolverSp_batchQR(
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
  Initialize();
}

void CasadiSolverCuda_cuSolverSp_batchQR::sync_device() {
  rtn = cudaDeviceSynchronize();
  if (rtn != cudaSuccess)
    throw std::runtime_error("cudaDeviceSynchronize: Failed with code " + std::to_string(rtn));
}

void CasadiSolverCuda_cuSolverSp_batchQR::SetLinearSolver() {
}

void CasadiSolverCuda_cuSolverSp_batchQR::AllocateVectors() {
  // Device vectors
  yy = N_VNewManaged_Cuda(number_of_states, sunctx);
  yp = N_VNewManaged_Cuda(number_of_states, sunctx);
  avtol = N_VNewManaged_Cuda(number_of_states, sunctx);
  id = N_VNewManaged_Cuda(number_of_states, sunctx);
}

void CasadiSolverCuda_cuSolverSp_batchQR::SetMatrix() {
}

void CasadiSolverCuda_cuSolverSp_batchQR::Initialize() {
  DEBUG("CasadiSolver_cuSolverSp_batchQR::Initialize");
  auto atol = atol_np.unchecked<1>();

  printf("Number of parameters: %d\n", number_of_parameters);

  // Create SUNDIALS context object
  if (SUNContext_Create(NULL, &sunctx)) {
    throw std::runtime_error("SUNContext_Create: failed");
  }
  
  // Initialize cuSPARSE
  cusp_status = cusparseCreate(&cusp_handle);
  if (cusp_status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error("cusparseCreate: could not create cuSPARSE handle");
  }

  // Initialize cuSOLVER
  cusol_status = cusolverSpCreate(&cusol_handle);
  if (cusol_status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error("cusolverSpCreate: could not create cuSOLVER handle");
  }

  // Allocate memory for solver
  ida_mem = IDACreate(sunctx);
  if (ida_mem == nullptr)
    throw std::runtime_error("IDACreate: could not create memory for solver");

  // Create and initialize vectors
  AllocateVectors();
  realtype *atval = N_VGetArrayPointer(avtol);
  for (int i = 0; i < number_of_states; i++)
    atval[i] = atol[i];
  
  // Sensitivity vectors
  if (number_of_parameters > 0)
  {
    yyS = N_VCloneVectorArray(number_of_parameters, yy);
    ypS = N_VCloneVectorArray(number_of_parameters, yp);
  }
  for (int is = 0; is < number_of_parameters; is++)
  {
    N_VConst(RCONST(0.0), yyS[is]);
    N_VConst(RCONST(0.0), ypS[is]);
  }
  
  // Create the device matrix
  nblocks = 1;
  block_nnz = jac_times_cjmass_nnz;
  J = SUNMatrix_cuSparse_NewBlockCSR(
    nblocks,
    number_of_states,
    number_of_states,
    block_nnz,
    cusp_handle,
    sunctx
  );
  if (J == NULL)
    throw std::runtime_error("ERROR: could not create J\n");


  

  std::cout << "asssi\n";
  std::cout << number_of_states << '\n'; //2800
  std::cout << SUNMatrix_cuSparse_BlockRows(J) << '\n'; // 2800
  std::cout << SUNMatrix_cuSparse_BlockNNZ(J) << '\n'; // 12388

  DEBUG(":");
  realtype *jac_data =  SUNMatrix_cuSparse_BlockData(J, 0);
  std::cout << SUNMatrix_cuSparse_Rows(J) << '\n'; // 2800
  std::cout << SUNMatrix_cuSparse_Columns(J) << '\n'; // 2800

  DEBUG("1");
  jac_data =
    (realtype*) malloc(
      SUNMatrix_cuSparse_NNZ(J) * sizeof(realtype)
    );
  sunindextype *jac_colptrs =
    (sunindextype*) malloc(
      (SUNMatrix_cuSparse_BlockRows(J)+1) * sizeof(sunindextype)
    );
  sunindextype *jac_rowvals =
    (sunindextype*) malloc(
      SUNMatrix_cuSparse_BlockNNZ(J) * sizeof(sunindextype)
    );
  DEBUG("2");
  if (SUNMatrix_cuSparse_CopyFromDevice(J, jac_data, jac_colptrs, jac_rowvals))
    throw std::runtime_error("SUNMatrix_cuSparse_CopyFromDevice: Failed");
  
  DEBUG("3");
/*  for (int i = 0; i < SUNMatrix_cuSparse_BlockRows(J); i++) {
    std::cout << jac_rowvals[i] << " ";
  }
  DEBUG("4");
  for (int i = 1; i < SUNMatrix_cuSparse_BlockNNZ(J); i++)
    std::cout << jac_colptrs[i] << " ";*/
  DEBUG("5");
  SUNMatrix_cuSparse_CopyToDevice(
    J,
    jac_data,
    jac_colptrs,
    jac_rowvals
  );
  DEBUG("6");
  cudaDeviceSynchronize();
  DEBUG("7");





  
  // Initialise solver
  rtn = IDAInit(ida_mem, residual_casadi_cuda, 0, yy, yp);
  if (rtn != IDA_SUCCESS)
    throw std::runtime_error("IDAInit: Return value: " + std::to_string(rtn));

  // Set tolerances
  rtol = RCONST(rel_tol);
  rtn = IDASVtolerances(ida_mem, rtol, avtol);
  if (rtn != IDA_SUCCESS)
    throw std::runtime_error("IDASVtolerances: Return value: " + std::to_string(rtn));

  // Set events
  rtn = IDARootInit(ida_mem, number_of_events, events_casadi_cuda);
  if (rtn != IDA_SUCCESS)
    throw std::runtime_error("IDARootInit: Return value: " + std::to_string(rtn));
  void *user_data = functions.get();
  rtn = IDASetUserData(ida_mem, user_data);
  if (rtn != IDA_SUCCESS)
    throw std::runtime_error("IDASetUserData: Return value: " + std::to_string(rtn));

  // Specify preconditioner type
  precon_type = SUN_PREC_NONE;
  if (options.preconditioner != "none") {
    precon_type = SUN_PREC_LEFT;
  }
  
  // Create linear solver object
  LS = SUNLinSol_cuSolverSp_batchQR(yy, J, cusol_handle, sunctx);
  if (LS == NULL)
    throw std::runtime_error("SUNLinSol_cuSolverSp_batchQR: returned NULL");

  // Attach the linear solver
  rtn = IDASetLinearSolver(ida_mem, LS, J);
  if (rtn != IDA_SUCCESS)
    throw std::runtime_error("IDASetLinearSolver: Return value: " + std::to_string(rtn));

  // Setup preconditioner
  if (options.preconditioner != "none")
  {
    DEBUG("\tsetting IDADDB preconditioner");
    rtn = IDABBDPrecInit(
      ida_mem, number_of_states, options.precon_half_bandwidth,
      options.precon_half_bandwidth, options.precon_half_bandwidth_keep,
      options.precon_half_bandwidth_keep, 0.0, residual_casadi_approx_cuda, NULL);
    if (rtn != IDA_SUCCESS)
      throw std::runtime_error("IDABBDPrecInit: Return value: " + std::to_string(rtn));
  }

  if (options.jacobian == "matrix-free")
  {
    rtn = IDASetJacTimes(ida_mem, NULL, jtimes_casadi_cuda);
    if (rtn != IDA_SUCCESS)
      throw std::runtime_error("IDASetJacTimes: Return value: " + std::to_string(rtn));
  }
  else if (options.jacobian != "none")
  {
    rtn = IDASetJacFn(ida_mem, jacobian_casadi_cuda);
    if (rtn != IDA_SUCCESS)
      throw std::runtime_error("IDASetJacJn: Return value: " + std::to_string(rtn));
  }

  if (number_of_parameters > 0)
  {
    rtn = IDASensInit(
      ida_mem,
      number_of_parameters,
      IDA_SIMULTANEOUS,
      sensitivities_casadi_cuda,
      yyS,
      ypS);
    if (rtn != IDA_SUCCESS)
      throw std::runtime_error("IDASensInit: Return value: " + std::to_string(rtn));
    rtn = IDASensEEtolerances(ida_mem);
    if (rtn != IDA_SUCCESS)
      throw std::runtime_error("IDASensEEtolerances: Return value: " + std::to_string(rtn));
  }

  rtn = SUNLinSolInitialize(LS);
  if (rtn)
      throw std::runtime_error("SUNLinSolInitialize: Return value: " + std::to_string(rtn));

  auto id_np_val = rhs_alg_id.unchecked<1>();
  realtype *id_val = N_VGetArrayPointer(id);
  for (int ii = 0; ii < number_of_states; ii++)
    id_val[ii] = id_np_val[ii];

  rtn = IDASetId(ida_mem, id);
  if (rtn != IDA_SUCCESS)
    throw std::runtime_error("IDASetId: Return value: " + std::to_string(rtn));
}

CasadiSolverCuda_cuSolverSp_batchQR::~CasadiSolverCuda_cuSolverSp_batchQR()
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

Solution CasadiSolverCuda_cuSolverSp_batchQR::solve(
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
  for (int i = 0; i < number_of_states; i++)
  {
    yval[i] = y0[i];
    ypval[i] = yp0[i];
  }

  // sensitivity parameters
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

  rtn = IDAReInit(ida_mem, t0, yy, yp);
  if (rtn)
    throw std::runtime_error("IDAReInit: Return value: " + std::to_string(rtn));
  if (number_of_parameters > 0) {
    rtn = IDASensReInit(ida_mem, IDA_SIMULTANEOUS, yyS, ypS);
    if (rtn)
      throw std::runtime_error("IDASensReInit: Return value: " + std::to_string(rtn));
  }

  // correct initial values
  DEBUG("IDACalcIC");
  rtn = IDACalcIC(ida_mem, IDA_YA_YDP_INIT, t(1));
  if (rtn)
    throw std::runtime_error("IDACalcIC: Return value: " + std::to_string(rtn));
  if (number_of_parameters > 0) {
    rtn = IDAGetSens(ida_mem, &t0, yyS);
    if (rtn)
      throw std::runtime_error("IDAGetSens: Return value: " + std::to_string(rtn));
  }

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
    IDASetStopTime(
      ida_mem,
      t_next);
    DEBUG("IDASolve");
    retval = IDASolve(
      ida_mem,
      t_final,
      &tret,
      yy,
      yp,
      IDA_NORMAL);

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
