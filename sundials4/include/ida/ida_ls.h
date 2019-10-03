/* ----------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ SMU
 *                Alan Hindmarsh, Radu Serban and
 *                Aaron Collier @ LLNL
 * ----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2019, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * ----------------------------------------------------------------
 * This is the header file for IDA's linear solver interface.
 * ----------------------------------------------------------------*/

#ifndef _IDALS_H
#define _IDALS_H

#include <sundials/sundials_direct.h>
#include <sundials/sundials_iterative.h>
#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif


/*=================================================================
  IDALS Constants
  =================================================================*/

#define IDALS_SUCCESS           0
#define IDALS_MEM_NULL         -1
#define IDALS_LMEM_NULL        -2
#define IDALS_ILL_INPUT        -3
#define IDALS_MEM_FAIL         -4
#define IDALS_PMEM_NULL        -5
#define IDALS_JACFUNC_UNRECVR  -6
#define IDALS_JACFUNC_RECVR    -7
#define IDALS_SUNMAT_FAIL      -8
#define IDALS_SUNLS_FAIL       -9


/*=================================================================
  IDALS user-supplied function prototypes
  =================================================================*/

typedef int (*IDALsJacFn)(realtype t, realtype c_j, N_Vector y,
                          N_Vector yp, N_Vector r, SUNMatrix Jac,
                          void *user_data, N_Vector tmp1,
                          N_Vector tmp2, N_Vector tmp3);

typedef int (*IDALsPrecSetupFn)(realtype tt, N_Vector yy,
                                N_Vector yp, N_Vector rr,
                                realtype c_j, void *user_data);

typedef int (*IDALsPrecSolveFn)(realtype tt, N_Vector yy,
                                N_Vector yp, N_Vector rr,
                                N_Vector rvec, N_Vector zvec,
                                realtype c_j, realtype delta,
                                void *user_data);

typedef int (*IDALsJacTimesSetupFn)(realtype tt, N_Vector yy,
                                    N_Vector yp, N_Vector rr,
                                    realtype c_j, void *user_data);

typedef int (*IDALsJacTimesVecFn)(realtype tt, N_Vector yy,
                                  N_Vector yp, N_Vector rr,
                                  N_Vector v, N_Vector Jv,
                                  realtype c_j, void *user_data,
                                  N_Vector tmp1, N_Vector tmp2);


/*=================================================================
  IDALS Exported functions
  =================================================================*/

SUNDIALS_EXPORT int IDASetLinearSolver(void *ida_mem,
                                       SUNLinearSolver LS,
                                       SUNMatrix A);


/*-----------------------------------------------------------------
  Optional inputs to the IDALS linear solver interface
  -----------------------------------------------------------------*/

SUNDIALS_EXPORT int IDASetJacFn(void *ida_mem, IDALsJacFn jac);
SUNDIALS_EXPORT int IDASetPreconditioner(void *ida_mem,
                                         IDALsPrecSetupFn pset,
                                         IDALsPrecSolveFn psolve);
SUNDIALS_EXPORT int IDASetJacTimes(void *ida_mem,
                                   IDALsJacTimesSetupFn jtsetup,
                                   IDALsJacTimesVecFn jtimes);
SUNDIALS_EXPORT int IDASetEpsLin(void *ida_mem, realtype eplifac);
SUNDIALS_EXPORT int IDASetIncrementFactor(void *ida_mem,
                                          realtype dqincfac);

/*-----------------------------------------------------------------
  Optional outputs from the IDALS linear solver interface
  -----------------------------------------------------------------*/

SUNDIALS_EXPORT int IDAGetLinWorkSpace(void *ida_mem,
                                       long int *lenrwLS,
                                       long int *leniwLS);
SUNDIALS_EXPORT int IDAGetNumJacEvals(void *ida_mem,
                                      long int *njevals);
SUNDIALS_EXPORT int IDAGetNumPrecEvals(void *ida_mem,
                                       long int *npevals);
SUNDIALS_EXPORT int IDAGetNumPrecSolves(void *ida_mem,
                                        long int *npsolves);
SUNDIALS_EXPORT int IDAGetNumLinIters(void *ida_mem,
                                      long int *nliters);
SUNDIALS_EXPORT int IDAGetNumLinConvFails(void *ida_mem,
                                          long int *nlcfails);
SUNDIALS_EXPORT int IDAGetNumJTSetupEvals(void *ida_mem,
                                          long int *njtsetups);
SUNDIALS_EXPORT int IDAGetNumJtimesEvals(void *ida_mem,
                                         long int *njvevals);
SUNDIALS_EXPORT int IDAGetNumLinResEvals(void *ida_mem,
                                         long int *nrevalsLS);
SUNDIALS_EXPORT int IDAGetLastLinFlag(void *ida_mem,
                                      long int *flag);
SUNDIALS_EXPORT char *IDAGetLinReturnFlagName(long int flag);


#ifdef __cplusplus
}
#endif

#endif
