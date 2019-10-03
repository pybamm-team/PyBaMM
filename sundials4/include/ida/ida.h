/* -----------------------------------------------------------------
 * Programmer(s): Allan G. Taylor, Alan C. Hindmarsh, Radu Serban,
 *                and Aaron Collier @ LLNL
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2019, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------
 * This is the header file for the main IDA solver.
 * -----------------------------------------------------------------*/

#ifndef _IDA_H
#define _IDA_H

#include <stdio.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_nonlinearsolver.h>
#include <ida/ida_ls.h>

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

/* -----------------
 * IDA Constants
 * ----------------- */

/* itask */
#define IDA_NORMAL           1
#define IDA_ONE_STEP         2

/* icopt */
#define IDA_YA_YDP_INIT      1
#define IDA_Y_INIT           2

/* return values */

#define IDA_SUCCESS          0
#define IDA_TSTOP_RETURN     1
#define IDA_ROOT_RETURN      2

#define IDA_WARNING          99

#define IDA_TOO_MUCH_WORK   -1
#define IDA_TOO_MUCH_ACC    -2
#define IDA_ERR_FAIL        -3
#define IDA_CONV_FAIL       -4

#define IDA_LINIT_FAIL      -5
#define IDA_LSETUP_FAIL     -6
#define IDA_LSOLVE_FAIL     -7
#define IDA_RES_FAIL        -8
#define IDA_REP_RES_ERR     -9
#define IDA_RTFUNC_FAIL     -10
#define IDA_CONSTR_FAIL     -11

#define IDA_FIRST_RES_FAIL  -12
#define IDA_LINESEARCH_FAIL -13
#define IDA_NO_RECOVERY     -14
#define IDA_NLS_INIT_FAIL   -15
#define IDA_NLS_SETUP_FAIL  -16

#define IDA_MEM_NULL        -20
#define IDA_MEM_FAIL        -21
#define IDA_ILL_INPUT       -22
#define IDA_NO_MALLOC       -23
#define IDA_BAD_EWT         -24
#define IDA_BAD_K           -25
#define IDA_BAD_T           -26
#define IDA_BAD_DKY         -27
#define IDA_VECTOROP_ERR    -28

#define IDA_UNRECOGNIZED_ERROR -99


/* ------------------------------
 * User-Supplied Function Types
 * ------------------------------ */

typedef int (*IDAResFn)(realtype tt, N_Vector yy, N_Vector yp,
                        N_Vector rr, void *user_data);

typedef int (*IDARootFn)(realtype t, N_Vector y, N_Vector yp,
                         realtype *gout, void *user_data);

typedef int (*IDAEwtFn)(N_Vector y, N_Vector ewt, void *user_data);

typedef void (*IDAErrHandlerFn)(int error_code,
                                const char *module, const char *function,
                                char *msg, void *user_data);

/* -------------------
 * Exported Functions
 * ------------------- */

/* Initialization functions */
SUNDIALS_EXPORT void *IDACreate(void);

SUNDIALS_EXPORT int IDAInit(void *ida_mem, IDAResFn res, realtype t0,
                            N_Vector yy0, N_Vector yp0);
SUNDIALS_EXPORT int IDAReInit(void *ida_mem, realtype t0, N_Vector yy0,
                              N_Vector yp0);

/* Tolerance input functions */
SUNDIALS_EXPORT int IDASStolerances(void *ida_mem, realtype reltol,
                                    realtype abstol);
SUNDIALS_EXPORT int IDASVtolerances(void *ida_mem, realtype reltol,
                                    N_Vector abstol);
SUNDIALS_EXPORT int IDAWFtolerances(void *ida_mem, IDAEwtFn efun);

/* Initial condition calculation function */
SUNDIALS_EXPORT int IDACalcIC(void *ida_mem, int icopt, realtype tout1);

/* Initial condition calculation optional input functions */
SUNDIALS_EXPORT int IDASetNonlinConvCoefIC(void *ida_mem, realtype epiccon);
SUNDIALS_EXPORT int IDASetMaxNumStepsIC(void *ida_mem, int maxnh);
SUNDIALS_EXPORT int IDASetMaxNumJacsIC(void *ida_mem, int maxnj);
SUNDIALS_EXPORT int IDASetMaxNumItersIC(void *ida_mem, int maxnit);
SUNDIALS_EXPORT int IDASetLineSearchOffIC(void *ida_mem, booleantype lsoff);
SUNDIALS_EXPORT int IDASetStepToleranceIC(void *ida_mem, realtype steptol);
SUNDIALS_EXPORT int IDASetMaxBacksIC(void *ida_mem, int maxbacks);

/* Optional input functions */
SUNDIALS_EXPORT int IDASetErrHandlerFn(void *ida_mem, IDAErrHandlerFn ehfun,
                                       void *eh_data);
SUNDIALS_EXPORT int IDASetErrFile(void *ida_mem, FILE *errfp);
SUNDIALS_EXPORT int IDASetUserData(void *ida_mem, void *user_data);
SUNDIALS_EXPORT int IDASetMaxOrd(void *ida_mem, int maxord);
SUNDIALS_EXPORT int IDASetMaxNumSteps(void *ida_mem, long int mxsteps);
SUNDIALS_EXPORT int IDASetInitStep(void *ida_mem, realtype hin);
SUNDIALS_EXPORT int IDASetMaxStep(void *ida_mem, realtype hmax);
SUNDIALS_EXPORT int IDASetStopTime(void *ida_mem, realtype tstop);
SUNDIALS_EXPORT int IDASetNonlinConvCoef(void *ida_mem, realtype epcon);
SUNDIALS_EXPORT int IDASetMaxErrTestFails(void *ida_mem, int maxnef);
SUNDIALS_EXPORT int IDASetMaxNonlinIters(void *ida_mem, int maxcor);
SUNDIALS_EXPORT int IDASetMaxConvFails(void *ida_mem, int maxncf);
SUNDIALS_EXPORT int IDASetSuppressAlg(void *ida_mem, booleantype suppressalg);
SUNDIALS_EXPORT int IDASetId(void *ida_mem, N_Vector id);
SUNDIALS_EXPORT int IDASetConstraints(void *ida_mem, N_Vector constraints);

SUNDIALS_EXPORT int IDASetNonlinearSolver(void *ida_mem,
                                          SUNNonlinearSolver NLS);

/* Rootfinding initialization function */
SUNDIALS_EXPORT int IDARootInit(void *ida_mem, int nrtfn, IDARootFn g);

/* Rootfinding optional input functions */
SUNDIALS_EXPORT int IDASetRootDirection(void *ida_mem, int *rootdir);
SUNDIALS_EXPORT int IDASetNoInactiveRootWarn(void *ida_mem);

/* Solver function */
SUNDIALS_EXPORT int IDASolve(void *ida_mem, realtype tout, realtype *tret,
                             N_Vector yret, N_Vector ypret, int itask);

/* Dense output function */
SUNDIALS_EXPORT int IDAGetDky(void *ida_mem, realtype t, int k, N_Vector dky);

/* Optional output functions */
SUNDIALS_EXPORT int IDAGetWorkSpace(void *ida_mem, long int *lenrw,
                                    long int *leniw);
SUNDIALS_EXPORT int IDAGetNumSteps(void *ida_mem, long int *nsteps);
SUNDIALS_EXPORT int IDAGetNumResEvals(void *ida_mem, long int *nrevals);
SUNDIALS_EXPORT int IDAGetNumLinSolvSetups(void *ida_mem, long int *nlinsetups);
SUNDIALS_EXPORT int IDAGetNumErrTestFails(void *ida_mem, long int *netfails);
SUNDIALS_EXPORT int IDAGetNumBacktrackOps(void *ida_mem, long int *nbacktr);
SUNDIALS_EXPORT int IDAGetConsistentIC(void *ida_mem, N_Vector yy0_mod,
                                       N_Vector yp0_mod);
SUNDIALS_EXPORT int IDAGetLastOrder(void *ida_mem, int *klast);
SUNDIALS_EXPORT int IDAGetCurrentOrder(void *ida_mem, int *kcur);
SUNDIALS_EXPORT int IDAGetActualInitStep(void *ida_mem, realtype *hinused);
SUNDIALS_EXPORT int IDAGetLastStep(void *ida_mem, realtype *hlast);
SUNDIALS_EXPORT int IDAGetCurrentStep(void *ida_mem, realtype *hcur);
SUNDIALS_EXPORT int IDAGetCurrentTime(void *ida_mem, realtype *tcur);
SUNDIALS_EXPORT int IDAGetTolScaleFactor(void *ida_mem, realtype *tolsfact);
SUNDIALS_EXPORT int IDAGetErrWeights(void *ida_mem, N_Vector eweight);
SUNDIALS_EXPORT int IDAGetEstLocalErrors(void *ida_mem, N_Vector ele);
SUNDIALS_EXPORT int IDAGetNumGEvals(void *ida_mem, long int *ngevals);
SUNDIALS_EXPORT int IDAGetRootInfo(void *ida_mem, int *rootsfound);
SUNDIALS_EXPORT int IDAGetIntegratorStats(void *ida_mem, long int *nsteps,
                                          long int *nrevals,
                                          long int *nlinsetups,
                                          long int *netfails,
                                          int *qlast, int *qcur,
                                          realtype *hinused, realtype *hlast,
                                          realtype *hcur, realtype *tcur);
SUNDIALS_EXPORT int IDAGetNumNonlinSolvIters(void *ida_mem, long int *nniters);
SUNDIALS_EXPORT int IDAGetNumNonlinSolvConvFails(void *ida_mem,
                                                 long int *nncfails);
SUNDIALS_EXPORT int IDAGetNonlinSolvStats(void *ida_mem, long int *nniters,
                                          long int *nncfails);
SUNDIALS_EXPORT char *IDAGetReturnFlagName(long int flag);

/* Free function */
SUNDIALS_EXPORT void IDAFree(void **ida_mem);


#ifdef __cplusplus
}
#endif

#endif
