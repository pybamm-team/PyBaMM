/*
 * -----------------------------------------------------------------
 * Programmer(s): Daniel Reynolds @ SMU
 * Based on sundials_klu_impl.h and arkode_klu.h/cvode_klu.h/... 
 *     code, written by Carol S. Woodward @ LLNL
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
 * This is the header file for the KLU implementation of the 
 * SUNLINSOL module, SUNLINSOL_KLU.
 * 
 * Note:
 *   - The definition of the generic SUNLinearSolver structure can 
 *     be found in the header file sundials_linearsolver.h.
 * -----------------------------------------------------------------
 */

#ifndef _SUNLINSOL_KLU_H
#define _SUNLINSOL_KLU_H

#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sunmatrix/sunmatrix_sparse.h>
#ifndef _KLU_H
#include <klu.h>
#endif

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

/* Default KLU solver parameters */
#define SUNKLU_ORDERING_DEFAULT  1    /* COLAMD */
#define SUNKLU_REINIT_FULL       1
#define SUNKLU_REINIT_PARTIAL    2

/* Interfaces to match 'sunindextype' with the correct KLU types/functions */
#if defined(SUNDIALS_INT64_T)
#define sun_klu_symbolic      klu_l_symbolic
#define sun_klu_numeric       klu_l_numeric
#define sun_klu_common        klu_l_common
#define sun_klu_analyze       klu_l_analyze
#define sun_klu_factor        klu_l_factor
#define sun_klu_refactor      klu_l_refactor
#define sun_klu_rcond         klu_l_rcond
#define sun_klu_condest       klu_l_condest
#define sun_klu_defaults      klu_l_defaults
#define sun_klu_free_symbolic klu_l_free_symbolic
#define sun_klu_free_numeric  klu_l_free_numeric
#elif defined(SUNDIALS_INT32_T)
#define sun_klu_symbolic      klu_symbolic
#define sun_klu_numeric       klu_numeric
#define sun_klu_common        klu_common
#define sun_klu_analyze       klu_analyze
#define sun_klu_factor        klu_factor
#define sun_klu_refactor      klu_refactor
#define sun_klu_rcond         klu_rcond
#define sun_klu_condest       klu_condest
#define sun_klu_defaults      klu_defaults
#define sun_klu_free_symbolic klu_free_symbolic
#define sun_klu_free_numeric  klu_free_numeric
#else  /* incompatible sunindextype for KLU */
#error  Incompatible sunindextype for KLU
#endif

#if defined(SUNDIALS_DOUBLE_PRECISION)
#else
#error  Incompatible realtype for KLU
#endif

/* --------------------------------------
 * KLU Implementation of SUNLinearSolver
 * -------------------------------------- */

/* Create a typedef for the KLU solver function pointer to suppress compiler
 * warning messages about sunindextype vs internal KLU index types. */

typedef sunindextype (*KLUSolveFn)(sun_klu_symbolic*, sun_klu_numeric*,
                                   sunindextype, sunindextype,
                                   double*, sun_klu_common*);
 
struct _SUNLinearSolverContent_KLU {
  long int         last_flag;
  int              first_factorize;
  sun_klu_symbolic *symbolic;
  sun_klu_numeric  *numeric;
  sun_klu_common   common;
  KLUSolveFn       klu_solver;
};

typedef struct _SUNLinearSolverContent_KLU *SUNLinearSolverContent_KLU;

  
/* -------------------------------------
 * Exported Functions for SUNLINSOL_KLU
 * ------------------------------------- */

SUNDIALS_EXPORT SUNLinearSolver SUNLinSol_KLU(N_Vector y, SUNMatrix A);
SUNDIALS_EXPORT int SUNLinSol_KLUReInit(SUNLinearSolver S, SUNMatrix A,
                                        sunindextype nnz, int reinit_type);
SUNDIALS_EXPORT int SUNLinSol_KLUSetOrdering(SUNLinearSolver S,
                                             int ordering_choice);

/* deprecated */
SUNDIALS_EXPORT SUNLinearSolver SUNKLU(N_Vector y, SUNMatrix A);
/* deprecated */
SUNDIALS_EXPORT int SUNKLUReInit(SUNLinearSolver S, SUNMatrix A,
                                 sunindextype nnz, int reinit_type);
/* deprecated */
SUNDIALS_EXPORT int SUNKLUSetOrdering(SUNLinearSolver S,
                                      int ordering_choice);

SUNDIALS_EXPORT SUNLinearSolver_Type SUNLinSolGetType_KLU(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolInitialize_KLU(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolSetup_KLU(SUNLinearSolver S, SUNMatrix A);
SUNDIALS_EXPORT int SUNLinSolSolve_KLU(SUNLinearSolver S, SUNMatrix A,
                                       N_Vector x, N_Vector b, realtype tol);
SUNDIALS_EXPORT long int SUNLinSolLastFlag_KLU(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolSpace_KLU(SUNLinearSolver S,
                                       long int *lenrwLS,
                                       long int *leniwLS);
SUNDIALS_EXPORT int SUNLinSolFree_KLU(SUNLinearSolver S);
  

#ifdef __cplusplus
}
#endif

#endif
