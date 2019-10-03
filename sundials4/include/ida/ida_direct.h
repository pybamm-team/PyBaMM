/* -----------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ SMU
 *                Radu Serban @ LLNL
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
 * Header file for the deprecated direct linear solver interface in
 * IDA; these routines now just wrap the updated IDA generic
 * linear solver interface in ida_ls.h.
 * -----------------------------------------------------------------*/

#ifndef _IDADLS_H
#define _IDADLS_H

#include <ida/ida_ls.h>

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif


/*=================================================================
  Function Types (typedefs for equivalent types in ida_ls.h)
  =================================================================*/

typedef IDALsJacFn IDADlsJacFn;

/*===================================================================
  Exported Functions (wrappers for equivalent routines in ida_ls.h)
  ===================================================================*/

int IDADlsSetLinearSolver(void *ida_mem, SUNLinearSolver LS,
                          SUNMatrix A);

int IDADlsSetJacFn(void *ida_mem, IDADlsJacFn jac);

int IDADlsGetWorkSpace(void *ida_mem, long int *lenrwLS,
                       long int *leniwLS);

int IDADlsGetNumJacEvals(void *ida_mem, long int *njevals);

int IDADlsGetNumResEvals(void *ida_mem, long int *nrevalsLS);

int IDADlsGetLastFlag(void *ida_mem, long int *flag);

char *IDADlsGetReturnFlagName(long int flag);


#ifdef __cplusplus
}
#endif

#endif
