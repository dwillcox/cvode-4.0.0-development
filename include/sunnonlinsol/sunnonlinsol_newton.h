/* -----------------------------------------------------------------------------
 * Programmer(s): David J. Gardner @ LLNL
 * -----------------------------------------------------------------------------
 * LLNS Copyright Start
 * Copyright (c) 2014, Lawrence Livermore National Security
 * This work was performed under the auspices of the U.S. Department
 * of Energy by Lawrence Livermore National Laboratory in part under
 * Contract W-7405-Eng-48 and in part under Contract DE-AC52-07NA27344.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 * For details, see the LICENSE file.
 * LLNS Copyright End
 * -----------------------------------------------------------------------------
 * This is the header file for the SUNNonlinearSolver module implementation of
 * Newton's method.
 * 
 * Part I defines the solver-specific content structure.
 * 
 * Part II contains prototypes for the solver constructor and operations.
 * ---------------------------------------------------------------------------*/

#ifndef _SUNNONLINSOL_NEWTON_H
#define _SUNNONLINSOL_NEWTON_H

#include "sundials/sundials_types.h"
#include "sundials/sundials_nvector.h"
#include "sundials/sundials_nonlinearsolver.h"

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

/* -----------------------------------------------------------------------------
 * I. Content structure
 * ---------------------------------------------------------------------------*/

struct _SUNNonlinearSolverContent_Newton {

  /* functions provided by the integrator */
  SUNNonlinSolSysFn      Sys;    /* nonlinear system residual function         */
  SUNNonlinSolLSetupFn   LSetup; /* linear solver setup function               */
  SUNNonlinSolLSolveFn   LSolve; /* linear solver solve function               */
  SUNNonlinSolConvTestFn CTest;  /* nonlinear solver convergence test function */

  /* nonlinear solver variables */
  N_Vector    delta;     /* Newton update vector                                   */
  booleantype jcur;      /* Jacobian status, current = SUNTRUE / stale = SUNFALSE  */
  int         curiter;   /* current number of iterations in a solve attempt        */
  int         maxiters;  /* maximum number of iterations in a solve attempt        */
  long int    niters;    /* total number of nonlinear iterations across all solves */
};

typedef struct _SUNNonlinearSolverContent_Newton *SUNNonlinearSolverContent_Newton;

/* -----------------------------------------------------------------------------
 * II: Exported functions
 * ---------------------------------------------------------------------------*/
 
/* Constructor to create solver and allocates memory */
SUNDIALS_EXPORT SUNNonlinearSolver SUNNonlinSol_Newton(N_Vector y);

/* core functions */
SUNDIALS_EXPORT SUNNonlinearSolver_Type SUNNonlinSolGetType_Newton(SUNNonlinearSolver NLS);

SUNDIALS_EXPORT int SUNNonlinSolInitialize_Newton(SUNNonlinearSolver NLS);

SUNDIALS_EXPORT int SUNNonlinSolSolve_Newton(SUNNonlinearSolver NLS,
                                             N_Vector y0, N_Vector y,
                                             N_Vector w, realtype tol,
                                             booleantype callLSetup, void *mem);

SUNDIALS_EXPORT int SUNNonlinSolFree_Newton(SUNNonlinearSolver NLS);

/* set functions */
SUNDIALS_EXPORT int SUNNonlinSolSetSysFn_Newton(SUNNonlinearSolver NLS,
                                                SUNNonlinSolSysFn SysFn);

SUNDIALS_EXPORT int SUNNonlinSolSetLSetupFn_Newton(SUNNonlinearSolver NLS,
                                                   SUNNonlinSolLSetupFn LSetupFn);

SUNDIALS_EXPORT int SUNNonlinSolSetLSolveFn_Newton(SUNNonlinearSolver NLS,
                                                   SUNNonlinSolLSolveFn LSolveFn);

SUNDIALS_EXPORT int SUNNonlinSolSetConvTestFn_Newton(SUNNonlinearSolver NLS,
                                                     SUNNonlinSolConvTestFn CTestFn);

SUNDIALS_EXPORT int SUNNonlinSolSetMaxIters_Newton(SUNNonlinearSolver NLS,
                                                   int maxiters);

/* get functions */
SUNDIALS_EXPORT int SUNNonlinSolGetNumIters_Newton(SUNNonlinearSolver NLS,
                                                   long int *niters);

SUNDIALS_EXPORT int SUNNonlinSolGetCurIter_Newton(SUNNonlinearSolver NLS,
                                                  int *iter);

SUNDIALS_EXPORT int SUNNonlinSolGetSysFn_Newton(SUNNonlinearSolver NLS,
                                                SUNNonlinSolSysFn *SysFn);

#ifdef __cplusplus
}
#endif

#endif
