/* -----------------------------------------------------------------------------
 * Programmer(s): David Gardner @ LLNL
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
 * This is the implementation file for a generic SUNNonlinerSolver package. It
 * contains the implementation of the SUNNonlinearSolver operations listed in
 * the 'ops' structure in sundials_nonlinearsolver.h
 * ---------------------------------------------------------------------------*/

#include <stdlib.h>
#include <sundials/sundials_nonlinearsolver.h>

/* -----------------------------------------------------------------------------
 * core functions
 * ---------------------------------------------------------------------------*/

SUNNonlinearSolver_Type SUNNonlinSolGetType(SUNNonlinearSolver NLS)
{
  return(NLS->ops->gettype(NLS));
}

int SUNNonlinSolInitialize(SUNNonlinearSolver NLS)
{
  if (NLS->ops->initialize)
    return((int) NLS->ops->initialize(NLS));
  else
    return(SUN_NLS_SUCCESS);
}

int SUNNonlinSolSetup(SUNNonlinearSolver NLS, N_Vector y, void* mem)
{
  if (NLS->ops->setup)
    return((int) NLS->ops->setup(NLS, y, mem));
  else
    return(SUN_NLS_SUCCESS);
}

int SUNNonlinSolSolve(SUNNonlinearSolver NLS,
                      N_Vector y0, N_Vector y,
                      N_Vector w, realtype tol,
                      booleantype callLSetup, void* mem)
{
  return((int) NLS->ops->solve(NLS, y0, y, w, tol, callLSetup, mem));
}

int SUNNonlinSolFree(SUNNonlinearSolver NLS)
{
  if (NLS == NULL) return(SUN_NLS_SUCCESS);
  if (NLS->ops == NULL) return(SUN_NLS_SUCCESS);

  if (NLS->ops->free) {
    return(NLS->ops->free(NLS));
  } else {
    /* free the content structure */
    if (NLS->content) {
      free(NLS->content);
      NLS->content = NULL;
    }
    /* free the ops structure */
    if (NLS->ops) {
      free(NLS->ops);
      NLS->ops = NULL;
    }
    /* free the nonlinear solver */
    free(NLS);
    return(SUN_NLS_SUCCESS);
  }
}

/* -----------------------------------------------------------------------------
 * set functions
 * ---------------------------------------------------------------------------*/

/* set the nonlinear system function (required) */
int SUNNonlinSolSetSysFn(SUNNonlinearSolver NLS, SUNNonlinSolSysFn SysFn)
{
  return((int) NLS->ops->setsysfn(NLS, SysFn));
}

/* set the linear solver setup function (optional) */
int SUNNonlinSolSetLSetupFn(SUNNonlinearSolver NLS, SUNNonlinSolLSetupFn LSetupFn)
{
  if (NLS->ops->setlsetupfn)
    return((int) NLS->ops->setlsetupfn(NLS, LSetupFn));
  else
    return(SUN_NLS_SUCCESS);
}

/* set the linear solver solve function (optional) */
int SUNNonlinSolSetLSolveFn(SUNNonlinearSolver NLS, SUNNonlinSolLSolveFn LSolveFn)
{
  if (NLS->ops->setlsolvefn)
    return((int) NLS->ops->setlsolvefn(NLS, LSolveFn));
  else
    return(SUN_NLS_SUCCESS);
}

/* set the convergence test function (optional) */
int SUNNonlinSolSetConvTestFn(SUNNonlinearSolver NLS, SUNNonlinSolConvTestFn CTestFn)
{
  if (NLS->ops->setctestfn)
    return((int) NLS->ops->setctestfn(NLS, CTestFn));
  else
    return(SUN_NLS_SUCCESS);
}

int SUNNonlinSolSetMaxIters(SUNNonlinearSolver NLS, int maxiters)
{
  if (NLS->ops->setmaxiters)
    return((int) NLS->ops->setmaxiters(NLS, maxiters));
  else
    return(SUN_NLS_SUCCESS);
}

/* -----------------------------------------------------------------------------
 * get functions
 * ---------------------------------------------------------------------------*/

/* get the total number on nonlinear iterations (optional) */
int SUNNonlinSolGetNumIters(SUNNonlinearSolver NLS, long int *niters)
{
  if (NLS->ops->getnumiters) {
    return((int) NLS->ops->getnumiters(NLS, niters));
  } else {
    *niters = 0;
    return(SUN_NLS_SUCCESS);
  }
}


/* get the iteration count for the current nonlinear solve */
int SUNNonlinSolGetCurIter(SUNNonlinearSolver NLS, int *iter)
{
  if (NLS->ops->getcuriter) {
    return((int) NLS->ops->getcuriter(NLS, iter));
  } else {
    *iter = -1;
    return(SUN_NLS_SUCCESS);
  }
}
