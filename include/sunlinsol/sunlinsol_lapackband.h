/*
 * -----------------------------------------------------------------
 * Programmer(s): Daniel Reynolds @ SMU
 * -----------------------------------------------------------------
 * LLNS/SMU Copyright Start
 * Copyright (c) 2017, Southern Methodist University and 
 * Lawrence Livermore National Security
 *
 * This work was performed under the auspices of the U.S. Department 
 * of Energy by Southern Methodist University and Lawrence Livermore 
 * National Laboratory under Contract DE-AC52-07NA27344.
 * Produced at Southern Methodist University and the Lawrence 
 * Livermore National Laboratory.
 *
 * All rights reserved.
 * For details, see the LICENSE file.
 * LLNS/SMU Copyright End
 * -----------------------------------------------------------------
 * This is the header file for the LAPACK band implementation of the 
 * SUNLINSOL module.
 * 
 * Part I contains declarations specific to the LAPACK band 
 * implementation of the supplied SUNLINSOL module.
 * 
 * Part II contains the prototype for the constructor 
 * SUNLapackBand as well as implementation-specific prototypes 
 * for various useful solver operations.
 *
 * Notes:
 *
 *   - The definition of the generic SUNLinearSolver structure can 
 *     be found in the header file sundials_linearsolver.h.
 *
 * -----------------------------------------------------------------
 */

#ifndef _SUNLINSOL_LAPBAND_H
#define _SUNLINSOL_LAPBAND_H

#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_lapack.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sunmatrix/sunmatrix_band.h>

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

/* Interfaces to match 'realtype' with the correct LAPACK functions */
#if defined(SUNDIALS_DOUBLE_PRECISION)
#define xgbtrf_f77    dgbtrf_f77
#define xgbtrs_f77    dgbtrs_f77
#elif defined(SUNDIALS_SINGLE_PRECISION)
#define xgbtrf_f77    sgbtrf_f77
#define xgbtrs_f77    sgbtrs_f77
#else
#error  Incompatible realtype for LAPACK; disable LAPACK and rebuild
#endif

/* Catch to disable LAPACK linear solvers with incompatible sunindextype */
#if defined(SUNDIALS_INT32_T)
#else  /* incompatible sunindextype for LAPACK */
#error  Incompatible sunindextype for LAPACK; disable LAPACK and rebuild
#endif

/*
 * -----------------------------------------------------------------
 * PART I: LAPACK band implementation of SUNLinearSolver
 *
 * The LAPACK band implementation of the SUNLinearSolver 'content' 
 * structure contains:
 *     N -- size of the linear system
 *     pivots -- index array for partial pivoting in LU factorization
 *     last_flag -- last error return flag from internal setup/solve
 * -----------------------------------------------------------------
 */
  
struct _SUNLinearSolverContent_LapackBand {
  sunindextype N;
  sunindextype *pivots;
  long int last_flag;
};

typedef struct _SUNLinearSolverContent_LapackBand *SUNLinearSolverContent_LapackBand;

  
/*
 * -----------------------------------------------------------------
 * PART II: functions exported by sunlinsol_lapackband
 * 
 * CONSTRUCTOR:
 *    SUNLapackBand creates and allocates memory for a LAPACK banded 
 *      matrix solver
 *
 * OTHER:
 * -----------------------------------------------------------------
 */

SUNDIALS_EXPORT SUNLinearSolver SUNLapackBand(N_Vector y, SUNMatrix A);

/*
 * -----------------------------------------------------------------
 * LAPACK band implementations of required linear solver operations
 * -----------------------------------------------------------------
 */

SUNDIALS_EXPORT SUNLinearSolver_Type SUNLinSolGetType_LapackBand(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolInitialize_LapackBand(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolSetup_LapackBand(SUNLinearSolver S, SUNMatrix A);
SUNDIALS_EXPORT int SUNLinSolSolve_LapackBand(SUNLinearSolver S, SUNMatrix A,
                                              N_Vector x, N_Vector b, realtype tol);
SUNDIALS_EXPORT long int SUNLinSolLastFlag_LapackBand(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolSpace_LapackBand(SUNLinearSolver S,
                                              long int *lenrwLS,
                                              long int *leniwLS);
SUNDIALS_EXPORT int SUNLinSolFree_LapackBand(SUNLinearSolver S);
  
#ifdef __cplusplus
}
#endif

#endif
