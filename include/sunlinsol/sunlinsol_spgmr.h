/*
 * -----------------------------------------------------------------
 * Programmer(s): Daniel Reynolds @ SMU
 * Based on code sundials_spgmr.h by: Scott D. Cohen, 
 *      Alan C. Hindmarsh and Radu Serban @ LLNL
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
 * This is the header file for the SPGMR implementation of the 
 * SUNLINSOL module.  The SPGMR algorithm is based on the
 * Scaled Preconditioned GMRES (Generalized Minimal Residual)
 * method.
 *
 * The SPGMR algorithm solves a linear system A x = b.
 * Preconditioning is allowed on the left, right, or both.
 * Scaling is allowed on both sides, and restarts are also allowed.
 * We denote the preconditioner and scaling matrices as follows:
 *   P1 = left preconditioner
 *   P2 = right preconditioner
 *   S1 = diagonal matrix of scale factors for P1-inverse b
 *   S2 = diagonal matrix of scale factors for P2 x
 * The matrices A, P1, and P2 are not required explicitly; only
 * routines that provide A, P1-inverse, and P2-inverse as
 * operators are required.
 *
 * In this notation, SPGMR applies the underlying GMRES method to
 * the equivalent transformed system
 *   Abar xbar = bbar , where
 *   Abar = S1 (P1-inverse) A (P2-inverse) (S2-inverse) ,
 *   bbar = S1 (P1-inverse) b , and   xbar = S2 P2 x .
 *
 * The scaling matrices must be chosen so that vectors S1
 * P1-inverse b and S2 P2 x have dimensionless components.
 * If preconditioning is done on the left only (P2 = I), by a
 * matrix P, then S2 must be a scaling for x, while S1 is a
 * scaling for P-inverse b, and so may also be taken as a scaling
 * for x.  Similarly, if preconditioning is done on the right only
 * (P1 = I, P2 = P), then S1 must be a scaling for b, while S2 is
 * a scaling for P x, and may also be taken as a scaling for b.
 *
 * The stopping test for the SPGMR iterations is on the L2 norm of
 * the scaled preconditioned residual:
 *      || bbar - Abar xbar ||_2  <  delta
 * with an input test constant delta.
 *
 * The usage of this SPGMR solver involves supplying three routines
 * and making a variety of calls.  The user-supplied routines are
 *    atimes (A_data, x, y) to compute y = A x, given x,
 *    psolve (P_data, y, x, lr) to solve P1 x = y or P2 x = y for 
 *           x, given y,
 *    psetup (P_data) to perform any 'setup' operations in 
 *           preparation for calling psolve.
 * The user calls are:
 *    SUNLinearSolver LS = SUNSPGMR(y, pretype, maxl);
 *           to create the linear solver structure,
 *    flag = SUNLinSolSetATimes(LS, A_data, atimes);
 *           to set the matrix-vector product setup/apply routines,
 *    flag = SUNLinSolSetPreconditioner(LS, P_data, psetup, psolve);
 *           to *optionally* set the preconditioner setup/apply routines,
 *    flag = SUNLinSolInitialize(LS);
 *           to perform internal solver memory allocations,
 *    flag = SUNLinSolSetup(LS, NULL);
 *           to call the psetup routine (if non-NULL);
 *    flag = SUNLinSolSolve(LS, NULL, x, b, w, tol);
 *           to solve the linear system to the tolerance 'tol'
 *    long int nli = SUNLinSolNumIters(LS);
 *           to *optionally* retrieve the number of linear iterations 
 *           performed by the solver,
 *    long int lastflag = SUNLinSolLastFlag(LS);
 *           to *optionally* retrieve the last internal solver error flag,
 *    flag = SUNLinSolFree(LS);
 *           to free the solver memory.
 * Complete details for specifying atimes, psetup and psolve 
 * and for the usage calls are given below.
 *
 * -----------------------------------------------------------------
 * 
 * Part I contains declarations specific to the SPGMR implementation
 * of the supplied SUNLINSOL module.
 * 
 * Part II contains the prototype for the constructor 
 * SUNSPGMR as well as implementation-specific prototypes 
 * for various useful solver operations.
 *
 * Notes:
 *
 *   - The definition of the generic SUNLinearSolver structure can 
 *     be found in the header file sundials_linearsolver.h.
 *
 * -----------------------------------------------------------------
 */

#ifndef _SUNLINSOL_SPGMR_H
#define _SUNLINSOL_SPGMR_H

#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_spgmr.h>

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

/* Default SPGMR solver parameters */
#define SUNSPGMR_MAXL_DEFAULT    5
#define SUNSPGMR_MAXRS_DEFAULT   0
#define SUNSPGMR_GSTYPE_DEFAULT  MODIFIED_GS

/*
 * -----------------------------------------------------------------
 * PART I: SPGMR implementation of SUNLinearSolver
 *
 * The SPGMR implementation of the SUNLinearSolver 'content' 
 * structure contains:
 *     maxl -- number of GMRES basis vectors to use
 *     pretype -- flag for type of preconditioning to employ
 *     gstype -- flag for type of Gram-Schmidt orthogonalization
 *     max_restarts -- number of GMRES restarts to allow
 *     numiters -- number of iterations from most-recent solve
 *     resnorm -- final linear residual norm from most-recent solve
 *     last_flag -- last error return flag from internal setup/solve
 *     ATimes -- function pointer to ATimes routine
 *     ATData -- pointer to structure for ATimes
 *     Psetup -- function pointer to preconditioner setup routine
 *     Psolve -- function pointer to preconditioner solve routine
 *     PData -- pointer to structure for Psetup/Psolve
 *     s1, s2 -- vector pointers for supplied scaling matrices
 *     V -- the array of Krylov basis vectors v_1, ..., v_(maxl+1),
 *         stored in V[0], ..., V[maxl]. Each v_i is a vector of 
 *         type N_Vector.
 *     Hes -- the (maxl+1) x maxl Hessenberg matrix. It is stored
 *         row-wise so that the (i,j)th element is given by Hes[i][j].
 *     givens -- a length 2*max array which represents the Givens 
 *         rotation matrices that arise in the algorithm. The Givens 
 *         rotation matrices F_0, F_1, ..., F_j, where F_i is
 *
 *             1
 *               1
 *                 c_i  -s_i      <--- row i
 *                 s_i   c_i
 *                           1
 *                             1
 *
 *         are represented in the givens vector as
 *         givens[0]=c_0, givens[1]=s_0, givens[2]=c_1, givens[3]=s_1,
 *         ..., givens[2j]=c_j, givens[2j+1]=s_j.
 *     xcor -- a vector (type N_Vector) which holds the scaled,
 *         preconditioned correction to the initial guess
 *     yg -- a length (maxl+1) array of realtype used to hold "short"
 *         vectors (e.g. y and g).
 *     vtemp -- a vector used as temporary vector storage
 * -----------------------------------------------------------------
 */
  
struct _SUNLinearSolverContent_SPGMR {
  int maxl;
  int pretype;
  int gstype;
  int max_restarts;
  int numiters;
  realtype resnorm;
  long int last_flag;

  ATimesFn ATimes;
  void* ATData;
  PSetupFn Psetup;
  PSolveFn Psolve;
  void* PData;

  N_Vector s1;
  N_Vector s2;
  N_Vector *V;
  realtype **Hes;
  realtype *givens;
  N_Vector xcor;
  realtype *yg;
  N_Vector vtemp;

  realtype *cv;
  N_Vector *Xv;
};

typedef struct _SUNLinearSolverContent_SPGMR *SUNLinearSolverContent_SPGMR;

/*
 * -----------------------------------------------------------------
 * PART II: functions exported by sunlinsol_spgmr
 * 
 * CONSTRUCTOR:
 *    SUNSPGMR creates and allocates memory for a SPGMR solver
 *
 * "SET" ROUTINES:
 *    SUNSPGMRSetPrecType updates the type of preconditioning to 
 *       use.  Supported values are PREC_NONE, PREC_LEFT, PREC_RIGHT 
 *       and PREC_BOTH.
 *    SUNSPGMRSetGSType sets the type of Gram-Schmidt 
 *       orthogonalization to use.  Supported values are MODIFIED_GS 
 *       and CLASSICAL_GS.
 *    SUNSPGMRSetMaxRestarts sets the number of GMRES restarts to 
 *       allow.  A negative input will result in the default of 0.
 * 
 * -----------------------------------------------------------------
 */

SUNDIALS_EXPORT SUNLinearSolver SUNSPGMR(N_Vector y, int pretype, int maxl);
SUNDIALS_EXPORT int SUNSPGMRSetPrecType(SUNLinearSolver S, int pretype);
SUNDIALS_EXPORT int SUNSPGMRSetGSType(SUNLinearSolver S, int gstype);
SUNDIALS_EXPORT int SUNSPGMRSetMaxRestarts(SUNLinearSolver S, int maxrs);

/*
 * -----------------------------------------------------------------
 * SPGMR implementations of various useful linear solver operations
 * -----------------------------------------------------------------
 */

SUNDIALS_EXPORT SUNLinearSolver_Type SUNLinSolGetType_SPGMR(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolInitialize_SPGMR(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolSetATimes_SPGMR(SUNLinearSolver S, void* A_data,
                                             ATimesFn ATimes);
SUNDIALS_EXPORT int SUNLinSolSetPreconditioner_SPGMR(SUNLinearSolver S,
                                                     void* P_data,
                                                     PSetupFn Pset,
                                                     PSolveFn Psol);
SUNDIALS_EXPORT int SUNLinSolSetScalingVectors_SPGMR(SUNLinearSolver S,
                                                     N_Vector s1,
                                                     N_Vector s2);
SUNDIALS_EXPORT int SUNLinSolSetup_SPGMR(SUNLinearSolver S, SUNMatrix A);
SUNDIALS_EXPORT int SUNLinSolSolve_SPGMR(SUNLinearSolver S, SUNMatrix A,
                                         N_Vector x, N_Vector b, realtype tol);
SUNDIALS_EXPORT int SUNLinSolNumIters_SPGMR(SUNLinearSolver S);
SUNDIALS_EXPORT realtype SUNLinSolResNorm_SPGMR(SUNLinearSolver S);
SUNDIALS_EXPORT N_Vector SUNLinSolResid_SPGMR(SUNLinearSolver S);
SUNDIALS_EXPORT long int SUNLinSolLastFlag_SPGMR(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolSpace_SPGMR(SUNLinearSolver S, 
                                         long int *lenrwLS, 
                                         long int *leniwLS);
SUNDIALS_EXPORT int SUNLinSolFree_SPGMR(SUNLinearSolver S);


#ifdef __cplusplus
}
#endif

#endif
