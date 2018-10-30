/*=================================================================
  IMPORTED HEADER FILES
  =================================================================*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "../cvode/cvode_impl.h"
#include "cvode_cusolver_spqr_impl.h"
#include <sundials/sundials_math.h>
#include <nvector/nvector_cuda.h>

#include <cusolverSp.h>
#include <cusparse.h>
#include <cuda_runtime_api.h>

/*=================================================================
  FUNCTION SPECIFIC CONSTANTS
  =================================================================*/

#define ZERO         RCONST(0.0)
#define ONE          RCONST(1.0)
#define TWO          RCONST(2.0)

#define PRINT_CUSOLVER_DEBUGGING 0

/*=================================================================
  EXPORTED FUNCTIONS -- REQUIRED
  =================================================================*/

/*------------------------------------------------------------------
 cv_cuSolver_SetLinearSolver sets which cuSolver solver to use.
-------------------------------------------------------------------*/

int cv_cuSolver_SetLinearSolver(void *cvode_mem, cuSolver_method cus_method,
				bool store_jacobian = true,
				int num_steps_save_jacobian = 0)
{
  CVodeMem cv_mem;
  CV_cuSolver_Mem cv_cus_mem;

  /* Return immediately if any input is NULL */
  /* if (cvode_mem == NULL) { */
  /*   cvProcessError(NULL, CVDLS_MEM_NULL, "CVDLS",  */
  /*                  "CVDlsSetLinearSolver", MSGD_CVMEM_NULL); */
  /*   return(CVDLS_MEM_NULL); */
  /* } */
  /* if ( (LS == NULL)  || (A == NULL) ) { */
  /*   cvProcessError(NULL, CVDLS_ILL_INPUT, "CVDLS",  */
  /*                  "CVDlsSetLinearSolver", */
  /*                   "Both LS and A must be non-NULL"); */
  /*   return(CVDLS_ILL_INPUT); */
  /* } */
  cv_mem = (CVodeMem) cvode_mem;

  /* Test if solver and vector are compatible with DLS */
  /* if (SUNLinSolGetType(LS) != SUNLINEARSOLVER_DIRECT) { */
  /*   cvProcessError(cv_mem, CVDLS_ILL_INPUT, "CVDLS",  */
  /*                  "CVDlsSetLinearSolver",  */
  /*                  "Non-direct LS supplied to CVDls interface"); */
  /*   return(CVDLS_ILL_INPUT); */
  /* } */
  /* if (cv_mem->cv_tempv->ops->nvgetarraypointer == NULL || */
  /*     cv_mem->cv_tempv->ops->nvsetarraypointer == NULL) { */
  /*   cvProcessError(cv_mem, CVDLS_ILL_INPUT, "CVDLS",  */
  /*                  "CVDlsSetLinearSolver", MSGD_BAD_NVECTOR); */
  /*   return(CVDLS_ILL_INPUT); */
  /* } */

  /* free any existing system solver attached to CVode */
  if (cv_mem->cv_lfree)  cv_mem->cv_lfree(cv_mem);

  /* Set four main system linear solver function fields in cv_mem */
  cv_mem->cv_linit  = cv_cuSolver_Initialize;
  cv_mem->cv_lsetup = cv_cuSolver_Setup;
  cv_mem->cv_lsolve = cv_cuSolver_Solve;
  cv_mem->cv_lfree  = cv_cuSolver_Free;
  
  /* Get memory for CV_cuSolver_MemRec */
  cv_cus_mem = NULL;
  cv_cus_mem = (CV_cuSolver_Mem) malloc(sizeof(struct CV_cuSolver_MemRec));
  /* if (cvdls_mem == NULL) { */
  /*   cvProcessError(cv_mem, CVDLS_MEM_FAIL, "CVDLS",  */
  /*                   "CVDlsSetLinearSolver", MSGD_MEM_FAIL); */
  /*   return(CVDLS_MEM_FAIL); */
  /* } */

  /* store the type of cuSolver to use */
  cv_cus_mem->method = cus_method;
  
  /* Initialize Jacobian-related data */
  cv_cus_mem->jac = NULL; // put the function here
  cv_cus_mem->J_data = cv_mem;

  /* Allocate memory for x and other stuff */
  cv_cus_mem->x = N_VClone(cv_mem->cv_tempv);
  if (cv_cus_mem->x == NULL) {
    cvProcessError(cv_mem, CV_CUSOLVER_MEM_FAIL, "CV_CUSOLVER", 
                    "CV_cuSolver_SetLinearSolver", MSGD_MEM_FAIL);
    free(cv_cus_mem); cv_cus_mem = NULL;
    return(CV_CUSOLVER_MEM_FAIL);
  }

  /* Allocate cuSolver system and workspace structs */
  cv_cus_mem->cus_work = (CV_cuSolver_workspace_QR) malloc(sizeof(struct CV_cuSolver_workspace_QR_t));
  cv_cus_mem->csr_sys  = (CV_cuSolver_csr_sys) malloc(sizeof(struct CV_cuSolver_csr_sys_t));

  cv_cus_mem->store_jacobian = static_cast<booleantype>(store_jacobian || (num_steps_save_jacobian >= 0));

  cv_cus_mem->CV_CUSOLVER_MSBJ = num_steps_save_jacobian;

  if (cv_cus_mem->store_jacobian) {
    cv_cus_mem->saved_jacobian = (CV_cuSolver_csr_sys) malloc(sizeof(struct CV_cuSolver_csr_sys_t));
  } else {
    cv_cus_mem->saved_jacobian = NULL;
  }
  
  /* Set pointers in cv_cuSolver memory structure to NULL */
  cv_cus_mem->cus_work->workspace = NULL;

  cv_cus_mem->csr_sys->d_csr_row_count = NULL;
  cv_cus_mem->csr_sys->d_csr_col_index = NULL;
  cv_cus_mem->csr_sys->d_csr_values = NULL;

  if (cv_cus_mem->store_jacobian) {
    cv_cus_mem->saved_jacobian->d_csr_row_count = NULL;
    cv_cus_mem->saved_jacobian->d_csr_col_index = NULL;
    cv_cus_mem->saved_jacobian->d_csr_values = NULL;
  }

  /* Attach linear solver memory to integrator memory */
  cv_mem->cv_lmem = cv_cus_mem;

#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Created CV_cuSolver_Mem object." << std::endl;
#endif

  return(CV_CUSOLVER_SUCCESS);
}


/* 
 * =================================================================
 * EXPORTED FUNCTIONS -- OPTIONAL
 * =================================================================
 */
              
/* CV_cuSolver_SetJacFun specifies the Jacobian function. */
int cv_cuSolver_SetJacFun(void *cvode_mem, CV_cuSolver_JacFn jac)
{
  CVodeMem cv_mem;
  CV_cuSolver_Mem cv_cus_mem;

  /* Return immediately if cvode_mem or cv_mem->cv_lmem are NULL */
  if (cvode_mem == NULL) {
    cvProcessError(NULL, CV_CUSOLVER_MEM_NULL, "CV_CUSOLVER",
                   "CV_cuSolver_SetJacFn", MSGD_CVMEM_NULL);
    return(CV_CUSOLVER_MEM_NULL);
  }
  cv_mem = (CVodeMem) cvode_mem;
  if (cv_mem->cv_lmem == NULL) {
    cvProcessError(cv_mem, CV_CUSOLVER_LMEM_NULL, "CV_CUSOLVER",
                   "CV_cuSolver_SetJacFn", MSGD_LMEM_NULL);
    return(CV_CUSOLVER_LMEM_NULL);
  }
  cv_cus_mem = (CV_cuSolver_Mem) cv_mem->cv_lmem;

  cv_cus_mem->jac = jac;
  
  return(CV_CUSOLVER_SUCCESS);
}


/* CV_cuSolver_GetWorkSpace returns the length of workspace allocated for the
   CV_CUSOLVER linear solver. */
int cv_cuSolver_GetWorkSpace(void *cvode_mem, long int *lenrwLS,
			     long int *leniwLS)
{
  CVodeMem cv_mem;
  CV_cuSolver_Mem cv_cus_mem;
  sunindextype lrw1, liw1;
  long int lrw, liw;
  int flag;

  /* Return immediately if cvode_mem or cv_mem->cv_lmem are NULL */
  if (cvode_mem == NULL) {
    cvProcessError(NULL, CV_CUSOLVER_MEM_NULL, "CV_CUSOLVER",
                   "CV_cuSolver_GetWorkSpace", MSGD_CVMEM_NULL);
    return(CV_CUSOLVER_MEM_NULL);
  }
  cv_mem = (CVodeMem) cvode_mem;
  if (cv_mem->cv_lmem == NULL) {
    cvProcessError(cv_mem, CV_CUSOLVER_LMEM_NULL, "CV_CUSOLVER",
                   "CV_cuSolver_GetWorkSpace", MSGD_LMEM_NULL);
    return(CV_CUSOLVER_LMEM_NULL);
  }
  cv_cus_mem = (CV_cuSolver_Mem) cv_mem->cv_lmem;

  /* initialize outputs with requirements from CV_cuSolver_Mem structure */
  *lenrwLS = 0;
  *leniwLS = 0;

  /* add NVector size */
  if (cv_cus_mem->x->ops->nvspace) {
    N_VSpace(cv_cus_mem->x, &lrw1, &liw1);
    *lenrwLS = lrw1;
    *leniwLS = liw1;
  }

  // CSR System storage
  if (cv_cus_mem->csr_sys != NULL) {
    *lenrwLS += cv_cus_mem->csr_sys->csr_number_nonzero*cv_cus_mem->csr_sys->number_subsystems;
    *leniwLS += cv_cus_mem->csr_sys->csr_number_nonzero;
    *leniwLS += cv_cus_mem->csr_sys->size_per_subsystem + 1;
  }

  // Saved J storage
  if (cv_cus_mem->saved_jacobian != NULL) {
    *lenrwLS += cv_cus_mem->saved_jacobian->csr_number_nonzero*cv_cus_mem->saved_jacobian->number_subsystems;
    *leniwLS += cv_cus_mem->saved_jacobian->csr_number_nonzero;
    *leniwLS += cv_cus_mem->saved_jacobian->size_per_subsystem + 1;
  }

  // cuSolver workspace
  if (cv_cus_mem->cus_work != NULL) {
    *lenrwLS += cv_cus_mem->cus_work->workspace_size;
  }

  return(CV_CUSOLVER_SUCCESS);
}


// Revise this
/* CV_cuSolver_GetReturnFlagName returns the name associated with a CV_CUSOLVER
   return value. */
char *cv_cuSolver_GetReturnFlagName(long int flag)
{
  char *name;

  name = (char *)malloc(30*sizeof(char));

  switch(flag) {
  case CV_CUSOLVER_SUCCESS:
    sprintf(name,"CV_CUSOLVER_SUCCESS");
    break;   
  case CV_CUSOLVER_MEM_NULL:
    sprintf(name,"CV_CUSOLVER_MEM_NULL");
    break;
  case CV_CUSOLVER_LMEM_NULL:
    sprintf(name,"CV_CUSOLVER_LMEM_NULL");
    break;
  case CV_CUSOLVER_ILL_INPUT:
    sprintf(name,"CV_CUSOLVER_ILL_INPUT");
    break;
  case CV_CUSOLVER_MEM_FAIL:
    sprintf(name,"CV_CUSOLVER_MEM_FAIL");
    break;
  case CV_CUSOLVER_JACFUNC_UNRECVR:
    sprintf(name,"CV_CUSOLVER_JACFUNC_UNRECVR");
    break;
  case CV_CUSOLVER_JACFUNC_RECVR:
    sprintf(name,"CV_CUSOLVER_JACFUNC_RECVR");
    break;
  case CV_CUSOLVER_SCALEADDI_FAIL:
    sprintf(name,"CV_CUSOLVER_SCALEADDI_FAIL");
    break;
  default:
    sprintf(name,"NONE");
  }

  return(name);
}



/*=================================================================
  CV_CUSOLVER PRIVATE FUNCTIONS
  =================================================================*/

/*-----------------------------------------------------------------
  cv_cuSolver_Initialize
  -----------------------------------------------------------------
  This routine performs remaining initializations specific
  to the cuSolver linear solver interface (and solver itself)
  -----------------------------------------------------------------*/
int cv_cuSolver_Initialize(CVodeMem cvode_mem)
{
  
#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Doing cv_cuSolver_Initialize" << std::endl;
#endif

  CV_cuSolver_Mem cv_cus_mem;

  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;

  /* Return immediately if cvode_mem or cvode_mem->cv_lmem are NULL */
  if (cvode_mem == NULL) {
    cvProcessError(NULL, CV_CUSOLVER_MEM_NULL, "CV_CUSOLVER", 
                    "cv_cuSolver_Initialize", MSGD_CVMEM_NULL);
    return(CV_CUSOLVER_MEM_NULL);
  }
  if (cvode_mem->cv_lmem == NULL) {
    cvProcessError(cvode_mem, CV_CUSOLVER_LMEM_NULL, "CV_CUSOLVER", 
                    "cv_cuSolver_Initialize", MSGD_LMEM_NULL);
    return(CV_CUSOLVER_LMEM_NULL);
  }
  cv_cus_mem = (CV_cuSolver_Mem) cvode_mem->cv_lmem;

  /* Set Jacobian evaluation user data */
  cv_cus_mem->J_data = cvode_mem->cv_user_data;
  cv_cus_mem->nstlj  = 0;
  cv_cuSolver_InitializeCounters(cv_cus_mem);


  // Make handle for cuSolver if it doesn't already exist
#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Creating cuSolver Handle" << std::endl;  
#endif
  cusolver_status = cusolverSpCreate(&cv_cus_mem->cus_work->cusolverHandle);
  cv_cuSolver_check_cusolver_status(cusolver_status);  
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  // Setup Sparse system description
#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Creating Matrix Descriptor" << std::endl;  
#endif
  cusparse_status = cusparseCreateMatDescr(&cv_cus_mem->cus_work->system_description);
  cv_cuSolver_check_cusparse_status(cusparse_status);
  assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "In Matrix Descriptor, setting Matrix Type" << std::endl;
#endif
  cusparse_status = cusparseSetMatType(cv_cus_mem->cus_work->system_description, CUSPARSE_MATRIX_TYPE_GENERAL);
  cv_cuSolver_check_cusparse_status(cusparse_status);
  assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

#if PRINT_CUSOLVER_DEBUGGING  
  std::cout << "In Matrix Descriptor, setting Matrix Index Base" << std::endl;
#endif
  cusparse_status = cusparseSetMatIndexBase(cv_cus_mem->cus_work->system_description, CUSPARSE_INDEX_BASE_ONE);
  cv_cuSolver_check_cusparse_status(cusparse_status);
  assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

  // Create an info object
#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Creating info object" << std::endl;  
#endif
  cusolver_status = cusolverSpCreateCsrqrInfo(&cv_cus_mem->cus_work->info);
  cv_cuSolver_check_cusolver_status(cusolver_status);  
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Finished cv_cuSolver_Initialize" << std::endl;
#endif
  return(0);
}



/*-----------------------------------------------------------------
  cv_cuSolver_Setup
  -----------------------------------------------------------------
  This must be called AFTER THE USER CALLS cv_cuSolver_CSR_SetSizes

  This routine first checks to see if we can use a stored
  Jacobian using the same criteria as for the CVODE direct solver.

  This routine calculates a new Jacobian matrix if needed and makes
  the system matrix A from J, the 'gamma' factor and the identity:
    A = I-gamma*J.
  This routine then initializes the cuSolver linear solver memory.
  -----------------------------------------------------------------*/
int cv_cuSolver_Setup(CVodeMem cvode_mem, int convfail, N_Vector ypred,
		      N_Vector fpred, booleantype *jcurPtr, N_Vector vtemp1,
		      N_Vector vtemp2, N_Vector vtemp3)
{

#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Doing cv_cuSolver_Setup" << std::endl;  
#endif

  booleantype jbad, jgood;
  realtype dgamma;
  CV_cuSolver_Mem cv_cus_mem;
  int retval;
  cudaError_t cuda_status = cudaSuccess;

  /* Return immediately if cvode_mem or cvode_mem->cv_lmem are NULL */
  if (cvode_mem == NULL) {
    cvProcessError(NULL, CV_CUSOLVER_MEM_NULL, "CV_CUSOLVER", 
                    "cv_cuSolver_Setup", MSGD_CVMEM_NULL);
    return(CV_CUSOLVER_MEM_NULL);
  }
  if (cvode_mem->cv_lmem == NULL) {
    cvProcessError(cvode_mem, CV_CUSOLVER_LMEM_NULL, "CV_CUSOLVER", 
                    "cv_cuSolver_Setup", MSGD_LMEM_NULL);
    return(CV_CUSOLVER_LMEM_NULL);
  }
  cv_cus_mem = (CV_cuSolver_Mem) cvode_mem->cv_lmem;

  // This is where we could decide whether to use a stored J instead
  /* Use nst, gamma/gammap, and convfail to set J eval. flag jgood */
  dgamma = SUNRabs((cvode_mem->cv_gamma/cvode_mem->cv_gammap) - ONE);
  jbad = (cvode_mem->cv_nst == 0) ||
    (cvode_mem->cv_nst > cv_cus_mem->nstlj + cv_cus_mem->CV_CUSOLVER_MSBJ) ||
    ((convfail == CV_FAIL_BAD_J) && (dgamma < CV_CUSOLVER_DGMAX)) ||
    (convfail == CV_FAIL_OTHER);
  jgood = (!jbad) && cv_cus_mem->store_jacobian;

  /* If jgood = SUNTRUE, use saved copy of J */
  if (jgood) {
    *jcurPtr = SUNFALSE;
    
    // Copy saved J into the linear system to solve
    cuda_status = cudaMemcpy(cv_cus_mem->csr_sys->d_csr_values,
			     cv_cus_mem->saved_jacobian->d_csr_values,
			     sizeof(realtype) * cv_cus_mem->csr_sys->csr_number_nonzero * cv_cus_mem->csr_sys->number_subsystems,
			     cudaMemcpyDeviceToDevice);

    if (cuda_status != cudaSuccess) {
      cvProcessError(cvode_mem, CV_CUSOLVER_JACCOPY_UNRECVR, "CV_CUSOLVER",
		     "cv_cuSolver_Setup",  MSGD_JACCOPY_FAILED);
      cv_cus_mem->last_flag = CV_CUSOLVER_JACCOPY_UNRECVR;
      return(-1);
    }
    /* If jgood = SUNFALSE, call jac routine for new J value */
  } else {
    // Calculate Jacobian matrix  
    cv_cus_mem->nje++;
    cv_cus_mem->nstlj = cvode_mem->cv_nst;    
    *jcurPtr = SUNTRUE;

    retval = cv_cus_mem->jac(cvode_mem->cv_tn, ypred, 
			     fpred, cv_cus_mem->csr_sys, 
			     cv_cus_mem->J_data);
    if (retval < 0) {
      cvProcessError(cvode_mem, CV_CUSOLVER_JACFUNC_UNRECVR, "CV_CUSOLVER", 
		     "cv_cuSolver_Setup",  MSGD_JACFUNC_FAILED);
      cv_cus_mem->last_flag = CV_CUSOLVER_JACFUNC_UNRECVR;
#if PRINT_CUSOLVER_DEBUGGING
      std::cout << "Jacobian evaluation error. Returning." << std::endl;
      std::cout << "Finished cv_cuSolver_Setup" << std::endl;
#endif
      return(-1);
    }
    if (retval > 0) {
      cv_cus_mem->last_flag = CV_CUSOLVER_JACFUNC_RECVR;
#if PRINT_CUSOLVER_DEBUGGING
      std::cout << "Jacobian evaluation recoverable error. Returning." << std::endl;
      std::cout << "Finished cv_cuSolver_Setup" << std::endl;
#endif
      return(1);
    }

    if (cv_cus_mem->store_jacobian) {
      // Copy J to keep it around after we're done with this solve
      cuda_status = cudaMemcpy(cv_cus_mem->saved_jacobian->d_csr_values,
			       cv_cus_mem->csr_sys->d_csr_values,
			       sizeof(realtype) * cv_cus_mem->csr_sys->csr_number_nonzero * cv_cus_mem->csr_sys->number_subsystems,
			       cudaMemcpyDeviceToDevice);
      assert(cuda_status == cudaSuccess);
    }

  }
  
  /* Scale and add I to get A = I - gamma*J */
  retval = cv_cuSolver_ScaleAddI(-cvode_mem->cv_gamma, cv_cus_mem);
  if (retval) {
    cvProcessError(cvode_mem, CV_CUSOLVER_SCALEADDI_FAIL, "CV_CUSOLVER", 
                   "cv_cuSolver_Setup",  MSGD_MATSCALEADDI_FAILED);
    cv_cus_mem->last_flag = CV_CUSOLVER_SCALEADDI_FAIL;
#if PRINT_CUSOLVER_DEBUGGING
    std::cout << "Scale/Add I Failure. Returning." << std::endl;
    std::cout << "Finished cv_cuSolver_Setup" << std::endl;
#endif
    return(-1);
  }

  /* Call cuSolver linear solver 'setup' with this system matrix, and
     return success/failure flag */
  cv_cus_mem->last_flag = cv_cuSolver_SolverInitialize(cv_cus_mem);
#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Finished cv_cuSolver_Setup" << std::endl;
#endif
  return(cv_cus_mem->last_flag);

}


/*-----------------------------------------------------------------
  cv_cuSolver_Solve
  -----------------------------------------------------------------
  This routine interfaces between CVode and the generic 
  cuSolver solver, by calling the solver and scaling 
  the solution appropriately when gamrat != 1.
  -----------------------------------------------------------------*/
int cv_cuSolver_Solve(CVodeMem cv_mem, N_Vector b, N_Vector weight,
               N_Vector ycur, N_Vector fcur)
{
  
#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Doing cv_cuSolver_Solve" << std::endl;
#endif

  int retval;
  CV_cuSolver_Mem cv_cus_mem;

  /* Return immediately if cv_mem or cv_mem->cv_lmem are NULL */
  if (cv_mem == NULL) {
    cvProcessError(NULL, CV_CUSOLVER_MEM_NULL, "CV_CUSOLVER", 
		    "cv_cuSolver_Solve", MSGD_CVMEM_NULL);
    return(CV_CUSOLVER_MEM_NULL);
  }
  if (cv_mem->cv_lmem == NULL) {
    cvProcessError(cv_mem, CV_CUSOLVER_LMEM_NULL, "CV_CUSOLVER", 
		    "cv_cuSolver_Solve", MSGD_LMEM_NULL);
    return(CV_CUSOLVER_LMEM_NULL);
  }
  cv_cus_mem = (CV_cuSolver_Mem) cv_mem->cv_lmem;

  /* call the cuSolver linear system solver, and copy x to b */
  retval = cv_cuSolver_SolveSystem(cv_cus_mem, b);
  N_VScale(ONE, cv_cus_mem->x, b);
  
  /* scale the correction to account for change in gamma */
  if ((cv_mem->cv_lmm == CV_BDF) && (cv_mem->cv_gamrat != ONE))
    N_VScale(TWO/(ONE + cv_mem->cv_gamrat), b, b);
  
  /* store solver return value and return */
  cv_cus_mem->last_flag = retval;
  
#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Finished cv_cuSolver_Solve" << std::endl;
#endif
  
  return(retval);
}


/*-----------------------------------------------------------------
  cv_cuSolver_Free
  -----------------------------------------------------------------
  This routine frees memory associates with the CV_cuSolver solver 
  interface.
  -----------------------------------------------------------------*/
int cv_cuSolver_Free(CVodeMem cv_mem)
{
  
#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Doing cv_cuSolver_Free" << std::endl;
#endif

  CV_cuSolver_Mem cv_cus_mem;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;  

  /* Return immediately if cv_mem or cv_mem->cv_lmem are NULL */
  if (cv_mem == NULL)  return (CV_CUSOLVER_SUCCESS);
  if (cv_mem->cv_lmem == NULL)  return(CV_CUSOLVER_SUCCESS);
  cv_cus_mem = (CV_cuSolver_Mem) cv_mem->cv_lmem;

  /* Free x vector */
  if (cv_cus_mem->x) {
    N_VDestroy(cv_cus_mem->x);
    cv_cus_mem->x = NULL;
  }

  /* Destroy cuSolver memory */
  cusolver_status = cusolverSpDestroy(cv_cus_mem->cus_work->cusolverHandle);
  cv_cuSolver_check_cusolver_status(cusolver_status);  
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  cusolver_status = cusolverSpDestroyCsrqrInfo(cv_cus_mem->cus_work->info);
  cv_cuSolver_check_cusolver_status(cusolver_status);  
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  /* Cleanup system workspace */
  cv_cuSolver_WorkspaceFree(cv_cus_mem);

  /* free CV_cuSolver interface structure */
  free(cv_mem->cv_lmem);
  
#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Finished cv_cuSolver_Free" << std::endl;
#endif
  
  return(CV_CUSOLVER_SUCCESS);
}


/*-----------------------------------------------------------------
  cv_cuSolver_InitializeCounters
  -----------------------------------------------------------------
  This routine resets the counters inside the CV_cuSolver_Mem object.
  -----------------------------------------------------------------*/
int cv_cuSolver_InitializeCounters(CV_cuSolver_Mem cv_cus_mem)
{

  // Number of Jacobian evaluations
  cv_cus_mem->nje = 0;

  return(0);
}


/*-----------------------------------------------------------------
  cv_cuSolver_GetNumJacEvals
  -----------------------------------------------------------------
  Get the number of Jacobian evaluations we have done.
  -----------------------------------------------------------------*/
int cv_cuSolver_GetNumJacEvals(void* cv_mem, int* nje)
{
  CVodeMem cvode_mem = (CVodeMem) cv_mem;
  CV_cuSolver_Mem cv_cus_mem = (CV_cuSolver_Mem) cvode_mem->cv_lmem;
  *nje = cv_cus_mem->nje;

  return(0);
}


/*-----------------------------------------------------------------
  cv_cuSolver_ScaleAddI
  -----------------------------------------------------------------
  This routine scales the CSR-formatted system and adds the identity
  -----------------------------------------------------------------*/
int cv_cuSolver_ScaleAddI(realtype scale, CV_cuSolver_Mem cv_cus_mem)
{
  
#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Doing ScaleAddI" << std::endl;
#endif
  
  cudaError_t cuda_status = cudaSuccess;
  cuda_status = cudaGetLastError();
  assert(cuda_status == cudaSuccess);

#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Got CUDA Last Error of: ";
  std::cout << cudaGetErrorString(cuda_status) << std::endl;
#endif
  
  CV_cuSolver_csr_sys csr_sys = cv_cus_mem->csr_sys;
  const int system_size = csr_sys->number_subsystems;
  const int numThreads = (system_size < 32) ? system_size : 32;
  const int numBlocks = (int)(ceil(((double) system_size)/((double) numThreads)));
  const int csr_base = (CUSPARSE_INDEX_BASE_ONE == cusparseGetMatIndexBase(cv_cus_mem->cus_work->system_description)) ? 1:0;

#if PRINT_CUSOLVER_DEBUGGING
  // print out the first two systems before ScaleAddI
  realtype sysmat[csr_sys->csr_number_nonzero];
  cuda_status = cudaMemcpy(&sysmat[0], csr_sys->d_csr_values,
			   sizeof(realtype) * csr_sys->csr_number_nonzero,
			   cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);
  std::cout << "first system is: " << std::endl;
  for (int i = 0; i < csr_sys->csr_number_nonzero; i++)
    std::cout << sysmat[i] << " ";
  std::cout << std::endl;

  cuda_status = cudaMemcpy(&sysmat[0], csr_sys->d_csr_values + csr_sys->csr_number_nonzero,
			   sizeof(realtype) * csr_sys->csr_number_nonzero,
			   cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);
  std::cout << "second system is: " << std::endl;
  for (int i = 0; i < csr_sys->csr_number_nonzero; i++)
    std::cout << sysmat[i] << " ";
  std::cout << std::endl;

  std::cout << "scale factor is: " << scale << std::endl;
#endif
  
  cv_cuSolver_ScaleAddI_kernel<<<numBlocks, numThreads>>>(scale,
							  csr_sys->d_csr_values,
							  csr_sys->d_csr_col_index,
							  csr_sys->d_csr_row_count,
							  csr_base,
							  csr_sys->csr_number_nonzero,
							  csr_sys->size_per_subsystem,
							  csr_sys->number_subsystems);
  cuda_status = cudaDeviceSynchronize();
#if PRINT_CUSOLVER_DEBUGGING  
  std::cout << cudaGetErrorString(cuda_status) << std::endl;
#endif
  assert(cuda_status == cudaSuccess);  

#if PRINT_CUSOLVER_DEBUGGING
  // print out the first two systems after ScaleAddI
  cuda_status = cudaMemcpy(&sysmat[0], csr_sys->d_csr_values,
			   sizeof(realtype) * csr_sys->csr_number_nonzero,
			   cudaMemcpyDeviceToHost);
  std::cout << cudaGetErrorString(cuda_status) << std::endl;
  assert(cuda_status == cudaSuccess);
  std::cout << "first system is: " << std::endl;
  for (int i = 0; i < csr_sys->csr_number_nonzero; i++)
    std::cout << sysmat[i] << " ";
  std::cout << std::endl;

  cuda_status = cudaMemcpy(&sysmat[0], csr_sys->d_csr_values + csr_sys->csr_number_nonzero,
			   sizeof(realtype) * csr_sys->csr_number_nonzero,
			   cudaMemcpyDeviceToHost);
  std::cout << cudaGetErrorString(cuda_status) << std::endl;
  assert(cuda_status == cudaSuccess);
  std::cout << "second system is: " << std::endl;
  for (int i = 0; i < csr_sys->csr_number_nonzero; i++)
    std::cout << sysmat[i] << " ";
  std::cout << std::endl;
#endif

  return(0);
}


__global__ void cv_cuSolver_ScaleAddI_kernel(const realtype scale, realtype* csr_values,
					     int* csr_col_index, int* csr_row_count,
					     const int csr_base, const int nnz,
					     const int size, const int nbatched)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int sys_offset = tid * nnz;

  if (tid < nbatched) {
    realtype* Asystem = csr_values + sys_offset;

    for (int i = 0; i < nnz; i++)
      Asystem[i] = Asystem[i] * scale;

    int loc = 0;
    int ninrow = 0;
    int col = 0;
    for (int row = 1; row <= size; row++) {
      ninrow = csr_row_count[row] - csr_row_count[row-1];
      for (int j = 0; j < ninrow; j++) {
	col = csr_col_index[loc];
	if (row == col)
	  Asystem[loc] += ONE;
	loc++;
      }
    }
  }
}


// __global__ void cv_cuSolver_ScaleAddI_kernel(const realtype scale, realtype* csr_values,
// 					     int* csr_col_index, int* csr_row_count,
// 					     const int csr_base, const int nnz,
// 					     const int size, const int nbatched)
// {
//   const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   const int sys_id = tid / nnz;
//   const int sys_offset = tid - nnz * sys_id;

//   if (tid < nnz * nbatched) {
//     realtype* Asystem = csr_values + sys_id * nnz;

//     Asystem[sys_offset] = Asystem[sys_offset] * scale;

//     // If we are on a diagonal then add 1.0 for the identity
//     const int column = csr_col_index[sys_offset];
//     int row = -1;
//     for (int i = 1; i <= size; i++) {
//       if (sys_offset <= csr_row_count[i] - csr_base) {
// 	row = i;
// 	break;
//       }
//     }
//     if (row == column)
//       Asystem[sys_offset] = Asystem[sys_offset] + ONE;
//   }
// }


/*-----------------------------------------------------------------
  cv_cuSolver_CSR_SetSizes
  -----------------------------------------------------------------
  This routine sets the sizes of the matrix subsystems in CSR format

  User MUST call this before calling cv_cuSolver_SystemInitialize
  -----------------------------------------------------------------*/
int cv_cuSolver_CSR_SetSizes(void* cv_mem, int size_per_subsystem,
			     int csr_number_nonzero, int number_subsystems)
{
  
#if PRINT_CUSOLVER_DEBUGGING  
  std::cout << "Doing CSR_SetSizes" << std::endl;
#endif
  
  CVodeMem cvode_mem = (CVodeMem) cv_mem;
  CV_cuSolver_Mem cv_cus_mem = (CV_cuSolver_Mem) cvode_mem->cv_lmem;

  int retval1 = CV_CUSOLVER_SUCCESS;
  int retval2 = CV_CUSOLVER_SUCCESS;

  retval1 = cv_cuSolver_CSR_SetSizes_Matrix(cv_cus_mem->csr_sys, size_per_subsystem,
					    csr_number_nonzero, number_subsystems);

  if (cv_cus_mem->store_jacobian) {
    retval2 = cv_cuSolver_CSR_SetSizes_Matrix(cv_cus_mem->saved_jacobian, size_per_subsystem,
					      csr_number_nonzero, number_subsystems);
  }

  if (retval1 != CV_CUSOLVER_SUCCESS || retval2 != CV_CUSOLVER_SUCCESS) {
    cvProcessError(cvode_mem, CV_CUSOLVER_MEM_FAIL, "CV_CUSOLVER", 
                   "CV_cuSolver_CSR_SetSizes", MSGD_MEM_FAIL);
    cv_cuSolver_WorkspaceFree(cv_cus_mem);
    free(cv_cus_mem); cv_cus_mem = NULL;
    return(CV_CUSOLVER_MEM_FAIL);
  }

#if PRINT_CUSOLVER_DEBUGGING  
  std::cout << "Finished CSR_SetSizes" << std::endl;
#endif
  
  return(0);
  
}


int cv_cuSolver_CSR_SetSizes_Matrix(CV_cuSolver_csr_sys csr_sys, int size_per_subsystem,
				    int csr_number_nonzero, int number_subsystems)
{
  // Return with an error if memory is already allocated.
  // That is, you cannot change these values after setting them
  // to avoid memory leaks.  
  if (csr_sys->d_csr_values == NULL &&
      csr_sys->d_csr_col_index == NULL &&
      csr_sys->d_csr_row_count == NULL) {
  
    csr_sys->size_per_subsystem = size_per_subsystem;
    csr_sys->csr_number_nonzero = csr_number_nonzero;
    csr_sys->number_subsystems  = number_subsystems;

    return(CV_CUSOLVER_SUCCESS);

  } else {

    return(CV_CUSOLVER_MEM_FAIL);

  }
}


/*-----------------------------------------------------------------
  cv_cuSolver_SystemInitialize
  -----------------------------------------------------------------
  This routine initializes the CSR matrix system memory.

  USER MUST CALL THIS AFTER cv_cuSolver_CSR_SetSizes.
  -----------------------------------------------------------------*/
int cv_cuSolver_SystemInitialize(void* cv_mem, int* csr_row_count, int* csr_col_index)
{
  CVodeMem cvode_mem = (CVodeMem) cv_mem;
  CV_cuSolver_Mem cv_cus_mem = (CV_cuSolver_Mem) cvode_mem->cv_lmem;

  cv_cuSolver_csr_sys_initialize(cv_cus_mem->csr_sys, csr_row_count, csr_col_index);

  if (cv_cus_mem->store_jacobian) cv_cuSolver_csr_sys_initialize(cv_cus_mem->saved_jacobian, csr_row_count, csr_col_index);

  return(0);
}


void cv_cuSolver_csr_sys_initialize(CV_cuSolver_csr_sys csr_sys, int* csr_row_count, int* csr_col_index)
{
  cudaError_t cuda_status = cudaSuccess;

  if (csr_sys->d_csr_values == NULL) {
    cuda_status = cudaMalloc((void**) &csr_sys->d_csr_values,
			     sizeof(realtype) * csr_sys->csr_number_nonzero * csr_sys->number_subsystems);
    assert(cuda_status == cudaSuccess);
#if PRINT_CUSOLVER_DEBUGGING    
    std::cout << "Allocated device memory for d_csr_values" << std::endl;
#endif
  }

  if (csr_sys->d_csr_col_index == NULL) {
    cuda_status = cudaMalloc((void**) &csr_sys->d_csr_col_index,
			     sizeof(int) * csr_sys->csr_number_nonzero);
    assert(cuda_status == cudaSuccess);
    cuda_status = cudaMemcpy(csr_sys->d_csr_col_index, csr_col_index,
			     sizeof(int) * csr_sys->csr_number_nonzero,
			     cudaMemcpyHostToDevice);
    assert(cuda_status == cudaSuccess);
#if PRINT_CUSOLVER_DEBUGGING
    std::cout << "Allocated device memory for d_csr_col_index and initialized" << std::endl;
#endif
  }

  if (csr_sys->d_csr_row_count == NULL) {
    cuda_status = cudaMalloc((void**) &csr_sys->d_csr_row_count,
			     sizeof(int) * (csr_sys->size_per_subsystem + 1));
    assert(cuda_status == cudaSuccess);
    cuda_status = cudaMemcpy(csr_sys->d_csr_row_count, csr_row_count,
			     sizeof(int) * (csr_sys->size_per_subsystem + 1),
			     cudaMemcpyHostToDevice);
    assert(cuda_status == cudaSuccess);
#if PRINT_CUSOLVER_DEBUGGING
    std::cout << "Allocated device memory for d_csr_row_count and initialized" << std::endl;
#endif
  }
}


void cv_cuSolver_check_cusolver_status(cusolverStatus_t status)
{
#if PRINT_CUSOLVER_DEBUGGING
  if (status == CUSOLVER_STATUS_SUCCESS)
    std::cout << "CUSOLVER_STATUS_SUCCESS" << std::endl;
  if (status == CUSOLVER_STATUS_NOT_INITIALIZED)
    std::cout << "CUSOLVER_STATUS_NOT_INITIALIZED" << std::endl;
  if (status == CUSOLVER_STATUS_ALLOC_FAILED)
    std::cout << "CUSOLVER_STATUS_ALLOC_FAILED" << std::endl;
  if (status == CUSOLVER_STATUS_INVALID_VALUE)
    std::cout << "CUSOLVER_STATUS_INVALID_VALUE" << std::endl;
  if (status == CUSOLVER_STATUS_ARCH_MISMATCH)
    std::cout << "CUSOLVER_STATUS_ARCH_MISMATCH" << std::endl;
  if (status == CUSOLVER_STATUS_EXECUTION_FAILED)
    std::cout << "CUSOLVER_STATUS_EXECUTION_FAILED" << std::endl;
  if (status == CUSOLVER_STATUS_INTERNAL_ERROR)
    std::cout << "CUSOLVER_STATUS_INTERNAL_ERROR" << std::endl;
  if (status == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
    std::cout << "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED" << std::endl;
#endif
}


void cv_cuSolver_check_cusparse_status(cusparseStatus_t status)
{
#if PRINT_CUSOLVER_DEBUGGING
  if (status == CUSPARSE_STATUS_SUCCESS)
    std::cout << "CUSPARSE_STATUS_SUCCESS" << std::endl;
  if (status == CUSPARSE_STATUS_NOT_INITIALIZED)
    std::cout << "CUSPARSE_STATUS_NOT_INITIALIZED" << std::endl;
  if (status == CUSPARSE_STATUS_ALLOC_FAILED)
    std::cout << "CUSPARSE_STATUS_ALLOC_FAILED" << std::endl;
  if (status == CUSPARSE_STATUS_INVALID_VALUE)
    std::cout << "CUSPARSE_STATUS_INVALID_VALUE" << std::endl;
  if (status == CUSPARSE_STATUS_ARCH_MISMATCH)
    std::cout << "CUSPARSE_STATUS_ARCH_MISMATCH" << std::endl;
  if (status == CUSPARSE_STATUS_MAPPING_ERROR)
    std::cout << "CUSPARSE_STATUS_MAPPING_ERROR" << std::endl;
  if (status == CUSPARSE_STATUS_EXECUTION_FAILED)
    std::cout << "CUSPARSE_STATUS_EXECUTION_FAILED" << std::endl;
  if (status == CUSPARSE_STATUS_INTERNAL_ERROR)
    std::cout << "CUSPARSE_STATUS_INTERNAL_ERROR" << std::endl;
  if (status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
    std::cout << "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED" << std::endl;
#endif
}


/*-----------------------------------------------------------------
  cv_cuSolver_SolverInitialize
  -----------------------------------------------------------------
  This routine initializes the cuSolver and sets up workspace.
  -----------------------------------------------------------------*/
int cv_cuSolver_SolverInitialize(CV_cuSolver_Mem cv_cus_mem)
{
  cudaError_t cuda_status = cudaSuccess;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

  // Analyze system structure
#if PRINT_CUSOLVER_DEBUGGING  
  std::cout << "Calling analysis routine with ..." << std::endl;
  std::cout << "   cusolverHandle = " << cv_cus_mem->cus_work->cusolverHandle << std::endl;
  std::cout << "   size = " << cv_cus_mem->csr_sys->size_per_subsystem << std::endl;
  std::cout << "   nnz = " << cv_cus_mem->csr_sys->csr_number_nonzero << std::endl;
#endif

  if (cv_cus_mem->cus_work->workspace == NULL) {
    cusolver_status = cusolverSpXcsrqrAnalysisBatched(cv_cus_mem->cus_work->cusolverHandle,
						      cv_cus_mem->csr_sys->size_per_subsystem,
						      cv_cus_mem->csr_sys->size_per_subsystem,
						      cv_cus_mem->csr_sys->csr_number_nonzero,
						      cv_cus_mem->cus_work->system_description,
						      cv_cus_mem->csr_sys->d_csr_row_count,
						      cv_cus_mem->csr_sys->d_csr_col_index,
						      cv_cus_mem->cus_work->info);
    cv_cuSolver_check_cusolver_status(cusolver_status);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);


    // Get the workspace size we will need
    cusolver_status = cusolverSpDcsrqrBufferInfoBatched(cv_cus_mem->cus_work->cusolverHandle,
							cv_cus_mem->csr_sys->size_per_subsystem,
							cv_cus_mem->csr_sys->size_per_subsystem,
							cv_cus_mem->csr_sys->csr_number_nonzero,
							cv_cus_mem->cus_work->system_description,
							cv_cus_mem->csr_sys->d_csr_values,
							cv_cus_mem->csr_sys->d_csr_row_count,
							cv_cus_mem->csr_sys->d_csr_col_index,
							cv_cus_mem->csr_sys->number_subsystems,
							cv_cus_mem->cus_work->info,
							&cv_cus_mem->cus_work->internal_size,
							&cv_cus_mem->cus_work->workspace_size);
    cv_cuSolver_check_cusolver_status(cusolver_status);  
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    // Allocate working space on the device
    cuda_status = cudaMalloc((void**) &cv_cus_mem->cus_work->workspace, cv_cus_mem->cus_work->workspace_size);
    assert(cuda_status == cudaSuccess);
  }

  return(0);
}


/*-----------------------------------------------------------------
  cv_cuSolver_SolveSystem
  -----------------------------------------------------------------
  This routine actually solves the system
  -----------------------------------------------------------------*/
int cv_cuSolver_SolveSystem(CV_cuSolver_Mem cv_cus_mem, N_Vector b)
{

#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Doing cv_cuSolver_SolveSystem" << std::endl;
#endif
  
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cuda_status = cudaSuccess;
  cuda_status = cudaGetLastError();

#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Got CUDA Last Error of: ";
  std::cout << cudaGetErrorString(cuda_status) << std::endl;
#endif

  assert(cuda_status == cudaSuccess);

  realtype* device_b = N_VGetDeviceArrayPointer_Cuda(b);
  realtype* device_x = N_VGetDeviceArrayPointer_Cuda(cv_cus_mem->x);

#if PRINT_CUSOLVER_DEBUGGING
  // print out the first system
  realtype sysmat[cv_cus_mem->csr_sys->csr_number_nonzero];
  cuda_status = cudaMemcpy(&sysmat[0], cv_cus_mem->csr_sys->d_csr_values,
			   sizeof(realtype) * cv_cus_mem->csr_sys->csr_number_nonzero,
			   cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);
  std::cout << "first system is: A = " << std::endl;
  for (int i = 0; i < cv_cus_mem->csr_sys->csr_number_nonzero; i++)
    std::cout << sysmat[i] << " ";
  std::cout << std::endl;


  realtype sysmat_b[cv_cus_mem->csr_sys->size_per_subsystem];
  cuda_status = cudaMemcpy(&sysmat_b[0], device_b,
			   sizeof(realtype) * cv_cus_mem->csr_sys->size_per_subsystem,
			   cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);
  std::cout << "first system is: b = " << std::endl;
  for (int i = 0; i < cv_cus_mem->csr_sys->size_per_subsystem; i++)
    std::cout << sysmat_b[i] << " ";
  std::cout << std::endl;


  // print out the second system
  realtype sysmat2[cv_cus_mem->csr_sys->csr_number_nonzero];
  cuda_status = cudaMemcpy(&sysmat2[0], cv_cus_mem->csr_sys->d_csr_values + cv_cus_mem->csr_sys->csr_number_nonzero,
			   sizeof(realtype) * cv_cus_mem->csr_sys->csr_number_nonzero,
			   cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);
  std::cout << "second system is: A = " << std::endl;
  for (int i = 0; i < cv_cus_mem->csr_sys->csr_number_nonzero; i++)
    std::cout << sysmat2[i] << " ";
  std::cout << std::endl;


  realtype sysmat2_b[cv_cus_mem->csr_sys->size_per_subsystem];
  cuda_status = cudaMemcpy(&sysmat2_b[0], device_b + cv_cus_mem->csr_sys->size_per_subsystem,
			   sizeof(realtype) * cv_cus_mem->csr_sys->size_per_subsystem,
			   cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);
  std::cout << "second system is: b = " << std::endl;
  for (int i = 0; i < cv_cus_mem->csr_sys->size_per_subsystem; i++)
    std::cout << sysmat2_b[i] << " ";
  std::cout << std::endl;
#endif
  

  cusolver_status = cusolverSpDcsrqrsvBatched(cv_cus_mem->cus_work->cusolverHandle,
					      cv_cus_mem->csr_sys->size_per_subsystem,
					      cv_cus_mem->csr_sys->size_per_subsystem,
					      cv_cus_mem->csr_sys->csr_number_nonzero,
					      cv_cus_mem->cus_work->system_description,
					      cv_cus_mem->csr_sys->d_csr_values,
					      cv_cus_mem->csr_sys->d_csr_row_count,
					      cv_cus_mem->csr_sys->d_csr_col_index,
					      device_b,
					      device_x,
					      cv_cus_mem->csr_sys->number_subsystems,
					      cv_cus_mem->cus_work->info,
					      cv_cus_mem->cus_work->workspace);
  cv_cuSolver_check_cusolver_status(cusolver_status);  
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
#if PRINT_CUSOLVER_DEBUGGING
  std::cout << "Finished cv_cuSolver_SolveSystem" << std::endl;
#endif
  return(0);
}



/*-----------------------------------------------------------------
  cv_cuSolver_WorkspaceFree
  -----------------------------------------------------------------
  This routine frees memory associated with the solver and system
  -----------------------------------------------------------------*/
int cv_cuSolver_WorkspaceFree(CV_cuSolver_Mem cv_cus_mem)
{
  cv_cuSolver_SystemFree(cv_cus_mem);
  cv_cuSolver_SolverFree(cv_cus_mem);
  return(0);
}



/*-----------------------------------------------------------------
  cv_cuSolver_SystemFree
  -----------------------------------------------------------------
  This routine frees memory associated with the CSR matrix system
  -----------------------------------------------------------------*/
int cv_cuSolver_SystemFree(CV_cuSolver_Mem cv_cus_mem)
{
  cv_cuSolver_csr_sys_free(cv_cus_mem->csr_sys);
  free(cv_cus_mem->csr_sys);
  cv_cus_mem->csr_sys = NULL;

  if (cv_cus_mem->store_jacobian) {
    cv_cuSolver_csr_sys_free(cv_cus_mem->saved_jacobian);
    free(cv_cus_mem->saved_jacobian);
    cv_cus_mem->saved_jacobian = NULL;
  }

  return(0);
}


void cv_cuSolver_csr_sys_free(CV_cuSolver_csr_sys csr_sys)
{
  cudaError_t cuda_status = cudaSuccess;  
  if (csr_sys->d_csr_row_count != NULL) {
    cuda_status = cudaFree(csr_sys->d_csr_row_count);
    assert(cuda_status == cudaSuccess);
    csr_sys->d_csr_row_count = NULL;    
  }

  if (csr_sys->d_csr_col_index != NULL) {
    cuda_status = cudaFree(csr_sys->d_csr_col_index);
    assert(cuda_status == cudaSuccess);
    csr_sys->d_csr_col_index = NULL;
  }

  if (csr_sys->d_csr_values != NULL) {
    cuda_status = cudaFree(csr_sys->d_csr_values);
    assert(cuda_status == cudaSuccess);
    csr_sys->d_csr_values = NULL;    
  }
}


/*-----------------------------------------------------------------
  cv_cuSolver_SolverFree
  -----------------------------------------------------------------
  This routine frees memory associated with the cuSolver solver
  -----------------------------------------------------------------*/
int cv_cuSolver_SolverFree(CV_cuSolver_Mem cv_cus_mem)
{

  if (cv_cus_mem->cus_work->workspace != NULL) {
    cudaFree(cv_cus_mem->cus_work->workspace);
    cv_cus_mem->cus_work->workspace = NULL;    
  }

  free(cv_cus_mem->cus_work);
  cv_cus_mem->cus_work = NULL;

  return(0);
}
