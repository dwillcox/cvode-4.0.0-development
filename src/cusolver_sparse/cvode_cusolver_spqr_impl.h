#ifndef _CVCUSOLVER_SPQR_IMPL_H
#define _CVCUSOLVER_SPQR_IMPL_H

#include "../cvode/cvode_impl.h"
#include <cusolverSp.h>
#include <cusparse.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

/*-----------------------------------------------------------------
  CV_CUSOLVER solver constants
  -----------------------------------------------------------------
  CV_CUSOLVER_DGMAX  maximum change in gamma between Jacobian evaluations
  -----------------------------------------------------------------*/

#define CV_CUSOLVER_DGMAX RCONST(0.2)

  /*-----------------------------------------------------------------
    Types : CV_cuSolver_MemRec, CV_cuSolver_Mem
    -----------------------------------------------------------------
    CV_cuSolver_Mem is pointer to a CV_cuSolver_MemRec structure.
    -----------------------------------------------------------------*/

  enum cuSolver_return_flags{CV_CUSOLVER_SUCCESS,
			     CV_CUSOLVER_MEM_NULL,
			     CV_CUSOLVER_LMEM_NULL,
			     CV_CUSOLVER_ILL_INPUT,
			     CV_CUSOLVER_MEM_FAIL,
			     CV_CUSOLVER_JACFUNC_UNRECVR,
			     CV_CUSOLVER_JACCOPY_UNRECVR,
			     CV_CUSOLVER_JACFUNC_RECVR,
			     CV_CUSOLVER_SCALEADDI_FAIL};

  enum cuSolver_method_t{QR}; // Add RF later

  typedef enum cuSolver_method_t cuSolver_method;

  typedef struct CV_cuSolver_workspace_QR_t {
    // Data structures for the QR cuSolver workspace
    size_t workspace_size;
    size_t internal_size;
    cusparseMatDescr_t system_description;
    cusolverSpHandle_t cusolverHandle;
    cusparseHandle_t cusparseHandle;
    csrqrInfo_t info;
    void*  workspace;
  }* CV_cuSolver_workspace_QR;

  typedef struct CV_cuSolver_csr_sys_t {
    // Data structures for the CSR sparse matrix system
    int size_per_subsystem, csr_number_nonzero, number_subsystems;

    // This data will live on the device.
    int* d_csr_row_count;
    int* d_csr_col_index;
    realtype* d_csr_values;
  }* CV_cuSolver_csr_sys;


  typedef int (*CV_cuSolver_JacFn)(realtype t, N_Vector y, N_Vector fy,
				   CV_cuSolver_csr_sys Jac, void* user_data);

  typedef struct CV_cuSolver_MemRec {

    int CV_CUSOLVER_MSBJ = 50;

    cuSolver_method method;            /* which cuSolver method we are using */

    CV_cuSolver_workspace_QR cus_work; /* cuSolver workspace structure */

    CV_cuSolver_csr_sys csr_sys;       /* CSR formatted matrix system to solve          */

    CV_cuSolver_csr_sys saved_jacobian;/* Saved Jacobian */

    CV_cuSolver_JacFn jac;             /* Jacobian routine to be called                 */
    void *J_data;                      /* user data is passed to jac                    */

    N_Vector x;                        /* solution vector */

    int nje;                           /* number of Jacobian evaluations */

    int nstlj;                         /* step where we last calculated Jacobian */

    int last_flag;                     /* last return flag */

    booleantype store_jacobian;        /* Should we store a Jacobian? */

  }* CV_cuSolver_Mem;

  /*-----------------------------------------------------------------
    Function prototypes
    -----------------------------------------------------------------*/

  /* generic linit/lsetup/lsolve/lfree interface routines for CVode to call */
  int cv_cuSolver_Initialize(CVodeMem cv_mem);

  int cv_cuSolver_Setup(CVodeMem cv_mem, int convfail, N_Vector ypred,
			N_Vector fpred, booleantype *jcurPtr,
			N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3);

  int cv_cuSolver_Solve(CVodeMem cv_mem, N_Vector b, N_Vector weight,
			N_Vector ycur, N_Vector fcur);

  int cv_cuSolver_Free(CVodeMem cv_mem);

  /* user functions for working with the CSR formatted system and cuSolver */

  int cv_cuSolver_SetLinearSolver(void *cvode_mem, cuSolver_method cus_method,
				  bool store_jacobian, int num_steps_save_jacobian);

  int cv_cuSolver_SetJacFun(void *cvode_mem, CV_cuSolver_JacFn jac);

  int cv_cuSolver_GetWorkSpace(void *cvode_mem, long int *lenrwLS,
			       long int *leniwLS);

  char* cv_cuSolver_GetReturnFlagName(long int flag);

  int cv_cuSolver_CSR_SetSizes(void* cv_mem, int size_per_subsystem,
			       int csr_number_nonzero, int number_subsystems);

  int cv_cuSolver_CSR_SetSizes_Matrix(CV_cuSolver_csr_sys csr_sys, int size_per_subsystem,
				      int csr_number_nonzero, int number_subsystems);

  int cv_cuSolver_SystemInitialize(void* cv_mem, int* csr_row_count, int* csr_col_index);

  void cv_cuSolver_csr_sys_initialize(CV_cuSolver_csr_sys csr_sys, int* csr_row_count, int* csr_col_index);

  int cv_cuSolver_GetNumJacEvals(void* cv_mem, int* nje);

  /* internal functions */

  int cv_cuSolver_InitializeCounters(CV_cuSolver_Mem cv_cus_mem);

  int cv_cuSolver_SolverInitialize(CV_cuSolver_Mem cv_cus_mem);

  int cv_cuSolver_ScaleAddI(realtype scale, CV_cuSolver_Mem cv_cus_mem);

__global__ void cv_cuSolver_ScaleAddI_kernel(const realtype scale, realtype* csr_values,
					     int* csr_col_index, int* csr_row_count,
					     const int csr_base, const int nnz,
					     const int size, const int nbatched);

  int cv_cuSolver_SolveSystem(CV_cuSolver_Mem cv_cus_mem, N_Vector b);

  int cv_cuSolver_WorkspaceFree(CV_cuSolver_Mem cv_cus_mem);

  int cv_cuSolver_SystemFree(CV_cuSolver_Mem cv_cus_mem);

  void cv_cuSolver_csr_sys_free(CV_cuSolver_csr_sys csr_sys);

  int cv_cuSolver_SolverFree(CV_cuSolver_Mem cv_cus_mem);

  void cv_cuSolver_check_cusolver_status(cusolverStatus_t status);

  void cv_cuSolver_check_cusparse_status(cusparseStatus_t status);

/*-----------------------------------------------------------------
  Error Messages
  -----------------------------------------------------------------*/

#define MSGD_CVMEM_NULL "Integrator memory is NULL."
#define MSGD_BAD_NVECTOR "A required vector operation is not implemented."
#define MSGD_MEM_FAIL "A memory request failed."
#define MSGD_LMEM_NULL "Linear solver memory is NULL."
#define MSGD_JACCOPY_FAILED "The stored Jacobian failed to copy in an unrecoverable manner."
#define MSGD_JACFUNC_FAILED "The Jacobian routine failed in an unrecoverable manner."
#define MSGD_MATSCALEADDI_FAILED "The ScaleAddI routine failed in an unrecoverable manner."

#ifdef __cplusplus
}
#endif

#endif
