# ---------------------------------------------------------------
# Programmer:  David J. Gardner @ LLNL
# ---------------------------------------------------------------
# LLNS Copyright Start
# Copyright (c) 2014, Lawrence Livermore National Security
# This work was performed under the auspices of the U.S. Department 
# of Energy by Lawrence Livermore National Laboratory in part under 
# Contract W-7405-Eng-48 and in part under Contract DE-AC52-07NA27344.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see the LICENSE file.
# LLNS Copyright End
# ---------------------------------------------------------------
# Print warning is the user sets a deprecated CMake variable and
# copy the value into the correct CMake variable
# ---------------------------------------------------------------

# macro to print warning for deprecated CMake variable
MACRO(PRINT_DEPRECATED old_variable new_variable)
  PRINT_WARNING("${old_variable} is deprecated and will be removed in the future."
                "Copying value to ${new_variable}.")
ENDMACRO()

IF(DEFINED EXAMPLES_ENABLE)
  PRINT_DEPRECATED(EXAMPLES_ENABLE EXAMPLES_ENABLE_C)
  FORCE_VARIABLE(EXAMPLES_ENABLE_C BOOL "Build SUNDIALS C examples" ${EXAMPLES_ENABLE})
  UNSET(EXAMPLES_ENABLE CACHE)
ENDIF()

IF(DEFINED CXX_ENABLE)
  PRINT_DEPRECATED(CXX_ENABLE EXAMPLES_ENABLE_CXX)
  FORCE_VARIABLE(EXAMPLES_ENABLE_CXX BOOL "Build ARKode C++ examples" ${CXX_ENABLE})
  UNSET(CXX_ENABLE CACHE)
ENDIF()

IF(DEFINED F90_ENABLE)
  PRINT_DEPRECATED(F90_ENABLE EXAMPLES_ENABLE_F90)
  FORCE_VARIABLE(EXAMPLES_ENABLE_F90 BOOL "Build ARKode Fortran90 examples" ${F90_ENABLE})
  UNSET(F90_ENABLE CACHE)
ENDIF()

if(DEFINED MPI_MPICC)
  print_deprecated(MPI_MPICC MPI_C_COMPILER)
  force_variable(MPI_C_COMPILER FILEPATH "MPI C compiler" ${MPI_MPICC})
  unset(MPI_MPICC CACHE)
endif()

if(DEFINED MPI_MPICXX)
  print_deprecated(MPI_MPICXX MPI_CXX_COMPILER)
  force_variable(MPI_CXX_COMPILER FILPATH "MPI C++ compiler" ${MPI_MPICXX})
  unset(MPI_MPICXX CACHE)
endif()

if((DEFINED MPI_MPIF77) OR (DEFINED MPI_MPIF90))
  if(DEFINED MPI_MPIF90)
    print_warning("MPI_MPIF77 and MPI_MPIF90 are deprecated and will be removed in the future." "Copying MPI_MPIF90 value to MPI_Fortran_COMPILER")
    force_variable(MPI_Fortran_COMPILER FILEPATH "MPI Fortran compiler" ${MPI_MPIF90})
  else()
    print_warning("MPI_MPIF77 and MPI_MPIF90 are deprecated and will be removed in the future." "Copying MPI_MPIF77 value to MPI_Fortran_COMPILER")
    force_variable(MPI_Fortran_COMPILER FILEPATH "MPI Fortran compiler" ${MPI_MPIF77})
  endif()
  unset(MPI_MPIF77 CACHE)
  unset(MPI_MPIF90 CACHE)
endif()

if(DEFINED MPI_RUN_COMMAND)
  print_deprecated(MPI_RUN_COMMAND MPIEXEC_EXECUTABLE)
  force_variable(MPIEXEC_EXECUTABLE FILEPATH "MPI run command" ${MPI_RUN_COMMAND})
  unset(MPI_RUN_COMMAND CACHE)
endif()
