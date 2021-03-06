# ---------------------------------------------------------------
# Programmer:  Radu Serban @ LLNL
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
# CMakeLists.txt file for CVODE parhyp examples
# ---------------------------------------------------------------

# Example lists are tuples "name\;nodes\;tasks\;type" where the
# type is develop for examples excluded from 'make test' in releases

# Examples using SUNDIALS linear solvers
SET(CVODE_examples
  "cvAdvDiff_non_ph\;2\;2\;develop"
  )

IF(MPI_C_COMPILER)
  # use MPI wrapper as the compiler
  SET(CMAKE_C_COMPILER ${MPI_C_COMPILER})
ELSE()
  # add MPI_INCLUDE_PATH to include directories
  INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
ENDIF()

# Specify libraries to link against (through the target that was used to
# generate them) based on the value of the variable LINK_LIBRARY_TYPE
IF(LINK_LIBRARY_TYPE MATCHES "static")
  SET(CVODE_LIB sundials_cvode_static)
  SET(NVECP_LIB sundials_nvecparhyp_static)
ELSE()
  SET(CVODE_LIB sundials_cvode_shared)
  SET(NVECP_LIB sundials_nvecparhyp_shared)
ENDIF()

# Set-up linker flags and link libraries
SET(SUNDIALS_LIBS ${CVODE_LIB} ${NVECP_LIB} ${EXTRA_LINK_LIBS})


# Add the build and install targets for each example
FOREACH(example_tuple ${CVODE_examples})

  # parse the example tuple
  LIST(GET example_tuple 0 example)
  LIST(GET example_tuple 1 number_of_nodes)
  LIST(GET example_tuple 2 number_of_tasks)
  LIST(GET example_tuple 3 example_type)

  # example source files
  ADD_EXECUTABLE(${example} ${example}.c)

  SET_TARGET_PROPERTIES(${example} PROPERTIES FOLDER "Examples")

  # add example to regression tests
  SUNDIALS_ADD_TEST(${example} ${example}
    MPI_NPROCS ${number_of_tasks}
    ANSWER_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    ANSWER_FILE ${example}.out
    EXAMPLE_TYPE ${example_type})

  # libraries to link against
  TARGET_LINK_LIBRARIES(${example} ${SUNDIALS_LIBS} ${HYPRE_LIBRARIES})

  IF(NOT MPI_C_COMPILER)
    TARGET_LINK_LIBRARIES(${example} ${MPI_LIBRARY} ${MPI_EXTRA_LIBRARIES})
  ENDIF()

  # install example
  IF(EXAMPLES_INSTALL)
    INSTALL(FILES ${example}.c ${example}.out
      DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/parhyp)
    #INSTALL(FILES vis.c DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/parhyp)
  ENDIF()

ENDFOREACH(example_tuple ${CVODE_examples})


IF(EXAMPLES_INSTALL)

  # Install the README file
  INSTALL(FILES README DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/parhyp)

  # Prepare substitution variables for Makefile and/or CMakeLists templates
  SET(SOLVER "CVODE")
  SET(SOLVER_LIB "sundials_cvode")

  EXAMPLES2STRING(CVODE_examples EXAMPLES)

  # Regardless of the platform we're on, we will generate and install
  # CMakeLists.txt file for building the examples. This file  can then
  # be used as a template for the user's own programs.

  # generate CMakelists.txt in the binary directory
  CONFIGURE_FILE(
    ${PROJECT_SOURCE_DIR}/examples/templates/cmakelists_parhyp_C_ex.in
    ${PROJECT_BINARY_DIR}/examples/cvode/parhyp/CMakeLists.txt
    @ONLY
    )

  # install CMakelists.txt
  INSTALL(
    FILES ${PROJECT_BINARY_DIR}/examples/cvode/parhyp/CMakeLists.txt
    DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/parhyp
    )

  # On UNIX-type platforms, we also  generate and install a makefile for
  # building the examples. This makefile can then be used as a template
  # for the user's own programs.

  IF(UNIX)
    # generate Makefile and place it in the binary dir
    CONFIGURE_FILE(
      ${PROJECT_SOURCE_DIR}/examples/templates/makefile_parhyp_C_ex.in
      ${PROJECT_BINARY_DIR}/examples/cvode/parhyp/Makefile_ex
      @ONLY
      )
    # install the configured Makefile_ex as Makefile
    INSTALL(
      FILES ${PROJECT_BINARY_DIR}/examples/cvode/parhyp/Makefile_ex
      DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/parhyp
      RENAME Makefile
      )
  ENDIF(UNIX)

ENDIF(EXAMPLES_INSTALL)
