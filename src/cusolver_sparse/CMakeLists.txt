# CMakeLists.txt file for the cuda cuSolver sparse library

INSTALL(CODE "MESSAGE(\"\nInstall CUSOLVER_SPARSE\n\")")

SET(cusolver_sparse_SOURCES cvode_cusolver_spqr.cu)

# tells compiler it is a CUDA source
set_source_files_properties(${cusolver_sparse_SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

# Add variable shared_SOURCES with the common SUNDIALS sources which will
# also be included in the NVECCUDA library
SET(shared_SOURCES 
  sundials_math.c
  cvode.c
)
ADD_PREFIX(${sundials_SOURCE_DIR}/src/sundials/ shared_SOURCES)
ADD_PREFIX(${sundials_SOURCE_DIR}/src/cvode/ shared_SOURCES)

SET(cusolver_sparse_HEADERS 
  cvode_cusolver_spqr.h
)
ADD_PREFIX(${sundials_SOURCE_DIR}/include/cusolver_sparse/ cusolver_sparse_HEADERS)

# Add source directory to include directories
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(/usr/local/cuda-9.2/include)

# Define C preprocessor flag -DBUILD_SUNDIALS_LIBRARY 
ADD_DEFINITIONS(-DBUILD_SUNDIALS_LIBRARY)

# Rules for building and installing the static library:
#  - Add the build target
#  - Set the library name and make sure it is not deleted
#  - Install the library
IF(BUILD_STATIC_LIBS)
  CUDA_ADD_LIBRARY(sundials_cusolver_sparse_static STATIC ${cusolver_sparse_SOURCES})
  SET_TARGET_PROPERTIES(sundials_cusolver_sparse_static
    PROPERTIES OUTPUT_NAME sundials_cusolver_sparse CLEAN_DIRECT_OUTPUT 1
  )
  INSTALL(TARGETS sundials_cusolver_sparse_static DESTINATION lib)
ENDIF(BUILD_STATIC_LIBS)

# Rules for building and installing the shared library:
#  - Add the build target for the library
#  - Set the library name and make sure it is not deleted
#  - Set VERSION and SOVERSION for shared libraries
#  - Install the library
IF(BUILD_SHARED_LIBS)
  CUDA_ADD_LIBRARY(sundials_cusolver_sparse_shared SHARED ${cusolver_sparse_SOURCES})
  SET_TARGET_PROPERTIES(sundials_cusolver_sparse_shared
    PROPERTIES OUTPUT_NAME sundials_cusolver_sparse CLEAN_DIRECT_OUTPUT 1
  )
  SET_TARGET_PROPERTIES(sundials_cusolver_sparse_shared
    PROPERTIES VERSION ${nveclib_VERSION} SOVERSION ${nveclib_SOVERSION}
  )
  INSTALL(TARGETS sundials_cusolver_sparse_shared DESTINATION lib)
ENDIF(BUILD_SHARED_LIBS)

# Install the header files
INSTALL(FILES ${cusolver_sparse_HEADERS} DESTINATION include/cusolver)

MESSAGE(STATUS "Added NVECTOR_CUDA module")
