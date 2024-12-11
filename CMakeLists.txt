cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(FLUX LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/")

# cmake global settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(BUILD_THS ON CACHE INTERNAL "Build Torch op")
set(BUILD_TEST ON CACHE INTERNAL "Build unit tests")
set(ENABLE_NVSHMEM ON CACHE INTERNAL "Use NVSHMEM to transfer data")
set(CUTLASS_TRACE OFF CACHE INTERNAL "Print CUTLASS Host Trace info")
set(FLUX_DEBUG OFF CACHE INTERNAL "Define FLUX_DEBUG")
OPTION(WITH_PROTOBUF "build with protobuf" OFF)
message("PYTHONPATH: ${PYTHONPATH}")
message("NVShmem Support: ${ENABLE_NVSHMEM}")

# find cuda
find_package(CUDAToolkit REQUIRED)

string(REPLACE " " ";" CUDAARCHS "${FLUX_CUDAARCHS}")

message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
if(CUDAToolkit_VERSION VERSION_LESS "11.0")
  message(FATAL_ERROR "requires cuda to be >= 11.0")
elseif(CUDAToolkit_VERSION VERSION_LESS "12.0")
  set(CUDAARCHS "80" CACHE STRING "CUDA Architectures")
elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.4")
  set(CUDAARCHS "80;89;90" CACHE STRING "CUDA Architectures")
else()
  set(CUDAARCHS "80;90" CACHE STRING "CUDA Architectures")
endif()

set(CMAKE_CUDA_ARCHITECTURES ${CUDAARCHS})
message(STATUS "CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

set(CUDA_ARCH_FLAGS)
foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
  list(APPEND CUDA_ARCH_FLAGS "-gencode=arch=compute_${ARCH},code=\\\"sm_${ARCH},compute_${ARCH}\\\"")
endforeach()

string(JOIN " " JOINED_CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${JOINED_CUDA_ARCH_FLAGS}")
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wno-psabi")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler=-fno-strict-aliasing")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -DCUTLASS_DEBUG_TRACE_LEVEL=0")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

set(CMAKE_CXX_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")

if(CUTLASS_TRACE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUTLASS_DEBUG_TRACE_LEVEL=1")
endif()

if(FLUX_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFLUX_DEBUG")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFLUX_DEBUG")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# force use sm90a for cutlass
string(REGEX REPLACE "sm_90" "sm_90a" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
string(REGEX REPLACE "compute_90" "compute_90a" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/library/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/profiler/include
)

set(COMMON_LIB_DIRS "")
list(APPEND COMMON_LIB_DIRS "${CUDAToolkit_LIBRARY_DIR}")
message(ENABLE_NVSHMEM "ENABLE_NVSHMEM is set to: ${ENABLE_NVSHMEM}")
if(ENABLE_NVSHMEM)
  add_definitions(-DFLUX_SHM_USE_NVSHMEM)
  set(NVSHMEM_BUILD_DIR ${PROJECT_SOURCE_DIR}/3rdparty/nvshmem/build)
  message(STATUS "NVSHMEM build dir: ${NVSHMEM_BUILD_DIR}")
  if(NOT EXISTS ${NVSHMEM_BUILD_DIR})
      message(FATAL_ERROR "NVSHMEM not found. Please run ./build_nvshmem.sh first.")
  endif()
  list(APPEND COMMON_HEADER_DIRS "${NVSHMEM_BUILD_DIR}/src/include")
  list(APPEND COMMON_LIB_DIRS "${NVSHMEM_BUILD_DIR}/src/lib")
endif()

# append headers explicitly for .cu files, in order to enable vscode clangd intellisense
foreach(inc_dir ${COMMON_HEADER_DIRS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -I${inc_dir}")
endforeach()
message("final CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

add_subdirectory(src)

add_library(flux SHARED capi/flux_api.cu)

target_include_directories(flux PRIVATE
    src
    include)

INSTALL(TARGETS flux
    LIBRARY DESTINATION "lib")

INSTALL(FILES capi/flux_api.h DESTINATION "include" RENAME flux_api.h)

target_link_libraries(flux PUBLIC -Wl,--whole-archive flux_cuda_all_gather -Wl,--no-whole-archive -Wl,--whole-archive flux_cuda_reduce_scatter -Wl,--no-whole-archive -Wl,--whole-archive flux_cuda -Wl,--no-whole-archive)
