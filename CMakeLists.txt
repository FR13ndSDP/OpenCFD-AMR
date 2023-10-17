cmake_minimum_required(VERSION 3.22.1)

message(STATUS "CMake version: ${CMAKE_VERSION}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Util/CMake")

#
# Base options
#
set(EBAMR_DIM             "3"        CACHE STRING "Number of physical dimensions")
set(EBAMR_PRECISION       "DOUBLE"   CACHE STRING "Floating point precision SINGLE or DOUBLE")
set(EBAMR_CASE            "Wing"    CACHE STRING "Case folder")

#
# Physics options
#
option(OPTION_USE_CHEM      "Enable chemical reaction"     OFF)

#
# Misc options
#
option(OPTION_ENABLE_TINY_PROFILE "Enable AMReX tiny profiler" OFF)
option(OPTION_ENABLE_HDF5         "Enable HDF5 PlotFiles"      OFF)
option(OPTION_ENABLE_HYPRE        "Enable Hypre"               OFF)
option(OPTION_ENABLE_SUNDIALS     "Enable Sundials integrator" OFF)

#
# HPC options
#
option(OPTION_MPI    "Enable MPI"    OFF)
option(OPTION_OPENMP "Enable OpenMP" OFF)
option(OPTION_CUDA   "Enable CUDA"   ON)
option(OPTION_HIP    "Enable HIP"    OFF)
option(OPTION_SYCL   "Enable SyCL"   OFF)

# C++ Options
if (OPTION_HIP)
   set(CMAKE_CXX_COMPILER clang++) # or use hipcc for DTK <=23.04
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

#
# Setup Languages
#
if (OPTION_CUDA)
   enable_language(CUDA)
   # TODO: just a workaround
   set(AMREX_CUDA_ARCH 80)
   set(AMREX_CUDA_ARCHS 80)
endif ()
if (OPTION_HIP)
   enable_language(HIP)
   # Hygon DCU
   set(AMD_ARCH gfx906 CACHE STRING "DCU AMD ARCH")
endif ()

# project
project( ebamr
         DESCRIPTION "An AMR code for high order methods"
         HOMEPAGE_URL "https://github.com/FR13ndSDP/OpenCFD-AMR"
         VERSION 1.0
         LANGUAGES CXX
)

# Checks
if(EBAMR_DIM EQUAL 1)
    message(FATAL_ERROR "EBAMR does not support 1D.")
endif()

#
# Cases
#
set(EBAMR_EXEC_SUBDIRS Exec/)
list(TRANSFORM EBAMR_EXEC_SUBDIRS PREPEND "${CMAKE_CURRENT_LIST_DIR}/")
set(EXEC_DIR "")
foreach(subdir ${EBAMR_EXEC_SUBDIRS})
    FILE(GLOB caselist ${subdir}/*)
    foreach(case ${caselist})
        if(IS_DIRECTORY ${case})
            get_filename_component( case_dir ${case} NAME)
            LIST(APPEND EBAMR_CASE_LIST ${case_dir})
            if( ${EBAMR_CASE} STREQUAL ${case_dir})
                set(EXEC_DIR ${case})
            endif()
        endif()
    endforeach()
endforeach()
if (NOT EXEC_DIR) 
    message(FATAL_ERROR "EBAMR_CASE=${EBAMR_CASE} is not a valid case !")
endif()

########################### EBAMR #####################################
message(STATUS "Exec Configure Section")
add_executable(ebamr "")
set_target_properties(ebamr PROPERTIES OUTPUT_NAME "EBR${EBAMR_DIM}d.out")

set(SRC_DIR ${CMAKE_SOURCE_DIR}/Source)
add_subdirectory(${SRC_DIR})
add_subdirectory(${EXEC_DIR})

if (OPTION_MPI)
   find_package(MPI REQUIRED CXX)
endif()

# detect if hipcc is used
if ( ${CMAKE_CXX_COMPILER} MATCHES "(.*)hipcc$" )
   # this is for DTK<=23.04 with hipcc
   set(CMAKE_CXX_FLAGS "-pthread --offload-arch=${AMD_ARCH}")
endif()

if ( NOT CMAKE_CXX_FLAGS )
  target_link_libraries(ebamr PUBLIC AMReX::Flags_CXX)
endif ()

if ( OPTION_USE_CHEM )
   set(CMAKE_CXX_FLAGS "-DCHEM ${CMAKE_CXX_FLAGS}")
   if ( OPTION_CUDA )
      set(CMAKE_CUDA_FLAGS "-DCHEM ${CMAKE_CXX_FLAGS}")
   endif ()
endif ()
########################### AMReX #####################################
message(STATUS "AMReX Configure Section")
set(AMREX_SUBMOD_LOCATION "${CMAKE_SOURCE_DIR}/Submodules/amrex")
include(SetAMReXOptions)
list(APPEND CMAKE_MODULE_PATH "${AMREX_SUBMOD_LOCATION}/Tools/CMake")
add_subdirectory(${AMREX_SUBMOD_LOCATION})
include(AMReXBuildInfo)
generate_buildinfo(ebamr ${CMAKE_SOURCE_DIR})
target_include_directories(ebamr SYSTEM PUBLIC ${AMREX_SUBMOD_LOCATION}/Tools/C_scripts)
if (OPTION_CUDA)
  setup_target_for_cuda_compilation(ebamr)
endif ()
target_link_libraries(ebamr PUBLIC AMReX::amrex)
########################### Summary ##################################
include(AMReXGenexHelpers)
include(AMReXTargetHelpers)

get_target_properties_flattened(ebamr  _includes _defines _flags _link_line)
set(_lang CXX)
set(_prop _includes _defines _flags _link_line)

# Loop over all combinations of language and property and extract
# what you need
foreach( _p IN LISTS _prop )
   foreach( _l IN LISTS _lang )

      string(TOLOWER ${_l} _ll) # Lower case language name

      # _${_ll}${_p} is a variable named as _lang_property,
      # both lower case.
      set(_${_ll}${_p} "${${_p}}")
      eval_genex( _${_ll}${_p} ${_l} ${CMAKE_${_l}_COMPILER_ID}
         COMP_VERSION ${CMAKE_${_l}_COMPILER_VERSION}
         CONFIG       ${CMAKE_BUILD_TYPE}
         INTERFACE    BUILD)

      if (_${_ll}${_p})

         list(REMOVE_DUPLICATES _${_ll}${_p})

         if ("${_p}" STREQUAL "_defines")
            string(REPLACE ";" " -D" _${_ll}${_p} "-D${_${_ll}${_p}}")
         elseif ("${_p}" STREQUAL "_includes")
            string(REPLACE ";" " -I" _${_ll}${_p} "-I${_${_ll}${_p}}")
         else()
            string(REPLACE ";" " " _${_ll}${_p} "${_${_ll}${_p}}")
         endif ()

      endif ()

   endforeach()
endforeach ()

string ( TOUPPER "${CMAKE_BUILD_TYPE}"  AMREX_BUILD_TYPE )
set(_cxx_flags "${CMAKE_CXX_FLAGS_${AMREX_BUILD_TYPE}} ${CMAKE_CXX_FLAGS} ${_cxx_flags}")
message( STATUS "   C++ flags                = ${_cxx_flags}")
message( STATUS "   Link line                = ${_cxx_link_line}")
message( STATUS "   C++ include paths        = ${_cxx_includes}")
