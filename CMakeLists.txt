cmake_minimum_required(VERSION 3.22.1)

project( ebamr-GPU
         DESCRIPTION "An AMR code for high order methods"
         HOMEPAGE_URL "https://github.com/FR13ndSDP/ebamr-GPU"
         VERSION 1.0
         LANGUAGES CXX
       )

message(STATUS "CMake version: ${CMAKE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Util/CMake")

#
# Base options
#
set(PELELMEX_DIM             "3"        CACHE STRING "Number of physical dimensions")
set(PELELMEX_PRECISION       "DOUBLE"   CACHE STRING "Floating point precision SINGLE or DOUBLE")
set(PELELMEX_CASE            "0_Sod"     CACHE STRING "Case folder")
#
# Physics options
#
option(PELELMEX_USE_EB        "Enable Embedded Boundary"     ON)

#
# Misc options
#
option(PELELMEX_ENABLE_TINY_PROFILE "Enable AMReX tiny profiler" OFF)
option(PELELMEX_ENABLE_HDF5         "Enable HDF5 PlotFiles"      OFF)

#
# HPC options
#
option(PELELMEX_MPI    "Enable MPI"    OFF)
option(PELELMEX_OPENMP "Enable OpenMP" OFF)
option(PELELMEX_CUDA   "Enable CUDA"   OFF)
option(PELELMEX_HIP    "Enable HIP"    OFF)
option(PELELMEX_SYCL   "Enable SyCL"   OFF)

# C++ Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#
# Setup Languages
#
if (PELELMEX_CUDA)
   enable_language(CUDA)
endif ()
if (PELELMEX_HIP)
   enable_language(HIP)
endif ()

# Checks
if(PELELMEX_DIM EQUAL 1)
    message(FATAL_ERROR "PeleLMeX does not support 1D.")
endif()

#
# Cases
#
set(PELELMEX_EXEC_SUBDIRS Exec/)
list(TRANSFORM PELELMEX_EXEC_SUBDIRS PREPEND "${CMAKE_CURRENT_LIST_DIR}/")
set(EXEC_DIR "")
foreach(subdir ${PELELMEX_EXEC_SUBDIRS})
    FILE(GLOB caselist ${subdir}/*)
    foreach(case ${caselist})
        if(IS_DIRECTORY ${case})
            get_filename_component( case_dir ${case} NAME)
            LIST(APPEND PELELMEX_CASE_LIST ${case_dir})
            if( ${PELELMEX_CASE} STREQUAL ${case_dir})
                set(EXEC_DIR ${case})
            endif()
        endif()
    endforeach()
endforeach()
if (NOT EXEC_DIR)
    message(FATAL_ERROR "PELELMEX_CASE=${PELELMEX_CASE} is not a valid case !")
endif()

########################### PeleLMeX ##################################
add_executable(pelelmex "")
set_target_properties(pelelmex PROPERTIES OUTPUT_NAME "PeleLMeX${PELELMEX_DIM}d.exe")

set(SRC_DIR ${CMAKE_SOURCE_DIR}/Source)
add_subdirectory(${SRC_DIR})
add_subdirectory(${EXEC_DIR})
include(SetRpath)

if ( NOT CMAKE_CXX_FLAGS )
  target_link_libraries(pelelmex PUBLIC AMReX::Flags_CXX)
endif ()
target_link_libraries(pelelmex PUBLIC amrex)
########################### AMReX #####################################

message(STATUS "AMReX Configure Section")
set(AMREX_SUBMOD_LOCATION "${CMAKE_SOURCE_DIR}/Submodules/amrex")
include(SetAMReXOptions)
list(APPEND CMAKE_MODULE_PATH "${AMREX_SUBMOD_LOCATION}/Tools/CMake")
add_subdirectory(${AMREX_SUBMOD_LOCATION})
include(AMReXBuildInfo)
generate_buildinfo(pelelmex ${CMAKE_SOURCE_DIR})
target_include_directories(pelelmex SYSTEM PUBLIC ${AMREX_SUBMOD_LOCATION}/Tools/C_scripts)

########################### Summary ##################################
include(AMReXGenexHelpers)
include(AMReXTargetHelpers)

get_target_properties_flattened(pelelmex  _includes _defines _flags _link_line)
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