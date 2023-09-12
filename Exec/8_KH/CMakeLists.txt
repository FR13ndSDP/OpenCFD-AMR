target_include_directories(ebamr PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

target_sources(ebamr
     PRIVATE
          prob.H
          prob.cpp
          prob_parm.H
          prob_bcfill.cpp
)