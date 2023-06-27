target_include_directories(ebamr PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

target_sources(ebamr
     PRIVATE
          ebr_prob.H
          ebr_prob_parm.H
          ebr_prob.cpp
          ebr_bcfill.cpp
          ebr_prob_parm.cpp
)