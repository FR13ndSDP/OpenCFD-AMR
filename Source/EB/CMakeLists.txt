target_include_directories(ebamr PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

target_sources(ebamr
     PRIVATE
          AdvanceBox.cpp
          EBkernels.H
          EBdiffusion.H
)