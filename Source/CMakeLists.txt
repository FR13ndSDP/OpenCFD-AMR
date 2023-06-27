target_include_directories(ebamr PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

target_sources(ebamr
    PRIVATE
        main.cpp
        EBR.H
        EBR_index_macros.H
        EBR_parm.H
        EBR_tagging.H
        LevelBld.cpp
        EBR.cpp
        EBR_setup.cpp
        EBR_geom.cpp
        EBR_io.cpp
        EBR_advance.cpp
        EBR_tagging.cpp
        EBR_parm.cpp
)

add_subdirectory(kernel)