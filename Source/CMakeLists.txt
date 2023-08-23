target_include_directories(ebamr PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

target_sources(ebamr
    PRIVATE
        main.cpp
        LevelBld.cpp
        EBR.H
        EBR.cpp
        IndexDefines.H
        Constants.H
        Constants.cpp
        Tagging.H
        Tagging.cpp
        Setup.cpp
        Geometry.cpp
        IO.cpp
        Advance.cpp
)

add_subdirectory(kernel)
add_subdirectory(EB)
if (OPTION_USE_CHEM)
    add_subdirectory(CHEM)
endif ()