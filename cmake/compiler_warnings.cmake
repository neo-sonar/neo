add_library(neosonar.compiler_warnings INTERFACE)
add_library(neosonar::compiler_warnings ALIAS neosonar.compiler_warnings)

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(neosonar.compiler_warnings INTERFACE "/W4")
else ()
    target_compile_options(neosonar.compiler_warnings
        INTERFACE
            "-Wall"
            "-Wextra"
            "-Wpedantic"

            "-Wcast-align"
            "-Wconversion"
            "-Woverloaded-virtual"
            "-Wredundant-decls"
            "-Wreorder"
            "-Wshadow"
            "-Wsign-compare"
            "-Wsign-conversion"
            "-Wstrict-aliasing"
            "-Wswitch-enum"
            "-Wuninitialized"
            "-Wunreachable-code"
            "-Wunused-parameter"
            "-Wzero-as-null-pointer-constant"

            "-Wno-switch-enum"
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(WIN32)
            # Warns for [[no_unique_address]] being ignored for ABI reasons
            target_compile_options(neosonar.compiler_warnings INTERFACE "-Wno-unknown-attributes")
        endif()
        if(EMSCRIPTEN)
            # Warnings in xsimd
            target_compile_options(neosonar.compiler_warnings INTERFACE "-Wno-implicit-int-conversion-on-negation")
        endif()
    endif()
endif ()
