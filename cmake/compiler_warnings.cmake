add_library(neosonar.compiler_warnings INTERFACE)
add_library(neosonar::compiler_warnings ALIAS neosonar.compiler_warnings)

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(neosonar.compiler_warnings INTERFACE "/W4")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
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
elseif(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    target_compile_options(neosonar.compiler_warnings
        INTERFACE
            "-Weverything"

            "-Wno-c++20-compat"
            "-Wno-c++98-compat-pedantic"
            "-Wno-ctad-maybe-unsupported"
            "-Wno-disabled-macro-expansion"
            "-Wno-double-promotion"
            "-Wno-float-equal"
            "-Wno-padded"
            "-Wno-switch-default"
            "-Wno-switch-enum"
            "-Wno-unsafe-buffer-usage"
            "-Wno-unused-member-function"
    )
    if(WIN32)
        target_compile_options(neosonar.compiler_warnings
            INTERFACE
                "-Wno-unknown-attributes" # Warns for [[no_unique_address]] being ignored for ABI reasons
        )
    endif()
    if(EMSCRIPTEN)
        target_compile_options(neosonar.compiler_warnings
            INTERFACE
                "-Wno-implicit-int-conversion-on-negation" # xsimd
        )
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "22.0.0")
        target_compile_options(neosonar.compiler_warnings
            INTERFACE
                "-Wno-nrvo"
                "-Wno-unique-object-duplication" # xsimd
        )
    endif()
endif()
