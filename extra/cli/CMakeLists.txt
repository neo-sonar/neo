project(neosonar-neo-cli)

FetchContent_Declare(dr_libs GIT_REPOSITORY "https://github.com/mackron/dr_libs" GIT_TAG "master" GIT_SHALLOW TRUE)
FetchContent_Declare(fmt GIT_REPOSITORY "https://github.com/fmtlib/fmt" GIT_TAG "11.2.0" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(dr_libs fmt)

add_executable(neo-cli)
target_sources(neo-cli
    PRIVATE
        "src/convolver.cpp"
        "src/wav.cpp"
        "src/wav.hpp"
)

target_include_directories(neo-cli
    PRIVATE
        ${dr_libs_SOURCE_DIR}
)

target_link_libraries(neo-cli
    PRIVATE
        neosonar::neo
        neosonar::compiler_warnings
        fmt::fmt-header-only
)
