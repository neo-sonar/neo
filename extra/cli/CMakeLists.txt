project(neosonar-neo-cli)

FetchContent_Declare(dr_libs GIT_REPOSITORY "https://github.com/mackron/dr_libs" GIT_TAG "master" GIT_SHALLOW TRUE)
FetchContent_Declare(fmt GIT_REPOSITORY "https://github.com/fmtlib/fmt" GIT_TAG "11.2.0" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(dr_libs fmt)

add_library(neo.wav)
add_library(neo::wav ALIAS neo.wav)
target_sources(neo.wav PRIVATE "src/wav.cpp" "src/wav.hpp")
target_include_directories(neo.wav PUBLIC ${dr_libs_SOURCE_DIR})
target_link_libraries(neo.wav
    PUBLIC
        fmt::fmt-header-only
        neosonar::neo
        neosonar::compiler_warnings
)

add_executable(neo-cli)
target_sources(neo-cli PRIVATE "src/convolver.cpp")
target_link_libraries(neo-cli PRIVATE neo::wav)
