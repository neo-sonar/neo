project(neosonar.neo-cli)

FetchContent_Declare(dr_libs GIT_REPOSITORY "https://github.com/mackron/dr_libs" GIT_TAG "master" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(dr_libs)

add_library(neosonar.wav)
add_library(neosonar::wav ALIAS neosonar.wav)
target_sources(neosonar.wav PRIVATE "src/wav.cpp" "src/wav.hpp")
target_include_directories(neosonar.wav PUBLIC ${dr_libs_SOURCE_DIR})
target_link_libraries(neosonar.wav PUBLIC neosonar::neo neosonar::compiler_warnings)

add_executable(neo-cli)
target_sources(neo-cli PRIVATE "src/main.cpp")
target_link_libraries(neo-cli PRIVATE neosonar::wav)
