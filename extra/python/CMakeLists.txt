project(neosonar-neo-python LANGUAGES CXX VERSION ${CMAKE_PROJECT_VERSION})

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_neo src/main.cpp)
target_link_libraries(_neo PRIVATE pybind11::headers neosonar::neo)
target_compile_definitions(_neo PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS _neo LIBRARY DESTINATION neo)
