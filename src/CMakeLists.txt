project(neosonar.neo VERSION ${CMAKE_PROJECT_VERSION})

add_library(neosonar.neo INTERFACE)
add_library(neosonar::neo ALIAS neosonar.neo)

target_link_libraries(neosonar.neo INTERFACE mdspan::mdspan)
target_compile_features(neosonar.neo INTERFACE cxx_std_23)
target_compile_definitions(neosonar.neo INTERFACE MDSPAN_USE_BRACKET_OPERATOR=0 MDSPAN_USE_PAREN_OPERATOR=1)
target_include_directories(neosonar.neo INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(NEO_ENABLE_APPLE_ACCELERATE AND APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    target_link_libraries(neosonar.neo INTERFACE ${ACCELERATE_LIBRARY})
    target_compile_definitions(neosonar.neo INTERFACE NEO_HAS_APPLE_ACCELERATE=1)
    target_compile_definitions(neosonar.neo INTERFACE ACCELERATE_NEW_LAPACK=1 ACCELERATE_LAPACK_ILP64=1)
endif()

if(NEO_ENABLE_INTEL_IPP)
    find_package(IPP)
    if(IPP_FOUND)
        target_link_libraries(neosonar.neo INTERFACE ${IPP_LIBRARIES})
        target_compile_definitions(neosonar.neo INTERFACE NEO_HAS_INTEL_IPP=1)
    else()
        message(FATAL_ERROR "Intel IPP was not found")
    endif()
endif()

if(NEO_ENABLE_INTEL_MKL)
    find_package(MKL)
    if(MKL_FOUND)
        target_link_libraries(neosonar.neo INTERFACE MKL::MKL)
        target_compile_definitions(neosonar.neo INTERFACE NEO_HAS_INTEL_MKL=1)
    else()
        message(FATAL_ERROR "Intel MKL was not found")
    endif()
endif()

if(NEO_ENABLE_XSIMD)
    target_link_libraries(neosonar.neo INTERFACE xsimd)
    target_compile_definitions(neosonar.neo INTERFACE NEO_HAS_XSIMD=1)
endif()
