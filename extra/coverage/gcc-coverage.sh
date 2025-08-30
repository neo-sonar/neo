#!/bin/bash
set -euxo pipefail

export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
: "${GCOV:=gcov}"

export CXXFLAGS="-fprofile-arcs -ftest-coverage -march=native"
export CMAKE_BUILD_TYPE="Debug"
export CMAKE_GENERATOR="Ninja"

BUILD_DIR=cmake-build-coverage-gcc
BASE_DIR=$PWD

# Clean up old build
rm -rf "$BUILD_DIR"

# Configure
cmake -S . -B "$BUILD_DIR" -D NEO_ENABLE_INTEL_IPP=ON -D NEO_ENABLE_INTEL_MKL=ON -D NEO_ENABLE_BENCHMARKS=OFF -D NEO_ENABLE_TOOLS=OFF -D NEO_ENABLE_PLUGIN=OFF

# Build
cmake --build "$BUILD_DIR" --target neo-tests

# Enter build directory
cd "$BUILD_DIR"

# Clean-up counters for any previous run.
lcov --zerocounters --directory .

# Run tests
ctest --test-dir . -C Debug --output-on-failure -j $(nproc)

# Create coverage report by taking into account only the files contained in src/
lcov --ignore-errors mismatch --capture --directory . -o coverage.info --include "$BASE_DIR/src/*" --exclude "*_test.cpp" --gcov-tool $GCOV

# Create HTML report in the out/ directory
genhtml coverage.info --output-directory out

# Show coverage report to the terminal
lcov --list coverage.info
