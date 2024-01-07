#!/bin/bash
set -euxo pipefail

# Record the base directory
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
: "${GCOV:=gcov}"

BASE_DIR=$PWD
BUILD_DIR=cmake-build-coverage

# Clean up old build
# rm -rf "$BUILD_DIR"

# Configure
cmake -S . -B "$BUILD_DIR" -G Ninja -D CMAKE_BUILD_TYPE=Debug -D CMAKE_CXX_FLAGS="-fprofile-arcs -ftest-coverage -march=native" -D NEO_ENABLE_INTEL_IPP=ON -D NEO_ENABLE_INTEL_MKL=ON

# Build
cmake --build "$BUILD_DIR" --target neosonar-neo-tests

# Enter build directory
cd "$BUILD_DIR"

# Clean-up counters for any previous run.
lcov --zerocounters --directory .

# Run tests
ctest --test-dir . -C Debug --output-on-failure -j 4

# Create coverage report by taking into account only the files contained in src/
lcov --ignore-errors mismatch --capture --directory . -o coverage.info --include "$BASE_DIR/src/*" --exclude "*_test.cpp" --gcov-tool $GCOV

# Create HTML report in the out/ directory
genhtml coverage.info --output-directory out

# Show coverage report to the terminal
lcov --list coverage.info
