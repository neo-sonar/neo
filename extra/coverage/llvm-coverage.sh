#!/bin/bash

set -euxo pipefail

BASE_DIR=$PWD
BUILD_DIR=cmake-build-coverage-clang

export CC="clang-20"
export CXX="clang++-20"

export CXXFLAGS="-march=native -stdlib=libc++ -fprofile-instr-generate -fcoverage-mapping"
export CMAKE_BUILD_TYPE="Debug"
export CMAKE_GENERATOR="Ninja"
export LLVM_PROFILE_FILE="$BUILD_DIR/raw/%p-%m.profraw"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/raw"

cmake -S . -B "$BUILD_DIR" -D NEO_ENABLE_INTEL_IPP=ON -D NEO_ENABLE_INTEL_MKL=ON -D NEO_ENABLE_BENCHMARKS=OFF -D NEO_ENABLE_TOOLS=OFF -D NEO_ENABLE_PLUGIN=OFF
cmake --build "$BUILD_DIR" --target all
ctest --test-dir "$BUILD_DIR" --output-on-failure -j $(nproc)

llvm-profdata-20 merge -sparse $BUILD_DIR/raw/*.profraw -o "$BUILD_DIR/combined.profdata"
llvm-cov-20 show \
  -object $BUILD_DIR/test/neosonar-neo-tests \
  -instr-profile="$BUILD_DIR/combined.profdata" \
  -format=html \
  -output-dir="$BUILD_DIR/html" \
  -show-expansions \
  -show-line-counts-or-regions \
  -ignore-filename-regex='(^/usr/|/catch2/|/_deps/)'


llvm-cov-20 export \
  -object $BUILD_DIR/test/neosonar-neo-tests \
  -instr-profile="$BUILD_DIR/combined.profdata" \
  -format=lcov \
  -ignore-filename-regex='(^/usr/|/catch2/|/_deps/)' > "$BUILD_DIR/coverage.info"

genhtml $BUILD_DIR/coverage.info -o $BUILD_DIR/lcov-html --ignore-errors unmapped
