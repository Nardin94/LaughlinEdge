#!/bin/bash
set -e

echo "Loading CUDA 12.4.0..."
module load cuda/12.4.0

OUTPUT_BIN="LaughlinEdge"
ARCH_FLAG="sm_86"

# Find source files while excluding eigen-3.4, build directories, and the
# standalone system_parameters.cpp utility (has its own main(), generates
# modules/sys_params.h, and is not part of the main executable)
SOURCES=$(find . \( -name "*.cpp" -o -name "*.cu" \) -not -path "*/eigen-3.4/*" -not -path "*/build/*" -not -name "system_parameters.cpp")

if [ -z "$SOURCES" ]; then
    echo "Error: No source files found."
    exit 1
fi

echo "Compiling sources:"
echo "$SOURCES"

# Added -rdc=true to allow CUDA device-code linking across multiple .cu files
nvcc -O3 -std=c++17 -arch=$ARCH_FLAG -rdc=true -x cu -DFMT_HEADER_ONLY -I ./eigen-3.4 $SOURCES -o "$OUTPUT_BIN"

echo "Compilation successful! Executable created: ./$OUTPUT_BIN"
