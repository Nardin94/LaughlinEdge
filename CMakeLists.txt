cmake_minimum_required(VERSION 3.18)

project(LaughlinEdge
    VERSION 1.0
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

# Source files
set(SOURCES
    ./main.cu
    
    # Homemade headears implementing vectors, matrices and complex numbers
    ./modules/complex_numbers.h
    ./modules/tensors.h

    # System parameters as constexpr
    ./modules/sys_params.h

    # Integer partitions
    ./modules/integer_partitions.h
    ./modules/integer_partitions.cu

    # Monte Carlo
    ./modules/edge_montecarlo.h
    ./modules/em_initialize_rnd.cu # Initialization of cuRAND
    ./modules/em_markovchain.cu # Contains most of the definitions for the montecarlo class
    ./modules/em_parameters.cu # All the definitions of the parameters class
    ./modules/em_bisection.cu # Bisects to get the good acceptance
    ./modules/em_spectrum.cu # Contains spectrum calculation functions
    ./modules/em_dsf.cu # Contains edge dsf calculation functions
    ./modules/em_edgeResponse.tpp # Contains excitation matrix elements calculations
    
    # Generalized eigensolver
    ./modules/generalized_eigenproblem.h
    ./modules/generalized_eigenproblem.cpp

    # Dynamics
    ./modules/dynamics.h
    ./modules/dynamics.cpp
)

add_executable(LaughlinEdge ${SOURCES})

set_target_properties(LaughlinEdge PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# cudart and cuRAND, as well as fmt for string formatting (std::format is not cuda friendly)
find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)

target_link_libraries(LaughlinEdge PRIVATE 
    CUDA::cudart
    CUDA::curand
    fmt::fmt
)

# Host compiler and device flags (-Xptxas -O3)
target_compile_options(LaughlinEdge PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xptxas=-O3>
)
