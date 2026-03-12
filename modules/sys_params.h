#ifndef SYSPARAMS_H
#define SYSPARAMS_H

namespace sys_params {
    __device__ constexpr int particlesNumber = 25;
    __device__ constexpr int inverseFilling = 2;

    __device__ constexpr int confinementExponent = 4;
    __device__ constexpr float confinementStrength = 1.000000;

    __device__ constexpr float RCl = 10.000000;
    __device__ constexpr float DCl = 20.000000;
    __device__ constexpr float RCl_reciprocal = 0.100000;
    __device__ constexpr float R0_reciprocal = 0.200000;

    __device__ constexpr int ansatzExtendedMaxDegree = 11;
    __device__ constexpr int ansatzMaxPartitionSize = 5;
    __device__ constexpr int ansatzSubspaceDimension = 50;

    __device__ constexpr int ansatzMultipletSize = 5;

}

#endif