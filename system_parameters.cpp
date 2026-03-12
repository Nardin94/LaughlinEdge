// Generates sys_params.h
// The file will contain all the relevant simulation parameters as constant expressions

#include <cmath>
#include <iomanip>
#include <fstream>

// System parameters
int particlesNumber = 25; // Number of particles
int inverseFilling = 2; // Filling factor ν = 1/m

int	confinementExponent = 4; // Confining potential exponent, VConf * |z / R_{cl}|^conf
double confinementStrength = 1.; // Strength parameter of the confining potential

// These are ansatz-values. In any realistic simulation they will not exceed these values. There is a bit of overhead in having them larger than the actual values - but the code will be much faster (avoids using global memory)
// If they are too-small, the mcedge code will complain at run-time.
int ansatzMaxDegree = 10; // With this we can arrive up to L=10
int ansatzMaxPartitionSize = 5; // The longest partition at a given L has K = floor[ (sqrt(1+8L)-1)/2 ] distinct elements. With K=4 we can cover all the L up to 15 (excluded)
int ansatzSubspaceDimension = 50; // The size of the edge Hilbert space should be smaller than this parameter

int ansatzMultipletSize = 5;

void generate_config_file(){
    std::ofstream out("./modules/sys_params.h");

    // 1. Compute stuff
	double RCl = std::sqrt(2.*inverseFilling*particlesNumber);
	double DCl = 2.*RCl;
	double RCl_reciprocal = 1. / RCl;
	double R0_reciprocal = 2./RCl;		

    // 2. 
    out << std::fixed << std::setprecision(6);
    out << "#ifndef SYSPARAMS_H\n";
    out << "#define SYSPARAMS_H\n\n";

    out << "namespace sys_params {\n";
    out << "    __device__ constexpr int particlesNumber = " << particlesNumber << ";\n";
    out << "    __device__ constexpr int inverseFilling = " << inverseFilling << ";\n\n";

    out << "    __device__ constexpr int confinementExponent = " << confinementExponent << ";\n";
    out << "    __device__ constexpr float confinementStrength = " << confinementStrength << ";\n\n";

    out << "    __device__ constexpr float RCl = " << RCl << ";\n";
    out << "    __device__ constexpr float DCl = " << DCl << ";\n";
    out << "    __device__ constexpr float RCl_reciprocal = " << RCl_reciprocal << ";\n";
    out << "    __device__ constexpr float R0_reciprocal = " << R0_reciprocal << ";\n\n";  

    out << "    __device__ constexpr int ansatzExtendedMaxDegree = " << ansatzMaxDegree + 1 << ";\n";
    out << "    __device__ constexpr int ansatzMaxPartitionSize = " << ansatzMaxPartitionSize << ";\n";
    out << "    __device__ constexpr int ansatzSubspaceDimension = " << ansatzSubspaceDimension << ";\n\n";

    out << "    __device__ constexpr int ansatzMultipletSize = " << ansatzMultipletSize << ";\n\n";

    out << "}\n\n";

    out << "#endif";


    return;
}

int main(){

    generate_config_file();

    return 0;
}