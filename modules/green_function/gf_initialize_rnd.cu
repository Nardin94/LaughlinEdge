#include "../green_function.h"

namespace greenFunctionMC {

	unsigned long long makeSeed(){
		std::random_device rd;

		// build a 64-bit seed. 
		// Cast the 32-bit rd() to 64-bit: (unsigned long long)rd() --> [00000000|rd1]
		// [00000000|rd1] << 32 --> [rd1|00000000]
		// | rd() puts a second draw into the lower half: [rd1|00000000] --> [rd1|rd2]
		unsigned long long s = ((unsigned long long)rd() << 32) | rd(); 
		
		// Take the current time with high resolution
		// cast to unsigned long long
		// XOR into s (mixes in the current time, and since it is reversible it does not reduce entropy)
		s ^= (unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count();
		
		// getpid() returns the process id. Multiplying by 0x9e3779b97f4a7c15 spreads the pid allover the 64 bits
		s ^= (unsigned long long)getpid() * 0x9e3779b97f4a7c15ULL;

		return s;
	}

	__global__ void initializeRandom(curandState *state, unsigned long long seed){
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		curand_init(seed, tid, 0, &state[tid]);
		return;
	}
}
