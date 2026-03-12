#include "./edge_montecarlo.h"

namespace edgeMC {

	__global__ void initializeRandom(curandState *state, uint seed){
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		curand_init(seed, tid, 0, &state[tid]);
		return;
	}
}
