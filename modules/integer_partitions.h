// Here the partitions of the integers [dM_1, dM_2, ..., dM_n] are generated
// These are used to generate the symmetric polynomials
// e.g. [0,2,4] ---> partitions of 0 -> {{}} (just the Laughlin state)
//					 partitions of 2 -> {{2}, {1,1}}
//					 partitions of 4 -> {{4}, {3,1}, {2,2}, {2,1,1}, {1,1,1,1}}

#ifndef IPARTS_H
#define IPARTS_H

#include "./complex_numbers.h"
#include "./tensors.h"	

namespace iparts{	

	// Check for CUDA errors
	#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(const cudaError_t &code, const char *file, const int &line, bool abort=false){
		if(code != cudaSuccess){
			fprintf(stderr, "\n\n\nGPUassert - CUDA returned error code: %s %s %d (error code %d)\n\n\n", cudaGetErrorString(code), file, line, code);
			if (abort) exit(code);
		}
	}	

	// Integer partitions
	class integer_partitions{
		public:
			// Personalized types for the partition (useful to compress it)
			// Structures
			struct partition_element{
				// Encodes the elements of a partition in compressed form
				// e.g. 12 contains [2,2,2,2,2,2], but can be compressed as (2, 6)=(num, rep) i.e. the number 2 is repeated 6 times
				// This corresponds to the PSSP (\sum_i z_i^num)^rep
				
				int number;
				int repetitions;
			};

			struct partition{
				// Wraps an array of all the partitions, together with ΔM (the associated angular momentum) and the number of partitions contained

				int dM; // Angular momentum carried by the partition
				int subspaceSize; // Reminder for the size of the subspace
				
				int size; // Number of elements in the partition vector
				partition_element* partitionArray; // Vector containing all the (distingushable) elements of a given partition andb the number of times each element appears
			};
		
			
			// Total dimension of the edge Hilbert space, and largest degree of the edge polynomial appearing
			int subspaceDimension;
			int maximalDegree;
			int longestPartitionLenght;
			
			// Initialization and destructor
			integer_partitions();
			integer_partitions(int particles_number, vector<int> &angular_momenta_list);
			
			~integer_partitions();
			
			// Printing stuff
			void print_raw_partitions(vector<vector<int>> &raw_partitions);
			void printEdgePartition(int subspace_row);
			void print_partitions(vector<vector<partition_element>> &regrouped_partitions);

			// Get the pointers
			partition* getDevPartitions() const;
			partition* getHostPartitions() const;

			// Get a vector of angular momenta: returns the L associated to the i-th partition
			vector<int> getAngularMomenta() const;

		private:
			vector<int> dM_list; // List of the desired partitions
			int partition_cutoff; // Cutoff in the number of partitions
			
			// Pointers to the partitions
			partition* host_partitions; // Only host
			partition* device_partitions; // Only device

			// Some functions to fill the partitions in
			vector<vector<int>> integer_partitions_generate(int n); // Generate the partitions of n, without restrictions
			vector<int> occurrences_count(vector<int> &p, int min, int max); // Counts how many times a certain element appears in a given partition
			void reorder(vector<partition_element> &v); // Reorders the partitions
			vector<vector<partition_element>> group_N_particles_restriced_partitions(vector<vector<int>> &p); // filters out certain partitions - not all of them are linearly independent
			
			// Copy the partitions to the GPU
			void partitions_host2device(integer_partitions::partition *host_partitions, integer_partitions::partition *&device_partitions);
			
			// Initialization of the partitions
			void partitions_initialize(vector<vector<vector<partition_element>>> &partitions);
			vector<vector<vector<partition_element>>> createVectorPartitionsOnHost();
			partition* createPartitionsOnHost(vector<vector<vector<partition_element>>>& partitions);
	};
	
	// Printing from the device (global functions cannot be member functions)
	__global__ void printPartitions(integer_partitions::partition* partitions, int subspace_dim);
	__global__ void print_row_info(integer_partitions::partition* device_partitions, int subspace_row, int subspace_dim);
		
}

#endif
