#include "./complex_numbers.h"
#include "./tensors.h"
#include "./integer_partitions.h"

namespace iparts {
	////////////////////////////////////////////////////////////////////
	// Members of the integer_partitions class

		// Actual calculations
		vector<vector<int>> integer_partitions::integer_partitions_generate(int n){
			// Generate all the integer partitions of n, withaout restriction
				
			vector<vector<int>> p;
			vector<int> a(n+1, 0);
				
			int k = 1;
			int x, y = n - 1;
			int l;
				
			while(k != 0){
				x = a[k - 1] + 1;
				k -= 1;
				while(2 * x <= y){
					a[k] = x;
					y   -= x;
					k   += 1;
				}
				l = k + 1;
				while(x <= y){
					a[k] = x;
					a[l]= y;
					
					vector<int> sub = extract(a, 0, k+1);
					p.push_back(sub);	
						
					x += 1;
					y -= 1;
				}
				a[k] = x + y;
				y    = x + y - 1;
					
				vector<int> sub = extract(a, 0, k);  	
				p.push_back(sub);
		   }
			   
		   return p;
		}

		vector<int> integer_partitions::occurrences_count(vector<int> &p, int min, int max){
			// Counts how many times a certain element appears in a given partition
			// min = smallest element appearing
			// max = largest element appearing
			
			vector<int> occurrences(max-min+1, 0);

			for(int i=0; i<p.size(); i++){
				occurrences[p[i]-min] ++;
			}
				
			return occurrences;
		}

		void integer_partitions::reorder(vector<partition_element> &v){
			// Reorders a partition using a bubble sort algorithm. The sorting criterios is increasing partition number repetition
				
			for(int i=0; i<v.size(); i++){
				for(int j=0; j<v.size()-i-1; j++){
					if(v[j].repetitions > v[j+1].repetitions){
						partition_element tmp = v[j+1];
						v[j+1] = v[j];
						v[j] = tmp;
					}
				}
			}
				
			return;
		}
		
		vector<vector<integer_partitions::partition_element>> integer_partitions::group_N_particles_restriced_partitions(vector<vector<int>> &p){
			// Filters out a certain number of partitions - not all the symmetric polynomials are linearly independent if there are only N particles in the system			
			// Lambdas for extracting smallest and largest elements
			auto smallest = []( vector<int> &v ){
				int min = v[0];
				
				for(int i=1; i<v.size(); i++){
					if		(v[i] < min) min = v[i];
				}
					
				return min;
			};

			auto largest = []( vector<int> &v ){
				int max = v[0];
				
				for(int i=1; i<v.size(); i++){
					if     (v[i] > max) max = v[i];
				}
					
				return max;
			};
			
			
			// Compute the size of the regrouped partitions
			int size = 0;	
			for(int i=0; i<p.size(); i++){
				if(p[i].size() <= partition_cutoff){
					size++;
				}
			}
			
			
			// Fill them	
			vector<vector<partition_element>> regrouped_partitions(size);
			int index = 0;
			
			for(int i=0; i<p.size(); i++){	
				if(p[i].size() <= partition_cutoff){
					int min = smallest(p[i]);
					int max = largest(p[i]);
							
					vector<partition_element> v;
					v.reserve(max-min+1);
							
					vector<int> occurrences = occurrences_count(p[i], min, max);
								
					for(int m0=min; m0<=max; m0++){
						if(occurrences[m0-min] != 0){
							partition_element elem = {m0, occurrences[m0-min]};
							v.push_back(elem);
						}
					}
						
					reorder(v);
						
					regrouped_partitions[index++] = v;
				}
			}	
			
			return regrouped_partitions;
		}

		void integer_partitions::partitions_initialize(vector<vector<vector<partition_element>>> &partitions){
			// Here the partitions are initialized
			
			// Some lambdas
			auto largest_element = [](vector<vector<partition_element>> &p){
				int M = 0;
				
				for(int i=0; i<p.size(); i++){
					for(int j=0; j<p[i].size(); j++){
						if(p[i][j].number > M){
								M = p[i][j].number;
						}
					}
				}
				
				return M;
			};

			auto max_between = [](int a, int b){
				if(a>b){
					return a;
				}
				return b;
			};
						
			// Initialization
			subspaceDimension = 0; // Dimension of the subspace in which the Hamiltonian matrix elements are computed 
			maximalDegree = 0; // Maximal polynomial degree
			
			partitions.reserve(dM_list.size());
			
			for(int j=0; j<dM_list.size(); j++){
				vector<vector<int>> raw_partitions = integer_partitions_generate( dM_list[j] );
				partitions[j] = group_N_particles_restriced_partitions(raw_partitions);
				
				subspaceDimension += partitions[j].size();		
				maximalDegree = max_between(maximalDegree, largest_element(partitions[j]));
			}
			
			return;
		}

		vector<vector<vector<integer_partitions::partition_element>>> integer_partitions::createVectorPartitionsOnHost(){	
			vector<vector<vector<integer_partitions::partition_element>>> partitions;
			partitions_initialize(partitions);

			/*
			// Printing from host
			for(int j=0; j<dM_list.size(); j++){
				printf("Number of elements in the %d partition (cutoff at %d elements)\t # = %d\n", dM_list[j], partition_cutoff, partitions[j].size());
				print_partitions(partitions[j]);
			}
			//*/
				
			return partitions;
		}
			
		integer_partitions::partition* integer_partitions::createPartitionsOnHost(vector<vector<vector<integer_partitions::partition_element>>>& partitions){
			partition* host_partitions = new partition[subspaceDimension];
			
			longestPartitionLenght = 0;
			
			int k0 = 0;
			for(int j=0; j<dM_list.size(); j++){
				for(int i=0; i<partitions[j].size(); i++){
					host_partitions[k0].dM = dM_list[j];
					host_partitions[k0].subspaceSize = partitions[j].size();
					host_partitions[k0].size = partitions[j][i].size();

					if( partitions[j][i].size() > longestPartitionLenght ){
						longestPartitionLenght = partitions[j][i].size();
					}
			
					host_partitions[k0].partitionArray = new partition_element[partitions[j][i].size()];
					for(int k=0; k<partitions[j][i].size(); k++){
						host_partitions[k0].partitionArray[k].number = partitions[j][i][k].number;
						host_partitions[k0].partitionArray[k].repetitions = partitions[j][i][k].repetitions;
					}
					
					k0++;
				}
			}
			
			return host_partitions;
		}

		// Copy to the GPU
		void integer_partitions::partitions_host2device(integer_partitions::partition *host_partitions, integer_partitions::partition *&device_partitions){
			// Allocate temporary array of partitions
			partition* tmp_partitions = new integer_partitions::partition[subspaceDimension];

			// For each partition, allocate device memory for its partitionArray
			for(int k0 = 0; k0 < subspaceDimension; k0++) {
				tmp_partitions[k0].dM = host_partitions[k0].dM;
				tmp_partitions[k0].size = host_partitions[k0].size;
				tmp_partitions[k0].subspaceSize = host_partitions[k0].subspaceSize;

				gpuErrchk( cudaMalloc(&tmp_partitions[k0].partitionArray, sizeof(partition_element) * host_partitions[k0].size) );
				gpuErrchk( cudaMemcpy(tmp_partitions[k0].partitionArray, host_partitions[k0].partitionArray, sizeof(partition_element) * host_partitions[k0].size, cudaMemcpyHostToDevice) );
			}
			
			// Allocate device array of partitions and copy
			gpuErrchk( cudaMalloc(&device_partitions, subspaceDimension * sizeof(partition)) );
			gpuErrchk( cudaMemcpy(device_partitions, tmp_partitions, subspaceDimension * sizeof(partition), cudaMemcpyHostToDevice) );

			// Free temporary mixed host-device array of structs (do not free the partitionArray-s yet)
			delete[] tmp_partitions;
			
			return;
		}

		// Printing stuff
		void integer_partitions::print_raw_partitions(vector<vector<int>> &raw_partitions){
			// Partitions printing
			for(int i=0; i<raw_partitions.size(); i++){
				for(int j=0; j<raw_partitions[i].size(); j++){
					printf("%d ", raw_partitions[i][j]);
				}
				printf("\n");
			}
			printf("\n");
					
			return;	
		}

		void integer_partitions::printEdgePartition(int subspace_row){
			for(int k=0; k<host_partitions[subspace_row].size; k++){
				printf("%d(x%d) ", host_partitions[subspace_row].partitionArray[k].number, host_partitions[subspace_row].partitionArray[k].repetitions);
			}
						  
			return;
		}

		void integer_partitions::print_partitions(vector<vector<partition_element>> &regrouped_partitions){
			// Partitions printing		
			for(int i=0; i<regrouped_partitions.size(); i++){
				for(int j=0; j<regrouped_partitions[i].size(); j++){
					printf("%d(x%d)\t", regrouped_partitions[i][j].number, regrouped_partitions[i][j].repetitions);
				}
				printf("\n");
			}
			printf("\n\n");	
				
			return;
		}	

		__global__ void printPartitions(integer_partitions::partition* partitions, int subspace_dim){
			// Printing partitions from the GPU
			int j0=-1;
			
			for(int k0=0; k0<subspace_dim; k0++){
				if(j0 != partitions[k0].dM){
					j0 = partitions[k0].dM;		
					printf("\n\tAngular momentum: dM=%d -- Edgespace dimension: %d\n\t\t", partitions[k0].dM, partitions[k0].subspaceSize);
				}
				printf("P(%d):\t", k0);
				for(int k=0; k<partitions[k0].size; k++){
					printf("%d(x%d)\t", partitions[k0].partitionArray[k].number, partitions[k0].partitionArray[k].repetitions);
				}
				printf("\n\t\t");
			}
			
			printf("\nTotal dimension of the edge subspace: %d\n\n", subspace_dim);
			
			return;
		}

		__global__ void print_row_info(integer_partitions::partition* device_partitions, int subspace_row, int subspace_dim){
			// Printing partitions from the GPU
			printf("\nDiagonalizing the row associated with the  ");
			for(int k=0; k<device_partitions[subspace_row].size; k++){
				printf("%d(x%d) ", device_partitions[subspace_row].partitionArray[k].number, device_partitions[subspace_row].partitionArray[k].repetitions);
			}			  
			printf("partition (%d out of %d)\n", subspace_row+1, subspace_dim);
						  
			return;
		}

		// Initialization
		integer_partitions::integer_partitions(){
			return;
		}

		integer_partitions::integer_partitions(int particles_number, vector<int> &angular_momenta_list){
			// Copy stuff to class members
			dM_list = angular_momenta_list;
			partition_cutoff = particles_number;
			
			// Initialize
			std::cout << "Generating the partitions and moving them on the GPU..." << std::endl;	
			vector<vector<vector<partition_element>>> vectorPartitionsOnHost = createVectorPartitionsOnHost(); // The partitions (with repetions -- i.e. uncompressed) are computed on the CPU
			host_partitions = createPartitionsOnHost(vectorPartitionsOnHost); // Rearranged in a compressed way (e.g. [1,1,1,1,2,2] ---> [1x4,2x2])

			partitions_host2device(host_partitions, device_partitions); // Copy the partitions to the gpu
		
			std::cout << "Printing the partitions (on GPU). The cutoff is at " << partition_cutoff << "...\n";
			printPartitions<<<1,1>>>( device_partitions, subspaceDimension );
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			
			return;
		}

		integer_partitions::~integer_partitions(){
			// The destructor. Free up all the used memory
			
			for(int i=0; i<subspaceDimension; i++){
				if(host_partitions[i].partitionArray){
					delete[] host_partitions[i].partitionArray;
				}
				
				if(device_partitions) {
					partition p;
					gpuErrchk( cudaMemcpy(&p, &device_partitions[i], sizeof(partition), cudaMemcpyDeviceToHost) ); // Get the pointer on the host
					gpuErrchk( cudaFree(p.partitionArray) );
				}
			}
			delete[] host_partitions;
			gpuErrchk( cudaFree(device_partitions) );
			
			return;
		}

		// Getters
		integer_partitions::partition* integer_partitions::getDevPartitions() const{
			return device_partitions;
		}
		
		integer_partitions::partition* integer_partitions::getHostPartitions() const{
			return host_partitions;
		}

		// Get a vector of angular momenta: returns the L associated to the i-th partition
		vector<int> integer_partitions::getAngularMomenta() const{
			vector<int> angular_momenta(subspaceDimension);

			for(int state=0; state<subspaceDimension; state++){
				angular_momenta[state] = host_partitions[state].dM;
			}

			return angular_momenta;
		}

		
	////////////////////////////////////////////////////////////////////

}
