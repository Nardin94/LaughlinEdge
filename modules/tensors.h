#ifndef _TENSORS_
#define _TENSORS_

#include <stdio.h>
#include <complex.h>
#include <cuda.h>
#include <iostream>
#include <vector>

template<typename T>
class vector
{
private:
    int sz, cap;
    T* vec;

public:    
	// Vector constructor ...
	__host__ __device__
    vector(const int &input_size) : sz(input_size), cap(input_size) {
		vec = new T[input_size];
    }  

	__host__ __device__
    vector(const int &input_size, const T &x0) : sz(input_size), cap(input_size) {
		vec = new T[input_size];
		for(int i=0; i<input_size; i++){
			vec[i] = x0;
		}
    }

	__host__ __device__
    vector(){
		sz  = 0;
		cap = 0;
		vec = new T[0];
    }        

   __host__
	vector(std::initializer_list<T> list): sz(list.size()), cap(list.size()){
		vec = new T[sz];
		int i = 0;
		for (const T &item : list) {
			vec[i++] = item;
		}
	} 	

	vector(const vector<T> &other): sz(other.sz), cap(other.cap){
		vec = new T[cap];
		for(int i=0; i<sz; i++){
			vec[i] = other.vec[i];
		}
	}

	// ... and destructors  (look at https://www2.cs.sfu.ca/CourseCentral/125/tjd/vector_example.html)
	__host__ __device__
	~vector() {
		if(vec){
			//printf("out of scope\n");
			delete[] vec;
		}
	}
    
    
    // Accessors
    __host__ __device__
    T & operator[](const int &index){
		return vec[index];
    }
    
    __host__ __device__
    const T & operator[](const int &index) const{
		return vec[index];
    }


    // Overloading the assignment operator (=)
    __host__ __device__
	vector<T> & operator=(vector<T> const &new_vec){	
		if(new_vec.size() > cap){										// If the size of new_vec is larger than the capacity of vec, need to allocate some space. If the capacity is large enough,
			reserve(new_vec.size());									//  even if the size is not, there is no need to allocate new space
		}
		sz = new_vec.size();											// The size of vec is set to the same size of new_vec
																		// The content is copied
		for(int i=0; i<new_vec.size(); i++){
			vec[i] = new_vec[i];
		}
		
		return *this;													// Returns a reference to vec
	}	 		 		       


	// Returning the class members
	__host__ __device__
	int  size() const{
        return sz;
	}  
	
	__host__ __device__
	int capacity() const{
        return cap;
	}
	
	__host__ __device__
	T* vec_pointer() const{
        return vec;
	}


	// Changing the class members
	__host__ __device__
	void update(const int new_sz, const int new_cap, T* new_vec){
		sz  = new_sz;
		cap = new_cap;
		vec = new_vec;
		return;
    }
   
   
	// Swapping two vectors
	__host__ __device__
	friend void swap(vector<T> &v1, vector<T> &v2){
		int sz1   = v1.size();
		int cap1  = v1.capacity();
		T *vecP1   = v1.vec_pointer();
     
		v1.update(v2.size(), v2.capacity(), v2.vec_pointer());  
		v2.update(sz1, cap1, vecP1);
     
		return;
	}


	// Reserve the space
	__host__ __device__
	void reserve(const int &new_cap){		
		if(cap >= new_cap){
			return;
		}
		
		// Allocate a chunk of memory of the required size
		T *new_vec = new T[new_cap];
		
		// Copy the old vector onto the new one
		for(int k=0; k<sz; k++){
			new_vec[k] = vec[k];
		}
		
		// Update the capacity
		cap = new_cap;
		
		// Swap the two arrays
		T *tmp  = vec;
		vec     = new_vec;
		new_vec = tmp;
		
		// Delete the old one
		delete []new_vec;
		
		
		return;
	}
	
	
	// Resize the vector
	__host__ __device__
	void resize(const int new_sz){
		if(new_sz <= sz || new_sz <= cap){								// In these two cases just change the size parameter, for there is enough allocated space
			sz = new_sz;
		}
		else{															// In this case the vector has to be reallocated though
			reserve(new_sz);
			sz = new_sz;
		}
		
		return;
	}
	
	
	// Push and pop_back operations
	__host__ __device__
	void push_back(const T &x){
		if(sz == cap){
			reserve(2*cap+1);
		}
		vec[sz++] = x;
				
		return;
	}	

	__host__ __device__
	void push_back(vector<T> &x){
		int sizeX = x.size();
		
		if(sz + sizeX > cap){
			reserve(sz + sizeX);
		}
		
		for(int i=0; i<sizeX; i++){
			vec[sz+i] = x[i];
		}
		
		return;
	}

	__host__ __device__
	void pop_back(){
		if(sz>0) sz--;
		return;
	}

	// Convert to std::vector
	template<typename T2>
	__host__
    operator std::vector<T2>() const{
		std::vector<T2> tmp(vec, vec + sz);

        return tmp; 
    }	

	// Extract a portion of a vector
	__host__ __device__
	friend vector<T> extract(vector<T> &v, int b, int e){	
		vector<T> sub(e-b+1);
		
		for(int i=b; i<=e; i++){
			sub[i-b] = v[i];
		}
		
		return sub;
	}	
   
};

template <typename T>
class matrix
{
	private:
		int nRows;
		int nCols;
		vector<T> vec;
		
	public:
		// Vector constructors ...
		__host__ __device__
		matrix(const int &N, const int &M) : nRows(N), nCols(M), vec(N*M) {
		}

		__host__ __device__
		matrix() : nRows(0), nCols(0), vec(0) {
		}

		__host__ __device__
		matrix(const int &N, const int &M, const T &x0) : nRows(N), nCols(M), vec(N*M, x0) {
		}

		__host__ __device__
		matrix<T> & operator=(matrix<T> const &new_mat){
			nRows = new_mat.rows();
			nCols  = new_mat.cols();
			
			vec.resize(nRows * nCols);
			
			for(int i=0; i<nRows * nCols; i++){
				vec[i] = new_mat.vec[i];
			}
		
			return *this;										
		}	 				
		
		// Returning the content of the array
		__host__ __device__
		T & operator()(const int &row, const int &col){
			return vec[row * nCols + col];
		};
		
		__host__ __device__
		T operator()(const int &row, const int &col) const{
			return vec[row * nCols + col];
		};	
		
		// Returning dimensions
		__host__ __device__
		int rows() const
		{
			return nRows;
		}
		__host__ __device__
		int cols() const
		{
			return nCols;
		}		
		
		// Basic matrix operations
		__host__ __device__
		matrix<T>  transpose() const{
			matrix<T> u(nCols, nRows);
			
			for(int row=0; row<nCols; row++){
					for(int col=0; col<nRows; col++){
							u(row, col) = vec[col * nCols + row];
						}
			}
			
			return u;
		}
			
		__host__ __device__
		matrix<T> operator+(matrix<T> const &mat){	
			matrix<T> u(nRows, nCols);
						
			for(int i=0; i<nRows * nCols; i++){
				u.vec[i] = vec[i] + mat.vec[i];
			}
		
			return u;										
		}	
			
		__host__ __device__
		matrix<T> operator-(matrix<T> const &mat){	
			matrix<T> u(nRows, nCols);
						
			for(int i=0; i<nRows * nCols; i++){
				u.vec[i] = vec[i] - mat.vec[i];
			}
		
			return u;										
		}					
			
		__host__ __device__
		matrix<T> operator*(matrix<T> const &mat){	
			matrix<T> u(nRows, mat.cols());
						
			for(int row=0; row<nRows; row++){
				for(int col=0; col<mat.cols(); col++){
					u(row, col) = 0;
					for(int j=0; j<nCols; j++){
						u(row, col) += vec[row * nCols + j] *	mat.vec[j * mat.cols() + col];
					}
					
				}
			}
		
			return u;										
		}								
			
};

template <typename T>
class cmatrix
{
	private:
		int nRows;
		int nCols;
		vector<complex<T>> vec;
		
	public:
		// Vector constructors ...
		__host__ __device__
		cmatrix(const int &N, const int &M) : nRows(N), nCols(M), vec(N*M) {
		}

		__host__ __device__
		cmatrix() : nRows(0), nCols(0), vec(0) {
		}

		__host__ __device__
		cmatrix(const int &N, const int &M, const T &x0) : nRows(N), nCols(M), vec(N*M, x0) {
		}

		__host__ __device__
		cmatrix<T> & operator=(cmatrix<T> const &new_mat){
			nRows = new_mat.rows();
			nCols  = new_mat.cols();
			
			vec.resize(nRows * nCols);
			
			for(int i=0; i<nRows * nCols; i++){
				vec[i] = new_mat.vec[i];
			}
		
			return *this;										
		}	 				
		
		// Returning the content of the array
		__host__ __device__
		complex<T> & operator()(const int &row, const int &col){
			return vec[row * nCols + col];
		};
		
		__host__ __device__
		complex<T> operator()(const int &row, const int &col) const{
			return vec[row * nCols + col];
		};	
		
		// Returning dimensions
		__host__ __device__
		int rows() const
		{
			return nRows;
		}
		__host__ __device__
		int cols() const
		{
			return nCols;
		}		
		
		// Basic matrix operations
		__host__ __device__
		cmatrix<T>  transpose() const{
			cmatrix<T> u(nCols, nRows);
			
			for(int row=0; row<nCols; row++){
					for(int col=0; col<nRows; col++){
							u(row, col) = vec[col * nCols + row];
					}
			}
			
			return u;
		}
	
		__host__ __device__
		cmatrix<T>  conj() const{
			cmatrix<T> u(nRows, nCols);
			
			for(int row=0; row<nRows; row++){
					for(int col=0; col<nCols; col++){
							u(row, col) = vec[row * nCols + col].conj();
						}
			}
			
			return u;
		}		
		
		__host__ __device__
		cmatrix<T> adjoint() const{
			cmatrix<T> u(nCols, nRows);
			
			for(int row=0; row<nCols; row++){
					for(int col=0; col<nRows; col++){
							u(row, col) = vec[col * nCols + row].conj();
						}
			}
			
			return u;
		}		
		
		// Overloading matrix operations	
		__host__ __device__
		cmatrix<T> operator+(cmatrix<T> const &mat){	
			cmatrix<T> u(nRows, nCols);
						
			for(int i=0; i<nRows * nCols; i++){
				u.vec[i] = vec[i] + mat.vec[i];
			}
		
			return u;										
		}	
			
		__host__ __device__
		cmatrix<T> operator-(cmatrix<T> const &mat){	
			cmatrix<T> u(nRows, nCols);
						
			for(int i=0; i<nRows * nCols; i++){
				u.vec[i] = vec[i] - mat.vec[i];
			}
		
			return u;										
		}					
			
		__host__ __device__
		cmatrix<T>  operator*(cmatrix<T>  const &mat){	
			cmatrix<T>  u(nRows, mat.cols());
						
			for(int row=0; row<nRows; row++){
				for(int col=0; col<mat.cols(); col++){
					u(row, col) = 0;
					for(int j=0; j<nCols; j++){
						u(row, col) += vec[row * nCols + j] *	mat.vec[j * mat.cols() + col];
					}
					
				}
			}
		
			return u;										
		}
		
		
		__host__ __device__
		cmatrix<T> operator*(const T& x) const{
			cmatrix<T>  u(nRows, nCols);
						
			for(int row=0; row<nRows; row++){
				for(int col=0; col<nCols; col++){
						u(row, col) = vec[row * nCols + col] * x;					
				}
			}
		
			return u;		
		}		
	
		__host__ __device__
		cmatrix<T> operator*(const complex<T>& z) const{
  			cmatrix<T> u(nRows, nCols);

			for(int row=0; row<nRows; row++){
				for(int col=0; col<nCols; col++){
					u(row, col) = vec[row * nCols + col] * z;
				}
			}
			
			return u;
		}

		__host__ __device__
		cmatrix<T> operator/(const T& x) const{
			cmatrix<T>  u(nRows, nCols);
						
			for(int row=0; row<nRows; row++){
				for(int col=0; col<nCols; col++){
						u(row, col) = vec[row * nCols + col] / x;					
				}
			}
		
			return u;		
		}									
};

template <typename T>
class rank3tensor
{
	private:
		int n1, n2, n3;
		vector<T> vec;
		
	public:
		// Vector constructors ...
		__host__ __device__
		rank3tensor(const int &N1, const int &N2, const int &N3) : n1(N1), n2(N2), n3(N3), vec(N1*N2*N3) {
		}	

		__host__ __device__
		rank3tensor(const int &N1, const int &N2, const int &N3, const T &x0) : n1(N1), n2(N2), n3(N3), vec(N1*N2*N3, x0) {
		}	


		// Returning the content of the array
		__host__ __device__
		T & operator()(const int &i1, const int &i2, const int &i3){
			return vec[i1 + n1 * (i2 + n2 * i3)];
		};
		
		
		__host__ __device__
		T operator()(const int &i1, const int &i2, const int &i3) const{
			return vec[i1 + n1 * (i2 + n2 * i3)];
		};	
		
		// Returning dimensions
		__host__ __device__
		int s1() const
		{
			return n1;
		}
		__host__ __device__
		int s2() const
		{
			return n2;
		}
		__host__ __device__
		int s3() const
		{
			return n3;
		}			
};

template <typename T>
class vector_wrapper
{
private:
    int sz;
    T* vec;

public:
	__host__ __device__
    vector_wrapper(T* pVec, const int &input_size){
		sz    = input_size;
		vec = pVec;
    }     
    
    
    __host__ __device__
    T & operator[](const int &index){
		return vec[index];
    }
    
    __host__ __device__
    const T & operator[](const int &index) const{
		return vec[index];
    }



	__host__ __device__
	int  size() const{
        return sz;
	}  
	
	__host__ __device__
	T* vec_pointer() const{
        return vec;
	}
   
};

template <typename T, int rk>
class tensor_wrapper
{
private:
    int sz[rk];
    T* vec;

public:
	__host__ __device__
    tensor_wrapper(T* pVec, const int (&input_sizes)[rk]){
		for(int i=0; i<rk; i++){
			sz[i]    = input_sizes[i];
		}
		
		vec = pVec;
    }



    __host__ __device__
    T & operator[](const int (&indices)[rk]){
		int index;
		
		index = indices[0]; 
		for(int i=1; i<rk; i++){
			index  *= sz[i];
			index += indices[i];
		}
		
		return vec[index];
    }
    
    __host__ __device__
    const T & operator[](const int (&indices)[rk]) const{
		int index;
		
		index = indices[0]; 
		for(int i=1; i<rk; i++){
			index  *= sz[i];
			index += indices[i];
		}
		
		return vec[index];
    }



	__host__ __device__
	vector<int>  shape() const{
		vector<int> sh(rk);
		
		for(int i=0; i<rk; i++){
			sh[i] = sz[i];
		}
				
        return sh;
	}  

	__host__ __device__
	int  rank() const{
        return rk;
	}  
	
	__host__ __device__
	T* vec_pointer() const{
        return vec;
	}

};

#endif
