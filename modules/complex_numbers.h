#ifndef _COMPLEX_NUMBERS_
#define _COMPLEX_NUMBERS_

#include <iostream>
#include <complex>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>


template <typename T>
class complex											
{

private:
    T re;
    T im;
    
public:
	__host__ __device__
	complex(){
	}

	__host__ __device__
    complex(const T &x, const T &y) :
            re(x), im(y)
	{
    }

	__host__ __device__
    complex(const T &x) :
            re(x), im(0)
    {
    }    
    
	//Returns real part
	__host__ __device__
	T real(){
		return re;
	}
	__host__ __device__
	T real() const{
		return re;
	}	
	__host__ __device__
	friend T re(const complex<T> &z){
		return z.real();
	}


	__host__ __device__
	T* pointer_to_real(){
		return &re;
	}
		
	//Returns imaginary part
	__host__ __device__
	T imag(){
		return im;
	}
	__host__ __device__
	T imag() const{
		return im;
	}	
	__host__ __device__
	friend T im(const complex<T> &z){
		return z.imag();
	}


	__host__ __device__
	T* pointer_to_imag(){
		return &im;
	}
			
	//Returns the norm
	__host__ __device__
	T norm(){
		return re*re + im*im;
	}
	__host__ __device__
	T norm() const{
		return re*re + im*im;
	}	
	__host__ __device__
	friend T norm(const complex<T> &z){
		return z.norm();
	}		

	//Returns the modulus
	__host__ __device__
	T cabs(){
		return sqrt(re*re + im*im);
	}	
	__host__ __device__
	T cabs() const{
		return sqrt(re*re + im*im);
	}	
	__host__ __device__
	friend T cabs(const complex<T> &z){
		return z.cabs();
	}
	
	//Returns the argument
	__host__ __device__
	T arg(){
		return atan2(im, re);
	}
	__host__ __device__
	T arg() const{
		return atan2(im, re);
	}	
	__host__ __device__
	friend T arg(const complex<T> &z){
		return z.arg();
	}
	
	//Complex conjugation
	__host__ __device__
	complex<T> conj(){
		return complex<T>(re, -im);
	}
	__host__ __device__
	complex<T> conj() const{
		return complex<T>(re, -im);
	}	
	__host__ __device__
	friend complex<T> conj(const complex<T> &z){
		return z.conj();
	}	
	
	// Conversions
	template<typename T2>
	__host__ __device__
    operator complex<T2>() const{ 
        return complex<T2>(T2(re), T2(im)); 
    }	
    
	template<typename T2>
	__host__
    operator std::complex<T2>() const{ 
        return std::complex<T2>(T2(re), T2(im)); 
    }	    	

	//Basic operations between complex numbers		
		//Sum
		__host__ __device__
		complex<T> operator+(const complex<T> &z){
			return complex<T>(re + z.real(), im + z.imag());
		} 	  
		__host__ __device__
		complex<T> operator+(const complex<T> &z) const{
			return complex<T>(re + z.real(), im + z.imag());
		} 	 

		__host__ __device__
		complex<T> operator+(const T &x){
			return complex<T>(re + x, im);
		} 	  
		__host__ __device__
		complex<T> operator+(const T &x) const{
			return complex<T>(re + x, im);
		} 

		__host__ __device__
		friend complex<T> operator+(const T &x, const complex<T> &z){
			return complex<T>(z.real() + x, z.imag());
		}	
		
		//Re-sum
		__host__ __device__
		void operator+=(const complex<T> &z){
			*this = *this + z;
			return;
		}			 		 		
		    
		//Difference
		__host__ __device__
		complex<T> operator-(const complex<T> &z){
			return complex<T>(re - z.real(), im - z.imag());
		} 		   
		__host__ __device__
		complex<T> operator-(const complex<T> &z) const{
			return complex<T>(re - z.real(), im - z.imag());
		} 

		__host__ __device__
		complex<T> operator-(const T &x){
			return complex<T>(re - x, im);
		} 	  
		__host__ __device__
		complex<T> operator-(const T &x) const{
			return complex<T>(re - x, im);
		} 	
		__host__ __device__
		friend complex<T> operator-(const T &x, const complex<T> &z){
			return complex<T>(-z.real() + x, -z.imag());
		}

		//Re-subtract
		__host__ __device__
		void operator-=(const complex<T> &z){
			*this = *this - z;
			return;
		}
					
		//Multiplication	
		__host__ __device__	
		complex<T> operator*(const complex<T> &z){
			return complex<T>(re*z.real() - im*z.imag(), re*z.imag() + im*z.real());
		}
		__host__ __device__
		complex<T> operator*(const complex<T> &z) const{
			return complex<T>(re*z.real() - im*z.imag(), re*z.imag() + im*z.real());
		}		
		
		__host__ __device__
		complex<T> operator*(const T& x){
			return complex<T>(x * re, x * im);
		}
		__host__ __device__
		complex<T> operator*(const T& x) const{
			return complex<T>(x * re, x * im);
		}		
		
		__host__ __device__
		friend complex<T> operator*(const T& x, const complex<T> &z1){
			return complex<T>(x * z1.real(), x * z1.imag());
		}						

		//Re-multiply
		__host__ __device__
		void operator*=(const complex<T> &z){
			*this = *this * z;
			return;
		}
					
		//Division
		__host__ __device__
		complex<T> operator/(const complex<T> &z){
			T norm = z.norm();
			return complex<T>( (re*z.real() + im*z.imag()) / norm, (- re*z.imag() + im*z.real()) / norm );
		}
		__host__ __device__
		complex<T> operator/(const complex<T> &z) const{
			T norm = z.norm();
			return complex<T>( (re*z.real() + im*z.imag()) / norm, (- re*z.imag() + im*z.real()) / norm );
		}	
		__host__ __device__
		complex<T> operator/(const T& x){
			return complex<T>( re / x, im / x );
		}
		__host__ __device__
		complex<T> operator/(const T& x) const{
			return complex<T>( re / x, im / x );
		}						
		__host__ __device__
		friend complex<T> operator/(const T& x, const complex<T> &z1){
			T norm = z1.norm();
			return complex<T>(x * z1.real() / norm, -x * z1.imag()  / norm);
		}

		//Re-divide
		__host__ __device__
		void operator/=(const complex<T> &z){
			*this = *this / z;
			return;
		}
		
    //Some elementary functions of complex argument
    __host__ __device__
    friend complex<T> cexp(const complex<T> &z){
		return exp(z.real()) * complex<T>(cos(z.imag()), sin(z.imag()));
	}
	
	__host__ __device__
    friend complex<T> csin(const complex<T> &z){
		return complex<T>(sin(z.real())*cosh(z.imag()), cos(z.real())*sinh(z.imag()));
	}
	
	__host__ __device__
    friend complex<T> ccos(const complex<T> &z){
		return complex<T>(cos(z.real())*cosh(z.imag()), - sin(z.real())*sinh(z.imag()));
	}		

	__host__ __device__
    friend complex<T> clog(const complex<T> &z){
		return complex<T>(log(z.cabs()), z.arg());
	}
    
    //Power of a complex number: principal value  
    __host__ __device__  
    friend complex<T> icpow(const complex<T> &z, int exponent){	

		if(exponent == 0){
			return 1;
		}			
		else if(exponent == 1){
			return z;
		}	
		else if(exponent == 2){
			return z*z;
		}
		else{
			T theta = z.arg() * exponent;

			T result 	= 1;
			T base		= z.cabs();
				
			for (;;){
				if (exponent & 1)
					result *= base;
				exponent >>= 1;
				if (!exponent)
					break;
				base *= base;
			}	
			
			return complex<T>(cos(theta), sin(theta)) * result;	
		}		
	}

    __host__ __device__  
    friend complex<T> cpow(const complex<T> &z, const T &alpha){
		T theta = z.arg() * alpha;
		
		return complex<T>(cos(theta), sin(theta)) * pow(z.cabs(), alpha);
	}
	
	
	__host__ __device__
	friend bool isnan(const complex<T> &z){
		if(isnan(z.real()) == true or isnan(z.imag()) == true){
			return true;
		}
		return false;
	}
    
};

template <typename T>
std::ostream &operator <<(std::ostream &os, complex<T> z) {
	if(z.imag() > 0){
		return os << z.real() << "+" << +z.imag() << "i";
	}
	else if(z.imag() < 0){
		return os << z.real() << "-" << -z.imag() << "i";
	}
	else{
		return os << z.real();
	}
}

#endif
