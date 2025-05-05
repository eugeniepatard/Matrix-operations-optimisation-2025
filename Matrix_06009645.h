// Step 1: RENAME THE _ADV_PROG_MATRIX_H_ Label To _ADV_PROG_MATRIX_%Your CID Nr%_H_. So, if your CID is 00112233, then the label should be named: _ADV_PROG_MATRIX_00112233_H_
#ifndef _ADV_PROG_MATRIX_06009645_H_
#define _ADV_PROG_MATRIX_06009645_H_

#include <vector>
#include <iostream>

namespace adv_prog_cw 
{
	// Step 2: RENAME THE MATRIX CLASS To Matrix_%Your CID Nr%. So, if your CID is 00112233, then the class should be named: Matrix_00112233
	template<typename fT>
	class Matrix_06009645 {
	public:
		Matrix_06009645();
		Matrix_06009645(size_t m, size_t n);
		Matrix_06009645(size_t m, size_t n, fT val);
        Matrix_06009645(size_t m, size_t n, const std::vector<std::vector<fT>> &values);
        Matrix_06009645(const Matrix_06009645 &M);
        ~Matrix_06009645();

		size_t Rows() const;
		size_t Cols() const;
		void Resize(size_t m, size_t n);
		// accessors M(i,j)
		fT& operator()(size_t m, size_t n);
		const fT& operator()(size_t m, size_t n) const;
		// assignment
		Matrix_06009645& operator=(const Matrix_06009645& M);
		Matrix_06009645& operator=(fT val);

		Matrix_06009645& operator+=(const Matrix_06009645& M);
		Matrix_06009645& operator-=(const Matrix_06009645& M);
		void Identity();
		void Zero();
		void Transposed(Matrix_06009645& RES) const;
		void Out(long digits = 5L) const;

		// Step 3: Implement the following methods. These will be tested for speed and accuracy for matrices of increasing size and complexity.
		
		// Step 3.1: Multiplication of the matrix by a scalar
		Matrix_06009645& operator*=(fT scalar);
		// Step 3.2:  Division of the matrix by a scalar
		Matrix_06009645& operator/=(fT scalar);
		// Step 3.3:  A method to compute the determinant of square matrices
		fT Determinant() const;

        // Step 3.4:  A method to compute the inverse of the matrix
        bool Inverse(Matrix_06009645& result) const;


    private:
		std::vector<std::vector<fT> >  data;
		size_t                   rows, cols;

		bool  CheckRange(size_t m, size_t n, const char* originator) const;
		bool  CheckSizes(const Matrix_06009645& mat, const char* originator) const;
	};

	// associated operators
	template<typename fT>
	Matrix_06009645<fT>  operator+(const Matrix_06009645<fT>& a, const Matrix_06009645<fT>& b);

	template<typename fT>
	Matrix_06009645<fT>  operator-(const Matrix_06009645<fT>& a, const Matrix_06009645<fT>& b);

	template<typename fT>
	Matrix_06009645<fT>  operator*(const Matrix_06009645<fT>& a, const Matrix_06009645<fT>& b);

	template<typename fT>
	std::vector<fT>  operator*(const Matrix_06009645<fT>& mat, const std::vector<fT>& v);

	// v^T = (v^T M)^T
	template<typename fT>
	Matrix_06009645<fT>  operator*(const std::vector<fT>& v, const Matrix_06009645<fT>& M);

	// <!-- ==== class Method documentation format ==================== -->

	// constructor (i,j)
	// ---------------------------------------
	//
	template<typename fT>
	inline Matrix_06009645<fT>::Matrix_06009645(size_t m, size_t n)
		: data(m, std::vector<fT>(n)), rows(m), cols(n)
	{
	}

	// operator (i,j)
	// ---------------------------------------
	//
	template<typename fT>
	inline fT& Matrix_06009645<fT>::operator()(size_t m, size_t n)
	{
#ifndef NDEBUG
		CheckRange(m, n, "Matrix_06009645<fT>::operator()");
#endif
		return data[m][n];
	}

	// operator (i,j) const
	// ---------------------------------------
	//
	template<typename fT>
	inline const fT& Matrix_06009645<fT>::operator()(size_t m, size_t n) const
	{
#ifndef NDEBUG
		CheckRange(m, n, "Matrix_06009645<fT>::operator()");
#endif
		return data[m][n];
	}

	// Rows()
	// ---------------------------------------------
	//
	template<typename fT>
	inline size_t Matrix_06009645<fT>::Rows() const { return rows; }

	// Cols()
	// ---------------------------------------------
	//
	template<typename fT>
	inline size_t Matrix_06009645<fT>::Cols() const { return cols; }

	// CheckRange(i,j, message )
	// ---------------------------------------
	//
	template<typename fT>
	inline bool Matrix_06009645<fT>::CheckRange(size_t m, size_t n,
		const char* originator) const
	{
		if (m >= rows) {
			std::cerr << "\n" << originator << " row index violation, index=" << m;
			std::cerr << " versus, row-max=" << rows << std::endl;
			throw std::length_error("Matrix_06009645<double,mn_max>::CheckRange");
			return false;
		}
		if (n >= cols) {
			std::cerr << "\n" << originator << " column index violation, index=" << n;
			std::cerr << " versus, column-max=" << cols << std::endl;
			throw std::length_error("Matrix_06009645<double,mn_max>::CheckRange");
			return false;
		}
		return true;
	}

	// CheckSizes (i,j,message)
	// ---------------------------------------
	template<typename fT>
	inline bool Matrix_06009645<fT>::CheckSizes(const Matrix_06009645& mat,
		const char* originator) const
	{
		if (rows != mat.rows) {
			std::cerr << "\n" << originator << " matrices have different sizes; rows1=" << rows;
			std::cerr << " versus, rows2=" << mat.rows << std::endl;
			throw std::length_error("Matrix_06009645<double,mn_max>::CheckSizes");
			return false;
		}
		if (cols != mat.cols) {
			std::cerr << "\n" << originator << " matrices have different sizes; columns1=" << cols;
			std::cerr << " versus, columns2=" << mat.cols << std::endl;
			throw std::length_error("Matrix_06009645<double,mn_max>::CheckSizes");
			return false;
		}
		return true;
	}
} // end scope

#endif
