#include "Matrix_06009645.h"
#include <cassert>
#include <omp.h>
#include <thread>

using namespace std;

typedef int                  int32;
typedef unsigned int         uint32;
typedef double               double64;

namespace adv_prog_cw {
	// default constructor
	// ---------------------------------------
	template<typename fT>
	Matrix_06009645<fT>::Matrix_06009645()
		: rows(0), cols(0)
	{
	}

	// operator=( DM )
	// ---------------------------------------
	template<typename fT>
	Matrix_06009645<fT>& Matrix_06009645<fT>::operator=(const Matrix_06009645<fT>& mat)
	{
		if (&mat != this) {
			rows = mat.rows;
			cols = mat.cols;

			data.resize(rows, vector<fT>(cols));

			for (size_t i = 0U; i < rows; i++)
				for (size_t j = 0U; j < cols; j++)
					data[i][j] = mat.data[i][j];
		}

		return *this;
	}

	template<typename fT>
	Matrix_06009645<fT>& Matrix_06009645<fT>::operator=(fT val)
	{
		for (size_t i = 0U; i < rows; i++)
			for (size_t j = 0U; j < cols; j++) data[i][j] = val;

		return *this;
	}

	// copy constructor
	// ---------------------------------------
	template<typename fT>
	Matrix_06009645<fT>::Matrix_06009645(const Matrix_06009645<fT>& mat)
	{
		*this = mat;
	}

	// constructor (i,j, value)
	// ---------------------------------------
	template<typename fT>
	Matrix_06009645<fT>::Matrix_06009645(size_t m, size_t n, fT val)
		: rows(m), cols(n)
	{
		data.resize(rows, vector<fT>(cols));

		for (size_t i = 0U; i < rows; i++)
			for (size_t j = 0U; j < cols; j++) data[i][j] = val;
	}

	// constructor (i,j, vector value)
	template<typename fT>
	Matrix_06009645<fT>::Matrix_06009645(size_t m, size_t n, const std::vector<std::vector<fT>>& values)
		: rows(m), cols(n), data(values)
	{
		// Check that the dimensions of the input values match the matrix dimensions
		if (values.size() != m || (values.size() > 0 && values[0].size() != n)) {
			throw std::invalid_argument("Matrix dimensions do not match provided values.");
		}
	}

	// destructor
	// ---------------------------------------
	template<typename fT>
	Matrix_06009645<fT>::~Matrix_06009645()
	{
	}

	// operator+=
	// ---------------------------------------
	template<typename fT>
	Matrix_06009645<fT>& Matrix_06009645<fT>::operator+=(const Matrix_06009645<fT>& mat)
	{
		CheckSizes(mat, "Matrix_06009645<fT>::operator+=");
		for (size_t i = 0U; i < rows; i++)
			for (size_t j = 0U; j < cols; j++) data[i][j] += mat.data[i][j];

		return *this;
	}

	// operator-=
	// ---------------------------------------
	template<typename fT>
	Matrix_06009645<fT>& Matrix_06009645<fT>::operator-=(const Matrix_06009645<fT>& mat)
	{
		CheckSizes(mat, "Matrix_06009645<fT>::operator-=");
		for (size_t i = 0U; i < rows; i++)
			for (size_t j = 0U; j < cols; j++) data[i][j] -= mat.data[i][j];

		return *this;
	}

	// operator +
	// ---------------------------------------
	template<typename fT>
	Matrix_06009645<fT>  operator+(const Matrix_06009645<fT>& a, const Matrix_06009645<fT>& b)
	{
		Matrix_06009645<fT> temp(a);
		temp += b;
		return temp;
	}

	// operator -
	// ---------------------------------------
	template<typename fT>
	Matrix_06009645<fT>  operator-(const Matrix_06009645<fT>& a, const Matrix_06009645<fT>& b)
	{
		Matrix_06009645<fT> temp(a);
		temp -= b;
		return temp;
	}

	// operator *
	// ---------------------------------------
	template<typename fT>
	Matrix_06009645<fT>  operator*(const Matrix_06009645<fT>& a, const Matrix_06009645<fT>& b)
	{
#ifndef NDEBUG
		if (a.Cols() != b.Rows()) {
			cout << "\nMatrix_06009645<" << typeid(fT).name() << "> operator*: Matrices cannot be multiplied ";
			cout << "because of incompatible sizes (A * B, see matrices below): " << endl;
			a.Out(3L);
			b.Out(3L);
			throw length_error("Matrix_06009645<double,mn_max>::operator*");
		}
#endif
		Matrix_06009645<fT>  temp(a.Rows(), b.Cols());

		for (size_t i = 0U; i < a.Rows(); i++)
			for (size_t j = 0U; j < b.Cols(); j++) {
				temp(i, j) = static_cast<fT>(0.0);
				for (size_t k = 0U; k < b.Rows(); k++)
					temp(i, j) += a(i, k) * b(k, j);
			}

		return temp;
	}

	// -------------------------------------------------------------------------------------------
	// OPERATOR FUNCTIONS
	// -------------------------------------------------------------------------------------------

	// vector^T = Matrix_06009645 * vector^T
	// ---------------------------------------
	template<typename fT>
	vector<fT>  operator*(const Matrix_06009645<fT>& mat, const vector<fT>& vec)
	{
		assert(mat.Cols() == vec.size());

		vector<fT> temp(mat.Rows(), static_cast<fT>(0.0));

		for (size_t i = 0; i < mat.Rows(); i++)
			for (size_t j = 0; j < mat.Cols(); j++) temp[i] += mat(i, j) * vec[j];

		return temp;
	} // end operator*

   // Matrix_06009645 = vector^T * Matrix_06009645
   // ---------------------------------------
	template<typename fT>
	Matrix_06009645<fT>  operator*(const vector<fT>& vec, const Matrix_06009645<fT>& mat)
	{
		if (vec.size() != mat.Rows()) {
			cerr << "\noperator*: vector cannot be multiplied with Matrix_06009645";
			cerr << "because of incompatible sizes (v * M): " << endl;
			throw length_error("Matrix_06009645<double,mn_max>::operator*");
		}
		Matrix_06009645<fT>  temp(vec.size(), mat.Cols());

		for (size_t i = 0U; i < vec.size(); i++)
			for (size_t j = 0U; j < mat.Cols(); j++) {
				temp(i, j) = static_cast<fT>(0.0);
				for (size_t k = 0U; k < mat.Rows(); k++)
					temp(i, j) += vec[k] * mat(k, j);
			}

		return temp;
	} // end operator

   // Resize()
   // ---------------------------------------------
	template<typename fT>
	void Matrix_06009645<fT>::Resize(size_t m, size_t n)
	{
		// resize Matrix_06009645 but keep storage as is if the new Matrix_06009645
		// is smaller than the old matrix
		if (m <= rows && n <= cols) {
			rows = m;
			cols = n;
			return;
		}

		// increase matrix size
		rows = m;
		cols = n;
		data.resize(m);
		for (size_t i = 0; i < m; i++) data[i].resize(n);
	}

	// Identity()
	// ---------------------------------------------
	template<typename fT>
	void Matrix_06009645<fT>::Identity()
	{
		for (size_t i = 0U; i < rows; i++)
			for (size_t j = 0U; j < cols; j++)
				if (i == j) data[i][j] = static_cast<fT>(1.0);
				else          data[i][j] = static_cast<fT>(0.0);
	}

	// Zero()
	// ---------------------------------------------
	template<typename fT>
	void Matrix_06009645<fT>::Zero()
	{
		for (size_t i = 0U; i < rows; i++)
			for (size_t j = 0U; j < cols; j++)
				data[i][j] = static_cast<fT>(0.0);
	}

	// Transposed()
	// ---------------------------------------------
	template<typename fT>
	void Matrix_06009645<fT>::Transposed(Matrix_06009645& M) const
	{
		M.Resize(cols, rows);

		for (size_t i = 0U; i < rows; i++)
			for (size_t j = 0U; j < cols; j++) M.data[j][i] = data[i][j];
	}
	
	// Out( digits )
	// ---------------------------------------
	template<typename fT>
	void Matrix_06009645<fT>::Out(long digits) const
	{
		std::streamsize  prec;
		cout << "\nMatrix<" << typeid(fT).name() << ">::Out(): m=" << rows << ", n=" << cols << endl;
		if (digits != 0U) {
			cout.setf(ios::scientific);
			prec = cout.precision(digits);
		}
		size_t row_break, split_after(10U);

		for (size_t i = 0; i < rows; i++)
		{
			row_break = 1;
			for (size_t j = 0; j < cols; j++, row_break++)
			{
				if (data[i][j] >= 0.) cout << " ";
				cout << data[i][j] << " ";
				if (row_break == split_after)
				{
					cout << endl;
					row_break = 0U;
				}
			}
			cout << endl;
		}

		if (digits != 0U) {
			cout.unsetf(ios::scientific);
			cout.precision(prec);
		}

		cout << endl;
	} // end Out()

	// ----------------------------------------------------
	// Step 3.1: Multiplication of the matrix by a scalar
	// ---------------------------------------------------
	// operator *= scalar
	// ---------------------------
	template<typename fT>
	Matrix_06009645<fT>& Matrix_06009645<fT>::operator*=(fT scalar)
	{
		const size_t parallel_threshold = 500;
		bool use_parallel = (rows >= parallel_threshold);
		if (use_parallel){
			std::cerr << "Using parallelisation\n";
			
			#pragma omp parallel for collapse(2)
			for (size_t i = 0U; i < rows; i++)
			{
				for (size_t j = 0U; j < cols; j++) 
				{
					data[i][j] *= scalar;
				}
			}
		}else{
			for (size_t i = 0U; i < rows; i++)
			{
				for (size_t j = 0U; j < cols; j++) 
				{
					data[i][j] *= scalar;
				}
			}
		}

		
	
		return *this;
	}

	// -----------------------------------------------
	// Step 3.2:  Division of the matrix by a scalar
	// ---------------------------------------------
	// operator/= ( scalar )
	// ----------------------------------------
	template<typename fT>
	Matrix_06009645<fT>& Matrix_06009645<fT>::operator/=(fT scalar)
	{
		if (scalar == 0)
		{
			throw std::invalid_argument("Cannot divide by zero");
		}

		const size_t parallel_threshold = 500;
		bool use_parallel = (rows >= parallel_threshold);
		if (use_parallel){
			std::cerr << "Using parallelisation\n";
			
			// Perform the division
			#pragma omp parallel for collapse(2)
			for (size_t i = 0U; i < rows; i++)
			{
				for (size_t j = 0U; j < cols; j++) 
				{
					data[i][j] /= scalar;
				}
			}
		}else{
			for (size_t i = 0U; i < rows; i++)
			{
				for (size_t j = 0U; j < cols; j++) 
				{
					data[i][j] /= scalar;
				}
			}
		}

		return *this;
	}

// --------------------------------------------------------------
	// Step 3.3:  A method to compute the determinant of square matrices
	// ------------------------------------------------------------
	// Determinant()
	// -----------------------------------
	template<typename fT>
	fT Matrix_06009645<fT>::Determinant() const
	{
		// Check if it is a square matrix
		if (rows != cols)
		{
			throw std::invalid_argument("Error: Matrix must be square to compute the determinant.");
		}

		// If matrix is empty
		if (rows == 0 or cols == 0)
		{
			throw std::invalid_argument("Error: Matrix should not be empty to compute the determinant.");
		}

		const size_t parallel_threshold = 500;
		bool use_parallel = (rows >= parallel_threshold);
		if (use_parallel){
			std::cerr << "Using parallelisation\n";
		}

		// Tolerance for floating-point comparison
		const fT tolerance = static_cast<fT>(1e-10);

		Matrix_06009645<fT> temp = *this; // Copy matrix to avoid modifying original matrix
		size_t n = temp.rows;  // must be a square matrix
		fT determinant = 1;
		bool is_zero = false;  // Flag to track if determinant should be zero

		// Gaussian elimination with partial pivoting (in-place)
		for (size_t i = 0U; i < n; ++i) {
			// Find the pivot row with the maximum element in the current column
			size_t pivot_row = i; // initiallise the 
			for (size_t j = i + 1; j < n; ++j) {
				if (std::abs(temp.data[j][i]) > std::abs(temp.data[pivot_row][i])) {
					pivot_row = j;
				}
			}

			// std::cerr << "balise 1\n";

			// If pivot is zero (within tolerance), the determinant is zero
			if (std::abs(temp.data[pivot_row][i]) < tolerance) {
				std::cerr << "The matrix is singular, the determinant is zero \n";
				return 0;
			}

			// Swap rows if the pivot is not already the current row
			if (pivot_row != i) {
				std::swap(temp.data[i], temp.data[pivot_row]);
				determinant *= -1; // Swap changes the sign of the determinant
			}

			// Perform row reduction (Gaussian elimination) in-place
			fT pivot_value = temp.data[i][i];
			vector<fT>& row_i = temp.data[i]; // Reference to row i (pivot row)

			if (use_parallel) {
				#pragma omp parallel for
				for (size_t j = i + 1; j < n; ++j) {
					vector<fT>& row_j = temp.data[j];  
					fT factor = row_j[i] / pivot_value;

					for (size_t k = i + 1; k < n; ++k) {
						row_j[k] -= factor * row_i[k];
					}
				}
			} else {
				for (size_t j = i + 1; j < n; ++j) {
					vector<fT>& row_j = temp.data[j];  
					fT factor = row_j[i] / pivot_value;

					for (size_t k = i + 1; k < n; ++k) {
						row_j[k] -= factor * row_i[k];
					}
				}
			}


		}

		// If determinant is zero, return 0
		/*if (is_zero) {
			return 0;
		}*/

		// The determinant is the product of the diagonal elements
		#pragma omp parallel for reduction(*:determinant)
		for (size_t i = 0; i < n; ++i) {
			determinant *= temp.data[i][i];
		}

		// Check if determinant is effectively zero, considering the tolerance
		if (std::abs(determinant) < tolerance) {
			return 0;
		}

		return determinant;
	}

	// --------------------------------------------------------
	// Step 3.4:  A method to compute the inverse of the matrix
	// --------------------------------------------------------
	// Inverse ( Matrix )
	// -------------------------------------------------------

	/* To compute the inverse, we use the augmented matrix [A | I], where A is the original matrix 
		and I is the identity matrix of the same size.
		We apply Gaussian elimination to transform matrix A into the identity matrix I, 
		while simultaneously applying the same row operations to the identity matrix.
		This process converts the right-hand side of the augmented matrix (initially the identity matrix)
		into the inverse of the matrix A (A^-1). */

	template<typename fT>
	bool Matrix_06009645<fT>::Inverse(Matrix_06009645& result) const
	{
		try {
			// Check if result has the same size as *this
			CheckSizes(result, "Matrix_06009645<fT>::Inverse");

			// Check if the matrix is square
			if (rows != cols) {
				throw std::invalid_argument("Error: Matrix must be square to compute the inverse.");
			}

			const size_t parallel_threshold = 500;
			bool use_parallel = (rows >= parallel_threshold);
			if (use_parallel){
				std::cerr << "Using parallelisation\n";
			}

			// Tolerance for floating-point comparison
			const fT tolerance = static_cast<fT>(1e-10);

			size_t n = rows; // because cols = rows
			std::vector<std::vector<fT>> augmented(n, std::vector<fT>(2 * n));

			// Augment the matrix with the identity matrix
			if(use_parallel)
			{
				#pragma omp parallel for collapse(2)
				for (size_t i = 0; i < n; ++i) {
					for (size_t j = 0; j < n; ++j) {
						augmented[i][j] = data[i][j]; // Copy the matrix we want to compute the inverse
						augmented[i][j + n] = (i == j) ? 1 : 0; // Forms the identity matrix on the right-hand side
					}
				}
			}else{
				for (size_t i = 0; i < n; ++i) {
					for (size_t j = 0; j < n; ++j) {
						augmented[i][j] = data[i][j]; // Copy the matrix we want to compute the inverse
						augmented[i][j + n] = (i == j) ? 1 : 0; // Forms the identity matrix on the right-hand side
					}
				}
			}

			// Perform Gaussian elimination
			for (size_t i = 0; i < n; ++i) {
				// Find the pivot row (Sequential)
				size_t pivot_row = i;
				for (size_t j = i + 1; j < n; ++j) {
					if (std::abs(augmented[j][i]) > std::abs(augmented[pivot_row][i])) {
						pivot_row = j;
					}
				}

				// If pivot is zero, matrix is singular
				if (std::abs(augmented[pivot_row][i]) < tolerance) {
					std::cerr << "Matrix is singular and cannot be inverted!" << std::endl;
					return false;
				}

				// Swap rows if necessary
				if (pivot_row != i) {
					std::swap(augmented[i], augmented[pivot_row]);
				}

				if (use_parallel){
					// Normalize the pivot row (Parallel)
					fT pivot = augmented[i][i];
					#pragma omp parallel for
					for (size_t j = 0; j < 2 * n; ++j) {
						augmented[i][j] /= pivot;
					}

					// Row elimination (Parallel)
					#pragma omp parallel for
					for (size_t j = 0; j < n; ++j) {
						if (i != j) {
							fT factor = augmented[j][i];
							fT* row_i = &augmented[i][0];  // Cache row i
							fT* row_j = &augmented[j][0];  // Cache row j
							// Parallelize over 'k' for row updates
							#pragma omp parallel for
							for (size_t k = 0; k < 2 * n; ++k) {
								row_j[k] -= factor * row_i[k];
							}
						}
					}

					// Extract inverse matrix from augmented matrix
					#pragma omp parallel for
					for (size_t i = 0; i < n; ++i) {
						std::copy(augmented[i].begin() + n, augmented[i].begin() + 2 * n, result.data[i].begin());
					}

				}else {
					// Normalize the pivot row
					fT pivot = augmented[i][i];
					for (size_t j = 0; j < 2 * n; ++j) {
						augmented[i][j] /= pivot;
					}

					// Row elimination
					for (size_t j = 0; j < n; ++j) {
						if (i != j) {
							fT factor = augmented[j][i];
							fT* row_i = &augmented[i][0];  // Cache row i
							fT* row_j = &augmented[j][0];  // Cache row j
							// Parallelize over 'k' for row updates
							for (size_t k = 0; k < 2 * n; ++k) {
								row_j[k] -= factor * row_i[k];
							}
						}
					}

					// Extract inverse matrix from augmented matrix
					for (size_t i = 0; i < n; ++i) {
						for (size_t j = 0; j < n; ++j) {
							result.data[i][j] = augmented[i][j + n];
						}
					}
				}
				
			}

			return true;
		}catch (const std::length_error& e) { // Catch an exception if results is not the same size as *this
			std::cerr << "Error in Matrix Inversion: " << e.what() << std::endl; 
			return false; // Handle the case and return false
		}
	}

   // ------------------------------------------------------------------------------
   // template instantiations
   // ------------------------------------------------------------------------------

	template class Matrix_06009645<double>;

	template Matrix_06009645<double>  operator+(
		const Matrix_06009645<double>& a,
		const Matrix_06009645<double>& b);

	template Matrix_06009645<double>  operator-(
		const Matrix_06009645<double>& a,
		const Matrix_06009645<double>& b);

	template Matrix_06009645<double>  operator*(
		const Matrix_06009645<double>& a,
		const Matrix_06009645<double>& b);
	template
		vector<double>  operator*(const Matrix_06009645<double>& mat,
			const vector<double>& vec);

	// extra operators
	template Matrix_06009645<double64>
		operator*(const vector<double64>& vec, const Matrix_06009645<double64>& mat);
}