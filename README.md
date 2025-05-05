# Matrix Operations - Imperial ACS 2025

## Project Overview
This repository contains the implementation of an advanced C++ Matrix class for a Master’s level coursework in Applied Computational Science and Engineering at Imperial College London. The objective of the project is to extend the provided Matrix class by implementing several core matrix operations, with a focus on efficiency, accuracy, and scalability for large datasets. 

### Key Features:
1. **Scalar Multiplication** – Implemented the `operator*=` to scale all elements of the matrix by a scalar value.
2. **Scalar Division** – Developed the `operator/=` to divide each element by a scalar, ensuring proper handling of edge cases like division by zero.
3. **Determinant Calculation** – Added the `Determinant()` function to compute the determinant of square matrices.
4. **Matrix Inversion** – Introduced the `Inverse()` function to compute the inverse of a matrix, with checks for invertibility.

### Performance Considerations:
- Multithreading with OpenMP is employed to optimize matrix operations and enhance performance for large matrices.
- The code is designed to efficiently scale with increasing matrix sizes, focusing on minimizing time complexity where possible.

## Coursework Details

- **Course**: Advanced C++ Programming
- **Institution**: Imperial College London, Applied Computational Science and Engineering
- **Year**: 2024-2025
- **Deadline**: 14th March 2025

### Instructions Followed:
1. **Header Pragma & Class Renaming**: The header guard and class were renamed to include my Candidate ID (CID).
2. **Method Implementations**: The project required the implementation of four key matrix operations—scalar multiplication, scalar division, determinant calculation, and matrix inversion.
3. **Edge Cases**: Special attention was given to edge cases, such as handling division by zero, ensuring the matrix is square for determinant computation, and checking matrix invertibility.

## Files in This Repository:
- **Matrix_00112233.h**: Header file containing the Matrix class declaration and method prototypes.
- **Matrix_00112233.cpp**: Source file with the implementation of matrix operations.

## Instructions for Building & Running the Code

1. **Dependencies**: The project requires a C++ compiler that supports OpenMP for multithreading optimization (e.g., GCC 10+).
2. **Compilation**: To compile the project, navigate to the project directory and run:
    ```bash
    g++ -fopenmp -o matrix_operations Matrix_00112233.cpp
    ```
3. **Running the Program**: After successful compilation, run the program with:
    ```bash
    ./matrix_operations
    ```

## Design Considerations

### 1. **Scalar Operations (`operator*=` and `operator/=`)**:
- These methods optimize matrix element-wise scalar operations by modifying the matrix in-place for memory efficiency.

### 2. **Determinant Calculation**:
- The determinant is calculated using recursive expansion by minors for smaller matrices. For larger matrices, more optimized techniques can be used, but recursion is sufficient for the scope of this coursework.

### 3. **Matrix Inversion**:
- Matrix inversion employs Gaussian elimination to compute the inverse. The function returns `false` if the matrix is singular (non-invertible).

### 4. **Multithreading with OpenMP**:
- OpenMP is used to parallelize computationally intensive tasks, such as matrix operations, allowing the code to scale better for large matrices.

## Challenges Encountered

- **Performance Optimization**: Optimizing matrix operations for performance, particularly for large matrices, presented challenges, especially in ensuring thread safety and managing multithreading overhead.
- **Edge Case Handling**: Ensuring robustness in edge cases, like division by zero and handling non-square matrices for determinant computation, required extra attention.

## Future Improvements

- **Optimized Determinant Calculation**: For larger matrices, more advanced algorithms like LU decomposition could replace the recursive approach.
- **Parallelization of Matrix Inversion**: Further optimizations could include parallel algorithms for matrix inversion to scale efficiently on multi-core processors.
  
## License
This project is for academic purposes and is part of the coursework for the **Advanced C++ Programming** class at Imperial College London.
