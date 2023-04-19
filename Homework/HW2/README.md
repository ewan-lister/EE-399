# Homework 2 Report - SVD on Yale Faces Data

**Author**:

Ewan Lister

**Abstract**:

In this assignment, we utilize correlation matrices, single value decomposition (SVD), and eigendecomposition to analyze the facial data contained within the Extended Yale B dataset. Using the aforementioned techniques, we seek to isolate base features which compose all human faces, and to estimate the similarities and differences betweeen faces. We will discuss the mathematical theory of the techniques, their application, and evaluate the results. By applying SVD to an easy to visualize data set, we can prepare our understanding, and generalize the technique for higher dimensions in future work.

## I. Introduction and Overview
Linear algebra is the mathematical framework by which we manipulate numerical and analyze numerical data. It is not just some tool for multiplying numbers easily. It aligns the data in such a way that one can quickly identify underlying relationships. 

In this assignment we begin by constructing a correlation matrix of the first 100 images in the `yalefaces.mat`, which provides us with a rough understanding of the similarity of each facial image relative to every other in the set of 100. Within this correlation matrix we identify the most similar and least similar images, based on the maximum magnitude and minimum magnitude entries found within the correlation matrix. We then construct an even smaller matrix taken from 10 images, to get a better understanding of its visual interpretation.

We follow by performing an eigendecomposition on a full correlation matrix $Y$ of all images found in `yalefaces.mat`. Because the eigenvectors and eigenvalues can be used to reconstruct their respective matrix, the example demonstrates that the yale faces are essentially composed linear combinations of so called base faces, or **eigenfaces**.

We conclude the algorithm implementation by executing an SVD on the matrix $Y$, an algorithm which is slightly similar, but more robust then eigendecomposition. Comparing the results we can see some of the similarities and differences between the two.

In the following sections, we plot and interpret our results, 

## Theoretical Background

In this section we will present mathematical theory covering the topics of correlation, eigenpairs, and singular value decomposition. Knowing the basic principles of these techniques will help us to investigate the facial image data.

### Correlation

In a statistical sense, correlation is a measurement of how closely related two variables are. In linear algebra we can express correlation between two vector as their dot product. Thus the correlation between two vectors $x_a$ and $x_b$ would be expressed as:

$$\rho_{ab} = x_a * x_b$$

If we wished to analyze the correlation in a matrix between each vector and every other, we can take the dot product for combination of vectors, and map that point to the column and row coordinated of the multiplied vectors, thus creating a **correlation matrix**.

### Eigenpairs

Many types of data can be processed so that underlying vectors which better describe the behavior of the data become visible. To illustrate, for square matrices, the transformation that the matrix achieves can be expressed by scalars called **eigenvalues** multiplying vectors called **eigenvectors**. If a given matrix $A$ operating on a vector $v$ causes $v$ to be scaled by some constant $\lambda$, then $v$ is an eigenvector of $A$ and $\lambda$ is the corresponsing eigenvalue, such that they form an **eigenpair**. All eigenpairs of $A$ satisfy the following equation:

$$ Av = \lambda v$$

### SVD

Singular Value Decomposition or SVD is a ubiquitous technique in data analysis that allows dimensional data to be decomposed into so called modes and singular values. Much like eigenvectors, modes provide a better description of what matrix data is actually composed of. SVD consists of decomposing a matrix $A$ in the following fashion:

$$ A = U\Sigma V^{T}$$

Where A can be any matrix of size $m \times n$, U is assumed to be unitary, S is a diagonal matrix with positive entries, and V is unitary. Thus 

\begin{equation*}
A = U\Sigma V^T =
\begin{bmatrix}
u_{11} & u_{12} \\
u_{21} & u_{22}
\end{bmatrix}
\begin{bmatrix}
\sigma_1 & 0 \\
0 & \sigma_2
\end{bmatrix}
\begin{bmatrix}
v_{11} & v_{12} \\
v_{21} & v_{22}
\end{bmatrix}^T
\end{equation*} 

Title/author/abstract Title, author/address lines, and short (100 words or less) abstract. 
Sec. I. Introduction and Overview
Sec. II. Theoretical Background
Sec. III. Algorithm Implementation and Development 
Sec. IV. Computational Results
Sec. V. Summary and Conclusions

![Fig. 1. Correlation of 100 Images](./Figures/correlation_matrix_100.png)
![Fig. 2. High and Low Correlation Images](./Figures/high_low_correlation_faces.png)
![Fig. 3. Correlation of 10 Images](.Ffigures/correlation_matrix_10.png)
![Fig. 4. Eigenfaces](./Figures/eigenfaces.png)
![Fig. 5. 1st Eigenvector vs. 1st Mode](./Figures/eigen_vs_1st_mode.png)
![Fig. 6. First 6 Modes](./Figures/first_6_modes.png)


