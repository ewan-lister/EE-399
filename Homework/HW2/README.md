# Homework 2 Report - SVD on Yale Faces Data

**Author**:

Ewan Lister

**Abstract**:

In this assignment, we utilize correlation matrices, single value decomposition (SVD), and eigendecomposition to analyze the facial data contained within the Extended Yale B dataset. Using the aforementioned techniques, we seek to isolate base features which compose all human faces, and to estimate the similarities and differences betweeen faces. We will discuss the mathematical theory of the techniques, their application, and evaluate the results. By applying SVD to an easy to visualize data set, we can prepare our understanding, and generalize the technique for higher dimensions in future work.

## Introduction and Overview
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

Where A can be any matrix of size $m \times n$, U is assumed to be unitary, S is a diagonal matrix with positive entries, and V is unitary. Thus the matrix is represented as a linear combination of basis vectors operating on a unitary matrix V.

In the following section we will explore the use of these techniques on the `yalefaces.mat` data.


## Algorithm Implementation and Development

import statements and loading data

    # import numpy, scipy, and yale faces data
    import numpy as np
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    results=loadmat('yalefaces.mat')
    X=results['X'] 

### (a) Compute a 100 × 100 correlation matrix $C$ where you will compute the dot product (correlation) between the first 100 images in the matrix $X$.

isolate first 100 vectors in matrix and compute dot product

    first_100_images = X[:, :100]

    correlation_matrix = np.dot(first_100_images.T, first_100_images)

plot correlation matrix

    # Plot the correlation matrix
plt.figure(figsize=(10, 10))
plt.imshow(correlation_matrix, cmap='viridis')
plt.colorbar()

    # Set the title and axes labels
    plt.title('Correlation Matrix of the First 100 Images')
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')

    # Adjust the axes range and ticks
    plt.xlim(-0.5, 99.5)
    plt.ylim(99.5, -0.5)
    plt.xticks(np.arange(0, 100, 10))
    plt.yticks(np.arange(0, 100, 10))

    # Show the plot
    plt.show()

### (b) From the correlation matrix for part (a), which two images are most highly correlated? Which are most uncorrelated? Plot these faces.

mask matrix data so that identical but low correlation value images are not used

    masked_corr_matrix = np.ma.array(correlation_matrix, mask=np.eye(correlation_matrix.shape[0], dtype=bool))

identify min and max correlation images and extract

    max_corr_indices = np.unravel_index(np.ma.argmax(masked_corr_matrix), masked_corr_matrix.shape)
    min_corr_indices = np.unravel_index(np.ma.argmin(masked_corr_matrix), masked_corr_matrix.shape)

    highest_corr_images = first_100_images[:, max_corr_indices]
    lowest_corr_images = first_100_images[:, min_corr_indices]

plot images 

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot the images with highest correlation
    axs[0, 0].imshow(highest_corr_images[:, 0].reshape(32, 32), cmap='viridis')
    axs[0, 0].set_title('Image {} (High Correlation)'.format(max_corr_indices[0]))
    axs[0, 0].axis('off')

    axs[0, 1].imshow(highest_corr_images[:, 1].reshape(32, 32), cmap='viridis')
    axs[0, 1].set_title('Image {} (High Correlation)'.format(max_corr_indices[1]))
    axs[0, 1].axis('off')

    # Plot the images with lowest correlation
    axs[1, 0].imshow(lowest_corr_images[:, 0].reshape(32, 32), cmap='viridis')
    axs[1, 0].set_title('Image {} (Low Correlation)'.format(min_corr_indices[0]))
    axs[1, 0].axis('off')

    axs[1, 1].imshow(lowest_corr_images[:, 1].reshape(32, 32), cmap='viridis')
    axs[1, 1].set_title('Image {} (Low Correlation)'.format(min_corr_indices[1]))
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

### (c) Repeat part (a) but now compute the 10 × 10 correlation matrix between images and plot the correlation matrix between them.

    images = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]

isolate images and calculate correlation matrix, account for 1 based indexing

    image_indices = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
    selected_images = X[:, [i - 1 for i in image_indices]]

    correlation_matrix = np.dot(selected_images.T, selected_images)

plot correlation matrix

    plt.figure(figsize=(8, 8))
    plt.imshow(correlation_matrix, cmap='viridis')
    plt.colorbar()

    plt.title('Correlation Matrix of Selected Images')
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')

    plt.xticks(np.arange(0, 10), image_indices)
    plt.yticks(np.arange(0, 10), image_indices)

    plt.show()


### (d) Create the matrix $Y = XX^{T}$ and find the first six eigenvectors with the largest magnitude eigenvalue.

create symmetric matrix, similar to correlation matrix of all images, and compute eigenpairs
    
    Y = np.dot(X,np.transpose(X))

    eigenvalues, eigenvectors = np.linalg.eigh(Y)

sort eigenvalues and vectors by eigenvalue magnitude, select first 6 eigen vectors in sorted list
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    first_6_eigenvectors = sorted_eigenvectors[:, :6]

plot and print first 6 eigenvectors

    print(first_6_eigenvectors)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(6):
        row = i // 3
        col = i % 3
        axs[row, col].imshow(first_6_eigenvectors[:, i].reshape(32, 32), cmap='viridis')
        axs[row, col].set_title('Eigenvector {}'.format(i+1))
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()

### (e) SVD the matrix X and find the first six principal component directions.

perform SVD using numpy function, and take first 6 principle component directions, print component directions

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    first_6_principal_components = Vt[:6, :]

    print(first_6_principal_components)

### (f) Compare the first eigenvector $v_{1}$ from (d) with the first SVD mode $u_{1}$ from (e) and compute the norm of difference of their absolute values.

compute absolute values and take norm difference

    abs_first_eigenvector = np.abs(sorted_eigenvectors[:, 0])
    abs_first_svd_mode = np.abs(U[:, 0])

    # Compute the norm difference of the absolute values
    norm_difference = np.linalg.norm(abs_first_eigenvector - abs_first_svd_mode)

plot and print
    print("Norm difference of the absolute values:", norm_difference)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first eigenvector
    axs[0].imshow(sorted_eigenvectors[:, 0].reshape(32, 32), cmap='viridis')
    axs[0].set_title('First Eigenvector')
    axs[0].axis('off')

    # Plot the first SVD mode from the U matrix
    axs[1].imshow(U[:, 0].reshape(32, 32), cmap='viridis')
    axs[1].set_title('First SVD Mode (U matrix)')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

### (g) Compute the percentage of variance captured by each of the first 6 SVD modes. Plot the first 6 SVD modes.

compute total variance captured by all modes and divide individual variances captured by total variance
    
    total_variance_captured = np.sum(S ** 2)

    variance_captured = (S ** 2) / total_variance_captured

convert to percentage and print

    percentage_variance_captured = variance_captured * 100

    for i in range(6):
        print("Percentage of the variance captured by Mode {}: {:.2f}%".format(i+1, percentage_variance_captured[i]))

plot first 6 modes

    for i in range(6):
        row = i // 3
        col = i % 3
        axs[row, col].imshow(U[:, i].reshape(32, 32), cmap='viridis')
        axs[row, col].set_title('Mode {}: {:.2f}%'.format(i+1, percentage_variance_captured[i]))
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()

## Computational Results

### Problem (a) 100 x 100 correlation matrix
![Fig. 1. Correlation of 100 Images](./Figures/correlation_matrix_100.png)

### Problem (b) Most and least correlated images
![Fig. 2. High and Low Correlation Images](./Figures/high_low_correlation_faces.jpg)

### Problem (c) 10 x 10 correlation matrix
![Fig. 3. Correlation of 10 Images](./Figures/correlation_matrix_10.jpg)

### Problem (d) First 6 eigenvectors of matrix $Y$
![Fig. 4. Eigenfaces](./Figures/eigenfaces.jpg)

### (f) Principal eigenvector and first mode comparison

    Norm difference of the absolute values: 5.688808695715053e-16
![Fig. 5. 1st Eigenvector vs. 1st Mode](./Figures/eigen_vs_1st_mode.jpg)

### (g) First 6 modes and respective percentage of variance captured
![Fig. 6. First 6 Modes](./Figures/first_6_modes.jpg)


## Summary and Conclusions

Applying the mathematical techniques of correlation, eigenfactorization, and SVD has allowed us to generate some interesting results from `yalefaces.mat`. In problem (a) we were able to identify two images that are likely to contain the same face simply based on the high correlation between the two, additionally we were able to find faces which are either unrelated, or simply have extremely varied lighting conditions based on their low correlation.

Isolating the eigenvectors of `yalefaces.mat` in problem (d) revealed a series of images that resemble primitive faces, becoming less and less endearing as the scaling value of the eigenvector decreases. Interestingly, some faces appear to resemble even functions, while others are odd. 

When comparing the highest eigenvalue eigenvector with the first mode of the SVD in problem (f), another interesting result arises in that the two images are nearly identical, having a miniscule norm difference. In fact, all 6 modes appear very similar, except that some have their sign flipped. This is explainable based on the fact that in SVD we only allow $\Sigma$ to contain positive singular values. This similarity leads one to conclude that the modes of the U matrix in SVD are similar to eigen values in a matrix. Except that they exist for all matrices.

As a general note, another special quality of the eigenpairs and SVD technique was that they allowed us to reduce the dimensionality of the data set, while still preserving most of the variance associated with the original data set. With only 6 SVD modes, we can reconstruct the individual features of each image within 93.89% of the original, which for all intents and purposes, makes these techniques robust for compression.

By applying mathematical techniques that allow for fast comparison and reduction in dimensionality of the data. We can store, understand, and utilize the data much faster, and much more clearly than without.
