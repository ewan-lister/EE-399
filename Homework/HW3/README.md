# Homework 3 Report - Comparing Linear Classifiers on MNIST Data Set

**Author**:

Ewan Lister

**Abstract**:

In this assignment, we conduct an analysis of the MNIST dataset, a collection of handwritten digits used for machine learning research. The analysis begins with an SVD analysis of the digit images, exploring the singular value spectrum and determining the rank necessary for good image reconstruction. The interpretation of the U, $\Sigma$, and V matrices is also discussed. The report then builds a classifier to identify individual digits in the training set, using LDA to classify two and three selected digits. The difficulty of separating different pairs of digits is quantified using LDA, SVM, and decision tree classifiers, with performance compared on both training and test sets. The report includes multiple visualizations to aid in understanding the results.

## Introduction and Overview

In this report, we present an analysis of the MNIST dataset, which contains handwritten digits that have been used extensively for machine learning research. We begin by discussing the theoretical background, including the concepts of SVD analysis, PCA space, and linear classifiers such as LDA and SVM. We then describe the algorithm implementation and development process, including reshaping the digit images into column vectors, performing SVD analysis, and building classifiers to identify individual digits.

Next, we present the computational results of our analysis, including the singular value spectrum and the number of modes necessary for good image reconstruction. We also discuss the accuracy of our classifiers on the training and test sets, as well as the difficulty of separating different pairs of digits. We compare the performance of LDA, SVM, and decision tree classifiers on the hardest and easiest pairs of digits to separate.

Finally, we provide a summary and conclusions, discussing the key findings of our analysis and the implications for future research. Throughout the report, we include visualizations to aid in understanding the results. Overall, our analysis demonstrates the power of SVD analysis and linear classifiers for identifying handwritten digits and highlights the challenges involved in separating certain pairs of digits.

## Theoretical Background

In this section we will present mathematical theory covering the topics of correlation, eigenpairs, and singular value decomposition. Knowing the basic principles of these techniques will help us to investigate the facial image data.

### Support Vector Machines (SVM)

Support Vector Machines (SVM) are a popular class of binary classifiers used in machine learning. The goal of SVM is to find a hyperplane that maximally separates two classes of data points. This hyperplane is chosen to maximize the margin between the two classes of points.

The SVM optimization problem can be written as follows:

$$min_{w,b,\\xi} \\frac{1}{2} \\lVert w \\rVert^2 + C\\sum_{i=1}^{n} \\xi_i$$

subject to the constraints:

$$y_i(w^Tx_i+b) \\geq 1 - \\xi_i$$

$$\\xi_i \\geq 0$$


where $w$ is the weight vector, $b$ is the bias term, $\xi_i$ is the slack variable, and $C$ is a hyperparameter that controls the trade-off between maximizing the margin and minimizing the classification error. The first constraint ensures that each data point is on the correct side of the hyperplane, while the second constraint ensures that the margin is not too wide.

### Linear Discriminant Analysis (LDA)
Linear Discriminant Analysis (LDA) is a popular method for dimensionality reduction and classification. The goal of LDA is to find a linear transformation of the data that maximizes the separation between two classes.

The LDA optimization problem can be written as follows:

$$max_{w} \\frac{w^TS_bw}{w^TS_ww}$$

where $S_b$ is the between-class scatter matrix and $S_w$ is the within-class scatter matrix. The between-class scatter matrix measures the distance between the class means, while the within-class scatter matrix measures the variability within each class.

The optimal weight vector $w$ is then used to project the data onto a lower-dimensional subspace, which can be used for classification.

### Decision Trees

Decision trees are a popular class of classifiers that use a tree structure to recursively partition the feature space. The goal of a decision tree is to find the optimal set of binary splits that minimize the classification error.

The decision tree optimization problem can be written as follows:

$$min_{\\Theta} \\sum_{i=1}^{n} I(y_i \\neq f(x_i; \\Theta))$$

where $\Theta$ is the set of binary splits, $f(x_i; \Theta)$ is the decision tree classifier, and $I(y_i \neq f(x_i; \Theta))$ is the classification error. The optimal set of binary splits can be found using various algorithms, such as greedy search or dynamic programming.

Overall, SVM, LDA, and decision trees are all popular and effective methods for classification in machine learning. Each method has its own strengths and weaknesses, and the choice of method often depends on the specific problem and the characteristics of the data.


## Algorithm Implementation and Development

import libraries and load mnist data

    import numpy as np
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score

    # fetch MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)

    # Convert the data and labels into numpy arrays
    data = np.array(mnist['data'])
    labels = np.array(mnist['target'])

    # take transpose of data to convert to short-wide matrix
    data = data.T

### 1. Do an SVD analysis of the digit images. You will need to reshape each image into a column vector and each column of your data matrix is a different image.

call numpy linear algebra SVD method, execute

    U, S, Vt = np.linalg.svd(data, full_matrices=False)

### 2. What does the singular value spectrum look like and how many modes are necessary for good image reconstruction? (i.e. what is the rank rof the digit space?)

plot and print first 50 SV

    # initialize white facecolor for plots
    w = 'white'

    # print first 50 singular values
    print(S[0:50])

    # Plot singular values
    plt.figure(figsize=(8, 8))
    plt.stem(np.arange(0, 50), S[0:50])

    # Set the title and axes labels
    plt.title('Singular Values of SVD on MNIST')
    plt.xlabel('Column Number')
    plt.ylabel('Singular Value')

    # Show the plot
    plt.savefig('./Figures/first_50_singular_values.png', facecolor=w)
    plt.show()

compute total variance captured by all modes

    # Compute the total variance captured by all modes
    total_variance_captured = np.sum(S ** 2)

    # Compute the variance captured by each mode
    variance_captured = (S ** 2) / total_variance_captured

    # Convert the variance captured to percentage
    percentage_variance_captured = variance_captured * 100

Plot first 12 modes

    # Print the percentage of the variance captured by each mode
    for i in range(6):
        print("Percentage of the variance captured by Mode {}: {:.2f}%".format(i+1, percentage_variance_captured[i]))

    # Plot the first six modes as images
    fig, axs = plt.subplots(4, 3, figsize=(12, 8))

    for i in range(12):
        row = i // 3
        col = i % 3
        axs[row, col].imshow(U[:, i].reshape(28, 28), cmap='viridis')
        axs[row, col].set_title('Mode {}: {:.2f}%'.format(i+1, percentage_variance_captured[i]))
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('./Figures/first_6_modes.jpg', facecolor=w)
    plt.show()


### 4. On a 3D plot, project onto three selected V-modes (columns) colored by their digit label. For example, columns 2,3, and 5.

isolate modes 2, 3, and 5 of $V$, project MNIST onto modes

    modes = [1, 2, 4]
    V = Vt.T
    V_modes = V[modes, :]
    proj_data = np.dot(mnist.data, V_modes.T)

plot graph

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    targets = np.unique(mnist.target.astype(np.int))
    colors = plt.cm.jet(np.linspace(0, 1, len(targets)))
    for target, color in zip(targets, colors):
        idx = np.where(mnist.target.astype(np.int) == target)
        ax.scatter(proj_data[idx, 0], proj_data[idx, 1], proj_data[idx, 2], c=color, label=target)
    ax.set_xlabel('Mode ' + str(modes[0]))
    ax.set_ylabel('Mode ' + str(modes[1]))
    ax.set_zlabel('Mode ' + str(modes[2]))
    ax.legend()
    plt.title('MNIST Projected onto Principal Modes')
    plt.savefig('./Figures/proj_3_modes_all.png', facecolor=w)
    plt.show()


### 5. Pick two digits. See if you can build a linear classifier (LDA) that can reasonable identify/classify them.

extract data and labels

    X = np.array(mnist['data'])
    X = X.T
    y = np.array(mnist['target'])

pick two digits (1 and 2)

    dig1 = '1'
    dig2 = '2'

create masks on dataset which select only data with specified digits, define a dictionary for digit pairs

    mask1 = y == dig1
    mask2 = y == dig2
    data_dict = {}

concatenate data and labels

    X_1 = X[:, mask1]
    X_2 = X[:, mask2]
    y_1 = np.zeros(len(X_1[0]))
    y_2 = np.ones(len(X_2[0]))

    X_dig = np.concatenate((X_1, X_2), axis=1)
    X_dig = X_dig.T
    y_dig = np.concatenate((y_1, y_2))

separate training and test data, organize

    X_train, X_test, y_train, y_test = train_test_split(X_dig, y_dig, test_size=0.3, random_state=42)
    data_dict[('1','2')] = (X_train, y_train, X_test, y_test)

    one_two = ('1', '2')
    X_train = data_dict[one_two][0]
    y_train = data_dict[one_two][1]
    X_test = data_dict[one_two][2]
    y_test = data_dict[one_two][3]

initialise linear discriminant analysis type classifier, train

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

classify test data, calculate accuracy, and print

    y_pred = lda.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy for digits {(1, 2)}: {acc:.2f}")

### 7. Which two digits in the data set appear to be the most difficult to separate? Quantify the accuracy of the separation with LDA on the test data.

define function which fits a classifier to training and test data for two given digits, and prints the accuracy

    def fit_and_err(lda, train_set, pair):
        X_train = train_set[0]
        y_train = train_set[1]
        X_test = train_set[2]
        y_test = train_set[3]   
        # Train a linear classifier
        lda.fit(X_train, y_train)
        # Evaluate the performance on the test set
        y_pred = lda.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy for digits {pair}: {acc:.5f}")
        return acc

ititialize all possible pairs of unique digits (10 choose 2)

    digit_pairs = [(i, j) for i in range(10) for j in range(i + 1, 10)]

loop over pairs of digits, generate mask, apply mask to data,
concatenate 

### 8. Which two digits in the data set are most easy to separate? Quantify the accuracy of the separation with LDA on the test data.

### 9. SVM (support vector machines) and decision tree classifiers were the state-of-the-art until about 2014. How well do these separate between all ten digits? (see code below to get started).

### 10. Compare the performance between LDA, SVM and decision trees on the hardest and easiest pair of digits to separate (from above).

## Computational Results and Interpretation

### **Analysis**

### 2. Singular Value Spectrum and Modes
![Fig. 1. First 50 Singular Values](./Figures/first_50_singular_values.png)

![Fig. 2. First 12 Modes of U](./Figures/first_12_modes.png)

![Fig. 3. First and Second Mode Added](./Figures/lin_combination_1_2.png)

### 3. What is the interpretation of the U, Σ, and V matrices?

The U, $\Sigma$ and V matrices represent the transformation that the facial images data causes to a vector. 


Let's refer to the facial images data as matrix A. The columns of U form an orthonormal basis for the vector space of A, and capture the directions of maximum variance in the row space of A. In other words, the first column of U is the vector which when reshaped to image size represents the most significant vector in the basis of A, which you would need to reconstruct the majority of images which are columns in A.

The singular values of the diagonal of $\Sigma$ represent the amount of variance captured by each respective vector in U. Consider that taking the dot product of $u_1$ with every column vector in A, and taking the sum of these dot products would be a related value, and $u_2$ would have a smaller value. In fact, in this specific decomposition, the first U mode $u_1$ captures 43.5% of the variance, while all others only capture below 5 percent. This makes sense, given that the data set contains images of integers that are only slightly rotated from vertical, and are typically centered. The rest of the image is white space, and you need only one mode to get almost halfway towards plotting a number.


### 4. Projection of images onto three modes

![Fig. 4. Projection of data onto first 3 principal modes](./Figures/proj_3_modes_all.png)

### **Building and Comparing Classifiers**

### Results of LDA Classifier on two arbitrary digits

| Classifier  | Digits | Accuracy     |
| ------------|  ---------   |   ---------- |
| LDA      | 1, 2       |  0.98  |


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
