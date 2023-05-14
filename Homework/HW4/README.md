# Homework 4 Report - Comparing Neural Networks with Traditional ML Classifiers

**Author**:

Ewan Lister

**Abstract**:

This report explores the construction of feedforward neural networks, or **FFNNs**, and how they compare to other canonical machine learning techniques. In the first part of the analysis, a three-layer feed-forward neural network is constructed and fitted to a dataset of 31 data points. The network is trained on different samples of the 31 points, and tested on the remainder. The least square errors of the network on the training and test data are computed and compared to the models fit in homework one. In the second part of the analysis, a feed-forward neural network is trained on a PCA compression of the MNIST dataset. The results of the neural network were compared against those of LSTM, SVM, and decision tree classifiers. The report includes code snippets, visualizations, and interpretations of the results.

## Introduction and Overview

Neural networks have become one of the most widely used and effective methods in data science for solving various machine learning problems, such as classification, regression, and image recognition. Neural networks are inspired by the functioning of the human brain and consist of multiple layers of interconnected neurons that can learn and extract complex patterns and relationships from data. The popularity of neural networks is due to their ability to automatically learn complex features from raw data, handle large amounts of data, and generalize well to unseen data.

In this report, we compare the performance of a feed-forward neural network with that of other popular machine learning models, such as LSTM, SVM, and decision tree classifiers. LSTM is a type of recurrent neural network that is particularly useful for handling sequential data, such as time series or natural language processing. SVM is a popular method for binary classification that tries to find the optimal decision boundary that maximally separates the two classes. Decision trees are another type of machine learning model that can be used for both classification and regression tasks and are particularly useful for generating interpretable models.

We will begin by covering the theoretical background of NNs, and LSTM, wish a short refresher on SVM and decision tree classifiers. This will follow with an implementation of these models into python, interpreting the results of training the models, followed by a summary and conclusion.

## Theoretical Background

### Neural Networks

A three-layer feedforward neural network consists of an input layer, a hidden layer, and an output layer. Each layer is composed of multiple neurons, also known as nodes, that receive inputs, perform computations, and generate outputs. The input layer receives the input data, and the output layer generates the final output of the network. The hidden layer(s) perform intermediate computations and extract relevant features from the input data.

Let $x$ be the input data, $y$ be the desired output, $W$ and $b$ be the weights and biases of the neurons, and $\sigma$ be the activation function of the neurons. The computation performed by a three-layer feedforward neural network can be expressed mathematically as follows:

$$h_1 = \sigma(W_1 x + b_1)$$

$$h_2 = \sigma(W_2 h_1 + b_2)$$

$$\hat{y} = \sigma(W_3 h_2 + b_3)$$

where $h_1$ and $h_2$ are the hidden layer outputs, and $\hat{y}$ is the predicted output of the network. $W_1$, $W_2$, and $W_3$ are the weight matrices between the layers, and $b_1$, $b_2$, and $b_3$ are the bias vectors of the neurons. The activation function $\sigma$ is typically a non-linear function that introduces non-linearity into the network and allows it to learn complex relationships between the input and output data. Some common activation functions include the sigmoid function, the ReLU function, and the hyperbolic tangent function.

In practice, the weights and biases of the network are learned through a process called backpropagation, where the network is trained on a set of training data to minimize the difference between the predicted outputs and the desired outputs. This process involves computing the gradient of the loss function with respect to the weights and biases, and updating them using an optimization algorithm such as stochastic gradient descent (SGD).

## Long Short-Term Memory (LSTM) Networks

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) that are designed to avoid the long-term dependency problem. This is achieved through their unique cell state structure which allows them to maintain and access information over long sequences, making them particularly effective for tasks involving sequential data such as time series analysis, natural language processing, and more.

The core idea behind LSTMs is the cell state, which runs straight down the entire chain, with only minor linear interactions. It's the LSTM's ability to regulate the cell state's information that makes it so special. At each step in the sequence, there are structures called gates that regulate the information flow into and out of the cell state. These gates are a way to optionally let information through, and they are composed out of a sigmoid neural net layer and a pointwise multiplication operation. The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means "let nothing through," while a value of one means "let everything through!" An LSTM has three of these gates: the forget gate, the input gate, and the output gate. These are defined mathematically as follows:

\[
    f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
\]
\[
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
\]
\[
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
\]

where $f_t$, $i_t$, and $o_t$ are the forget, input, and output gates at time $t$, respectively, $\sigma$ is the sigmoid function, $W$ and $b$ are the weight and bias parameters, $h_{t-1}$ is the hidden state from the previous time step, and $x_t$ is the input at the current time step. The forget gate determines how much of the past information (i.e., the cell state) to retain, the input gate decides how much of the current information to store in the cell state, and the output gate determines how much of the information in the cell state to reveal to the next layers in the network.

## Algorithm Implementation and Development

    import torch
    import torch.nn as nn
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import scipy.io as sio
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Lasso
    from sklearn.decomposition import PCA
    from scipy.io import loadmat
    from sklearn.datasets import fetch_openml

    X = torch.arange(0, 31, dtype=torch.float32).reshape(-1, 1)
    Y = torch.tensor([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
                    40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53],
                    dtype=torch.float32).reshape(-1, 1)

    data = dict(zip(X, Y))

### (i) Fit the data to a three layer feed forward neural network.

    # Define the neural network architecture
    class ThreeLayerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1, 20)  # input layer -> hidden layer
            self.fc2 = nn.Linear(20, 10) # hidden layer -> hidden layer
            self.fc3 = nn.Linear(10, 1)  # hidden layer -> output layer
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # initialize network
    net = ThreeLayerNet()

    # use SGD for fitting
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    # loss function
    criterion = nn.MSELoss()

    # create training data loader
    train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=1, shuffle=True)

    # Train the neural network using gradient descent
    num_epochs = 15
    for epoch in range(num_epochs):
        for i, (x) in enumerate(X):
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, Y[i])
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 31 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, 31, loss.item()))

### (ii) Using the first 20 data points as training data, fit the neural network. Compute the least-square error for each of these over the training points. Then compute the least square error of these models on the test data which are the remaining 10 data points.

    def check_train_test_error(x_train, y_train, x_test, y_test):
        for i, (x) in enumerate(x_train):
            outputs = net(x)
            error = criterion(outputs, y_train[i])
            print('Train error for x = {}, y = {} : {:.4f}'.format(x, y_train[i], error))
        print('\n')
        for i, (x) in enumerate(x_test):
            outputs = net(x)
            error = criterion(outputs, y_test[i])
            print('Test error for x = {}, y = {} : {:.4f}'.format(x, y_test[i], error))

    # isolate first 20 data points
    x_train = X[0:20]
    y_train = Y[0:20]
    x_test = X[20:31]
    y_test = Y[20:31]

    # train network on first 20 data points, examine progress of SGD via print statements
    num_epochs = 15
    for epoch in range(num_epochs):
        for i, (x) in enumerate(x_train):
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y_train[i])
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, 20, loss.item()))

    check_train_test_error(x_train, y_train, x_test, y_test)

## Computational Results and Interpretation

### **Analysis**

### 2. Singular Value Spectrum and Modes
![Fig. 1. First 50 Singular Values](./Figures/first_50_singular_values.png)

![Fig. 2. First 12 Modes of U](./Figures/first_12_modes.png)

Because the U modes of the MNIST matrix form a basis for the space in which all of the images exist, we can actually begin to construct legible images by simply taking linear combinations of the U modes. After all, an MNIST image is really just a 784 dimension vector. The first mode of U already accounts for about 44% of the variance between vectors, so by adding the first mode to the second mode, we get something that quite closely resembles a well known digit:

![Fig. 3. First and Second Mode Added](./Figures/lin_combination_1_2.png)

The addition or subtraction of other scaled U modes will allow us to perfectly recreate all of the images within the matrix. In theory we need 784 orthogonal modes to span every vector in A, but to the human eye we likely only need 30-40 modes to recreate images that resemble digits. Any subsequent modes are fitted to very slight variations between images.

### 3. What is the interpretation of the U, Σ, and V matrices?

The U, $\Sigma$ and V matrices represent the transformation that the facial images data causes to a vector. 

Let's refer to the facial images data as matrix A. The columns of U form an orthonormal basis for the vector space of A, and capture the directions of maximum variance in the row space of A. In other words, the first column of U is the vector which when reshaped to image size represents the most significant vector in the basis of A, which you would need to reconstruct the majority of images which are columns in A.

The singular values of the diagonal of $\Sigma$ represent the amount of variance captured by each respective vector in U. Consider that taking the dot product of $u_1$ with every column vector in A, and taking the sum of these dot products would be a related value, and $u_2$ would have a smaller value. In fact, in this specific decomposition, the first U mode $u_1$ captures 43.5% of the variance, while all others only capture below 5 percent. This makes sense, given that the data set contains images of integers that are only slightly rotated from vertical, and are typically centered. The rest of the image is white space, and you need only one mode to get almost halfway towards plotting a number.

As for V, V is simply the basis for the row space of A, where each column of V is a vector, and each vector is ranked according to how much variance it's direction captures. SVD forms of A would contain $V^T$, so instead the rows of V are a basis for the row space of A. They are equivalently the modes of V, and are perhaps slightly less intuitive in this example because they do not describe the column vectors, which are the images, but instead the vector constructed from taking one pixel from each image. 


### 4. Projection of images onto three modes

![Fig. 4. Projection of data onto first 3 principal modes](./Figures/proj_3_modes_all.png)

When we project onto modes 2, 3, and 5, we are essentially taking the dot product of one column in A with some label against mode 2, then mode 3, then mode 5. The results of these dot products form a 3d coordinate, we then plot this coordinate and label it according to the digit that described the image. Carry this out for all images in the data, and soon you have a graphical understanding of how all of the points relate to the modes. Notes that the labels on the graph are zero indexed, so Mode 1 on the graph corresponds to mode 2 in reality.

In the plot much of the data falls within similar areas, but some distinct features are also present. Examinine the 3d animation below. In general the cloud of points are contained in a cross formation, where a large cloud consisting mostly 8s, 7s, and 9s extends up into the positive levels of correlation to Mode 5. Looking at the two more linear patches of lines, most of the data is clustered at the intersection of these lines, or on the lines themselves. If we were to curve fit functions to these pairs of lines, then we could begin to identify specific parameters which described how to linearly combine the modes to create an image along that line. 


![Fig. 5. Gif of Projection of data onto first 3 principal modes](./Figures/proj_3_modes_all.gif)

If this were an unsupervised learning example, PCA would still be very useful for interpreting the data, as distinct spacial patterns are visible to the human eye even without the labeling of points.

### **Building and Comparing Classifiers**

### Results of LDA Classifier on two arbitrary digits

| Classifier  | Digits | Accuracy     |
| ------------|  ---------   |   ---------- |
| LDA      | 1, 2       |  0.98  |

Initially we selected two digits which are easy to separate as a benchmark for the LDA classifier. 1 has almost no correlation with the feature space we would associate with horizontal bars found in most numbers. Consider 7, 5, 9, 8, and 2. All of these numbers have some sort of extended line segment at their top. Thus it's likely trivial for an LDA classifier to separate simple vertical line segment 1s from 2s. In the case where images contained a 1 which contains a small bent serif at the top, this would extend the 1 into the aforementioned feature space, not to mention the large serif at the bottom, which resembles that at the base of a 2.

This is just an example of the similarities and differences between digits that would influence how they are classified, if LDA seeks to reduce the dimensionality of data while still specifying the most significant differences between distinct labels, these similarities can decrease its accuracy.


### Results of LDA Classifier on all 45 digit pairs

| Classifier | Digits | Accuracy |
|------------|--------|----------|
| LDA        | (0,1)  | 99.233%  |
| LDA        | (0,2)  | 98.129%  |
| LDA        | (0,3)  | 98.932%  |
| LDA        | (0,4)  | 99.150%  |
| LDA        | (0,5)  | 97.982%  |
| LDA        | (0,6)  | 98.936%  |
| LDA        | (0,7)  | 99.460%  |
| LDA        | (0,8)  | 98.446%  |
| LDA        | (0,9)  | 99.110%  |
| LDA        | (1,2)  | 98.207%  |
| LDA        | (1,3)  | 98.535%  |
| LDA        | (1,4)  | 99.229%  |
| LDA        | (1,5)  | 98.591%  |
| LDA        | (1,6)  | 99.164%  |
| LDA        | (1,7)  | 98.770%  |
| LDA        | (1,8)  | 96.418%  |
| LDA        | (1,9)  | 99.191%  |
| LDA        | (2,3)  | 96.509%  |
| LDA        | (2,4)  | 97.949%  |
| LDA        | (2,5)  | 97.494%  |
| LDA        | (2,6)  | 97.620%  |
| LDA        | (2,7)  | 97.736%  |
| LDA        | (2,8)  | 96.429%  |
| LDA        | (2,9)  | 98.399%  |
| LDA        | (3,4)  | 99.045%  |
| LDA        | (3,5)  | 95.665%  |
| LDA        | (3,6)  | 99.025%  |
| LDA        | (3,7)  | 98.291%  |
| LDA        | (3,8)  | 95.990%  |
| LDA        | (3,9)  | 97.825%  |
| LDA        | (4,5)  | 98.808%  |
| LDA        | (4,6)  | 98.856%  |
| LDA        | (4,7)  | 97.616%  |
| LDA        | (4,8)  | 98.755%  |
| LDA        | (4,9)  | 96.034%  |
| LDA        | (5,6)  | 96.790%  |
| LDA        | (5,7)  | 98.824%  |
| LDA        | (5,8)  | 95.840%  |
| LDA        | (5,9)  | 98.016%  |
| LDA | (6,7) | 99.624% |
| LDA | (6,8) | 98.103% |
| LDA | (6,9) | 99.470% |
| LDA | (7,8) | 98.442% |
| LDA | (7,9) | 95.580% |
| LDA | (8,9) | 97.364% |


Outlined above, we have 45 pairs of digits and the corresponding accuracy under an LDA classifier. Digit pairs with very low classification accuracy are 7 and 9, 5 and 8, 3 and 8, and 5 and 3. As you may know, 8 is a problem child in the digits. 6 and 7 are the easiest digits to separate. The accuracy matrix shown below provides a good illustration of the relationship between each digit. In a way, it is a similarity matrix for 10 digits. The digits with the least classification accuracy have similar features. Note that the diagonal of the matrix was set to 8 so as not to skew the data.

### Classification Accuracy Matrix for digit pairs

![Fig. 5. Accuracy Matrix for Digit Pairs](./Figures/acc_lda_mat.png)

### Performance of other classifiers

| Classifier | Digits | Accuracy |
|------------|--------|----------|
| LDA        | (7,9)  | 95.580%  |
| SVM        | (7,9)  | 98.901%  |
| DTC        | (7,9)  | 97.568%  |
| LDA        | (6,7)  | 99.624%  |
| SVM        | (6,7)  | 99.953%  |
| DTC        | (6,7)  | 99.459%  |

LDA, SVM, and DTC classifiers all facilitate the separation of data. However, some classifiers perform better than others in separating the "easiest" and most "difficult" digits. Across the board the SVM classifier outperforms all others. This may be due to the non-linearity of the process. As there is a large degree of variation even within a single digit class, and digits may also share features with other digits, any sort of linear separation will not create the sort of fine precision within feature spaces that non-linear functions could create. This is the power of SVM. 

The DTC is in close second, performing 2% better than LDA for the 7 and 9, but performing .2% worse than LDA for 6 and 7. Decision trees are recursive by nature, so once a preliminary division is made, the opportunity to retroactively fix a division between data is not as easy. 

## Summary and Conclusions

In conclusion, the analysis of the MNIST dataset using SVD, LDA, SVM, and decision tree classifiers provided valuable insights into the classification and separation of handwritten digits. The singular value spectrum analysis revealed the rank of the digit space, and the interpretation of the U, Σ, and V matrices further clarified the relationships between the digit images. The 3D projection onto selected V-modes offered a visual representation of the data in PCA space, which facilitated the development of linear classifiers for digit identification.

The classification performance varied depending on the chosen digits and classification techniques. Notably, digits 6 and 7 were the easiest to separate, while digits 9 and 7 presented the greatest challenge. Among the classifiers tested, the SVM classifier demonstrated the best overall performance in separating all ten digits, outperforming both LDA and decision tree classifiers, particularly for the hardest and easiest pairs of digits to separate.

The results underscore the importance of selecting appropriate classification techniques for specific tasks and highlight the effectiveness of SVM classifiers for this particular dataset. Additionally, the report emphasized the importance of visualizations and comparisons between training and test sets to ensure reliable and generalizable conclusions. Future work could explore other classification algorithms and feature extraction methods to further improve the performance of digit classification and recognition tasks.