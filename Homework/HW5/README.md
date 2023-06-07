# Homework 5 Report - Training ML Models on Lorenz System Data

**Author**:

Ewan Lister

**Abstract**:

This report presents a comparative study of different neural network architectures for forecasting the dynamics of the Lorenz equations. The study focuses on training a neural network to advance the solution from time t to t+∆t for three different values of the control parameter ρ (10, 28, and 40). The trained neural network models are then evaluated for future state prediction at ρ=17 and ρ=35. The comparative analysis includes feed-forward neural networks, LSTM (Long Short-Term Memory) networks, RNN (Recurrent Neural Networks), and Echo State Networks. The report provides an overview of the theoretical background, details of the algorithm implementation and development, computational results, and concludes with a summary and key findings.


## Introduction and Overview

Neural networks provide a greate deal of versatility. Exploiting the weak law of large numbers, as long as there is a large amount of data and enough layers in the network. The network can acheive at least a decent fit. However, systems exist where the output of an input datum is a function of multiple data points, instead of just one. This is where a NN begins to falter. Consider, for instance, well saturated trajectory data of a baseball, where the target we are training for is the next position of the ball after 1 second. If we are only given the ball's position as data and not its velocity, then a neural network would be hard pressed to predict the next position. In fact, a single position measurement of a moving object is a projection of acceleration and velocity (for which time is a basis vector) onto 3 dimensional space. Thus there is no way to capture time based relationships unless multiple positions are considered simultaneously. A neural network trains on data piece by piece. Thus it can't infer that the ball will travel in $\hat{x}$ because the last 3 steps showed a significant change in that direction, atleast if there are multiple trajectories to train on. 

What modification do we need to make in order to train time dependent models? We need memory. Recurrent Neural Networks, Long Short-Term Memory, and Echo State Networks each provide some capability for memory of previously trained data using their complex networks. In the following section we will explore their mathematical foundations.


## Theoretical Background

### Recursive Neural Networks

Recurrent Neural Networks (RNNs) are a class of neural networks designed to process sequential data by maintaining an internal hidden state that captures information from previous time steps. The mathematical theory behind RNNs involves the concept of recurrent connections and the unfolding of the network through time.

Let's consider a time series input sequence of length T, denoted by $\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T)$, where $\mathbf{x}_t$ represents the input at time step t. An RNN updates its hidden state $\mathbf{h}_t$ at each time step based on the current input $\mathbf{x_t}$ and the previous hidden state $\mathbf{h}_{t-1}$.

The hidden state $\mathbf{h}_t$ of an RNN is computed using the following equation:

$$
\mathbf{h_t} = \sigma(\mathbf{W}_{\text{in}} \mathbf{x_t} + \mathbf{W_{\text{rec}}} \mathbf{h}_{t-1} + \mathbf{b})
$$
 
Here, $\sigma(\cdot)$ represents an activation function such as the sigmoid or hyperbolic tangent, $\mathbf{W_{\text{in}}}$ is the input weight matrix, $\mathbf{W}_{\text{rec}}$ is the recurrent weight matrix, and $\mathbf{b}$ is a bias vector.

The recurrent connection allows the hidden state to retain information from previous time steps, enabling the RNN to model dependencies and capture temporal patterns in the input sequence. However, a common issue with traditional RNNs is the vanishing or exploding gradients problem, which can hinder their ability to capture long-term dependencies.

To address this issue, variations of RNNs have been developed, such as Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks. These architectures incorporate gating mechanisms to selectively update and pass information through time, mitigating the vanishing gradients problem.

The output of an RNN can be obtained by applying a linear transformation to the hidden state:

\[
\mathbf{y}_t = \mathbf{W}_{\text{out}} \mathbf{h}_t
\]

Here, $\mathbf{W}_{\text{out}}$ is the output weight matrix.

RNNs can be trained using backpropagation through time (BPTT), which extends the backpropagation algorithm to handle the sequential nature of the network. The objective is typically to minimize a suitable loss function, such as mean squared error or cross-entropy, by adjusting the network's parameters through gradient descent or its variants.

RNNs have shown remarkable success in various applications involving sequential data, including language modeling, machine translation, speech recognition, and sentiment analysis. They excel at modeling dependencies over variable-length sequences, making them powerful tools for tasks that involve sequential information processing.

### LSTM Networks

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) that are widely used in deep learning for sequence modeling tasks. The mathematical theory behind LSTM networks involves the concept of memory cells and gates.

Let's consider a time series input sequence of length T, denoted by $\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T)$, where $\mathbf{x}_t$ represents the input at time step t. An LSTM network consists of a set of memory cells, each responsible for capturing and propagating information across different time steps. At each time step t, a memory cell is updated based on the current input $\mathbf{x}_t$ and the previous hidden state $\mathbf{h}_{t-1}$.

The hidden state $\mathbf{h}_t$ of an LSTM network is computed using the following equations:

\[
\begin{align*}
\mathbf{i}_t &= \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i) \\
\mathbf{f}_t &= \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f) \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o) \\
\mathbf{g}_t &= \tanh(\mathbf{W}_g \mathbf{x}_t + \mathbf{U}_g \mathbf{h}_{t-1} + \mathbf{b}_g) \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\end{align*}
\]

Here, $\sigma(\cdot)$ is the sigmoid activation function, $\odot$ denotes element-wise multiplication, and $\mathbf{i}_t$, $\mathbf{f}_t$, $\mathbf{o}_t$, and $\mathbf{g}_t$ are called input gate, forget gate, output gate, and candidate cell state vectors, respectively. $\mathbf{W}$ and $\mathbf{U}$ are weight matrices, and $\mathbf{b}$ represents bias vectors.

The input gate $\mathbf{i}_t$ determines how much of the new input $\mathbf{x}_t$ should be added to the cell state $\mathbf{c}_t$. The forget gate $\mathbf{f}_t$ controls how much of the previous cell state $\mathbf{c}_{t-1}$ should be retained. The output gate $\mathbf{o}_t$ determines how much of the cell state $\mathbf{c}_t$ should be output as the hidden state $\mathbf{h}_t$. The candidate cell state $\mathbf{g}_t$ represents the information that can be potentially stored in the cell state.

In this way, LSTM networks are able to selectively retain or forget information from previous time steps, allowing them to capture long-range dependencies in sequential data. This property makes LSTM networks particularly effective for tasks such as speech recognition, machine translation, and sentiment analysis, where maintaining contextual information over long sequences is crucial.

### Echo State Networks

Echo State Networks (ESNs) are a type of recurrent neural network (RNN) that leverage the concept of reservoir computing. The mathematical theory behind ESNs involves the idea of a dynamic reservoir and a readout layer.

Let's consider a time series input sequence of length T, denoted by $\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T)$, where $\mathbf{x}_t$ represents the input at time step t. An ESN consists of a reservoir of recurrently connected neurons that exhibit complex dynamics. The reservoir is updated at each time step based on the current input $\mathbf{x}_t$ and the previous reservoir state $\mathbf{r}_{t-1}$.

The reservoir state $\mathbf{r}_t$ of an ESN is computed using the following equation:

\[
\mathbf{r}_t = \tanh(\mathbf{W}_{\text{in}} \mathbf{x}_t + \mathbf{W}_{\text{res}} \mathbf{r}_{t-1})
\]

Here, $\tanh(\cdot)$ represents the hyperbolic tangent activation function, $\mathbf{W}_{\text{in}}$ is the input weight matrix, and $\mathbf{W}_{\text{res}}$ is the reservoir weight matrix.

The reservoir acts as a high-dimensional dynamic memory, capturing and processing temporal information from the input sequence. The reservoir neurons have random or fixed weights, and they are typically sparsely connected to reduce computational complexity. Additionally, the reservoir weights are often set before training and remain fixed throughout the learning process.

The output of an ESN is obtained by feeding the reservoir state $\mathbf{r}_t$ into a linear readout layer:

\[
\mathbf{y}_t = \mathbf{W}_{\text{out}} \mathbf{r}_t
\]

Here, $\mathbf{W}_{\text{out}}$ is the readout weight matrix.

The readout layer learns to map the reservoir state to the desired output by minimizing a suitable objective function. This is typically achieved using techniques such as ridge regression or gradient-based optimization methods.

ESNs offer several advantages, including their simple training procedure and computational efficiency. The dynamic reservoir provides rich temporal dynamics that enable the network to effectively capture and process complex temporal patterns. ESNs have been successfully applied to various tasks, including time series prediction, speech recognition, and control problems.


###
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


$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

where $f_t$, $i_t$, and $o_t$ are the forget, input, and output gates at time $t$, respectively, $\sigma$ is the sigmoid function, $W$ and $b$ are the weight and bias parameters, $h_{t-1}$ is the hidden state from the previous time step, and $x_t$ is the input at the current time step. The forget gate determines how much of the past information (i.e., the cell state) to retain, the input gate decides how much of the current information to store in the cell state, and the output gate determines how much of the information in the cell state to reveal to the next layers in the network.

## Algorithm Implementation and Development

import in relevant libraries, we need all of the classifier models from canonical ML, as well as torch

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
    from sklearn.model_selection import train_test_split
    import torch.optim as optim
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC

initialize X and Y data in tensor form

    X = torch.arange(0, 31, dtype=torch.float32).reshape(-1, 1)
    Y = torch.tensor([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
                    40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53],
                    dtype=torch.float32).reshape(-1, 1)

    data = dict(zip(X, Y))

### (i) Fit the data to a three layer feed forward neural network.

define the neural network architecture

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

initialize learning rate, network, optimizer, and loss criterion

    lr = 0.0001
    net = ThreeLayerNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

train the neural network using gradient descent, no batching, and print loss after all pairs have been
back propogated

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

define a function which can be used to check error quickly on training and test data

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

isolate first 20 data points

    x_train = X[0:20]
    y_train = Y[0:20]
    x_test = X[20:31]
    y_test = Y[20:31]


train network on first 20 data points, examine progress of SGD via print statements

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

check error on training and test data

    check_train_test_error(x_train, y_train, x_test, y_test)

### (iii) Repeat (iii) but use the first 10 and last 10 data points as training data. Then fit the model to the test data (which are the 10 held out middle data points). Compare these results to (iii)

isolate first and last 10 training points

    x_train = torch.cat([X[0:10], X[20:31]])
    y_train = torch.cat([Y[0:10], Y[20:31]])
    x_test = X[10:20]
    y_test = Y[10:20]

set optimizer

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

train network on first and last 10 data points, examine progress of SGD via print statements

    num_epochs = 50
    for epoch in range(num_epochs):
        for i, (x) in enumerate(x_train):
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y_train[i])
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, 20, loss.item()))

check error on training data and middle 10 points

    check_train_test_error(x_train, y_train, x_test, y_test)

### (iv) Compare the models fit in homework one to the neural networks in (ii) and (iii)

Similarly to the curve fitting in homework 1, the neural network does a poor job of making any extrapolations about its test data if the data is outside of the domain of the training data. For example, the network did well when test data contained the 10 points between point 9 and point 20, but poorly when the test data was that from 20 to 31, which is unbounded by any training data. Thus is performs very similarly to the curve fitting in homework 1. However, the loss, for each value is still much greater in the case of the neural network.


## II Now train a feedforward neural network on the MNIST data set. You will start by performing the following analysis:

### (i) Compute the first 20 PCA modes of the digit images.

fetch MNIST dataset, convert data and labels into numpy arrays

    mnist = fetch_openml('mnist_784', version=1)
    data = np.array(mnist['data'])
    labels = np.array(mnist['target'])

apply PCA transformation onto the first 20 modes

    pca = PCA(n_components=20)
    data_pca_1 = pca.fit_transform(data)


### (ii) Build a feed-forward neural network to classify the digits. Compare the results of the neural network against LSTM, SVM (support vector machines) and decision tree classifiers.

separate training and test data for use in LSTM, SVM, and DTC classifiers. convert labels to ints

    data_train, data_test, label_train, label_test = train_test_split(data_pca_1, labels, test_size=0.3, random_state=42)
    label_train = label_train.astype(np.int16)
    label_test = label_test.astype(np.int16)

### testing neural network on MNIST data

define batch size, learning rate, and epoch number as the hyperparameters

    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10

define the model architecture

    class FeedforwardNN(nn.Module):
        def __init__(self):
            super(FeedforwardNN, self).__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
            
        def forward(self, x):
            x = x.view(-1, 784)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x


initialize the model and optimizer

    model = FeedforwardNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


train the model

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print('Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


evaluate the model on the test set

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

### testing LSTM on MNIST data

define the hyperparameters

    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10
    hidden_size = 128
    num_layers = 2


define the LSTM architecture

    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size=28, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 10)
            
        def forward(self, x):
            h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
            c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
            out, (h_n, c_n) = self.lstm(x, (h0, c0))
            out = self.fc(h_n[-1])
            return out

initialize the model and optimizer

    model = LSTM()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train the model

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.view(batch_size, 28, 28)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print('Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

evaluate the model on the test set

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.shape[0], 28, 28)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

### fitting an SVM classifier

train a support vector machine classifier

    clf = SVC()
    clf.fit(data_train, label_train)

evaluate the performance on the test set

    y_pred = clf.predict(data_test)
    acc = accuracy_score(label_test, y_pred)
    print(f"Accuracy for SVM: {acc:.2f}")

### fitting a DTC classifier

train a decision tree classifier

    clf = DecisionTreeClassifier()
    clf.fit(data_train, label_train)

evaluate the performance on the test set

    y_pred = clf.predict(data_test)
    acc = accuracy_score(label_test, y_pred)
    print(f"Accuracy for DTC: {acc:.2f}")

## Computational Results and Interpretation

### **Part 1**

### Training a 3 Layer FFNN on all 30 points

![Fig. 1. FNN Training on scatter distribution](./Figures/fnnepochs.jpg)

| ML Model Used| # of data points| MSE |
| Neural Network | 30
### Training a 3 Layer FFNN on first 20 points

![Fig. 2. FNN Training on x = {0 - 20} points in distribution](./Figures/fnnepochs2.jpg)

### Training a 3 Layer FFNN on First and Last 10 points

![Fig. 3. FNN Training on x = {0 - 10: 20 - 30} points in distribution](./Figures/fnnepochs3.jpg)

### Comparing models to HW1 performance


### **Part 2**

### Build a FFNN for classifying MNIST Data

![Fig. 4. FNN Training on MNIST data](./Figures/fnnMNIST1.jpg)

### Comparison with LSTM, SVM, and DTC

![Fig. 4. LSTM Training on MNIST data](./Figures/lstmMNIST1.jpg)

## Summary and Conclusions

In conclusion, the analysis of the MNIST dataset using SVD, LDA, SVM, and decision tree classifiers provided valuable insights into the classification and separation of handwritten digits. The singular value spectrum analysis revealed the rank of the digit space, and the interpretation of the U, Σ, and V matrices further clarified the relationships between the digit images. The 3D projection onto selected V-modes offered a visual representation of the data in PCA space, which facilitated the development of linear classifiers for digit identification.

The classification performance varied depending on the chosen digits and classification techniques. Notably, digits 6 and 7 were the easiest to separate, while digits 9 and 7 presented the greatest challenge. Among the classifiers tested, the SVM classifier demonstrated the best overall performance in separating all ten digits, outperforming both LDA and decision tree classifiers, particularly for the hardest and easiest pairs of digits to separate.

The results underscore the importance of selecting appropriate classification techniques for specific tasks and highlight the effectiveness of SVM classifiers for this particular dataset. Additionally, the report emphasized the importance of visualizations and comparisons between training and test sets to ensure reliable and generalizable conclusions. Future work could explore other classification algorithms and feature extraction methods to further improve the performance of digit classification and recognition tasks.