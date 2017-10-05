#### Classification using Naive Bayes

In this classification the classic MNIST data files are used from : [MNIST](http://yann.lecun.com/exdb/mnist/).
The task is to classify the digits given the labels and report the accuracy.

1. **naive_bayes.py** : Implements the classic Naive Bayes algorithm in pure python. This converts the discrete values images to binary using a threshold parameter.
2. **naive_bayes_gaussian.py** : Implements the Naive Bayes using Gaussian probability distribution. Here it is continuous instead of categorical. Type 1 and Type 2 errors are calculated and a ROC curve is plotted.
3. **nn_classifier.py** : Classic Neares Neighbor alogrithm to predict the labels. A pure python based k- fold cross validation is performed to determine the accurate k value.

##### Datasets
1. **t10k-images.idx3-ubyte** : Images files used for testing.
2. **t10k-labels.idx1-ubyte** : Image labels for verification.
3. **train-labels.idx1-ubyte** : Image labels used for training.
4. **train-images.idx3-ubyte** : Not attached due to size limitations. This is the training image data file. Train the classifiers using this file.


To test this classifier , run either **naive_bayes.py** , **naive_bayes_gaussian.py** or **nn_classifier.py**.
