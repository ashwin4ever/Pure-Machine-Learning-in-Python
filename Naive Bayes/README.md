#### Classification using Naive Bayes

In this classification the classic MNIST data files are used from : [MNIST](http://yann.lecun.com/exdb/mnist/)

1. **generate_data.py** : Generates the required randomm sample of data
2. **LinearClassify.py** : Computes the maximum likelihood least squares
3. **lin_class_multi_data.py** : Classifies the dataset as belonging to either class for the sampled datapoints. Here we generate 10 X 1000     samples.
4. **lin_class.py** : Similar as above, but generates a single sample of 5000 points.

To test the classifier , run either **lin_class.py** or **lin_class_multi_data.py**. Output is visualized using Matplotlibs plotting functions


##### KD Trees
1. **KDTRee.py** : A standard K-d tree implementation. Used to classify the datapoints generated from Gaussian samples by looking up the nearest neughbors.
2. **KDTree_Multi.py** : Similar implementation as above but classifies 10 X 1000 samples.


To test this classifier , run either **KDTRee.py** or **KDTree_Multi.py**. Output is visualized using Matplotlibs plotting functions
