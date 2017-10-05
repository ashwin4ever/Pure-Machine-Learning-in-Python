#### Linear Classifiers

The dataset is generated randomly from a 2D multivariate Gaussian distribution. The number of datapoints generated is 5000 with random mean and covariances.

1. **generate_data.py** : Generates the required randomm sample of data
2. **LinearClassify.py** : Computes the maximum likelihood least squares
3. **lin_class_multi_data.py** : Classifies the dataset as belonging to either class for the sampled datapoints. Here we generate 10 X 1000     samples.
4. **lin_class.py** : Similar as above, but generates a single sample of 5000 points.

To test the classifier , run either **lin_class.py** or **lin_class_multi_data.py**. Output is visualized using Matplotlibs plotting functions
