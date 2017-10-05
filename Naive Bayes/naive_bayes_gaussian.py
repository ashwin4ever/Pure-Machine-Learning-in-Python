'''
CS5793 - AI 2

2) Naive Bayes using Gaussian Distribution

'''
import numpy as np
import math
import matplotlib.pyplot as plt
import naive_bayes as NB
#from scipy import stats


def train(train_data , train_labels):

      '''
      Compute the mean and variance
      In this function the mean is calculated for each feature of each class and
      the variance is same for all the features. In this problem there are only two classes
      (i.e.) '5' or not a '5'. Mean will be calculated for both these classes and variance
      will be the same.

      '''

      mean = []
      
      # Get class 5
      label_5 = (train_labels == 5)
      image_5 = train_data[label_5]

      # Calculate mean for class 5
      mean_5 = ((np.sum(image_5, axis=0)+1) / (len(image_5)+10))

      # Get classes which is not 5
      label_rest = np.logical_not(label_5)
      image_rest = train_data[label_rest]

      # Calculate mean for classes not 5
      mean_rest = ((np.sum(image_rest, axis=0)+1) / (len(image_rest)+10))

      #mean_arr = np.vstack(mean)

      std_dev = np.std(train_data)

      # Calculate variance
      variance_5 = np.var(image_5)
      variance_rest = np.var(image_rest)

      return mean_5 , mean_rest , variance_5 , variance_rest


def gaussLikelihood(mean_5 , mean_rest , variance_5 , variance_rest , test_data):

      '''
      The Gaussian Likelihood is computed using the Probability Density function
      For each row of the test predict the class that the image belongs to

      pdf = (1 / sqrt(2 * pi * sigma ^2)) * exp(- (pow(x - mean , 2) / (2 * sigma ^ 2))
      exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
      return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

      '''
      mean_5 = mean_5.astype(np.float64)
      mean_rest = mean_rest.astype(np.float64)
      variance_5 = variance_5.astype(np.float64)
      variance_rest = variance_rest.astype(np.float64)
      test_data = test_data.astype(np.float64)

      prob_5 = 0.0
      prob_rest = 0.0
      prob_5 = (1 / np.sqrt(2 * math.pi * variance_5)) * np.exp(-(np.power(test_data - mean_5 , 2) / (2 * variance_5)))
      prob_rest = (1 / np.sqrt(2 * math.pi * variance_rest)) * np.exp(-(np.power(test_data - mean_rest , 2) / (2 * variance_rest)))

      # Caclulate log pdf
      prob_5 = np.sum(np.log2(prob_5) , axis = 1) 
      prob_rest = np.sum(np.log2(prob_rest) , axis = 1)

      return prob_5 , prob_rest


def buildRepresentatives(image , label):


      # Get representations of class 5
      # Extract label of class 5
      label_class_5 = (label == 5)
      label_class_rest = np.logical_not(label_class_5)

      # Extract image data of class 5
      image_arr_5 = image[label_class_5]
      label_arr_5 = label[label_class_5]

      # Get 1000 representations of class 5
      image_1000_5 = image_arr_5[np.random.choice(image_arr_5.shape[0] , 1000 , replace = False) , :]
      label_1000_5 = label_arr_5[np.random.choice(label_arr_5.shape[0] , 1000 , replace = False)]

      #Extract images and labels not 5
      image_arr_rest = image[label_class_rest]
      image_1000_rest = image_arr_rest[np.random.choice(image_arr_rest.shape[0] , 1000 , replace = False) , :]
      label_arr_rest = label[label_class_rest]
      label_1000_rest = label_arr_rest[np.random.choice(label_arr_rest.shape[0] , 1000 , replace = False)]      

      image_arr = np.vstack((image_1000_rest , image_1000_5))
      label_arr = np.vstack((label_1000_rest , label_1000_5))
      label_arr = label_arr.reshape(2000 , )

      # Shuffle image and labels
      shuffle_ind = np.arange(image_arr.shape[0])
      np.random.shuffle(shuffle_ind)
      image_arr = image_arr[shuffle_ind]
      label_arr = label_arr[shuffle_ind]

      # Get the split ratio
      # 90% -> training
      # 10% -> test data
      split = np.random.choice([True, False], image_arr.shape[0], p = [0.90, 0.10])
      train_data = image_arr[split == True]
      test_data = image_arr[split == False]

      train_label = label_arr[split == True]
      test_label = label_arr[split == False]


      
      '''
      # Plotting to check
      for i in range(5):
            plt.imshow(train_data[i , :].reshape(28 , 28))
            print(train_label[i])
            plt.colorbar()
            plt.show()
      
     '''
      return train_data , train_label , test_data , test_label




if __name__ == '__main__':
      # Process training data
      image_arr , label_arr = NB.processData('train-images.idx3-ubyte' , 'train-labels.idx1-ubyte', (60000 , 784))

      # Get N rows and D features
      N , D = image_arr.shape

      train_data , train_label , test_data , test_label = buildRepresentatives(image_arr , label_arr)

      mean_5 , mean_rest , variance_5 , variance_rest = train(train_data , train_label)

      '''
      Type 1 False positive : lambda 01
      
      Type 2 false negative : labda 10
      
      Type I errors are 5 times as costly as Type II errors : 5 (lambda_01 = 5lambda_02)
      
      Type I errors are twice as costly as Type II errors: 2 (lambda_01 = 2lambda10)
      
      Both types of errors are equally costly: 1 (lambda01 = lambda10)
      
      Type II errors are twice as costly as Type I errors: 1 / 2 
      
      Type II errors are 5 times as costly as Type I errors: 1 / 5
      
      '''
      # Error threshold
      thresh = [5 , 2 , 0.5 , 0.2]

      true_pos = 0
      true_neg = 0
      false_pos = 0
      false_neg = 0

      fpr = []
      tpr = []

      for t in thresh:

            prob_5 , prob_rest = gaussLikelihood(mean_5 , mean_rest , variance_5 , variance_rest , test_data)

            for i in range(test_data.shape[0]):

                  if prob_5[i] - prob_rest[i] > (t):

                        if test_label[i] == 5:
                              true_pos +=1
                        else:
                              false_pos += 1
                  else:
                        if test_label[i] == 5:
                              false_neg += 1

                        else:
                              true_neg += 1
      
            true_pos_rate = true_pos / (true_pos + false_neg)
            false_pos_rate  = false_pos / (true_neg + false_pos)
            fpr.append(false_pos_rate)
            tpr.append(true_pos_rate)
            print('True Positive: ' , true_pos)
            print('True Negative: ' , true_neg)
            print('False Positive: ' , false_pos)
            print('False Negative: ' , false_neg)
            print('{} {}'.format('True Positive Rate' , true_pos_rate))
            print('{} {}'.format('False Positive Rate' , false_pos_rate))
            print()

                        
      plt.plot(fpr , tpr)
      plt.title('ROC Curve')
      plt.xlabel('False Positive Rate', fontsize=15)
      plt.ylabel('True Positive Rate', fontsize=15)
      plt.show()      
