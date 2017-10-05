'''
CS5793 - AI 2

1) Naive Bayes Classifier

'''

import numpy as np
import math
import matplotlib.pyplot as plt



def calculatePrior(labels):

      '''
      Calculate the prior probabilty of each class label.
      It is calculated by # of times the class label appears in the
      data divided by the total number of training sets

      P(Y = y_k) = # Dataset(Y = y_k) / |Dataset|
      '''

      class_arr , class_count = np.unique(labels , return_counts = True)
      prior = np.divide(class_count , len(labels))
      
      return prior


def calculateLikelihood(labels , training_data):

      '''
      This calculates the maximum likelihood
      (i.e.) the proability of each attribute given a class
      theta = P(x_d = 1|k) or
      theta = P(X_i = x_ij | Y = y_k)

      For each attribute or feature and for each class calculate
      the probability by # of attribute values for each class divided
      by the number of that class sample.

      P(X_i = x_ij | Y = y_k) =
      # Dataset(X_i = x_ij for Y = y_k)/ |Dataset(y = y_k)|
      
      '''

      likelihood = []
      smoothing = 1

      # Caclulate the nuumber of black pixels
      # for each class of k = 0 to 9
      # Iterate through each digit and sum the black pixels (i.e.) value of 1
      for k in range(10):

            filter_pixels = (labels == k)
            train = training_data[filter_pixels]

            likelihood.append((np.sum(train, axis = 0) + smoothing)/ (len(train) + (smoothing * 10)))

      black_likelihood = np.vstack(likelihood)

      # Complement of the black pixel probabilties gives the white pixel likelihood
      white_likelihood = 1 - black_likelihood
      return (black_likelihood , white_likelihood)


def posterior(test_data , prior , theta_black , theta_white):

      '''

      This calculates the joint probability or posterior distribution over labels given
      images. Predicts the class label given an image.
      Uses log to compute and this avoids any numeric underflow.

      This uses the Naive Bayes classifier algorithm (i.e).
      find the class C_i that an object D most likely belongs.
      For this, we compute a probability for D belonging to each C_i,
      and choose the C_i that maximizes this probability.
      Choosing such a Ci is exactly what argmax does for you


      C_predict = argmax P(C = c_i) * Product(P(D|C = c_i))
      
      '''


      best_result = []

      
      # Run through each digit row
      # Find probability for white and black pixels   
      for r in range (0, 10000):
            
            row = test_data[r]
            predicted_prob = []

            # Iterate for every class (0 - 9)
            # For every image, get the probability of it belonging to class k
            # Pick the max probability for the image for class k
            for k in range(10):
                  
                  # Initial probability must be set to 0 for every image
                  prob = 0.0

                  # Extract the white and black pixels
                  white_pixel = (row == 0)
                  black_pixel = (row == 1)
                  
                  # Get the maximum likelihood
                  # for each class k for black and white pixels
                  white_prob = theta_white[k][white_pixel]
                  black_prob = theta_black[k][black_pixel]

                  # Sum the probabilities using log() instead of multiplying
                  prob += np.sum(np.log2(black_prob))
                  prob += np.sum(np.log2(white_prob))
                  
                  # Compute prior probabilty for class k
                  prob += np.log2(prior[k])

                  # Append all probabiltiies for class k
                  predicted_prob.append(prob)

            # Get the best probability for class k
            best_result.append(np.argmax(predicted_prob))

      return np.array(best_result)
            
      


def processData(image_nm , label_nm , shape):

      offset = [16 , 8]
      # Load the MNIST file
      # Load the trained images
      img_file = open(image_nm , 'rb')
      images = img_file.read()
      img_file.close()

      # Load the trained labels file
      label_file = open(label_nm , 'rb')
      label = label_file.read()
      label_file.close()

      # Extract from the 16th position to get image data
      images = bytearray(images)
      images = images[offset[0] : ]

      # COnvert the images data into numpy array
      # Shape the array into 60000 rows (N) and 784 features (D)
      image_arr = np.array([images])
      image_arr = image_arr.reshape(shape)

      # Extract the labels
      label = bytearray(label)
      label = label[offset[1] : ]


      # Convert the labels data into numpy array
      # Shape the array into 60000 rows (N) and 1 feature per image
      label_arr = np.array([label])
      label_arr = label_arr.reshape((shape[0] , ))
            


      return (image_arr , label_arr)


if __name__ == '__main__':
      # Process training data
      image_arr , label_arr = processData('train-images.idx3-ubyte' , 'train-labels.idx1-ubyte', (60000 , 784))

      # Get N rows and D features
      N , D = image_arr.shape

      # The byte values range from 0 - 255
      # Using a threshold, the range is modified to 0 - 1
      image_thresh_arr = np.where(image_arr > 0 , 1 , 0)


      # Get prior value
      prior = calculatePrior(label_arr)

      # Get likelihood for black and white pixels
      theta_black , theta_white = calculateLikelihood(label_arr , image_thresh_arr)


      # Process test data
      test_img_arr , test_label = processData('t10k-images.idx3-ubyte' , 't10k-labels.idx1-ubyte', (10000, 784))

      # The byte values range from 0 - 255
      # Using a threshold, the range is modified to 0 - 1
      test_image_thresh_arr = np.where(test_img_arr > 0 , 1 , 0)

      # Predict class label
      result_class = posterior(test_image_thresh_arr , prior , theta_black , theta_white)

      print('Test Label: ' , test_label)
      print('Result Label: ' , result_class)


      correct = np.sum(test_label == result_class)
      accuracy = (correct / len(test_label)) * 100.0

      print('{:.2f}'.format(accuracy))




      ##for i in range(2):
      ##      image_row = image_thresh_arr[i , :].reshape(28 , 28)
      ##      print(image_row)
      ##      plt.subplot(5 , 4 , i + 1)
      ##      print(label_arr[i][0])
      ##      plt.imshow(image_row , cmap = plt.cm.binary , interpolation = 'nearest')
      ##      plt.colorbar()
      ##plt.show()

