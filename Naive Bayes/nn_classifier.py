'''
CS5793 - AI 2

3) NN Classifier

'''
import numpy as np
import math
import matplotlib.pyplot as plt
import naive_bayes as NB


def buildData(images , labels):

      '''
      This function is used to extract images consisting of digits
      1 , 2 and 7. There will be 200 samples of these digits
      
      '''
      # Get class 7
      label_7_filter = (labels == 7)
      label_7 = labels[label_7_filter]
      image_7 = images[label_7_filter]

      # Get class 1
      label_1_filter = (labels == 1)
      label_1 = labels[label_1_filter]
      image_1 = images[label_1_filter]


      # Get class 2
      label_2_filter = (labels == 2)
      label_2 = labels[label_2_filter]
      image_2 = images[label_2_filter]     


      # Get 200 representations of class 7
      image_200_7 = image_7[np.random.choice(image_7.shape[0] , 200 , replace = False) , :]
      label_200_7 = label_7[np.random.choice(label_7.shape[0] , 200 , replace = False)]

      # Get 50 representations of class 7 for test data
      image_50_7 = image_7[200 : 250 , :]
      label_50_7 = label_7[200 : 250]

      # Get 200 representations of class 2
      image_200_2 = image_2[np.random.choice(image_2.shape[0] , 200 , replace = False) , :]
      label_200_2 = label_2[np.random.choice(label_2.shape[0] , 200 , replace = False)]

      # Get 50 representations of class 2 for test data
      image_50_2 = image_2[200 : 250 , :]
      label_50_2 = label_2[200 : 250]
     

      # Get 200 representations of class 1
      image_200_1 = image_1[np.random.choice(image_1.shape[0] , 200 , replace = False) , :]
      label_200_1 = label_1[np.random.choice(label_1.shape[0] , 200 , replace = False)]

      # Get 50 representations of class 1 for test data
      image_50_1 = image_1[200 : 250 , :]
      label_50_1 = label_1[200 : 250]

      image_arr = np.vstack((image_200_7 , image_200_2 , image_200_1))
      label_arr = np.vstack((label_200_7 , label_200_2 , label_200_1))
      label_arr = label_arr.reshape(600 ,)

     # print(image_arr.shape , label_arr.shape)

      test_image_50 = np.vstack((image_50_7 , image_50_1 ,image_50_2))
      test_label_50 = np.vstack((label_50_7 , label_50_1 , label_50_2))
      test_label_50 = test_label_50.reshape(150 ,)


      
      tot_image = np.concatenate([image_arr , test_image_50])
      tot_label = np.concatenate([label_arr , test_label_50])

      # Shuffle the arrays
      shuffle_ind = np.arange(tot_image.shape[0])
      np.random.shuffle(shuffle_ind)
      tot_image = tot_image[shuffle_ind]
      tot_label = tot_label[shuffle_ind]

      # Extract train images and labels
      train_image = tot_image[0 : 600 , :]
      train_label = tot_label[0 : 600]

      # Extract test images and labels
      test_image = tot_image[600 : 750 , :]
      test_label = tot_label[600 : 750]


      '''
      # Plotting to check
      for i in range(10):
            plt.imshow(train_image[i , :].reshape(28 , 28))
            print(train_label[i] , test_label[i])
            plt.colorbar()
            plt.show()
      
      '''


      return train_image , train_label , test_image , test_label

def knnClassify(k , train_data , test_data , train_labels):


      '''

      The method computes the K nearest point in the training data
      for each test data. The value of k is given as arg.

      Eucledian distance is used measure the distance between the points
      The labels corresponding to the points are taken and the majority
      of the label is chosen as the predicted class

      '''

      best_result = []
      # Cast as float for precision
      train_data = train_data.astype(np.float64)
      
      for test in test_data:
            # Calulate eucledian dist for every test image with training image
            # math.sqrt(sum([(a - b) ** 2
            # Cast as float for precision
            test = test.astype(np.float64)
            dist = np.sqrt(np.sum(np.power(train_data - test, 2) , axis = 1) )

            # sort the distances
            sorted_dist_idx = np.argsort(dist)

            predicted_labels = dict()

            # Choose k nearest distance idx
            k_dist = sorted_dist_idx[0 : k]


            for l in k_dist:
                  label = train_labels[l]

                  predicted_labels[label] = predicted_labels.get(label, 0) + 1


            predicted_labels = sorted(predicted_labels.items() , key = lambda x : x[1] , reverse = True)
            best_result.append(predicted_labels[0][0])
            

      return np.array(best_result)


def k_fold(folds , train_data , train_labels):

      '''

      This functions gives the K fold cross validation datasets
      The original datset is partiioned into k chunks

      # of folds is 5
      
      '''

      # Size of each chunk
      subset = train_data.shape[0] / folds

      
      # Partition groups for k chunks
      shapes = {0 : (0 , 120) , 1 : (120 , 240) ,
                2 : (240 , 360) , 3 : (360 , 480) , 4 : (480 , 600)}

      for i in range(folds):

            train = []
            test = []
            labels = []
            test_label = []

            for j in range(folds):

                  if j % folds == i:
                        #print('test: ' , j , i)
                        s , e = shapes[j]
                        test.append(train_data[s : e , :])
                        test_label.append(train_labels[s : e])
                        
                  if j % folds != i:
                        #print('train j: ', j , i)
                        start , end = shapes[j]
                        train.append(train_data[start : end , :])
                        labels.append(train_labels[start : end])

            
            yield np.vstack(train) , np.vstack(test), np.vstack(labels) , np.vstack(test_label)






if __name__ == '__main__':
      
      # Process training data
      image_arr , label_arr = NB.processData('train-images.idx3-ubyte' , 'train-labels.idx1-ubyte', (60000 , 784))

      train_image , train_label , test_image , test_label = buildData(image_arr , label_arr)

      '''
      for i in range(5):
            plt.imshow(train_image[i , :].reshape(28 , 28))
            plt.imshow(test_image[i , :].reshape(28 , 28))
            print(train_label[i] , test_label[i])
            plt.colorbar()
            plt.show()

      '''

      
      
      k = [1 , 3 , 5 , 7 , 9]
      idx = 0
      max_k = []
      accuracy = []
      
      # Create 5 fold cross validation
      for train , validation , label , t_label in k_fold(5 , train_image , train_label):

            label = label.reshape(train.shape[0] , )
            t_label = t_label.reshape(validation.shape[0] , )

            '''
            for i in range(5):
                  #plt.imshow(train[i , :].reshape(28 , 28))
                  plt.imshow(validation[i , :].reshape(28 , 28))
                  print(t_label[i] , label[i])
                  plt.colorbar()
                  plt.show()
            '''
      
            result = knnClassify(k[idx] , train , validation , label)
            
            result = result.reshape(validation.shape[0] , )

            
            '''
            # Plotting to check
            for i in range(5):
                  plt.imshow(validation[i , :].reshape(28 , 28))
                  plt.imshow(train_image[i , :].reshape(28 , 28))
                  print(test_label[i] , label[i] , result[i])
                  plt.colorbar()
                  plt.show()
            
           '''
            

            equal = np.sum(t_label == result)
            res = equal / len(t_label)
            accuracy.append(res)

            print(k[idx] , res)
            idx += 1


      max_k = k[accuracy.index(max(accuracy))]

      print('Best K is: ' , max_k)

      final_result = knnClassify(max_k , train_image , test_image , train_label)
      final_result = final_result.reshape(test_image.shape[0] , )

      # Plot some test cases
      for i in range(10):
            
            if final_result[i] == test_label[i]:
                  plt.imshow(test_image[i , :].reshape(28 , 28))
                  print('Predicted label: ' , final_result[i] , ' Actual Label: ' , test_label[i])
                  plt.colorbar()
                  plt.title('Correct')
                  plt.show()
            else:
                  plt.imshow(test_image[i , :].reshape(28 , 28))
                  print('Predicted label: ' , final_result[i] , ' Actual Label: ' , test_label[i])
                  plt.colorbar()
                  plt.title('Wrong')
                  plt.show()
                  
      final_acc = np.sum(test_label == final_result)

      print('Final accuracy: ' , final_acc / len(test_label))

    

      
