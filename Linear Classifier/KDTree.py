'''
CS 5793 AI 2

Basic Classifiers

1) KD Tree

'''

import numpy as np
import generate_data as gd
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from LinearClassify import LinearClassify
import matplotlib.pyplot as plt

class KDTree:


      def __init__(self , matrix , depth = 0):

            self.matrix = matrix
            self.k = 2

            if len(self.matrix) <= 0:

                  self.left = None
                  self.right = None
                  self.middle = None
                  self.axis = None

            else:
                  self.axis = depth % self.k

                  self.matrix = matrix[np.argsort(matrix[:, self.axis])]

                  median = len(self.matrix) // 2

                  self.middle = self.matrix[median]
                  self.left = KDTree(self.matrix[ : median] , depth + 1)
                  self.right = KDTree(self.matrix[median + 1 : ] , depth + 1)


      def find_nearest(self , vector):

            

           closest = self.find_nearest_helper(vector)

           return closest




      def find_nearest_helper(self , vector , depth = 0):

            first = ''
            other = ''

            if self.middle is None:
                  return None

            # Cycle through splitting axes (X or y)
            axis = depth % self.k

            # Check which direction to traverse to

            if vector[axis] < self.middle[axis]:
                  first = self.left
                  other = self.right

            else:
                  first = self.right
                  other = self.left


            # Returns a point from a deeper level
            dist = first.find_nearest_helper(vector , depth + 1)

            # Check if the current point is bettern than dist
            # Here we check if the point got by splitting on the axis
            # is better than some other point found at a deeper depth
            # This essentialy checks for best dist between our target and
            # either the splitting point or the point got by traversing
            # along the splitting axis

            if dist is None:
                  best = self.middle
            
            else:
                  
                  if (np.linalg.norm(vector - dist) < np.linalg.norm(vector - self.middle)):
                        best = dist
                  else:
                        best = self.middle

                        
            # Check on the other side of the tree
            # That is, if there is a point that is better than the current point
            # We check by comparing the dist between the best point and our target with the
            # splitting point
            # If we find it to be better, we traverse to the other side of the tree
            # And check the distance between the current node and our best node

            if (np.linalg.norm(vector - best)) > abs(vector[axis] - self.middle[axis]):
            
                  dist = other.find_nearest_helper(vector , depth + 1)

                  if dist is None:
                        best = self.middle


                  else:
                        if (np.linalg.norm(vector - dist) < np.linalg.norm(vector - best)):
                              best = dist
                        else:
                              best = self.middle
            

            return best





def plot(data1 , data2 , clr):

      plt.plot(data1[ : ,0] , data1[ : , 1 ] , 'x' , color = clr[0])
      plt.plot(data2[ : ,0] , data2[ : , 1 ] , 'x' , color = clr[1])
      plt.title('Gaussian Data points')
      plt.show()

def createFigures(x1 , x2 , x3 , x4 , x5 , x6 , clr):

      plt.plot(x1[ : ,0] , x1[ : , 1 ] , 'o' , ms = 5 , color = clr[0] , label = 'Tr Class 0')
      plt.plot(x2[ : ,0] , x2[ : , 1 ] , 'x' , color = clr[1] , label = 'Tr Class 1')
      plt.plot(x3[ : ,0] , x3[ : , 1 ] , '*' , ms = 5 , color = clr[2] , label = 'Crct Class 0')
      plt.plot(x4[ : ,0] , x4[ : , 1 ] , 'x' , color = clr[3] , label = 'Crct Class 1')
      plt.plot(x5[ : ,0] , x5[ : , 1 ] , '.' , ms = 4, color = clr[4] , label = 'Wr Class 0')
      plt.plot(x6[ : ,0] , x6[ : , 1 ] , '+' , ms = 3 , color = clr[5] , label = 'Wr Class 1')
      plt.legend()
      plt.title('Result')
      plt.show()


data_points = 5000 
rand_data = abs(np.random.randn(2 , 2) * 3)
mean = np.mean(rand_data , axis = 1)
cov = np.cov(rand_data)
dim = (2 , data_points)

x1 , y1 = gd.generateGaussData([3.17431445 , 2.9112456] , [[ 4.79610678, -3.67781644],[-3.67781644, 2.82027369]] , dim)
x2 , y2 = gd.generateGaussData([1.21862626 , 4.86495546]  , [[ 0.91761162, -1.49713279], [-1.49713279 , 2.44265278]] , dim)


x = np.vstack((x1 , x2))
r , c = x.shape
y = np.array([0 if row in x1 else 1 for row in x]).reshape(r , 1)

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.50)

kd = KDTree(x_train)


linClass = LinearClassify(x_train , y_train)
beta = linClass.classify()
thresh = 0.5



y_hat = []

for row in x_test:
      d = kd.find_nearest((row))
      i  = np.where((x_train == d).all(axis=1))
      val = y_train[i][0][0]
      y_hat.append(val)

yr , yc = y_test.shape
y_hat = np.array([y_hat]).reshape(yr , 1)

actual_y_hat = linClass.estimate(x_test , y_test , thresh , beta)

equal = (np.sum(y_hat == actual_y_hat))

plot(x1 , x2 , ['r' , 'b'])

accuracy = ((y_test.size - equal) / y_test.size) * 100

print('{}{:.2f}'.format('Accuracy: ',accuracy))

r_t , c_t = x_test.shape

correct_class0 = []
correct_class1 = []

wrong_class0 = []
wrong_class1 = []


for i in range(y_test.size):
      
      if actual_y_hat[i] == y_hat[i]:
            
            if y_test[i] == 0:
                  correct_class0.append(x_test[i])

            elif y_test[i] == 1:
                  correct_class1.append(x_test[i])

      else:
            if actual_y_hat[i] == 0:
                  wrong_class0.append(x_test[i])

            elif y_test[i] == 1:
                  wrong_class1.append(x_test[i])


train_class_0 = np.array([arr for arr in x_train if arr in x1])
train_class_1 = np.array([arr for arr in x_train if arr in x2])

correct_class0 = np.array(correct_class0)
correct_class1 = np.array(correct_class1)

wrong_class0 = np.array(wrong_class0)
wrong_class1 = np.array(wrong_class1)                  

if wrong_class1.size <= 0:
      wrong_class1 = np.zeros(x_test.shape)

if wrong_class0.size <= 0:
      wrong_class0 = np.zeros(x_test.shape)


if correct_class1.size <= 0:
      correct_class1 = np.zeros(x_test.shape)

if wrong_class0.size <= 0:
      correct_class1 = np.zeros(x_test.shape)      
      
createFigures(train_class_0 , train_class_1 , correct_class0 , correct_class1 , wrong_class0 , wrong_class1 , ['b' , 'g' , 'y' , 'r' , 'c' , 'k'] )
