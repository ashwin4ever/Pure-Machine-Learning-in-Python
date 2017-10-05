'''
CS 5793 AI 2

Basic Classifiers

5a) Linear Classifiers Multiple Data

'''
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import generate_data as gd
from LinearClassify import LinearClassify

#from lin_class import createFigures as CF



def plot(data1 , data2 , clr):

      plt.plot(data1[ : ,0] , data1[ : , 1 ] , 'x' , color = clr[0])
      plt.plot(data2[ : ,0] , data2[ : , 1 ] , 'x' , color = clr[1] , ms = 3)
      
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



data_x = []




for i in range(10):

      if i % 2:
            rand_data = abs(np.random.randn(2 , 2) * 3 + i)
      else:
            rand_data = abs(np.random.randn(2 , 2) * 3)
      
      mean = np.mean(rand_data , axis = 1)
      cov = np.cov(rand_data)
      
      dim = (2 , 1000)

      x1 , y1 = gd.generateGaussData(mean , cov , dim)

      data_x.append(x1)


# concatenate class 0 matrix
x_0 = np.vstack(data_x[0 : 5])
x_1 = np.vstack(data_x[5 : ])

plot(x_0 , x_1 , ['red' , 'blue'])

#print(x_0.shape , x_1.shape)

# Concatenate all the matrices
x = np.vstack(data_x)

y1 = np.zeros((5000 , 1))
y2 = np.ones((5000 , 1) , np.dtype('int32'))

y = np.vstack((y1 , y2))

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.50)
linClass = LinearClassify(x_train , y_train)

beta = linClass.classify()
thresh = 0.5

y_hat = linClass.estimate(x_test , y_test , thresh , beta)

equal = (np.sum(y_hat == y_test))
accuracy = ((equal) / (y_test.size)) * 100

print('{:.2f}'.format(accuracy))


correct_class0 = []
correct_class1 = []

wrong_class0 = []
wrong_class1 = []

for i in range(y_test.size):
      
      if y_test[i] == y_hat[i]:
            
            if y_test[i] == 0:
                  correct_class0.append(x_test[i])

            elif y_test[i] == 1:
                  #print(y_hat)
                  correct_class1.append(x_test[i])

      else:
            if y_test[i] == 0:
                  #print(y_test[i] , y_hat[i])
                  wrong_class0.append(x_test[i])

            elif y_test[i] == 1:
                  wrong_class1.append(x_test[i])


train_class_0 = np.array([arr for arr in x_train if arr in x_0])
train_class_1 = np.array([arr for arr in x_train if arr in x_1])

correct_class0 = np.array(correct_class0)
correct_class1 = np.array(correct_class1)

wrong_class0 = np.array(wrong_class0)
wrong_class1 = np.array(wrong_class1)

createFigures(train_class_0 , train_class_1 , correct_class0 , correct_class1 , wrong_class0 , wrong_class1 , ['b' , 'g' , 'y' , 'r' , 'c' , 'k'] )
