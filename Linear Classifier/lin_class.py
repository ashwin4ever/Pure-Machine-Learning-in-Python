'''
CS 5793 AI 2

Basic Classifiers

3) Linear Classifiers

'''
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import generate_data as gd
from LinearClassify import LinearClassify



X_list = []
Y_list = []


def plot(data1 , data2 , clr):

      plt.plot(data1[ : ,0] , data1[ : , 1 ] , 'x' , color = clr[0])
      plt.plot(data2[ : ,0] , data2[ : , 1 ] , 'x' , color = clr[1])
      plt.title('Gaussian Data points')
      plt.show()




def createFigures(x1 , x2 , x3 , x4 , x5 , x6 , clr):

      plt.plot(x1[ : ,0] , x1[ : , 1 ] , 'o' , ms = 5 , color = clr[0] , label = 'Tr Class 0')
      plt.plot(x2[ : ,0] , x2[ : , 1 ] , 'x' , color = clr[1] , label = 'Tr Class 1')
      plt.plot(x3[ : ,0] , x3[ : , 1 ] , '*' , ms = 5 , color = clr[2] , label = 'Crct Class 0')
      #plt.plot(x4[ : ,0] , x4[ : , 1 ] , 'x' , color = clr[3] , label = 'Crct Class 1')
      plt.plot(x5[ : ,0] , x5[ : , 1 ] , '.' , ms = 4, color = clr[4] , label = 'Wr Class 0')
      plt.plot(x6[ : ,0] , x6[ : , 1 ] , '+' , ms = 3 , color = clr[5] , label = 'Wr Class 1')
      plt.legend()
      plt.title('Result')
      plt.show()

      



def generateClassLabels(x , data_list):

      labels = []
      arr = np.random.shuffle(x)
      for d in data_list[0 : 1]:
            for row in x:
                  #print(row)
                  if row in d:
                        labels.append(1)
                  else:
                        labels.append(0)

      return np.array(labels)

data_points = 5000 # user input

rand_data = abs(np.random.randn(2 , 2) * 3)
mean1 = np.mean(rand_data , axis = 1)
cov1 = np.cov(rand_data)

rand_data1 = abs(np.random.randn(2 , 2) * 3.5)

mean2 = np.mean(rand_data1 , axis = 1)
cov2 = np.cov(rand_data1)

dim = (2 , data_points)



x1 , y1 = gd.generateGaussData([3.17431445 , 2.9112456] , [[ 4.79610678, -3.67781644],[-3.67781644, 2.82027369]] , dim)
x2 , y2 = gd.generateGaussData([1.21862626 , 4.86495546]  , [[ 0.91761162, -1.49713279], [-1.49713279 , 2.44265278]] , dim)

x = np.concatenate([x1 , x2] , axis = 0)

r , c = x.shape
np.random.shuffle(x)

y = np.array([0 if row in x1 else 1 for row in x]).reshape(r , 1)

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.50)

linClass = LinearClassify(x_train , y_train)
 
beta = linClass.classify()

plot(x1 , x2 , ['r' , 'b'])

thresh = 0.5
y_hat = linClass.estimate(x_test , y_test , thresh , beta)

            

equal = (np.sum(y_hat == y_test))



accuracy = ((y_test.size - equal) / y_test.size) * 100

print('{:.2f}'.format(accuracy))

r_t , c_t = x_test.shape

correct_class0 = []
correct_class1 = []

wrong_class0 = []
wrong_class1 = []

for i in range(y_test.size):
      
      if y_test[i] == y_hat[i]:
            
            if y_test[i] == 0:
                  correct_class0.append(x_test[i])

            elif y_test[i] == 1:
                  print(y_hat)
                  correct_class1.append(x_test[i])

      else:
            if y_test[i] == 0:
                  #print(y_test[i] , y_hat[i])
                  wrong_class0.append(x_test[i])

            elif y_test[i] == 1:
                  wrong_class1.append(x_test[i])
            


train_class_0 = np.array([arr for arr in x_train if arr in x1])
train_class_1 = np.array([arr for arr in x_train if arr in x2])

correct_class0 = np.array(correct_class0)
correct_class1 = np.array(correct_class1)

wrong_class0 = np.array(wrong_class0)
wrong_class1 = np.array(wrong_class1)

test_class_1 = np.array([arr for arr in x_test if arr in x2])

#fig = plt.figure(figsize=(18, 18))

#print(correct_class0.shape , correct_class1.shape , wrong_class0.shape , wrong_class1.shape , test_class_1.shape)
createFigures(train_class_0 , train_class_1 , correct_class0 , correct_class1 , wrong_class0 , wrong_class1 , ['b' , 'g' , 'y' , 'r' , 'c' , 'k'] )






      
      
