'''
CS 5793 AI 2

Basic Classifiers

3) Linear Classifiers

'''

import numpy as np

class LinearClassify:

      def __init__(self , x , y):

            self.x = x
            self.y = y


      def classify(self):
            beta_t = np.linalg.inv(self.x.T.dot(self.x)).dot(self.x.T)
            beta = beta_t.dot(self.y)
            return beta

      def estimate(self , x_t , y_t , thresh , beta):

            y_hat = []

            for row in x_t:
                  #print(row.T.dot(beta))
                  if row.T.dot(beta) < thresh:
                        y_hat.append(0)
                  else:
                        y_hat.append(1)

            return np.array(y_hat).reshape(y_t.size , 1)
