import numpy as np


features = np.array(
    [
        [1, 1],
        [-1, 1],
        [1, -1],
        [-1, -1]
    ])

target = np.array([1, -1, -1, -1])

print("our model : ")
print(features, target)
print("\n \n")

weight = [0.1, 0.1]
bias = 0.1
learning_rate = 0.1
epoch = 5

 
for i in range(epoch):
    print("epoch :", i+1)
    sum_squared_error = 0.0

    for j in range(features.shape[0]):
        actual = target[j]
        
        x1 = features[j][0]
        x2 = features[j][1]
        print( "X1 : " ,x1)
        print( "X2 : " ,x2)

        yin = (x1 * weight[0]) + (x2 * weight[1]) + bias

        print( "Yin : " ,yin)
  
        error = actual - yin
  
        print("error =", error)
  
        sum_squared_error += error * error
  
        weight[0] += learning_rate * error * x1
        weight[1] += learning_rate * error * x2
  
        bias += learning_rate * error

        print( "w1 : " ,weight[0])
        print( "w2 : " ,weight[1])
        print( "bias : " ,bias)
        
  
    print("sum of squared error = ", sum_squared_error/4, "\n\n")

