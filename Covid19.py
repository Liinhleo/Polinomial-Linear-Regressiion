import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#read dataset
dataset=pd.read_csv("Cov-Korean.csv")

# Split dataset to input X and outcome Y
X=dataset.iloc[:,1].values
Y=dataset.iloc[:,2].values
D=dataset.iloc[:,0].values

#Split dataset to trainning data & test data
X_train,X_test,Y_train,Y_test,D_train,D_test=train_test_split(X,Y,D,train_size=0.9,random_state=0)

#solution : sort dataset
X_train,Y_train,D_train =zip(*sorted(zip(X_train,Y_train,D_train)))
X_test,Y_test,D_test=zip(*sorted(zip(X_test,Y_test,D_test)))

#Trans list to matrix
X_train_after = np.array(X_train).reshape(-1,1)
Y_train_after = np.array(Y_train)
D_train_after = np.array(D_train)
X_test_after = np.array(X_test).reshape(-1,1)
Y_test_after = np.array(Y_test)
D_test_after = np.array(D_test)

# Visualize training data
plt.scatter(D_train_after,Y_train_after,color ="red")
plt.title("Prediction of n-Covid19 in Korea")
plt.xlabel("Date")
plt.ylabel("Num of cases")
plt.show()

# #================ PREDICTION TRAINING =============
# # Tien xu ly du lieu (de tinh x,x^2,x^3,x^4)
poly_transform = PolynomialFeatures(degree=5)
X_poly = poly_transform.fit_transform(X_train_after)

# Training model
poly_lin_reg=LinearRegression()
poly_lin_reg.fit(X_poly,Y_train_after)

# Visualize prediction training data
Y_train_pred=poly_lin_reg.predict(X_poly)
plt.scatter(D_train_after,Y_train_after,color="red")
plt.plot(D_train_after,Y_train_pred,color="blue")
plt.title("Prediction of n-Covid19 in Korea")
plt.xlabel("Date")
plt.ylabel("Num of cases")
plt.show()

#================ COMPARE TESTING & TRAINING =============
# Tien xu ly du lieu (de tinh x,x^2,x^3,x^4)
X_poly_test = poly_transform.fit_transform(X_test_after)

# Training model
poly_lin_reg_test=LinearRegression()
poly_lin_reg_test.fit(X_poly_test,Y_test_after)

#  Visualize testing
Y_test_pred=poly_lin_reg_test.predict(X_poly_test)
plt.scatter(D,Y,color="red") # ALL DATASET (X,Y,D)
plt.scatter(D_test_after,Y_test_pred,color="black") 
plt.plot(D_train_after,Y_train_pred,color="blue")
plt.title("Prediction of n-Covid19 in Korea")
plt.xlabel("Date")
plt.ylabel("Num of cases")
plt.show()
