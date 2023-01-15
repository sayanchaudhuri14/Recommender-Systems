
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from numpy.random import choice
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from time import time




ratings = pd.read_csv('../data/ratings.csv')




n_mov = len(np.unique(ratings['movieId']))
n_user = len(np.unique(ratings['userId']))
matrix = np.zeros([n_user,n_mov])
indicator = np.zeros_like(matrix)
hmap_mov = dict()
hmap_user = dict()
x = 0
y = 0
for i in range(len(ratings['userId'])):
    if ratings['userId'][i] not in hmap_user:
        hmap_user[ratings['userId'][i]] = x
        x+=1
for i in range(len(ratings['movieId'])):
    if ratings['movieId'][i] not in hmap_mov:
        hmap_mov[ratings['movieId'][i]] = y
        y+=1




for i in range(len(ratings['rating'])):
    matrix[hmap_user[ratings['userId'][i]],hmap_mov[ratings['movieId'][i]]] = ratings['rating'][i]
    indicator[hmap_user[ratings['userId'][i]],hmap_mov[ratings['movieId'][i]]] = 1


# ## Making train and test matrix



train, test = train_test_split(ratings, test_size=0.2)
train.reset_index(drop=True,inplace=True)
n_mov = len(np.unique(train['movieId']))
n_user = len(np.unique(train['userId']))
train_matrix = np.zeros([n_user,n_mov])
train_indicator = np.zeros_like(train_matrix)
hmap_mov_train = dict()
hmap_user_train = dict()
x = 0
y = 0
for i in range(len(train['userId'])):
    if train['userId'][i] not in hmap_user_train:
        hmap_user_train[train['userId'][i]] = x
        x+=1
for i in range(len(train['movieId'])):
    if train['movieId'][i] not in hmap_mov_train:
        hmap_mov_train[train['movieId'][i]] = y
        y+=1
for i in range(len(train['rating'])):
    train_matrix[hmap_user_train[train['userId'][i]],hmap_mov_train[train['movieId'][i]]] = train['rating'][i]
    train_indicator[hmap_user_train[train['userId'][i]],hmap_mov_train[train['movieId'][i]]] = 1




test.reset_index(drop=True,inplace=True)
n_mov = len(np.unique(test['movieId']))
n_user = len(np.unique(test['userId']))
test_matrix = np.zeros([n_user,n_mov])
test_indicator = np.zeros_like(test_matrix)
hmap_mov_test = dict()
hmap_user_test = dict()
x = 0
y = 0
for i in range(len(test['userId'])):
    if test['userId'][i] not in hmap_user_test:
        hmap_user_test[test['userId'][i]] = x
        x+=1
for i in range(len(test['movieId'])):
    if test['movieId'][i] not in hmap_mov_test:
        hmap_mov_test[test['movieId'][i]] = y
        y+=1
for i in range(len(test['rating'])):
    test_matrix[hmap_user_test[test['userId'][i]],hmap_mov_test[test['movieId'][i]]] = test['rating'][i]
    test_indicator[hmap_user_test[test['userId'][i]],hmap_mov_test[test['movieId'][i]]] = 1


# ## SVD Decomposition




MSE_SVD = []
Times_SVD = []
Size_SVD = []
U,S,VT = np.linalg.svd(matrix)

for i in range(610):
    start = time()
    new_matrix_svd = U[:,:i]@np.diag(S[:i])@VT[:i,:]
    MSE_SVD.append(mean_squared_error(matrix,new_matrix_svd))
    end = time()
    t = end-start
    Times_SVD.append(t)
    size = np.size(U[:,:i]) + np.size(S[:i])  + np.size(VT[:i,:])
    Size_SVD.append(size)
plt.plot(MSE_SVD)
plt.title('Error vs Number of Latent Factors for SVD')




plt.plot(Times_SVD)
plt.title('Times vs Number of Latent Factors for SVD')
plt.xlabel("Number of Latent Factors")
plt.ylabel("Time Taken in Sec")




plt.plot(Size_SVD)
plt.title('Size vs Number of Latent Factors for SVD')
plt.xlabel("Number of Latent Factors")
plt.ylabel("Size")



# ## CUR Decomposition



iterss = 5
MSE_CUR = []
Size_CUR = []
Time_CUR = []
Total_E = np.sum(matrix**2)
P_columns = np.sum(matrix**2,axis=0)/Total_E
P_rows = np.sum(matrix**2,axis=1)/Total_E 
for i in range(610):
    temp = [100]
    for _ in range(iterss):
        start = time()
        r,c = i,i        
        cols_chosen = choice(np.arange(0,P_columns.shape[0]), c, p=P_columns,replace = False)
        rows_chosen = choice(np.arange(0,P_rows.shape[0]), r, p=P_rows,replace = False)
        Cx = matrix[:,cols_chosen]/((P_columns[cols_chosen] * c)**0.5)
        Rx = matrix[rows_chosen,:].T/((P_rows[rows_chosen] * r)**0.5)
        Rx=Rx.T
        psi = Cx[rows_chosen,:].T/((P_rows[rows_chosen] * r)**0.5)
        Ux = np.linalg.pinv(psi)
        NEW_MAT = Cx@Ux@Rx
        temp.append(mean_squared_error(NEW_MAT,matrix))
        end = time()
        t = end - start
        Time_CUR.append(t)
        size = np.size(Cx) + np.size(Ux) + np.size(Rx)
        Size_CUR.append(size)
    MSE_CUR.append(min(temp))
plt.plot(MSE_CUR)



plt.plot(Time_CUR)
plt.title('Times vs Number of Latent Factors for CUR')
plt.xlabel("Number of Latent Factors")
plt.ylabel("Time Taken in Sec")




plt.plot(Size_CUR)
plt.title('Size vs Number of Latent Factors for CUR')
plt.xlabel("Number of Latent Factors")
plt.ylabel("Size")



plt.plot(Size_CUR,label = 'CUR')
plt.plot(Size_SVD , label = 'SVD')
plt.title('Size vs Number of Latent Factors Comparison')
plt.xlabel("Number of Latent Factors")
plt.ylabel("Size")
plt.legend()




plt.plot(Time_CUR,label = 'CUR')
plt.plot(Times_SVD , label = 'SVD')
plt.title('Time Taken vs Number of Latent Factors Comparison ')
plt.xlabel("Number of Latent Factors")
plt.ylabel("Time Taken in Sec")
plt.legend()
