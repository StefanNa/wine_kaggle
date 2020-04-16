import pandas as pd
import numpy as np

import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)

import sklearn.linear_model as lm
from sklearn import model_selection

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
#from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

df=pd.read_csv('winequality-white.csv',sep=';')
df_plain=pd.read_csv('winequality-white.csv',sep=';')

predict='sulphates'
remove_outliers=True

shape_orig=df.shape[0]
#removal of the 0-1 percentile and 99-100% of the values furthest away from the mean
#I will do my prediction of 'sulphates' therefore I do not remove its outliers
predict='sulphates'
# predict='density'
remove_lim=0.001

lim_high,lim_low=1-remove_lim,remove_lim

if remove_outliers==True:
    features=[cols for cols in df.columns if cols != predict]
    print(50*'_')
    index_names=[]
    #FOR all FEATURES get the percentile and add the indexes to a list (index_names)
    for feat in features:
        y = df[feat]
        removed_outliers = y.between(y.quantile(0.25)-3*(y.quantile(0.75)-y.quantile(0.25)),y.quantile(0.75)+3*(y.quantile(0.75)-y.quantile(0.25)))
#         print(feat,'\n',removed_outliers.value_counts())
        for index in list(df[~removed_outliers].index):
            if index not in index_names:
                index_names.append(index)
    #when all are added remove them
    df.drop(index_names, inplace=True)
    remove_outliers=False
print('% of data dropped',100*(1-(df.shape[0]/shape_orig)))


X=df.drop(columns=predict).values
y=df[predict].values
attributeNames = [name for name in df.drop(columns=predict)]
N, M = X.shape
#folds per level
K=10

#regression defs C1
lambdas = np.power(10.,np.linspace(-5,10,num=10))
lambdas_opt=np.zeros(K)
w_rlr = np.empty((M+1,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))

#ANN defs c1
activation=torch.nn.ReLU()
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000
max_iter_ = 1000
H = np.array([i for i in range(4,8)])
H_opt=np.empty(K)
fold_summary=[]
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
train_error_ann = np.empty((K,1))
test_error_ann = np.empty((K,1))

k=0
CV1 = model_selection.KFold(K, shuffle=True)
for train_index, test_index in CV1.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    CV2 = model_selection.KFold(K, shuffle=True)
    M_ = X_train.shape[1]
    
    #regression defs c2
    M_reg=M+1
    w = np.empty((M_reg,K,len(lambdas)))
    train_error_reg = np.empty((K,len(lambdas)))
    test_error_reg = np.empty((K,len(lambdas)))
    
    #ANN defs c2
    train_error_ann_ = np.empty((K,len(H)))
    test_error_ann_ = np.empty((K,len(H)))
    
    best_final_loss = 1e100
    
    f = 0
    y_train = y_train.squeeze()
    
    for train_index_, test_index_ in CV2.split(X_train,y_train):
        print('outerfold:',k,'/',K,'innerfold:',f,'/',K)
        X_train_ = X[train_index_]
        y_train_ = y[train_index_]
        X_test_ = X[test_index_]
        y_test_ = y[test_index_]
        
        #add w0 for reg
        X_train_reg=np.concatenate((np.ones((X_train_.shape[0],1)),X_train_),1)
        X_test_reg=np.concatenate((np.ones((X_test_.shape[0],1)),X_test_),1)
        
        #settings for NN
        logging_frequency = 250 # display the loss every 1000th iteration
        best_final_loss_ = 1e100
        tolerance=1e-6
        
        #standardize
        mu_ = np.mean(X_train_[:, 1:], 0)
        sigma_ = np.std(X_train_[:, 1:], 0)
        X_train_[:, 1:] = (X_train_[:, 1:] - mu_) / sigma_
        X_test_[:, 1:] = (X_test_[:, 1:] - mu_) / sigma_
        
        #regression
        Xty = X_train_reg.T @ y_train_
        XtX = X_train_reg.T @ X_train_reg
        #run regression model to find optimal lambds C2
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M_reg)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            train_error_reg[f,l] = np.power(y_train_-X_train_reg @ w[:,f,l].T,2).mean(axis=0)
            test_error_reg[f,l] = np.power(y_test_-X_test_reg @ w[:,f,l].T,2).mean(axis=0)
        
        #find optimal h ANN C2
        print('\t\t{}\t{}\t\t\t{}'.format('Iter', 'Loss','Rel. loss'))
        for count,h in enumerate(H):
            print('n_hidden_units:',h)
            model_ = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, h), #M features to n_hidden_units
                    activation,   # 1st transfer function,
                    torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
                    )
            net_ = model_()
        
            # initialize weights based on limits that scale with number of in- and
            # outputs to the layer, increasing the chance that we converge to 
            # a good solution
            torch.nn.init.xavier_uniform_(net_[0].weight)
            torch.nn.init.xavier_uniform_(net_[2].weight)
            optimizer = torch.optim.Adam(net_.parameters())
            
            learning_curve_ = [] # setup storage for loss at each step
            old_loss_ = 1e6
            
            for i in range(max_iter_):
                y_est = net_(torch.Tensor(X_train_)) # forward pass, predict labels on training set
                y_est_test = net_(torch.Tensor(X_test_))
                loss = loss_fn(y_est.squeeze(), torch.Tensor(y_train_)) # determine loss
                loss_value_ = loss.data.numpy() #get numpy array instead of tensor
                learning_curve_.append(loss_value_) # record loss for later display

                # Convergence check, see if the percentual loss decrease is within
                # tolerance:
                p_delta_loss = np.abs(loss_value_-old_loss_)/old_loss_
                if p_delta_loss < tolerance: break
                old_loss_ = loss_value_

                # display loss with some frequency:
                if (i != 0) & ((i+1) % logging_frequency == 0):
                    print_str = '\t\t' + str(i+1) + '\t' + str(loss_value_) + '\t' + str(p_delta_loss)
                    print(print_str)
                # do backpropagation of loss and optimize weights 
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            if loss_value_ < best_final_loss_:
                best_net_ = net_
                best_final_loss_ = loss_value_
                best_learning_curve_ = learning_curve_
                train_error_ann_[f,count] = np.power(y_train_-y_est.detach().numpy().squeeze(),2).mean(axis=0)
                test_error_ann_[f,count] = np.power(y_test_-y_est_test.detach().numpy().squeeze(),2).mean(axis=0)
                print('new best:',best_final_loss_)
        
        
        f=f+1
    #return optimal lambda
    opt_val_err = np.min(np.mean(test_error_reg,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error_reg,axis=0))]
    train_err_vs_lambda = np.mean(train_error_reg,axis=0)
    test_err_vs_lambda = np.mean(test_error_reg,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    lambdas_opt[k]=opt_lambda
    #return optimal h
    best_h=H[np.argmin(np.mean(test_error_ann_,axis=0))]
    H_opt[k]=best_h
    
    #baseline
    baseline=np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    
    #regression C1
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :]
    X_train_reg=np.concatenate((np.ones((X_train.shape[0],1)),X_train),1)
    X_test_reg=np.concatenate((np.ones((X_test.shape[0],1)),X_test),1)
 
    Xty = X_train_reg.T @ y_train
    XtX = X_train_reg.T @ X_train_reg
    
    lambdaI = opt_lambda * np.eye(M+1)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train_reg @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test_reg @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    
    #ANN C1
    model_ = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, best_h), #M features to n_hidden_units
        activation,   # 1st transfer function,
        torch.nn.Linear(best_h, 1), # n_hidden_units to 1 output neuron
        )
    net = model_()

    # initialize weights based on limits that scale with number of in- and
    # outputs to the layer, increasing the chance that we converge to 
    # a good solution
    torch.nn.init.xavier_uniform_(net[0].weight)
    torch.nn.init.xavier_uniform_(net[2].weight)
    optimizer = torch.optim.Adam(net.parameters())

    learning_curve = [] # setup storage for loss at each step
    old_loss = 1e6
    
    for i in range(max_iter):
        y_est = net(torch.Tensor(X_train)) # forward pass, predict labels on training set
        y_est_test = net(torch.Tensor(X_test))
        loss = loss_fn(y_est.squeeze(), torch.Tensor(y_train)) # determine loss
        loss_value = loss.data.numpy() #get numpy array instead of tensor
        learning_curve.append(loss_value) # record loss for later display

        # Convergence check, see if the percentual loss decrease is within
        # tolerance:
        p_delta_loss = np.abs(loss_value-old_loss)/old_loss
        if p_delta_loss < tolerance: break
        old_loss = loss_value

        # display loss with some frequency:
        if (i != 0) & ((i+1) % logging_frequency == 0):
            print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
            print(print_str)
        # do backpropagation of loss and optimize weights 
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    if loss_value < best_final_loss:
        best_net = net
        best_final_loss = loss_value
        best_learning_curve = learning_curve
        train_error_ann[k] = np.power(y_train-y_est.detach().numpy().squeeze(),2).mean(axis=0)
        test_error_ann[k] = np.power(y_test-y_est_test.detach().numpy().squeeze(),2).mean(axis=0)
        print('new best:',best_final_loss)
    
    summary_tags=['fold','h_opt','E_test_ann','lambda_opt','E_test_lrl','baseline']
    fold_summary.append([k+1,best_h,np.round(test_error_ann[k][0],4),opt_lambda,np.round(Error_test_rlr[k][0],4),np.round(baseline,4)])
    k+=1

summary_tags=['fold','h_opt','E_test_ann','lambda_opt','E_test_lrl','baseline']
print(summary_tags[0],'\t',summary_tags[1],'\t',summary_tags[2],'\t',summary_tags[3],'\t',summary_tags[4],'\t',summary_tags[5])
for i in fold_summary:
    print(i[0],'\t',i[1],'\t',i[2],'\t\t',i[3],'\t\t',i[4],'\t\t',i[5])    
#run both models with optimal lambda,h

#%% comparing

from sklearn import model_selection
import scipy.stats as st
test_proportion = 0.2

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)

#rlr
mu_= np.mean(X_train[:, 1:], 0)
sigma_= np.std(X_train[:, 1:], 0)
X_train[:, 1:] = (X_train[:, 1:] - mu_ ) / sigma_ 
X_test[:, 1:] = (X_test[:, 1:] - mu_ ) / sigma_

X_train_reg=np.concatenate((np.ones((X_train.shape[0],1)),X_train),1)
X_test_reg=np.concatenate((np.ones((X_test.shape[0],1)),X_test),1)

Xty = X_train_reg.T @ y_train
XtX = X_train_reg.T @ X_train_reg

lambdaI = opt_lambda * np.eye(M+1)
lambdaI[0,0] = 0 # Do no regularize the bias term
w_rlr_ = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
# Compute mean squared error with regularization with optimal lambda
# Error_train_rlr_ = np.square(y_train-X_train_reg @ w_rlr_).sum(axis=0)/y_train.shape[0]
# Error_test_rlr_ = np.square(y_test-X_test_reg @ w_rlr_).sum(axis=0)/y_test.shape[0]
Zrlr=np.square(y_test-X_test_reg @ w_rlr_)

#NN
y_est_test = net(torch.Tensor(X_test))
Zann=np.power(y_test-y_est_test.detach().numpy().squeeze(),2)

#baseline
Zbase=np.square(y_test-y_test.mean())

alpha = 0.05

CI=np.empty([3,2])
p=np.zeros(3)
Z = (Zbase-Zrlr,Zbase-Zann,Zrlr-Zann)

for c,z in enumerate(Z):
    CI[c] = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p[c] = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print(CI,p)