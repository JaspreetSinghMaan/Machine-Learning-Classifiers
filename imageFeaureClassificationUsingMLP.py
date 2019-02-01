###############################################################
# Image Feature classification using Multilayer perceptron    #
# classifier                                                  #
# Input: Training and Test feature file in *.csv format		  #
# Output: Accuracy 											  #
###############################################################



import pandas as pd
import numpy as np


# In[93]:
# Provide the absolute path of the Train and Test feature files written in .csv format

train_dataset = pd.read_csv('Color_ImageReadWriteWang_1000_ZM_F_OC_JMagn_7_Train.csv', header=None)
test_dataset = pd.read_csv('Color_ImageReadWriteWang_1000_ZM_F_OC_JMagn_7_Test.csv', header=None)
rc = train_dataset.shape
r = rc[0]
c = rc[1]


# In[94]:
# Arrange the dataset into training features, and training class labels and shows the sample of features

X_train = train_dataset.iloc[:,0:c-1].values
y_train = train_dataset.iloc[:,c-1].values
print('The independent features set: ')
print(X_train[:3,:])
print('The dependent variable: ')
print(y_train[:5])


# In[95]:
# Arrange the dataset into test features, and test class labels and shows the sample of features

X_test = test_dataset.iloc[:,0:c-1].values
y_test = test_dataset.iloc[:,c-1].values
print('The independent features set: ')
print(X_test[:3,:])
print('The dependent variable: ')
print(y_test[:5])


# In[96]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection  import GridSearchCV
import time
start_time = time.time()
# optimizing the results using grid search over the training features using cross validation 
clf = MLPClassifier()
param_grid = {'hidden_layer_sizes':[(200,), (400,), (600,), (800,), (1000,), (1200,), (1600,)],
              'solver':['lbfgs'],
              'activation':['relu'],
              'max_iter':[100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800 ]
             }
CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
CV_rfc.fit(X_train, y_train)
print (CV_rfc.best_params_)
print("--- %s seconds ---" % (time.time() - start_time))


# In[97]:


hlSize = CV_rfc.best_params_['hidden_layer_sizes']
maxIter = CV_rfc.best_params_['max_iter']
start_time = time.time()
# optimizing the results using grid search over the training features using cross validation 
clf = MLPClassifier()
param_grid = {'hidden_layer_sizes':[(hlSize[0]-100,), (hlSize[0],), (hlSize[0]+100,)],
              'solver':['lbfgs'],
              'activation':['relu'],
              'max_iter':[maxIter-100, maxIter-50, maxIter, maxIter+50, maxIter+100]
             }
CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
CV_rfc.fit(X_train, y_train)
print (CV_rfc.best_params_)
print("--- %s seconds ---" % (time.time() - start_time))


# In[98]:

# Intializing the parameters to fit the classifier on test dataset
hlSize = CV_rfc.best_params_['hidden_layer_sizes']
maxIter = CV_rfc.best_params_['max_iter']
start_time = time.time()
clf = MLPClassifier(solver='lbfgs')
clf.hidden_layer_sizes=(hlSize[0],)
clf.activation = 'relu'
clf.max_iter=maxIter;
clf.fit(X_train, y_train)

print("--- %s seconds ---" % (time.time() - start_time))


# In[99]:


y_pred = clf.predict(X_test)
print(y_pred)

# In[101]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

