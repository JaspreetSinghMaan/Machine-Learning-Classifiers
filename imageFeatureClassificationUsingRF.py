###############################################################
# Image Feature classification using Random Forest classifier #
# Input: Training and Test feature file in *.csv format		  #
# Output: Accuracy 											  #
###############################################################

import pandas as pd
import numpy as np


# In[43]:

# Provide the absolute path of the Train and Test feature files written in .csv format

train_dataset = pd.read_csv('Color_ImageReadWriteWang_1000_ZM_F_OC_JMagn_7_Train.csv', header=None)
test_dataset = pd.read_csv('Color_ImageReadWriteWang_1000_ZM_F_OC_JMagn_7_Test.csv', header=None)
rc = train_dataset.shape
r = rc[0]
c = rc[1]

# In[44]:
# Arrange the dataset into training features, and training class labels and shows the sample of features

X_train = train_dataset.iloc[:,0:c-1].values
y_train = train_dataset.iloc[:,c-1].values
print('The independent features set: ')
print(X_train[:3,:])
print('The dependent variable: ')
print(y_train[:5])


# In[45]:
# Arrange the dataset into test features, and test class labels and shows the sample of features

X_test = test_dataset.iloc[:,0:c-1].values
y_test = test_dataset.iloc[:,c-1].values
print('The independent features set: ')
print(X_test[:3,:])
print('The dependent variable: ')
print(y_test[:5])


# In[46]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time
start_time = time.time()
# optimizing the results using grid search over the training features using cross validation 
classifier = RandomForestClassifier()
param_grid = {'n_estimators':[200, 400, 600, 800, 1000, 1200, 1400, 1600],
              'criterion':['entropy'],
              #'random_state':[20, 40, 60, 80, 100],
              'random_state':[0],
              'n_jobs':[-1]
             }
CV_rfc = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
CV_rfc.fit(X_train, y_train)
print (CV_rfc.best_params_)
print("--- %s seconds ---" % (time.time() - start_time))


# In[47]:
# Intializing the parameters to fit the classifier on test dataset

bestParam = CV_rfc.best_params_
start_time = time.time()
classifier = RandomForestClassifier(n_estimators = bestParam['n_estimators'], 
                                    criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
print (CV_rfc.best_params_)
print("--- %s seconds ---" % (time.time() - start_time))


# In[50]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

