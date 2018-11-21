
# coding: utf-8

# In[135]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[136]:


data = pd.read_csv("Transformed Data Set - Sheet1.csv")


# In[137]:


data.info()


# In[138]:


data.head()


# In[139]:


data['Favorite Color'].unique()


# In[140]:


data['Favorite Music Genre'].unique()


# In[141]:


data['Favorite Beverage'].unique()


# In[142]:


data['Favorite Soft Drink'].unique()


# In[143]:


for label in ['Favorite Color','Favorite Music Genre','Favorite Beverage','Favorite Soft Drink']:
    data[label] = LabelEncoder().fit_transform(data[label])


# In[144]:


x = data[['Favorite Color', 'Favorite Music Genre', 'Favorite Beverage', 'Favorite Soft Drink']]


# In[145]:


y = data['Gender'].values.tolist()


# In[160]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[161]:


clf_tree = tree.DecisionTreeClassifier()


# In[162]:


clf_tree = clf_tree.fit(x_train,y_train)


# In[163]:


pred_tree = clf_tree.predict(x_test)
acc_tree = accuracy_score(y_test, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))


# In[164]:


clf_svm = SVC()


# In[165]:


clf_svm = clf_svm.fit(x_train,y_train)


# In[166]:


pred_svm = clf_svm.predict(x_test)
acc_svm = accuracy_score(y_test, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))


# In[167]:


clf_perceptron = Perceptron()


# In[168]:


clf_perceptron = clf_perceptron.fit(x_train,y_train)


# In[169]:


pred_perceptron = clf_perceptron.predict(x_test)
acc_perceptron = accuracy_score(y_test, pred_perceptron) * 100
print('Accuracy for Perceptron: {}'.format(acc_perceptron))


# In[170]:


clf_KNN = KNeighborsClassifier()


# In[171]:


clf_KNN = clf_KNN.fit(x_train,y_train)


# In[172]:


pred_KNN = clf_KNN.predict(x_test)
acc_KNN = accuracy_score(y_test, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))


# In[173]:


index = np.argmax([acc_tree,acc_svm, acc_perceptron, acc_KNN])
classifiers = {0: 'Tree', 1: 'SVM', 2: 'Perceptron', 3: 'KNN'}
print('Best gender classifier is {}'.format(classifiers[index]))

