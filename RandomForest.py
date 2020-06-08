# %%
from sklearn.model_selection import cross_val_score
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import confusion_matrix
import pylab as pl
from sklearn.metrics import precision_recall_fscore_support


# %%
data = pd.read_csv("pulsar_stars.csv",header=0)
data.head()

# diagnosis column is a object type so we can map it to integer value

data.head()

#data.to_csv(r'C:\Users\Deepak\Desktop\Masters Concordia\2019 Summer\INSE 6180\Project\Breast_Cancer_Sample.csv')

# %%
x = data.loc[:, data.columns != 'target_class']  # Independant variables
y = data['target_class'] # Dependant variables

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 5 )

print(f'X Traib ----- > {len(x_train)}')    
print(f'X TEST ----- > {len(x_test)}')
#x_train = x_train.values.tolist()
print(f'X Train type  {type(x_train)}')

print(x_train.shape)


# %%
y_test.head

# %%
#To identify best n_estimators value

from sklearn.ensemble import RandomForestClassifier
error_rm = []

# Calculating error for K values between 1 and 100
for i in range(1, 50):  
    rm = RandomForestClassifier(n_estimators=i)
    rm.fit(x_train,y_train)
    pred_i = rm.predict(x_test)
    error_rm.append(np.mean(pred_i != y_test))

#print(error_rm)


import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 100), error_rm, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate n_estimators Value')  
plt.xlabel('n_estimators Value')  
plt.ylabel('Mean Error')


# %%
#RandomForest classifier
start = time.time()

model=RandomForestClassifier(n_estimators=100, random_state=5)
model.fit(x_train,y_train)# now fit our model for training data
prediction=model.predict(x_test)# predict for the test data

end = time.time()

#prediction will contain the predicted value by our model predicted values of diagnosis column for test inputs
print(metrics.accuracy_score(prediction,y_test)) # to check the accuracy
# here we will use accuracy measurement between our predicted value and our test output values

print(end-start)




# %%
# Calculating confusion matrix

cm = confusion_matrix(y_test, prediction)

print(cm)
y_test.shape
#pl.matshow(cm)
#pl.title('Confusion matrix of the classifier')
#pl.colorbar()
#pl.show()



# %%
precision_recall_fscore_support(y_test, prediction, average='macro')

# %%
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(x_test.shape)
