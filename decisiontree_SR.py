#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing datasets
data_set = pd.read_csv('Data-sets/fact_bating_summary,,,...csv')
print(data_set)
# Extracting Independent and dependent Variable
x = data_set.iloc[:, [4,5]].values
y = data_set.iloc[:, 11].values

#split data into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#With random_state=0 , we get the same train and test sets across different executions.

#feature scaling for data normalization to normalize the features in dataset into finite range
from sklearn.preprocessing import StandardScaler
st=StandardScaler()
x_train=st.fit_transform(x_train)
x_test=st.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import accuracy_score
a=accuracy_score(y_test,y_pred)
print(a)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
bound1=np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01)
bound2=np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01)
x1,x2=np.meshgrid(bound1,bound2)
s=classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)
plt.contourf(x1,x2,s,alpha=0.75,cmap=ListedColormap(('red','blue')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_train)):
    plt.scatter(x_train[y_train==j,0],x_train[y_train==j,1],c=ListedColormap(('red','blue'))(i),label=j)
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
bound1=np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01)
bound2=np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01)
x1,x2=np.meshgrid(bound1,bound2)
s=classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)
plt.contourf(x1,x2,s,alpha=0.75,cmap=ListedColormap(('red','blue')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_train)):
    plt.scatter(x_train[y_train==j,0],x_train[y_train==j,1],c=ListedColormap(('red','blue'))(i),label=j)
plt.legend()
plt.show()



