import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_csv('Data-sets/fact_bating_summary,,,...csv')
dataset=dataset.loc[:,['balls','runs','4s','6s','SR','out/not_out','category']]
print(dataset)

# Pre-process the data
dataset = dataset.dropna() # Remove rows with missing values
dataset = dataset[dataset['balls'] != 0] # Remove rows where no balls were faced
dataset = dataset[dataset['runs'] != 0] # Remove rows where no runs were scored
dataset['strike_rate'] = dataset['runs'] / dataset['balls'] * 100 # Calculate strike rate

# Create new features
dataset['boundary_percentage'] = dataset['4s'] / dataset['balls'] * 100
dataset['six_percentage'] = dataset['6s'] / dataset['balls'] * 100
print(dataset)

# Split the data into training and testing sets
X = dataset[['boundary_percentage', 'six_percentage']]
y = dataset['category']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

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
