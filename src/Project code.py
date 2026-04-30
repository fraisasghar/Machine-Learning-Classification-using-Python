# %% [markdown]
# ICT Project part b

# %%
# Loading Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.tree import plot_tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 

# %%
df = pd.read_csv(r"C:\Users\msaad\Documents\Uni books\all_stocks_5yr.csv", header=None, names=['Stock Price','Trading Volume','Name'])

df.head(100)

# %%
# To find outliers
cols = df.columns[0:-1]
for i in cols:
    sns.boxplot(y=df[i])
    plt.show()

# %%
# To remove outliers from 'Trading Volume'
q1 = df['Trading Volume'].quantile(0.25)
q3 = df['Trading Volume'].quantile(0.75)
iqr = q3 - q1
df = df[(df['Trading Volume'] >= q1-0.5*iqr) & (df['Trading Volume'] <= q3+0.5*iqr)]
df.shape # To find out the number of rows and column after outlier treatment

# %%
sns.boxplot(y=df['Trading Volume'])
plt.show()

# %%
# To know number of rows and collumns
df.shape

# %%
# Check the dataframe information
df.info()

# %%
# To see summary statistics
df.describe().T

# %%
# Splitting the data into train and test sets
X = df[["Stock Price","Trading Volume"]]
y = df["Name"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state= 1)

# %%
# Defining an object for DTC and fitting for whole dataset
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=1 )
dt.fit(X, y)

# %%
from IPython.display import Image
from sklearn.tree import  plot_tree


features = X.columns
plt.figure(figsize=(12,9)) 
plot_tree(dt, feature_names=features,fontsize=7)
plt.show()

# %%
# Defining an object for DTC and fitting for train dataset
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train, y_train)

y_pred_train = dt.predict(X_train)
y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)

# %%
print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))
print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred, y_test))

# %%
#Classification for test before hyperparameter tuning
print(classification_report(y_test,y_pred))

# %%
# Hyperparameter Tuning of DTC

dt = DecisionTreeClassifier(random_state=1)

params = {'max_depth' : [2,3,4,5],
        'min_samples_split': [2,3,4,5],
        'min_samples_leaf': [1,2,3,4,5]}

gsearch = GridSearchCV(dt, param_grid=params, cv=3)

gsearch.fit(X,y)

gsearch.best_params_

# %%
# Passing best parameter for the Hyperparameter Tuning
dt = DecisionTreeClassifier(**gsearch.best_params_, random_state=1)

dt.fit(X_train, y_train)

y_pred_train = dt.predict(X_train)
y_prob_train = dt.predict_proba(X_train)[:,1]

y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:,1]

# %%
print('Confusion Matrix - Train:','\n',confusion_matrix(y_train,y_pred_train))
print('\n','Confusion Matrix - Test:','\n',confusion_matrix(y_test,y_pred))

# %%
#Classification for test after hyperparameter tuning
print(classification_report(y_test,y_pred))

# %%
print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))
print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred, y_test))

# %%
import seaborn as sns

# %%
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues, labels=None):
    plt.figure(figsize=(8, 6))  # Adjust figure size for better readability
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=True,
                xticklabels=labels if labels else np.arange(cm.shape[1]),
                yticklabels=labels if labels else np.arange(cm.shape[0]))
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout() # Adjust layout to prevent labels from being cut off
    plt.show()

# %%
class_names = ['AAL', 'AAPL', 'AAP', 'ABBV', 'ABC', 'ABT', 'ACN'] # Replace with your actual labels

plot_confusion_matrix(cm_train, title='Confusion Matrix - Train', labels=class_names)
plot_confusion_matrix(cm_test, title='Confusion Matrix - Test', labels=class_names)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Simulate Data (Replace this with your actual X_train and y_train)
# X_train = your feature dataset
# y_train = your target labels

# Example: Create some synthetic data (for demonstration purposes)
from sklearn.datasets import make_classification
X_train, y_train = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)

# Ensure X_train is a NumPy array with two features
if hasattr(X_train, "values"):  # Check if it's a DataFrame
    X_train = X_train.values  # Convert DataFrame to NumPy array
X_train = np.array(X_train)  # Ensure it's a NumPy array

# Ensure X_train has the correct shape for plotting
if X_train.ndim == 1:  # If it's a 1D array, reshape it
    X_train = X_train.reshape(-1, 1)

# Verify the shape
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")

# Train the Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train, y_train)

# Function to Plot Decision Boundary
def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel("Stock Price")
    plt.ylabel("Trading Volume")
    plt.show()

# Plot Decision Boundary if X_train has exactly 2 features
if X_train.shape[1] == 2:
    plot_decision_boundary(X_train, y_train, dt)
else:
    print(f"Cannot plot decision boundary: X_train has {X_train.shape[1]} features.")

# %%
