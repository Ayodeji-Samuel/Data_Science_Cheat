#1. Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


#2. Import the data-set
data = pd.read_csv('dataset.csv')
data.head()
data.shape
data.index
data.columns


#3. Check out the missing values
data.isnull().sum()
#3.1. Handling missing values
data.dropna(inplace=True)


#4. See the Categorical values
data.select_dtypes(exclude=['int', 'float']).columns
#4.1. Convert Categorical values into numerical values
data = pd.get_dummies(data)
#4.2. Convert Individual Categorical variable and Concat and drop Variable1
dummy = pd.get_dummies(data['Variable1'])
data = pd.concat([data, dummy], axis=1)
data.drop(['Variable1'], axis=1)
#4.3. Heat mapping of correlation Numerical variable or feature
correlation_matrix = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,vmax=0.8,square = True)
plt.show()


#5. Splitting the data-set into Training and Test set
#5.1. Spliting Data into X(Independent) and y(Dependent or Target)
X = data[['Variable2', 'Variable3', 'Variable4', 'Variable5']]
y = data[['Variable_target']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#6. Feature Scaling
#6.1. Plot the distribution of all the continuous variables in the data set for feature scaling
X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")].index.values].hist(figsize=[11,11])

### Feature scaling is the method to limit the range of variables so that they can be compared on common grounds.
### It is performed on continuous variables.
from sklearn.preprocessing import StandardScaler
standard_X = StandardScaler()

X_train = standard_X.fit_transform(X_train)
X_test = standard_X.fit_transform(X_test)
