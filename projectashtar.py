# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
####################### Loading the required libraries ###############################

import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Avoid Warnings
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

#Common model helpers

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, r2_score, accuracy_score
from sklearn.model_selection import (GridSearchCV, KFold, train_test_split, cross_val_score)

from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


################################ Read in water_potability file #########################################

waterDf = pd.read_csv('C:/Users/ASHISH BANSAL/OneDrive/Desktop/proj/water_potability.csv')

################################ Make a copy ################################################
waterDf.head()
waterData = waterDf.copy()

"""# INFORMATION ON WATER-POTATBILITY DATA"""

############################# About the data ###################################


print( waterData.info())

"""All 10 variables of the data are numerical. The target variable takes binary values 0 and 1. 
The feature variables are real numbers.
"""

print('Information about features\n')
print(waterData.describe())

################################ How does the data look like? ############################
print('How does the water-potability data look like?\n')
print(waterData.head())

###################### We work on the missing data ##############################
print('There are missing values within the data.\n')
print('The nature of the missing values within the features are as follows:\n')
print(waterData.isna().sum())

"""There are 491, 781 and 162 missing data in 'ph', 'Sulphate', and 'Trihalomethanes' respectively. 
We shall impute the missing values with the mean of the respective features by grouping them w.r.t 'Potability'.
"""

##################################### Imputing 'ph' value #####################################

phMean_0 = waterData[waterData['Potability'] == 0]['ph'].mean(skipna=True)
waterData.loc[(waterData['Potability'] == 0) & (waterData['ph'].isna()), 'ph'] = phMean_0
phMean_1 = waterData[waterData['Potability'] == 1]['ph'].mean(skipna=True)
waterData.loc[(waterData['Potability'] == 1) & (waterData['ph'].isna()), 'ph'] = phMean_1

##################################### Imputing 'Sulfate' value #####################################

SulfateMean_0 = waterData[waterData['Potability'] == 0]['Sulfate'].mean(skipna=True)
waterData.loc[(waterData['Potability'] == 0) & (waterData['Sulfate'].isna()), 'Sulfate'] = SulfateMean_0
SulfateMean_1 = waterData[waterData['Potability'] == 1]['Sulfate'].mean(skipna=True)
waterData.loc[(waterData['Potability'] == 1) & (waterData['Sulfate'].isna()), 'Sulfate'] = SulfateMean_1

################################ Imputing 'Trihalomethanes' value #####################################

TrihalomethanesMean_0 = waterData[waterData['Potability'] == 0]['Trihalomethanes'].mean(skipna=True)
waterData.loc[(waterData['Potability'] == 0) & (waterData['Trihalomethanes'].isna()), 'Trihalomethanes'] = TrihalomethanesMean_0
TrihalomethanesMean_1 = waterData[waterData['Potability'] == 1]['Trihalomethanes'].mean(skipna=True)
waterData.loc[(waterData['Potability'] == 1) & (waterData['Trihalomethanes'].isna()), 'Trihalomethanes'] = TrihalomethanesMean_1

########################################## Check ####################################
print('Checking to see any more missing data \n')
waterData.isna().sum()

######################### Convert 'Potability' to Category #######################

waterData['Potability'] = waterData['Potability'].astype('category')
waterData.info()

"""# EXPLORING DATA THROUGH VISUALS"""

print('Distribution of Target Variable within the sample data')

ax = plt.subplots(ncols=1, nrows=1, figsize=(16,6))
labels = list(waterData['Potability'].unique())
## use the wedgeprops and textprops arguments to style the wedges and texts, respectively
ax[1].pie(waterData['Potability'].value_counts(), labels=labels, autopct = '%1.1f%%',
          colors=['red', 'blue'], explode = [0.005]*len(labels),
          textprops={'size': 'x-large'},
         wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})

plt.show()

"""There are 1998 data with Potability=1 and 1278 with Potability=0. 
Hence the data is imbalanced. We shall balance the data through SMOTE.

Let's look at the correlation matrix of the features.
"""

######################################### Correlation Matrix #############################################

Corrmat = waterData.corr()
plt.subplots(figsize=(7,7))
sns.heatmap(Corrmat, cmap="YlGnBu", square = True, annot=True, fmt='.2f')
plt.show()

"""The Correlation graph shows absence of multicollinearity. """

print('Boxplot and density distribution of different features by Potability\n')

fig, ax = plt.subplots(ncols=2, nrows=9, figsize=(14, 28))

features = list(waterData.columns.drop('Potability'))
i=0
for cols in features:
    sns.kdeplot(waterData[cols], fill=True, alpha=0.4, hue = waterData.Potability, 
                palette=('indianred', 'steelblue'), multiple='stack', ax=ax[i,0])
    
    sns.boxplot(data= waterData, y=cols, x='Potability', ax=ax[i, 1],
                palette=('indianred', 'steelblue'))
    ax[i,0].set_xlabel(' ')
    ax[i,1].set_xlabel(' ')
    ax[i,1].set_ylabel(' ')
    ax[i,1].xaxis.set_tick_params(labelsize=14)
    ax[i,0].tick_params(left=False, labelleft=False)
    ax[i,0].set_ylabel(cols, fontsize=16)
    i=i+1
      
plt.show()

"""From the kdeplots there seems to be very less difference in mean values of the features among the levels of Potability."""

print('Correlation of Potability with feature variables:')
features = list(waterData.columns.drop('Potability'))

Corr = list()
for cols in features:
    Corr.append(waterData[cols].corr(waterData['Potability']))

corrDf = pd.DataFrame({'Features' : features, 'Corr' : Corr})
corrDf['Corr'] = corrDf['Corr'].abs()
corrDf.sort_values(by='Corr', ascending = True)
corrDf.head()

"""# PREPARING THE DATA FOR MODELLING"""

##################### Preparing the Data for Modelling ######################

X = waterData.drop('Potability', axis = 1).copy()
y = waterData['Potability'].copy()

############################# Train-Test split ############################
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

########################## Synthetic OverSampling ###########################
print('Balancing the data by SMOTE - Oversampling of Minority level\n')
smt = SMOTE()
counter = Counter(y_train)
print('Before SMOTE', counter)
X_train, y_train = smt.fit_resample(X_train, y_train)
counter = Counter(y_train)
print('\nAfter SMOTE', counter)

################################# Scaling #################################
ssc = StandardScaler()

X_train = ssc.fit_transform(X_train)
X_test = ssc.transform(X_test)

modelAccuracy = list()


model = [LogisticRegression(), RandomForestClassifier() ,DecisionTreeClassifier()
        ]
trainAccuracy = list()
testAccuracy = list()
kfold = KFold(n_splits=10, random_state=7, shuffle=True)

for mdl in model:
    trainResult = cross_val_score(mdl, X_train, y_train, scoring='accuracy', cv=kfold)
    trainAccuracy.append(trainResult.mean())
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    testResult = metrics.accuracy_score(y_test, y_pred)
    testAccuracy.append(testResult)

"""# RESULTS"""

print('The comparision\n')
modelScore = pd.DataFrame({'Model' : model, 'Train_Accuracy' : trainAccuracy, 'Test_Accuracy' : testAccuracy})
modelScore


########################################## RandomForestClassfier #############################
print('Random Forest Classifier\n')
Rfc = RandomForestClassifier()
Rfc.fit(X_train, y_train)

y_Rfc = Rfc.predict(X_test)
print(metrics.classification_report(y_test, y_Rfc))
print(modelAccuracy.append(metrics.accuracy_score(y_test, y_Rfc)))

sns.heatmap(confusion_matrix(y_test, y_Rfc), annot=True, fmt='d')
plt.show()

p=[[7.36876853,	213.1970192	,18452.02299,	8.292444792	,294.344858	,496.270851,	10.54509177,	94.8377427,	3.813881276]]

result=Rfc.predict(p)


print(result[0])
##################################For Deploying the model######################################
import pickle
pickle.dump(Rfc,open('Model.pkl','wb'))

