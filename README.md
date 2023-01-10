import pandas as pd 
import numpy as np 
df= pd.read_csv("health care diabetes.csv")
df.head()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
0	6	148	72	35	0	33.6	0.627	50	1
1	1	85	66	29	0	26.6	0.351	31	0
2	8	183	64	0	0	23.3	0.672	32	1
3	1	89	66	23	94	28.1	0.167	21	0
4	0	137	40	35	168	43.1	2.288	33	1
df.tail()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
763	10	101	76	48	180	32.9	0.171	63	0
764	2	122	70	27	0	36.8	0.340	27	0
765	5	121	72	23	112	26.2	0.245	30	0
766	1	126	60	0	0	30.1	0.349	47	1
767	1	93	70	31	0	30.4	0.315	23	0
1. Perform descriptive analysis. Understand the variables and their corresponding values. On the columns below, a value of zero does not make sense and thus indicates missing value:
• Glucose

• BloodPressure

• SkinThickness

• Insulin

• BMI

df.describe()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
count	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000
mean	3.845052	120.894531	69.105469	20.536458	79.799479	31.992578	0.471876	33.240885	0.348958
std	3.369578	31.972618	19.355807	15.952218	115.244002	7.884160	0.331329	11.760232	0.476951
min	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.078000	21.000000	0.000000
25%	1.000000	99.000000	62.000000	0.000000	0.000000	27.300000	0.243750	24.000000	0.000000
50%	3.000000	117.000000	72.000000	23.000000	30.500000	32.000000	0.372500	29.000000	0.000000
75%	6.000000	140.250000	80.000000	32.000000	127.250000	36.600000	0.626250	41.000000	1.000000
max	17.000000	199.000000	122.000000	99.000000	846.000000	67.100000	2.420000	81.000000	1.000000
df.isnull().sum()
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
dtype: int64
df.isna().sum()
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
dtype: int64
df.describe()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
count	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000
mean	3.845052	120.894531	69.105469	20.536458	79.799479	31.992578	0.471876	33.240885	0.348958
std	3.369578	31.972618	19.355807	15.952218	115.244002	7.884160	0.331329	11.760232	0.476951
min	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.078000	21.000000	0.000000
25%	1.000000	99.000000	62.000000	0.000000	0.000000	27.300000	0.243750	24.000000	0.000000
50%	3.000000	117.000000	72.000000	23.000000	30.500000	32.000000	0.372500	29.000000	0.000000
75%	6.000000	140.250000	80.000000	32.000000	127.250000	36.600000	0.626250	41.000000	1.000000
max	17.000000	199.000000	122.000000	99.000000	846.000000	67.100000	2.420000	81.000000	1.000000
from sklearn.impute import SimpleImputer
treat_columns = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
mean_imputer = SimpleImputer(missing_values=0,strategy='mean')
mean_imputer = mean_imputer.fit(df[treat_columns])
df[treat_columns] = mean_imputer.transform(df[treat_columns])
df.describe()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
count	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000
mean	3.845052	121.686763	72.405184	29.153420	155.548223	32.457464	0.471876	33.240885	0.348958
std	3.369578	30.435949	12.096346	8.790942	85.021108	6.875151	0.331329	11.760232	0.476951
min	0.000000	44.000000	24.000000	7.000000	14.000000	18.200000	0.078000	21.000000	0.000000
25%	1.000000	99.750000	64.000000	25.000000	121.500000	27.500000	0.243750	24.000000	0.000000
50%	3.000000	117.000000	72.202592	29.153420	155.548223	32.400000	0.372500	29.000000	0.000000
75%	6.000000	140.250000	80.000000	32.000000	155.548223	36.600000	0.626250	41.000000	1.000000
max	17.000000	199.000000	122.000000	99.000000	846.000000	67.100000	2.420000	81.000000	1.000000
df['Glucose'].describe()
count    768.000000
mean     121.686763
std       30.435949
min       44.000000
25%       99.750000
50%      117.000000
75%      140.250000
max      199.000000
Name: Glucose, dtype: float64
2. Visually explore these variables using histograms. Treat the missing values accordingly.
import matplotlib
import matplotlib.pyplot as plt
plt.hist(df['Glucose'], 
         facecolor='yellow', 
         edgecolor='blue',
         bins=10,
        align='mid')
plt.show()

df['BloodPressure'].describe()
count    768.000000
mean      72.405184
std       12.096346
min       24.000000
25%       64.000000
50%       72.202592
75%       80.000000
max      122.000000
Name: BloodPressure, dtype: float64
plt.hist(df['BloodPressure'], 
         facecolor='yellow', 
         edgecolor='blue',
         bins=10,
        align='mid')
plt.show()

df['SkinThickness'].describe()
count    768.000000
mean      29.153420
std        8.790942
min        7.000000
25%       25.000000
50%       29.153420
75%       32.000000
max       99.000000
Name: SkinThickness, dtype: float64
plt.hist(df['SkinThickness'], 
         facecolor='yellow', 
         edgecolor='blue',
         bins=10,
        align='mid')
plt.show()

df['Insulin'].describe()
count    768.000000
mean     155.548223
std       85.021108
min       14.000000
25%      121.500000
50%      155.548223
75%      155.548223
max      846.000000
Name: Insulin, dtype: float64
plt.hist(df['Insulin'], 
         facecolor='yellow', 
         edgecolor='blue',
         bins=10,
        align='mid')
plt.show()

df['BMI'].describe()
count    768.000000
mean      32.457464
std        6.875151
min       18.200000
25%       27.500000
50%       32.400000
75%       36.600000
max       67.100000
Name: BMI, dtype: float64
plt.hist(df['BMI'], 
         facecolor='yellow', 
         edgecolor='blue',
         bins=10,
        align='mid')
plt.show()

df.dtypes
Pregnancies                   int64
Glucose                     float64
BloodPressure               float64
SkinThickness               float64
Insulin                     float64
BMI                         float64
DiabetesPedigreeFunction    float64
Age                           int64
Outcome                       int64
dtype: object
3. There are integer and float data type variables in this dataset. Create a count (frequency) plot describing the data types and the count of variables.
LABELS=['NO_DIABETES','WITH_DIABETES']
#Create independent and Dependent Features
columns = df.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Outcome"]]
# Store the variable we are predicting 
target = "Outcome"
# Define a random state 
state = np.random.RandomState(42)
X = df[columns]
Y = df[target]
# Print the shapes of X & y
print(X.shape)
print(Y.shape)
(768, 8)
(768,)
df.isnull().values.any()
False
count_classes = pd.value_counts(df['Outcome'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Whether Patient has diabetes or not")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")
Text(0, 0.5, 'Frequency')

!pip install imblearn
Requirement already satisfied: imblearn in c:\users\hp\anaconda3\lib\site-packages (0.0)
Requirement already satisfied: imbalanced-learn in c:\users\hp\anaconda3\lib\site-packages (from imblearn) (0.9.1)
Requirement already satisfied: scikit-learn>=1.1.0 in c:\users\hp\anaconda3\lib\site-packages (from imbalanced-learn->imblearn) (1.1.2)
Requirement already satisfied: scipy>=1.3.2 in c:\users\hp\anaconda3\lib\site-packages (from imbalanced-learn->imblearn) (1.7.3)
Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\hp\anaconda3\lib\site-packages (from imbalanced-learn->imblearn) (2.2.0)
Requirement already satisfied: joblib>=1.0.0 in c:\users\hp\anaconda3\lib\site-packages (from imbalanced-learn->imblearn) (1.1.0)
Requirement already satisfied: numpy>=1.17.3 in c:\users\hp\anaconda3\lib\site-packages (from imbalanced-learn->imblearn) (1.21.5)
from imblearn.over_sampling import SMOTE 
oversample = SMOTE()
x_train, y_train = oversample.fit_resample(X, Y)
x=df[treat_columns] 
x.shape,y_train.shape
((768, 5), (1000,))
from collections import Counter
print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_train)))
Original dataset shape Counter({0: 500, 1: 268})
Resampled dataset shape Counter({1: 500, 0: 500})
#df1['Glucose'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.pairplot(df)
<seaborn.axisgrid.PairGrid at 0x1a0295dff70>

df
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
0	6	148.0	72.0	35.00000	155.548223	33.6	0.627	50	1
1	1	85.0	66.0	29.00000	155.548223	26.6	0.351	31	0
2	8	183.0	64.0	29.15342	155.548223	23.3	0.672	32	1
3	1	89.0	66.0	23.00000	94.000000	28.1	0.167	21	0
4	0	137.0	40.0	35.00000	168.000000	43.1	2.288	33	1
...	...	...	...	...	...	...	...	...	...
763	10	101.0	76.0	48.00000	180.000000	32.9	0.171	63	0
764	2	122.0	70.0	27.00000	155.548223	36.8	0.340	27	0
765	5	121.0	72.0	23.00000	112.000000	26.2	0.245	30	0
766	1	126.0	60.0	29.15342	155.548223	30.1	0.349	47	1
767	1	93.0	70.0	31.00000	155.548223	30.4	0.315	23	0
768 rows × 9 columns

sns.pairplot(df,hue='Outcome',palette='rainbow')
<seaborn.axisgrid.PairGrid at 0x1a02e404790>

df.corr()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
Pregnancies	1.000000	0.127911	0.208522	0.082989	0.056027	0.021565	-0.033523	0.544341	0.221898
Glucose	0.127911	1.000000	0.218367	0.192991	0.420157	0.230941	0.137060	0.266534	0.492928
BloodPressure	0.208522	0.218367	1.000000	0.192816	0.072517	0.281268	-0.002763	0.324595	0.166074
SkinThickness	0.082989	0.192991	0.192816	1.000000	0.158139	0.542398	0.100966	0.127872	0.215299
Insulin	0.056027	0.420157	0.072517	0.158139	1.000000	0.166586	0.098634	0.136734	0.214411
BMI	0.021565	0.230941	0.281268	0.542398	0.166586	1.000000	0.153400	0.025519	0.311924
DiabetesPedigreeFunction	-0.033523	0.137060	-0.002763	0.100966	0.098634	0.153400	1.000000	0.033561	0.173844
Age	0.544341	0.266534	0.324595	0.127872	0.136734	0.025519	0.033561	1.000000	0.238356
Outcome	0.221898	0.492928	0.166074	0.215299	0.214411	0.311924	0.173844	0.238356	1.000000
sns.heatmap(df.corr())
<AxesSubplot:>

sns.heatmap(df.corr(),cmap='coolwarm',annot=True)
<AxesSubplot:>

df['Outcome'].value_counts()
0    500
1    268
Name: Outcome, dtype: int64
df['Outcome'].shape
(768,)
# ## Get the Fraud and the normal dataset 

# WITH_DIABETES = df2['Outcome']==1

# NO_DIABETES = df2['Outcome']==0 
#print(WITH_DIABETES.shape,NO_DIABETES.shape)
outcome_name = ['Outcome']
outcome_labels = df[outcome_name]
from sklearn.linear_model import LogisticRegression  
# importing the class. 
# LogisticRegression is best suited for binary classification

import numpy as np
import warnings; warnings.simplefilter('ignore')  

# make the model or object
lr = LogisticRegression()  # making object of the LogisticRegression class.

#--fit the model
model = lr.fit(df[treat_columns], outcome_labels['Outcome'])

model

LogisticRegression
LogisticRegression()
# predictions = model.predict(prediction_features)

# ##--display results
# new_data['Outcome'] = predictions
# print(new_data)
X = df[treat_columns] 
y = df['Outcome']  

# the final preprocessing step is to divide data into training and test sets
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.20, 
                                                    random_state=0)

lr.fit(X_train, y_train)  
y_pred = lr.predict(X_test) 


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))
[[95 12]
 [21 26]]
              precision    recall  f1-score   support

           0       0.82      0.89      0.85       107
           1       0.68      0.55      0.61        47

    accuracy                           0.79       154
   macro avg       0.75      0.72      0.73       154
weighted avg       0.78      0.79      0.78       154

0.7857142857142857
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
fpr, tpr, thershold = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
rfc_roc = roc_auc_score(y_pred,y_test)
plt.figure()
plt.subplots(figsize=(15,10))
plt.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)'%rfc_roc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0,1.0])
plt.ylim([0,1.01])
plt.xlabel('False Positive Rate (1-specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.title('Receiver operating characteristic for Logistic Regression ')
plt.legend(loc ="lower right")
plt.show()
<Figure size 432x288 with 0 Axes>

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
 
# Create feature and target arrays
X = df[treat_columns] 
y = df['Outcome']  
 
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
 
knn = KNeighborsClassifier(n_neighbors=7)
 
knn.fit(X_train, y_train)
 
# Predict on dataset which model has not seen before
print(knn.predict(X_test))

print(knn.score(X_test, y_test))
[0 1 1 1 0 0 0 1 1 1 0 1 0 0 0 1 0 0 1 1 0 0 0 0 1 1 0 0 0 0 1 1 1 1 0 1 1
 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 1 1 0 0 0
 0 0 0 0 0 0 1 0 0 1 0 1 1 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 0 1 1 0 0 1 0 0 0
 1 0 1 1 1 1 1 0 1 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 0 0 1 1 0 0 0 1 1 0 0 0 0
 1 1 0 0 0 0]
0.7142857142857143
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
 
# Create feature and target arrays
X = df[treat_columns] 
y = df['Outcome']  
 
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
 
knn = KNeighborsClassifier(n_neighbors=5)
 
knn.fit(X_train, y_train)
 
# Predict on dataset which model has not seen before
print(knn.predict(X_test))

print(knn.score(X_test, y_test))
[0 1 1 1 0 0 0 1 1 1 0 1 0 0 0 1 0 0 1 1 1 0 0 0 0 1 0 0 0 0 1 1 1 1 0 1 1
 0 0 1 0 0 0 1 0 1 0 0 0 1 0 1 0 1 0 0 1 0 0 1 1 1 0 0 0 1 0 1 0 1 1 0 0 0
 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0
 1 1 1 1 1 1 1 0 1 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 0 0 1 1 0 0 0 1 1 0 0 0 0
 1 1 0 0 0 0]
0.7012987012987013
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
 
# Create feature and target arrays
X = df[treat_columns] 
y = df['Outcome']  
 
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
 
knn = KNeighborsClassifier(n_neighbors=11)
 
knn.fit(X_train, y_train)
 
# Predict on dataset which model has not seen before
print(knn.predict(X_test))

print(knn.score(X_test, y_test))
[0 1 0 0 0 0 0 1 1 1 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1 1 1 1 0 1 1
 0 0 1 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0 0 1 0 0 1 1 0 0 0 0 1 0 1 0 1 0 0 0 0
 0 0 0 0 0 0 1 0 0 1 0 1 1 0 0 0 0 0 0 1 1 1 0 0 1 0 0 0 0 0 1 0 1 1 0 1 0
 1 0 1 1 1 1 1 0 1 0 0 1 0 0 0 1 1 1 1 1 0 0 1 1 0 0 1 1 0 0 0 0 1 0 0 0 0
 1 1 0 0 0 0]
0.7402597402597403
https://public.tableau.com/app/profile/alluru.sumaneesh/viz/HealthCareDiabetes_DataReporting/DataReporting
 
