# Step: 1
# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Step: 2
# define the path 
# reading diabetes data from CSV File
group_8_df = pd.read_csv('diabetic_data.csv')

# Step: 3
print('Number of Records:', len(group_8_df))

# Overview of the data (columns, variable type and non-null values)
# Step: 4
print(group_8_df.info())

# After looking at the data columns, we can see some numerical and categorical columns
# Step: 5
# print top 5 records
print(group_8_df.head())

# CSV file contains some null or empty records with "?"
# Step: 6
# count the number of rows for each type for 'readmitted' column
print(group_8_df.groupby('readmitted').size())

# `discharge_disposition_id` column tells us where the patient went after the hospitalization.
# Step: 7
print(group_8_df.groupby('discharge_disposition_id').size())

# by looking at IDs_mapping.csv file we can see that (11,13,14,19,20,21) values are related to death or hospice so we can remove these unnecessary columns
# Step: 8
group_8_df = group_8_df.loc[~group_8_df.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])]

# add new columns as 'OUTPUT_CLASS' for our binary classification
# Here we will try to predict if a patient is likely to be re-admitted within 30 days of discharge.
# Step: 9
group_8_df['OUTPUT_CLASS'] = (group_8_df.readmitted == '<30').astype('int')


# create a function to calculate the prevalence of population that is readmitted with 30 days. 
# Step: 10
def calc_prevalence(y_actual):
    return (sum(y_actual) / len(y_actual))


# Step: 11
print('Prevalence:%.2f' % calc_prevalence(group_8_df['OUTPUT_CLASS'].values))

# Around 11% of the population is rehospitalized.
#  Pandas doesn't allow you to see all the columns at once, so we will view the data in the group of 10. 

# Display all column names
# Step: 12
print('Number of columns:', len(group_8_df.columns))

# View first 10 col values
# Step: 13
print(group_8_df[list(group_8_df.columns)[:10]].head())

# Step: 14
print(group_8_df[list(group_8_df.columns)[10:20]].head())

# Step: 15
print(group_8_df[list(group_8_df.columns)[20:30]].head())

# Step: 16
print(group_8_df[list(group_8_df.columns)[30:40]].head())

# Step: 17
print(group_8_df[list(group_8_df.columns)[40:]].head())

# Printing Unique values from each columns to find out the Categories of each columns
# Step: 18
# for loop - each column
for col in list(group_8_df.columns):

    # list of unique values
    n = group_8_df[col].unique()

    # if number of unique values is less than 30, print the values. 
    if len(n) < 30:
        print("Column Name: " + col)
        print(n)
    else:
        # else  print the number of unique values
        print("Column Name: " + col + ',  unique values: ' + str(len(n)))

# Step: 18
# replace ? with nan
group_8_df = group_8_df.replace('?', np.nan)

# Adding Numerical Features (int64) columns into cols_num list as it is easy to use
# Step: 19
cols_num = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

# find missing values in the numerical data.
# Step: 20
group_8_df[cols_num].isnull().sum()

# Adding Categorical Features (Object) columns into cols_cat list as it is easy to use
# Step: 21
cols_cat = ['race', 'gender',
            'max_glu_serum', 'A1Cresult',
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
            'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
            'tolazamide', 'insulin',
            'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone',
            'metformin-pioglitazone', 'change', 'diabetesMed', 'payer_code', 'medical_specialty']

# find missing values in the categorical data.
# Step: 22
group_8_df[cols_cat].isnull().sum()

# `race`, `payer_code`, and `medical_specialty` have alot of missing data, replace it with 'UNK' using fillna function.
# Step: 23
group_8_df['race'] = group_8_df['race'].fillna('UNK')
group_8_df['payer_code'] = group_8_df['payer_code'].fillna('UNK')
group_8_df['medical_specialty'] = group_8_df['medical_specialty'].fillna('UNK')

# review medical specialty data using grouby
# Step: 24
print('Number medical specialty:', group_8_df.medical_specialty.nunique())
group_8_df.groupby('medical_specialty').size().sort_values(ascending=False)

# Step: 25
# consider only top 10 samples and add one more category as other for rest sample data
top_10 = ['UNK', 'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice', 'Cardiology', 'Surgery-General',
          'Nephrology', 'Orthopedics',
          'Orthopedics-Reconstructive', 'Radiologist']

# Step: 26
# add new column as med_spec and copy medical_specialty column data into it
group_8_df['med_spec'] = group_8_df['medical_specialty'].copy()

# Step: 27
# replace all specialties not in top 10 with 'Other' category
group_8_df.loc[~group_8_df.med_spec.isin(top_10), 'med_spec'] = 'Other'

# Step: 28
# check total records of each type
group_8_df.groupby('med_spec').size()

# Step: 29
# Convert numerical data into strings
cols_cat_num = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
group_8_df[cols_cat_num] = group_8_df[cols_cat_num].astype('str')

# add categorical, numerical columns together
# Step: 30
group_8_df_cat = pd.get_dummies(group_8_df[cols_cat + cols_cat_num + ['med_spec']], drop_first=True)

# Step: 31
# print top five records to check the data
print(group_8_df_cat.head())

# using 'concat' function to merge the columns
# Step: 32
group_8_df = pd.concat([group_8_df, group_8_df_cat], axis=1)

# Save the column names of the categorical data.
# Step: 33
cols_all_cat = list(group_8_df_cat.columns)

# add Extra features using two more columns as `age` and `weight`
# Step: 34
print(group_8_df[['age', 'weight']].head())

# convert age to numerical data
# Step: 35
group_8_df.groupby('age').size()

age_id = {'[0-10)': 0,
          '[10-20)': 10,
          '[20-30)': 20,
          '[30-40)': 30,
          '[40-50)': 40,
          '[50-60)': 50,
          '[60-70)': 60,
          '[70-80)': 70,
          '[80-90)': 80,
          '[90-100)': 90}
group_8_df['age_group'] = group_8_df.age.replace(age_id)

# check weight column contains null values or not
# Step: 36
group_8_df.weight.notnull().sum()

# Step: 37
# add new column as has_weight and 1 or 0 based on if value exists or not
group_8_df['has_weight'] = group_8_df.weight.notnull().astype('int')

# Step: 38
# add into another list
cols_extra = ['age_group', 'has_weight']

# Overall Engineering Features Summary
# Step: 39

print('Total number of features:', len(cols_num + cols_all_cat + cols_extra))
print('Numerical Features:', len(cols_num))
print('Categorical Features:', len(cols_all_cat))
print('Extra features:', len(cols_extra))

# finally, check if we are missing any data.
# Step: 40
print(group_8_df[cols_num + cols_all_cat + cols_extra].isnull().sum().sort_values(ascending=False).head(10))

# create new dataframe which will only conatain usefull columns
# Step: 41
col2use = cols_num + cols_all_cat + cols_extra
group_8_df_data = group_8_df[col2use + ['OUTPUT_CLASS']]

# split data into 70% train, 15% validation, 15% test. 
# Step: 42
group_8_df_data = group_8_df_data.sample(n=len(group_8_df_data), random_state=42)
group_8_df_data = group_8_df_data.reset_index(drop=True)

# Save 30% of the data as validation and test data 
# Step: 43
group_8_df_valid_test = group_8_df_data.sample(frac=0.30, random_state=42)
print('Split size: %.3f' % (len(group_8_df_valid_test) / len(group_8_df_data)))

# split 30% of data into two parts:
# Step: 44
group_8_df_test = group_8_df_valid_test.sample(frac=0.5, random_state=42)
group_8_df_valid = group_8_df_valid_test.drop(group_8_df_test.index)

# drop testing data and rest add for train the model
# Step: 45
group_8_df_train_all = group_8_df_data.drop(group_8_df_valid_test.index)

# percent of our groups are hospitalized within 30 days.
# all three groups will show same percentage of prevalence. 
# Step: 46
print('Test prevalence(n = %d):%.3f' % (len(group_8_df_test), calc_prevalence(group_8_df_test.OUTPUT_CLASS.values)))
print('Valid prevalence(n = %d):%.3f' % (len(group_8_df_valid), calc_prevalence(group_8_df_valid.OUTPUT_CLASS.values)))
print('Train all prevalence(n = %d):%.3f' % (
len(group_8_df_train_all), calc_prevalence(group_8_df_train_all.OUTPUT_CLASS.values)))

# make sure we used all the data.
# Step: 47
print('all samples (n = %d)' % len(group_8_df_data))
assert len(group_8_df_data) == (
            len(group_8_df_test) + len(group_8_df_valid) + len(group_8_df_train_all)), 'math didnt work'

# create a balanced training data set that has 50% positive and 50% negative
# Step: 48
rows_pos = group_8_df_train_all.OUTPUT_CLASS == 1
group_8_df_train_pos = group_8_df_train_all.loc[rows_pos]
group_8_df_train_neg = group_8_df_train_all.loc[~rows_pos]

# merge the balanced data
# Step: 49
group_8_df_train = pd.concat(
    [group_8_df_train_pos, group_8_df_train_neg.sample(n=len(group_8_df_train_pos), random_state=42)], axis=0)

# pick the random sample
# Step: 50
group_8_df_train = group_8_df_train.sample(n=len(group_8_df_train), random_state=42).reset_index(drop=True)
print('Train balanced prevalence(n = %d):%.3f' % (
len(group_8_df_train), calc_prevalence(group_8_df_train.OUTPUT_CLASS.values)))

# save all traning, testing and validation data csv files.
# Step: 51
group_8_df_train_all.to_csv('group_8_df_train_all.csv', index=False)
group_8_df_train.to_csv('group_8_df_train.csv', index=False)
group_8_df_valid.to_csv('group_8_df_valid.csv', index=False)
group_8_df_test.to_csv('group_8_df_test.csv', index=False)

# create input matrix X and output vector y:
# Step: 52
X_train = group_8_df_train[col2use].values
X_train_all = group_8_df_train_all[col2use].values
X_valid = group_8_df_valid[col2use].values

y_train = group_8_df_train['OUTPUT_CLASS'].values
y_valid = group_8_df_valid['OUTPUT_CLASS'].values

print('Training All shapes:', X_train_all.shape)
print('Training shapes:', X_train.shape, y_train.shape)
print('Validation shapes:', X_valid.shape, y_valid.shape)

# Step: 53
# import scikit learn library for scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train_all)

# use scaler to test the data using `pickle` package.
# Step: 53
import pickle

# save scaler.csv file
scalerfile = 'scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))

# Step: 54
# read file
scaler = pickle.load(open(scalerfile, 'rb'))

# Step: 55
# matrix
X_train_tf = scaler.transform(X_train)
X_valid_tf = scaler.transform(X_valid)

# creating functions to evaluate the performance of the model.
# Step: 56
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


def calc_specificity(y_actual, y_pred, thresh):
    # calculates specificity
    return sum((y_pred < thresh) & (y_actual == 0)) / sum(y_actual == 0)


def print_report(y_actual, y_pred, thresh):
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    specificity = calc_specificity(y_actual, y_pred, thresh)
    print('AUC:%.3f' % auc)
    print('accuracy:%.3f' % accuracy)
    print('recall:%.3f' % recall)
    print('precision:%.3f' % precision)
    print('specificity:%.3f' % specificity)
    print('prevalence:%.3f' % calc_prevalence(y_actual))
    print(' ')
    return auc, accuracy, recall, precision, specificity


# Step: 57
# set threshold at 0.5
thresh = 0.5

# Model Selection: baseline models

# Logistic regression
# Step: 58

# logistic regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42, max_iter=10000)
lr.fit(X_train_tf, y_train)

y_train_preds = lr.predict_proba(X_train_tf)[:, 1]
y_valid_preds = lr.predict_proba(X_valid_tf)[:, 1]

print('Logistic Regression')
print('Training:')
lr_train_auc, lr_train_accuracy, lr_train_recall, lr_train_precision, lr_train_specificity = print_report(y_train,
                                                                                                          y_train_preds,
                                                                                                          thresh)
print('Validation:')
lr_valid_auc, lr_valid_accuracy, lr_valid_recall, lr_valid_precision, lr_valid_specificity = print_report(y_valid,
                                                                                                          y_valid_preds,
                                                                                                          thresh)

# Decision Tree Classifier
# Step: 59
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=25, random_state=42)
tree.fit(X_train_tf, y_train)

y_train_preds = tree.predict_proba(X_train_tf)[:, 1]
y_valid_preds = tree.predict_proba(X_valid_tf)[:, 1]

print('Decision Tree')
print('Training:')
tree_train_auc, tree_train_accuracy, tree_train_recall, tree_train_precision, tree_train_specificity = print_report(
    y_train, y_train_preds, thresh)
print('Validation:')
tree_valid_auc, tree_valid_accuracy, tree_valid_recall, tree_valid_precision, tree_valid_specificity = print_report(
    y_valid, y_valid_preds, thresh)

# Analyze results baseline models

# compare results and plot the outcomes using a package called seaborn 
# Step: 60
group_8_df_results = pd.DataFrame({'classifier': ['LR', 'LR', 'DT', 'DT'],
                                   'data_set': ['train', 'valid'] * 2,
                                   'auc': [lr_train_auc, lr_valid_auc, tree_train_auc, tree_valid_auc, ],
                                   'accuracy': [lr_train_accuracy, lr_valid_accuracy, tree_train_accuracy,
                                                tree_valid_accuracy, ],
                                   'recall': [lr_train_recall, lr_valid_recall, tree_train_recall, tree_valid_recall, ],
                                   'precision': [lr_train_precision, lr_valid_precision, tree_train_precision,
                                                 tree_valid_precision, ],
                                   'specificity': [lr_train_specificity, lr_valid_specificity, tree_train_specificity,
                                                   tree_valid_specificity, ]})

# import seaborn
# Step: 61
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

# Step: 62
ax = sns.barplot(x="classifier", y="auc", hue="data_set", data=group_8_df_results)
ax.set_xlabel('Classifier', fontsize=15)
ax.set_ylabel('AUC', fontsize=15)
ax.tick_params(labelsize=15)

# Step: 63
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=15)
plt.show()

# Step: 64
# Feature Importance: Logistic regression

feature_importances = pd.DataFrame(lr.coef_[0],
                                   index=col2use,
                                   columns=['importance']).sort_values('importance',
                                                                       ascending=False)
print(feature_importances.head())

# plot positive features
# Step: 65
num = 50
ylocs = np.arange(num)
# get the feature importance for top num and sort in reverse order
values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]
feature_labels = list(feature_importances.iloc[:num].index)[::-1]

plt.figure(num=None, figsize=(8, 15), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align='center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Positive Feature Importance Score - Logistic Regression')
plt.yticks(ylocs, feature_labels)
plt.show()

# plot negative features
# Step: 66
values_to_plot = feature_importances.iloc[-num:].values.ravel()
feature_labels = list(feature_importances.iloc[-num:].index)

plt.figure(num=None, figsize=(8, 15), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align='center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Negative Feature Importance Score - Logistic Regression')
plt.yticks(ylocs, feature_labels)
plt.show()

print(lr.get_params())

# optimize LR
# Step: 67
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

auc_scoring = make_scorer(roc_auc_score)

penalty = ['none', 'l2']
max_iter = range(1000, 20000, 1000)
c = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.0001, 0.0003]

random_grid_lr = {'penalty': penalty,
                  'max_iter': max_iter,
                  'C': c}
# create the randomized search cross-validation
lr_random = RandomizedSearchCV(estimator=lr, param_distributions=random_grid_lr,
                               n_iter=20, cv=2, scoring=auc_scoring, verbose=0,
                               random_state=42)

t1 = time.time()
lr_random.fit(X_train_tf, y_train)
t2 = time.time()
print(t2 - t1)

lr_random.best_params_

y_train_preds = lr.predict_proba(X_train_tf)[:, 1]
y_valid_preds = lr.predict_proba(X_valid_tf)[:, 1]

print('Baseline sgdc')
lr_train_auc_base = roc_auc_score(y_train, y_train_preds)
lr_valid_auc_base = roc_auc_score(y_valid, y_valid_preds)

print('Training AUC:%.3f' % (lr_train_auc_base))
print('Validation AUC:%.3f' % (lr_valid_auc_base))
print('Optimized sgdc')
y_train_preds_random = lr_random.best_estimator_.predict_proba(X_train_tf)[:, 1]
y_valid_preds_random = lr_random.best_estimator_.predict_proba(X_valid_tf)[:, 1]
lr_train_auc = roc_auc_score(y_train, y_train_preds_random)
lr_valid_auc = roc_auc_score(y_valid, y_valid_preds_random)

print('Training AUC:%.3f' % (lr_train_auc))
print('Validation AUC:%.3f' % (lr_valid_auc))

# Step: 68
# Hyperparameter tuning results
group_8_df_results = pd.DataFrame({'classifier': ['LR', 'LR'],
                                   'data_set': ['base', 'optimized'] * 1,
                                   'auc': [lr_valid_auc_base, lr_valid_auc, ],
                                   })

ax = sns.barplot(x="classifier", y="auc", hue="data_set", data=group_8_df_results)
ax.set_xlabel('Classifier', fontsize=15)
ax.set_ylabel('AUC', fontsize=15)
ax.tick_params(labelsize=15)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=15)
plt.show()

# Model selection
# Step: 69
# creating best model classifier pickle file
pickle.dump(lr_random.best_estimator_, open('best_classifier.pkl', 'wb'), protocol=4)

# # Model Evaluation
# Step: 70

# Evaluating performance of the test set.
X_test = group_8_df_test[col2use].values
y_test = group_8_df_test['OUTPUT_CLASS'].values

scaler = pickle.load(open('scaler.sav', 'rb'))
X_test_tf = scaler.transform(X_test)

# Step: 71
# reading pickle file
best_model = pickle.load(open('best_classifier.pkl', 'rb'))

y_train_preds = best_model.predict_proba(X_train_tf)[:, 1]
y_valid_preds = best_model.predict_proba(X_valid_tf)[:, 1]
y_test_preds = best_model.predict_proba(X_test_tf)[:, 1]

thresh = 0.5
print('-------------------------------predicted values--------------------------------------------------')
print(y_test_preds*100)

print('Training:')
train_auc, train_accuracy, train_recall, train_precision, train_specificity = print_report(y_train, y_train_preds,
                                                                                           thresh)
print('Validation:')
valid_auc, valid_accuracy, valid_recall, valid_precision, valid_specificity = print_report(y_valid, y_valid_preds,
                                                                                           thresh)
print('Test:')
test_auc, test_accuracy, test_recall, test_precision, test_specificity = print_report(y_test, y_test_preds, thresh)

# ROC curve
# Step: 72
# ploting ROC curve

from sklearn.metrics import roc_curve

fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_preds)
auc_train = roc_auc_score(y_train, y_train_preds)

fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_valid_preds)
auc_valid = roc_auc_score(y_valid, y_valid_preds)

fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_preds)
auc_test = roc_auc_score(y_test, y_test_preds)

plt.plot(fpr_train, tpr_train, 'r-', label='Train AUC:%.3f' % auc_train)
plt.plot(fpr_valid, tpr_valid, 'b-', label='Valid AUC:%.3f' % auc_valid)
plt.plot(fpr_test, tpr_test, 'g-', label='Test AUC:%.3f' % auc_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
