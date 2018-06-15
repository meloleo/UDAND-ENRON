#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
import sklearn.pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn import feature_selection
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Set to pandas dataframe
df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))

# set the index of df to be employee names
df.set_index(employees, inplace=True)

# replace NaN with 0
df.replace('NaN', 0, inplace=True)

### Task 2: Remove outliers
# find any employee names with more than 3 or less than 2 words
ls = []

for name in df.index:
    splitname = name.split(' ')
    if len(splitname) > 3:
        ls.append(name)
    elif len(splitname) < 2:
        ls.append(name)

print ls

#remove nonpersons
df.drop("THE TRAVEL AGENCY IN THE PARK", inplace = True)
df.drop('TOTAL', inplace = True)

# find employee with high numbers of null values
zeroval = (df == 0).astype(int).sum(axis=1).sort_values(ascending = False)
zeroval.head()

#drop employee with 21 null values
df.drop('LOCKHART EUGENE E', inplace = True)

### Task 3: Create new feature(s)

email_feat_list = ['to_messages', 'from_poi_to_this_person', 'from_messages', "email_address",
                   'from_this_person_to_poi', 'shared_receipt_with_poi']

fin_feat_list =  ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'deferred_income',
                  'expenses', 'other', 'long_term_incentive', 'director_fees', 'restricted_stock_deferred', 
                  'total_stock_value', 'exercised_stock_options', 'restricted_stock']



###FEATURE CREATION

#NEW FEATURE (# of poi emails divided by total # emails)
df['poi_all_emails_frac'] = (df['from_this_person_to_poi'] + df['from_poi_to_this_person']) / (df['to_messages'] + df['from_messages'])
#NEW FEATURE (# of emails from employee to poi divided by total # emails from employee)
df['poi_from_emails_frac'] = df['from_this_person_to_poi'] / df['from_messages']
#NEW FEATURE (# of emails from poi to employee divided by total # emails to employee)
df['poi_to_emails_frac'] = df['from_poi_to_this_person'] / df['to_messages']
#NEW FEATURE (fraction of amount of info in columns), subtract 1 to account for poi label
df["zero_values"] = zeroval / (len(df.columns) - 1)
#NEW FEATURE (boolean of whether email_address column is empty or not)
df['email_avail'] = np.where(df['email_address'] == 0, False, True)

# replace any infinity values with nan, fill nan values
df.replace([np.inf, -np.inf], np.nan, inplace = True)
df.fillna(0, inplace = True)

#create correlation matrix to determine superfluous features
def corrheatmap(dataframe):
    corr = dataframe.corr(method = "pearson") ** 2
    f, ax = plt.subplots(figsize=(11, 9))
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, mask=mask, vmax=.3, center=.25, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    c = corr
    os = (c.where(np.triu(np.ones(c.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
    print os.head()

corrheatmap(df)

#create dict of R squared values of features w/ POI label
poi_r2 = []
for feature in df.columns:
    try:
        rval = stats.pearsonr(df["poi"], df[feature])[0] ** 2 
        poi_r2.append([feature ,rval])
    except:
        print "Failed: ", feature
        pass

#sort by correlation, desecending
poi_r2.sort(key=lambda tup: -tup[1])

features_list = []
for k, v in poi_r2:
    if v > 0.02:
#        print k, ":", v
        features_list.append(k)

#analyze features with strong correlations with each other in heatmap, remove features with weaker POI correlations from feature list
remove = ["loan_advances", "total_stock_value", "to_messages", "other", "restricted_stock"]
        
for i in remove:
    try:
        features_list.remove(i)
    except:
        print "failed:", i
        pass
        
# after you create features, the column names will be your new features
# create a list of column names:
final_df = df[features_list]
# create a dictionary from the dataframe
df_dict = final_df.to_dict('index')

### Store to my_dataset for easy export below.
my_dataset = df_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

#RandomForest used to calculate feature_importances_
clf_RF = RandomForestClassifier(random_state = 42)
clf_RF.fit(features_train, labels_train)

#Features sorted by score
importances = sorted(zip(map(lambda x: round(x, 4), clf_RF.feature_importances_), 
                                   features_list), reverse=True)
#new feature list
RF_features_list = ['poi']

for importance, name in importances:
    if name == 'poi':
        pass
    elif importance > 0.02:
        RF_features_list.append(name)

print "RANDOM FOREST FEATURES_IMPORTANCES:"
print RF_features_list

### Extract features and labels from dataset using new features list
data = featureFormat(my_dataset, RF_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#create algorithm fit function
def untuned(name, clsfr):
    clsfr.fit(features_train, labels_train)
    predict = clsfr.predict(features_test)
    report = classification_report(labels_test, predict)

    print "UNTUNED CLASSIFICATION REPORT:", name
    print report


clf_GNB = GaussianNB()
clf_DT = tree.DecisionTreeClassifier()
clf_RF = RandomForestClassifier()
clf_ABC = AdaBoostClassifier()
clf_SVC = svm.SVC()

classifiers = [["Naive Bayes", clf_GNB],
               ["Decision Tree", clf_DT],
               ["Random Forest", clf_RF],
               ["AdaBoost", clf_ABC],
               ["Support Vector Machines", clf_SVC]]
for name, clsfr in classifiers:
    untuned(name, clsfr)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

skb = SelectKBest()
cv = StratifiedShuffleSplit(labels_train, 100, random_state = 42)
mms = MinMaxScaler()

#Tuned Naive Bayes
steps_NB = [('minmax', mms),
         ('SelectKBest', skb),
         ('naive_bayes', clf_GNB)]

parameters_NB = dict(SelectKBest__k = range(1, 10)
)

pipeline = sklearn.pipeline.Pipeline(steps_NB)
grid = GridSearchCV(pipeline, param_grid = parameters_NB, cv = cv, scoring = 'f1')
grid.fit(features_train, labels_train)
predict = grid.predict(features_test)
report = classification_report(labels_test, predict)
best_params = grid.best_params_
#print report
print "PARAMETERS USED:"
print best_params
print grid.best_score_
clf_GNB = grid.best_estimator_
print "TUNED CLASSIFICATION REPORT:"
test_classifier(clf_GNB, my_dataset, RF_features_list, folds = 1000)

#overwrite features_list
features_list = RF_features_list

#tuned Random Forest **without** SKB
steps_RF = [('minmax', mms),
         #('SelectKBest', skb),
         ('random_forest', clf_RF)]

parameters_RF = dict(#SelectKBest__k = [6],
                    random_forest__criterion = ['gini'],
                    random_forest__n_estimators = [9],
                    random_forest__min_samples_split = [2],
                    random_forest__random_state = [42])

pipeline = sklearn.pipeline.Pipeline(steps_RF)
grid = GridSearchCV(pipeline, param_grid = parameters_RF, cv = cv, scoring = 'f1')
grid.fit(features_train, labels_train)
predict = grid.predict(features_test)
report = classification_report(labels_test, predict)
best_params = grid.best_params_
print "TUNED CLASSIFICATION REPORT: RANDOM FOREST w/o SKBEST"
test_classifier(grid.best_estimator_, my_dataset, features_list, folds = 1000)
#print report
print "PARAMETERS USED:"
print best_params
print grid.best_score_



#tuned Random Forest with SKB
steps_RF = [('minmax', mms),
         ('SelectKBest', skb),
         ('random_forest', clf_RF)]

'''
#GridSearchCV parameter testing
parameters_RF = dict(SelectKBest__k = range(1, 10),
                    random_forest__criterion = ['gini', 'entropy'],
                    random_forest__n_estimators = range(1, 10),
                    random_forest__min_samples_split = range(2, 10),
                    random_forest__random_state = [42]
                 )
'''
#Final choice
parameters_RF = dict(SelectKBest__k = [6],
                    random_forest__criterion = ['gini'],
                    random_forest__n_estimators = [9],
                    random_forest__min_samples_split = [2],
                    random_forest__random_state = [42])

pipeline = sklearn.pipeline.Pipeline(steps_RF)
grid = GridSearchCV(pipeline, param_grid = parameters_RF, cv = cv, scoring = 'f1')
grid.fit(features_train, labels_train)
predict = grid.predict(features_test)
report = classification_report(labels_test, predict)
best_params = grid.best_params_
print "TUNED CLASSIFICATION REPORT: RANDOM FOREST w/ SKBEST"
test_classifier(grid.best_estimator_, my_dataset, features_list, folds = 1000)
#print report
print "PARAMETERS USED:"
print best_params
print grid.best_score_
clf = grid.best_estimator_


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)