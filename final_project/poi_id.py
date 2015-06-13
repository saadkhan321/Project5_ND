#!/usr/bin/python

import sys
import pickle
import os
import re
import string
import numpy as np
import pprint as pp
sys.path.append("../tools/")

### for measuring time taken

import time

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

print
print 'Features originally chosen:'

features_list = ['poi', 'bonus', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other','shared_receipt_with_poi'] # You will need to use more features

print
pp.pprint(features_list)

time.sleep(5)

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers

print
print 'Outliers removed:'
print
print '[GRAMM WENDY L]'
print
data_dict.pop("GRAMM WENDY L", None)
print '[LOCKHART EUGENE E]'
print
data_dict.pop("LOCKHART EUGENE E", None)
print '[WROBEL BRUCE]'
print
data_dict.pop("WROBEL BRUCE", None)
print '[THE TRAVEL AGENCY IN THE PARK]'
print
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", None)
print '[TOTAL]'
data_dict.pop("TOTAL", None)

time.sleep(5)

print
print 'Inconsistent records updated:'
print
print '[BELFER ROBERT]'
print
print '[BHATNAGAR SANJAY]'


data_dict['BELFER ROBERT'] = {'bonus': 'NaN',
                              'deferral_payments': 'NaN',
                              'deferred_income': -102500,
                              'director_fees': 102500,
                              'email_address': 'NaN',
                              'exercised_stock_options': 'NaN',
                              'expenses': 3285,
                              'from_messages': 'NaN',
                              'from_poi_to_this_person': 'NaN',
                              'from_this_person_to_poi': 'NaN',
                              'loan_advances': 'NaN',
                              'long_term_incentive': 'NaN',
                              'other': 'NaN',
                              'poi': False,
                              'restricted_stock': -44093,
                              'restricted_stock_deferred': 44093,
                              'salary': 'NaN',
                              'shared_receipt_with_poi': 'NaN',
                              'to_messages': 'NaN',
                              'total_payments': 3285,
                              'total_stock_value': 'NaN'}

data_dict['BHATNAGAR SANJAY'] = {'bonus': 'NaN',
                                 'deferral_payments': 'NaN',
                                 'deferred_income': 'NaN',
                                 'director_fees': 'NaN',
                                 'email_address': 'sanjay.bhatnagar@enron.com',
                                 'exercised_stock_options': 15456290,
                                 'expenses': 137864,
                                 'from_messages': 29,
                                 'from_poi_to_this_person': 0,
                                 'from_this_person_to_poi': 1,
                                 'loan_advances': 'NaN',
                                 'long_term_incentive': 'NaN',
                                 'other': 'NaN',
                                 'poi': False,
                                 'restricted_stock': 2604490,
                                 'restricted_stock_deferred': -2604490,
                                 'salary': 'NaN',
                                 'shared_receipt_with_poi': 463,
                                 'to_messages': 523,
                                 'total_payments': 137864,
                                 'total_stock_value': 15456290} 

### Task 3: Create new feature(s)

print
print 'Creating 1 new feature that is to be used as part of the final analysis'


### Store to my_dataset for easy export below.

my_dataset = data_dict
my_feature_list = features_list

np_my_feature_list = np.array(my_feature_list)

### computeFraction function used from lesson 11 exercise

def computeFraction(poi_messages, all_messages):

    if (poi_messages == 'NaN') or (all_messages == 'NaN'):
        fraction = 0.
    else:
        fraction = float(poi_messages)/float(all_messages)

    return fraction


for name in data_dict:
    
    data_point = data_dict[name]
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
   
    
    ### Adding values of new features to the modified dataset, 'my_dataset'
    
    my_dataset[name]['fraction_to_poi'] = fraction_to_poi

time.sleep(5)

print   
print '1 new feature, "fraction_to_poi", added to "my_dataset"'


### Adding new feature to the modified features list, 'my_feature_list'

fraction_features = ['fraction_to_poi']

my_feature_list = features_list + fraction_features

np_my_feature_list = np.array(my_feature_list)

np_my_feature_list = np_my_feature_list[1::]

print
print 'Number of features now:', len(my_feature_list), "and the final features list:"
print
pp.pprint(my_feature_list)


### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers

### Feature Selector utility

from sklearn.feature_selection import SelectKBest, f_classif

### Grid Fit/Transform utility

from sklearn.grid_search import GridSearchCV

### Algorithm

from sklearn.tree import DecisionTreeClassifier

### Cross Validation utility

from sklearn.cross_validation import StratifiedShuffleSplit


### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### from sklearn.naive_bayes import GaussianNB
### clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

folds = 1000
cv = StratifiedShuffleSplit(labels, folds)


### feature selection, because text is super high dimensional and 
### can be really computationally chewy as a result

k=4
selector = SelectKBest(f_classif, k)


### counter for number of iterations

count = 0

### dictionaries to document feature importances

feature_importances_frequency = {}
feature_importances_score = {}

### to measure duration of model training

t0 = time.time()

print
print 'Feature Selection is performed using SelectKBest and k=', k
time.sleep(5)
print 'Cross Validation is performed using StratifiedShuffleSplit with', folds, 'folds'
time.sleep(5)
print 'Algorithm used is Decision Tree Classifier'
time.sleep(5)
print 'Feature Importances are documented, frequency + average scores are printed out for further analysis'
time.sleep(5)
print
print 'Training the Model'
print

for train_indices, test_indices in cv:
    features_train = []
    features_test = []
    word_features_train = []
    word_features_test = []
    labels_train = []
    labels_test = []

    ### Partitioning of the data into training and testing datasets
    
    features_train = [features[ii] for ii in train_indices]
    labels_train = [labels[ii] for ii in train_indices]
       
    features_test = [features[jj] for jj in test_indices]
    labels_test = [labels[jj] for jj in test_indices]
        
    ### Fitting and transformation of the training dataset (vectorizer, scaling, feature selection)
    
    features_train_fit_transformed = selector.fit_transform(features_train, labels_train)
    
    ### Transformation of the test dataset (vectorizer, scaling, feature selection)
    
    features_test_transformed = selector.transform(features_test)

    ### Training the classifer
    
    ### Decision Tree Classifier code chunk
    
    dt = DecisionTreeClassifier()
    
    parameters = {'random_state': [1, 50], 'max_features':('auto', 'sqrt', 'log2'), 'criterion':('gini', 'entropy')}
    clf = GridSearchCV(dt, parameters)
    clf.fit(features_train_fit_transformed, labels_train)
    pred = clf.predict(features_test_transformed)
    
     ### Feature Importance documentation code chunk
    
    dt.fit(features_train_fit_transformed, labels_train)
    importances = dt.feature_importances_
    important_names_frequency = np_my_feature_list[importances > 0.05]
    important_names_score = np_my_feature_list
    
    count += 1
    if count%10 == 0:
        print '*',
    if count%100 == 0:
        print count/10,'%'
    

    for v in range(len(important_names_frequency)):
         
        if important_names_frequency[v] in feature_importances_frequency:
            
            feature_importances_frequency[important_names_frequency[v]] += 1
        else:
            feature_importances_frequency[important_names_frequency[v]] = 1
            
    for v in range(4):
         
        if important_names_score[v] in feature_importances_score:

            feature_importances_score[important_names_score[v]] += importances[v]
        else:
            feature_importances_score[important_names_score[v]] = 0.0
                
                
for key, value in feature_importances_score.items():
    feature_importances_score[key] = round((value / 1000), 3)       

print 
print 'Model Trained'
print    
print "Total time taken to train:", round(time.time()-t0, 3), "s"    
print
print 'Average Score of Feature Importances'
print
pp.pprint(feature_importances_score) 
print
print 'Frequency of Feature Importances'
print    
pp.pprint(feature_importances_frequency) 
print

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print
print 'Passing the result to tester.py to evaluate Performance Metrics'

test_classifier(clf, my_dataset, my_feature_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

print
print 'Pickle files dumped for anyone to run/check the results'
dump_classifier_and_data(clf, my_dataset, my_feature_list)