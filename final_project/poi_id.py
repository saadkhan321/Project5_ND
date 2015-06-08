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

features_list = ['poi', 'salary', 'bonus', 'total_stock_value',
                 'exercised_stock_options', 'shared_receipt_with_poi'] # You will need to use more features

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
print 'Creating 2 new features to be used as part of the final analysis'


### Store to my_dataset for easy export below.

my_dataset = data_dict
my_feature_list = features_list


### computeFraction function used from lesson 11 exercise

def computeFraction(poi_messages, all_messages):

    if (poi_messages == 'NaN') or (all_messages == 'NaN'):
        fraction = 0.
    else:
        fraction = float(poi_messages)/float(all_messages)

    return fraction


for name in data_dict:
    
    # print name
    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
      
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
   
    
    ### Adding values of new features to the modified dataset, 'my_dataset'
    
    my_dataset[name]['fraction_from_poi'] = fraction_from_poi
    my_dataset[name]['fraction_to_poi'] = fraction_to_poi

time.sleep(5)

print   
print '2 new features, "fraction_from_poi" and "fraction_to_poi", added to "my_dataset"'


### Adding new feature to the modified features list, 'my_feature_list'

fraction_features = ['fraction_from_poi', 'fraction_to_poi']

my_feature_list = features_list + fraction_features

print
print 'Number of features now:', len(my_feature_list), "and the final features list:"
print
pp.pprint(my_feature_list)



### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
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



### text vectorization--go from strings to lists of numbers

#vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)


#scaler = MinMaxScaler()

#skf = StratifiedKFold(labels, n_folds=6)
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
#cv = KFold( len(labels), 6, shuffle=False, random_state=None)

### feature selection, because text is super high dimensional and 
### can be really computationally chewy as a result
k=4
selector = SelectKBest(f_classif, k)
#selector = SelectPercentile(f_classif, percentile=20)

true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
count = 0
t0 = time.time()

print
print 'Feature Selection is performed using SelectKBest and k=4'
time.sleep(5)
print 'Cross Validation is performed using StratifiedShuffleSplit with 1000 folds'
time.sleep(5)
print 'Algorithm used is Decision Tree Classifier'
time.sleep(5)
print 'Feature Importances are supported and for every 100th iteration will be printed out importances for 4 best features are printed out'
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
    
    ### dt.feature_importances_
    
    dt.fit(features_train_fit_transformed, labels_train)
    importances = dt.feature_importances_
    
    
  
    ### Feature Importance documentation code chunk
    
    count += 1
    if count%5 == 0:
        print '*',
    if count%100 == 0:
        print count/10,'%'
    	indices = np.argsort(importances)[::-1]
	print
    	print 'Feature ranks for', count, 'th iteration:', indices
    	print
        print "Feature scores for", count, "th iteration:"
    	print
        for f in range(k):
            print "feature no.{}: {}".format(f+1,round(importances[indices[f]],3))
            
	
 	print
    if count == 1000:
	print 'Plotting importances for the last iteration'
	import pylab as pl
    	pl.figure()
    	pl.title("Feature importances")
    	pl.bar(xrange(k), importances[indices], color="r", align="center")
    	pl.xticks(xrange(k), indices)
    	pl.xlim([-1, k])
    	pl.ylabel('Importance Score')
    	pl.xlabel('Indices of the features')
	
    	



print
print 'Model Trained'
print    
print "Total time taken to train:", round(time.time()-t0, 3), "s"

pl.show()

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