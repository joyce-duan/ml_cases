'''
ipython credit-v3.py  400  # test run on a sample of 400 data row 
ipython credit-v3.py    run on all the data


to-do 
1.  tree: coding missing income as i.e. -1000

'''

import random
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.cross_validation import cross_val_score 

from configobj import ConfigObj
config = ConfigObj('config')

ml_home = config.get(
    'ml_home', '/')

sys.path.append('../util')
#sys.path.append('/Users/joyceduan/documents/git2/machine_learning')
sys.path.append(ml_home )
sys.path.append(ml_home + 'classification')
sys.path.append(ml_home + 'model_evaluation')

import myutil
from classifier_helper import ClassifierTestor, ClassifierSelector, ClassifierOptimizer
from ml_util import random_sample

def read_data(datafname):
	df = pd.read_csv(datafname)
	df.columns = [myutil.camel_to_snake(c) for c in df.columns]

	#yname = df.columns[1]
	#xnames = df.columns[2:]
	icol_y = 1
	icols_x = range(2,  len(df.columns))
	return df, icol_y, icols_x


def baseline_model_rf(df):
	# 0.78
	estimator = Pipeline([("imputer", Imputer(
                                          strategy="mean",
                                          axis=0)),
                      ("forest", RandomForestClassifier())])
	m = cross_val_score(estimator,\
            df.iloc[:,2:], \
               df.iloc[:,1], scoring = 'roc_auc', cv = 5)
	print m
	print m.mean()	

def baseline_model_gbc(df):
	# 0.84
	features = ['revolving_utilization_of_unsecured_lines', 'debt_ratio',
	            'number_of_times90_days_late', 'number_real_estate_loans_or_lines']
	clf = GradientBoostingClassifier()
	m = cross_val_score(clf, df[features], \
	           df.iloc[:,1], scoring = 'roc_auc', cv = 5)
	print m, m.mean()

def test_classifiers_by_accuracy():
	features = ['revolving_utilization_of_unsecured_lines', 'debt_ratio',
	            'number_of_times90_days_late', 'number_real_estate_loans_or_lines']

	X = df[features].values
	y = df.iloc[:, icol_y]
	print X[1]
	if test:
		X, y = random_sample(X, y)
	print X[1]
	print len(X), len(X[0])

	train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.25)
	models = [ 'LogisticRegression'
		, 'SGDClassifier'
		#, 'SVC'
		, 'RF'
		, 'AdaBoostClassifier'
		, 'KNN'
		#,'SVC_linear'
		, 'GBC']
	classifier_testor = ClassifierTestor(models)
	classifier_testor.fit(train_X, test_X, train_y, test_y)
	print classifier_testor.score()

def get_X_y(df, flag_test, n_samples = 0):
	features = ['revolving_utilization_of_unsecured_lines', 'debt_ratio',
	            'number_of_times90_days_late', 'number_real_estate_loans_or_lines']

	X = df[features].values
	y = df.iloc[:, icol_y]
	print X[1]
	if flag_test:
		if n_samples < 100: n_samples = 200 
		X, y = random_sample(X, y, k = n_samples)
	print X[1]
	print len(X), len(X[0])
	return X, y

def test_classifiers_by_auc(df, flag_test):
	features = ['revolving_utilization_of_unsecured_lines', 'debt_ratio',
	            'number_of_times90_days_late', 'number_real_estate_loans_or_lines']

	X = df[features].values
	y = df.iloc[:, icol_y]
	print X[1]
	if flag_test:
		X, y = random_sample(X, y)
	print X[1]
	print len(X), len(X[0])

	train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.20)

	#models = ['LogisticRegression']
	#'''
	models = [ #'LogisticRegression'
		#, 'SGDClassifier'
	#	, 'SGDC_lr'
		#, 'SVC'
		 'RF'
		, 'AdaBoostClassifier'
		, 'KNN'
		#,'SVC_linear'
		, 'GBC']
	#'''
	classifier_testor = ClassifierTestor(estimators = models, scoring = 'roc_auc')
	classifier_testor.fit_predict_proba(train_X, test_X, train_y, test_y)
	print classifier_testor.score()

def select_classifiers(df, flag_test, n_samples = 0):
	models = [ 
		 'rf'
		, 'adabc'
		, 'knn'
		, 'gbc']

	dict_params = {
    'rf': [
		{"clf__n_estimators": [10, 50, 250]
		, 'clf__max_depth':[20]
		, 'clf__class_weight': ['auto', None] }]
	, 'knn': [{"clf__n_neighbors": [ 10, 20]}]
	, 'gbc': [{'clf__learning_rate': [ 0.1, 0.01] # default 0.1
		, 'clf__n_estimators': [100, 500] #default 100
		}]
	,'adabc':[{'clf__n_estimators': [50, 200, 500] # default 50
		, 'clf__learning_rate': [0.1, 0.5] #, 1.0]   #default 1.0
		}]
    } 

	X, y = get_X_y(df, flag_test, n_samples)
	train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.20)

	select_clf = ClassifierSelector(models, dict_params=dict_params, scoring='roc_auc')
	select_clf.fit(train_X, train_y, cv=3, scoring= 'roc_auc')
	print select_clf.score()
	print select_clf.score_predict(test_X, test_y)


if __name__ == '__main__':
	pd.set_option('max_colwidth',500)
	sys.stdout.flush()

	cv = 5

	flag_test = 0
	n_samples = 0
	if len(sys.argv) > 1:
		flag_test = sys.argv[1]
		n_samples = int(sys.argv[1])
		cv = 5

	df, icol_y, icols_x = read_data('data/credit-training.csv')

	#cond = df['number_of_dependents'].isnull()
	df['number_of_dependents'] = df.number_of_dependents.fillna(0) 
	#test_classifiers_by_auc(df, flag_test)
	#select_classifiers(df, flag_test, n_samples)


	clf_gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100) 
	scores_allruns = []

	# run #3
	# all features, impute missing values
	#df = preprocess(df)

	df['debt_ratio_is_0'] = df['debt_ratio'].apply(lambda x: x == 0)
	df['revol_util_gt_1'] = df[ u'revolving_utilization_of_unsecured_lines'].apply(lambda x: x > 1)
	df['age_gt_85'] = df['age'].apply(lambda x: x > 85)
	df['num_pastdue'] = df[['number_of_time30-59_days_past_due_not_worse','number_of_times90_days_late'\
	 ,'number_of_time60-89_days_past_due_not_worse']].apply(lambda x: sum(x), axis = 1)
	df['no_credit'] = df['number_of_open_credit_lines_and_loans'].apply(lambda x: x == 0)
	df['zero_debt'] = df[ u'revolving_utilization_of_unsecured_lines'].apply(lambda x: x == 1)
	df['no_dependent'] = df['number_of_dependents'].apply(lambda x: x == 0)
	df['log_income'] = df['monthly_income'].apply(lambda x: np.log(x+1))
	df['log_debt_ratio'] = df['debt_ratio'].apply(lambda x: np.log(x+1))

	estimator = Pipeline([("imputer", Imputer(
                                          strategy="median",
                                          axis=0)),
                      ("clf", clf_gbc)])
             #strategy: mean (default), median, most_frequent
	t0 = time.time() # time it
	m = cross_val_score(estimator, df.iloc[:, icols_x], \
	           df.iloc[:,1], scoring = 'roc_auc', cv = cv)
	run = 'run 3 - more features'
	t1 = time.time() # time it
	time_taken = (t1-t0)/60	
	scores_allruns .append([run, np.mean(m), time_taken] + list(m) )
	print run, np.mean(m), time_taken, m

	# run #1
	# selcted featurs 
	#'''
	features = ['revolving_utilization_of_unsecured_lines', 'debt_ratio',\
  'number_of_times90_days_late', 'number_real_estate_loans_or_lines']	
	t0 = time.time() # time it		
	m = cross_val_score(clf_gbc, df[features], \
	           df.iloc[:,1], scoring = 'roc_auc', cv = cv)
	run = 'baseline-4features'
	t1 = time.time() # time it
	time_taken = (t1-t0)/60	
	scores_allruns .append([run, np.mean(m), time_taken] + list(m) )
	print run, np.mean(m), time_taken, m
	#'''
	#

	# run #2
	features = [u'revolving_utilization_of_unsecured_lines'\
	, u'age', u'number_of_time30-59_days_past_due_not_worse', u'debt_ratio'\
	, u'number_of_open_credit_lines_and_loans'\
	, u'number_of_times90_days_late', u'number_real_estate_loans_or_lines'\
	, u'number_of_time60-89_days_past_due_not_worse'
	, u'number_of_dependents'
	]	
	t0 = time.time() # time it	
	m = cross_val_score(clf_gbc, df[features], \
	           df.iloc[:,1], scoring = 'roc_auc', cv = cv)
	run = 'baseline- all features'
	t1 = time.time() # time it
	time_taken = (t1-t0)/60	
	scores_allruns .append([run, np.mean(m), time_taken] + list(m) )

	print run, np.mean(m), time_taken, m

	# all features, fill missing value
	# run #3
	t0 = time.time() # time it	

	estimator = Pipeline([("imputer", Imputer(
                                          strategy="median",
                                          axis=0)),
                      ("clf", clf_gbc)])
             #strategy: mean (default), median, most_frequent

	m = cross_val_score(estimator, df.iloc[:, icols_x], \
	           df.iloc[:,1], scoring = 'roc_auc', cv = cv)
	run = 'run 3 - fill na with meidan'
	t1 = time.time() # time it
	time_taken = (t1-t0)/60	
	scores_allruns .append([run, np.mean(m), time_taken] + list(m) )

	print run, np.mean(m), time_taken, m
	


	print run, np.mean(m), time_taken, m



	# all features; additonal features

	for s in scores_allruns:
		print s