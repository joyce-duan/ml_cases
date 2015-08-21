'''
ipython credit.py    run on all the data
'''
import random
import os.path
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from operator import itemgetter

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import cross_val_score 
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from configobj import ConfigObj
config = ConfigObj('config')

ml_home = config.get(
    'ml_home', '/')
import sys
sys.path.append('../../util')
sys.path.append(ml_home )
sys.path.append(ml_home + 'classification')
sys.path.append(ml_home + 'model_evaluation')

import myutil
from classifier_helper import ClassifierTestor, ClassifierSelector, ClassifierOptimizer
from ml_util import random_sample
sys.path.append('.')
from eda_uti import *

def baseline_model_rf(df, xnames, yname):
	'''
	estimator = RandomForestClassifier(n_estimators=100\
	, class_weight = {0:1, 1:4}
	, max_depth = 15, min_samples_leaf =5
	 )
	'''
	estimator = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_split=20, min_samples_leaf=5, \
	 max_features='auto')
	m = cross_val_score(estimator,\
            df[xnames], \
               df[yname], scoring = 'roc_auc', cv = 5)
	print m.mean(), m	

def baseline_model_gbc(df, xnames, yname):
	# 0.84
	clf = GradientBoostingClassifier()

	m = cross_val_score(clf, df[xnames], \
	           df[yname], scoring = 'roc_auc', cv = 5)
	print m.mean(), m

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def baseline_sgdc(df, xnames, yname):
	t0 = time.time() # time it
	'''
	clf =  Pipeline([("scalar",  StandardScaler())
			, ("clf", linear_model.SGDClassifier(loss='log'))])
	'''
	clf = linear_model.SGDClassifier(loss='log')
	m = cross_val_score(clf, df[xnames], \
	           df[yname], scoring = 'roc_auc', cv = 5)
	print m, np.mean(m)
	t1 = time.time() # time it
	print "finish in  %4.4fmin for %s " %((t1-t0)/60,'base run')

def optimize_sgdc():
	df, xnames, yname = load_df_from_pkl()
	baseline_sgdc(df, xnames, yname)
	t0 = time.time()
	params_grid = [{"clf__n_iter": [5, 8, 10, 15],
              "clf__alpha": [1e-7, 1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1],
              "clf__penalty": ["none", "l1", "l2"],
              'clf__loss': ['log']}]     
	optmizer = ClassifierOptimizer('sgdc') # linear_model.SGDClassifier(loss='log') 
	optmizer.set_params(params_grid)
	optmizer.add_pipleline([("scalar",  StandardScaler())])

	optmizer.optimize(df[xnames], df[yname], cv = 5, scoring = 'roc_auc')
	print optmizer.get_score_gridsearchcv()
	t1 = time.time() # time it
	print "finish in  %4.4fmin for %s " %((t1-t0)/60,'optimizer')

def test_classifiers_by_auc(df, xnames, yname):
	features = xnames
	X = df[features].values
	y = df[yname]

	train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.20)

	models = { 'LogisticRegression':linear_model.LogisticRegression()  
		, 'SGDC_lr': linear_model.SGDClassifier(loss='log') 
	,'RFR':RandomForestRegressor(random_state=0, n_estimators=100\
	, max_depth = 15, min_samples_leaf =5 )
	, 'RF':RandomForestClassifier(random_state=0, n_estimators=100\
	, max_depth = 15, min_samples_leaf =5 )
	, 'AdaBoostClassifier':AdaBoostClassifier(learning_rate=0.1, n_estimators=500)
	, 'GBC':GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
	}
	classifier_testor = ClassifierTestor(estimators = models, scoring = 'roc_auc')
	classifier_testor.fit_predict_proba(train_X, test_X, train_y, test_y)
	print classifier_testor.score()

def optimize_rf(df, xnames, yname):
	optmizer = ClassifierOptimizer('rf')
	params = [{"clf__n_estimators": [100, 300]
		, 'clf__max_depth':[5, 10, 20, 30]
		, 'clf__min_samples_leaf':[5]
		, 'clf__class_weight':[None, {0:1, 1:1}, {0:1,1:5}, {0:1,1:10}, 'auto'] }]
	optmizer.set_params(params)
	optmizer.optimize(df[xnames], df[yname], cv = 5, scoring = 'roc_auc')
	print optmizer.get_score_gridsearchcv()

def randomized_search(df, xnames, yname):
	# build a classifier
	clf = RandomForestClassifier(n_estimators=100)
	# specify parameters and distributions to sample from
	param_dist = {  #"max_depth": [3, 5, 7, 10, 15],
				   "max_depth": sp.stats.randint(3,15),
	              "max_features": sp.stats.randint(1, 11),
	              "min_samples_split": sp.stats.randint(1, 11),
	              "min_samples_leaf": sp.stats.randint(2, 11),
	              'class_weight':[None, {0:1, 1:1}, {0:1,1:5}, {0:1,1:10}, 'auto'],
	              "criterion": ["gini", "entropy"]}
	# run randomized search
	n_iter_search = 20
	grid_search = RandomizedSearchCV(clf, param_distributions=param_dist,
	                          verbose = 1, n_iter=n_iter_search, cv = 5, scoring= 'roc_auc',  n_jobs=1 )

	start = time.time()
	grid_search.fit(df[xnames], df[yname])
	print("RandomizedSearchCV took %.2f seconds for %d candidates"
	      " parameter settings." % ((time.time() - start), n_iter_search))
	report(grid_search.grid_scores_, n_top = 5)
	return grid_search

def randomized_search_gbc(df, xnames, yname):
	# build a classifier
	n_features = len(xnames)
 	clf = GradientBoostingClassifier(n_estimators=200)
	# specify parameters and distributions to sample from
	param_dist = { 'learning_rate': sp.stats.uniform(0.01, 0.1),
				   "max_depth": sp.stats.randint(3,7),
	              #"max_features": sp.stats.uniform(0.1, 1.0),
	              'max_features': sp.stats.randint(1, 11),
	              "min_samples_split": sp.stats.randint(3, 20),
	              "min_samples_leaf": sp.stats.randint(3, 18),
		}
	# run randomized search
	n_iter_search = 20
	grid_search = RandomizedSearchCV(clf, param_distributions=param_dist,
	                          verbose = 1, n_iter=n_iter_search, cv = 5, scoring= 'roc_auc',  n_jobs=1 )

	start = time.time()
	grid_search.fit(df[xnames], df[yname])
	print("RandomizedSearchCV took %.2f seconds for %d candidates"
	      " parameter settings." % ((time.time() - start), n_iter_search))
	report(grid_search.grid_scores_, n_top = 5)
	return grid_search

def build_gbc():
	clf = GradientBoostingClassifier(learning_rate=0.05, n_estimators=200 \
		, max_features = 6, min_samples_split = 15, max_depth = 5, min_samples_leaf = 16)
	return clf

def build_rfc():
	d = {'min_samples_leaf': 7, 'min_samples_split': 3, 'criterion': 'entropy', 'max_features': 9, 'max_depth': 11, 'class_weight': {0: 1, 1: 1} }
	clf = RandomForestClassifier(max_depth = 15, n_estimators = 250, min_samples_leaf =5 )
	clf.set_params(**d)
	return clf

def build_rfregressor():
	clf = RandomForestRegressor(random_state=0, n_estimators=100\
		, max_depth = 15, min_samples_leaf =5 )
	return clf

def build_sgdc():
	params = {u'n_iter': 10, u'penalty': u'l2', u'loss': u'log', u'alpha': 0.01}
	clf = linear_model.SGDClassifier(loss='log') 
	clf.set_params(**params)
	return clf

def init_all_models():
	clf1 = build_rfc()
	clf2 = build_rfregressor()
	clf3 = build_sgdc() #linear_model.SGDClassifier(loss='log') 
	clf4 = AdaBoostClassifier(learning_rate=0.1, n_estimators=500)
	clf5 = build_gbc() #GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
	#models = [clf1, clf2, clf3, clf4, clf5]
	models = [clf1, clf3,  clf4, clf5]
	return models

def init_calibrated_submodels(n_folds = 5):
	models = init_all_models()
	models_calibrated = [CalibratedClassifierCV(clf, cv=n_folds, method='isotonic') for clf in models] 
	model_names = [m.__class__.__name__ for m in models]
	return models_calibrated, model_names

def build_ensemble_models(df, xnames, yname, models, model_names, n_folds = 5, cv_pkl=True, warmstart = False, w = None, flag_print=True):
	'''
	build submodels, save as pickle files
	ensemble using weight w; only works for two classes 0/1
		INPUT
		- cv_pkl=True   read in cv from pkl file
		- warmstart = True   read in predicitons of submodels from pkl file
		- w: weight for each of the n_sub_models  
		- flag_print  write cv and test_predictions of each fold to pkl files
		OUTPUT
		- predictions_test  [i][j]: numpy array of [n, 1] : pred_proba for fold i and submodel j of class = 1
		- cv
	'''
	n_models = len(models)
	run_name = 'eval_ensemble_models'
	if w is None:
		w = [1.0/n_models] * n_models
	w = np.array(w)

	cv_fname = 'data/kfold_cv.pkl'
	if cv_pkl:
		cv = pickle.load(open(cv_fname))
	else:
		cv = KFold(df.shape[0], n_folds= n_folds) 
		if flag_print:
			pickle.dump(cv,open(cv_fname, 'wb'))

	print model_names
	predictions_test = []  # 2d list of np array: i: cv; j: model j ;  np_array: n_sample x n_classes
	scores, scores_all, scores_all_train =  [], [], []  #scores of ensemble; scores of all test and train
	t00 = time.time() # time it		
	t0 = t00
	i = 0
	for i_cv, (train, test) in enumerate(cv):
		predictions_test.append([])
		predictions_train_thiscv = []
		X_train, y_train = df.iloc[train][xnames], df.iloc[train][yname]
		X_test, y_test = df.iloc[test][xnames], df.iloc[test][yname]
		X_train, scaler = scale_X(X_train, xnames)
		X_test = scale_X_transform(X_test , xnames, scaler)

		if len(train) + len(test) != df.shape[0]:
			print 'ERROR: K fold and dataset mis-match'
			return 			
		scores_all.append([])
		scores_all_train.append([])		
		for j, clf in enumerate(models):
			model_name = model_names[j]
			probs, probs_train = get_predicted_prob(clf, i_cv, X_train, y_train, X_test, y_test, warmstart = warmstart, flag_print=flag_print, model_name = model_name)
			predictions_train_thiscv.append(probs_train)
			predictions_test[i_cv].append(probs)
			score_this_clf = roc_auc_score(y_test, probs[:, 1])
			print("testset score: %f" % score_this_clf)
			scores_all[i_cv].append(score_this_clf)
			scores_all_train[i_cv].append(roc_auc_score(y_train, probs_train[:,1]))
			t1 = time.time() # time it
			time_taken = (t1-t0)/60	
			print 'run %d %s time taken %.2fmin' % (i_cv, model_name, time_taken)
			t0 = t1 

		y_test_preds = np.array([p[:,1] for p in predictions_test[i_cv]])	
		y_test_ensemble = (y_test_preds.T).dot(w.T)
		y_test_ensemble = (y_test_ensemble - y_test_ensemble.min())/(y_test_ensemble.max() - \
		y_test_ensemble.min())

		y_train_preds = np.array([p[:,1] for p in predictions_train_thiscv])	
		y_train_ensemble = (y_train_preds.T).dot(w.T)
		y_train_ensemble = (y_train_ensemble - y_train_ensemble.min())/(y_train_ensemble.max() - \
		y_train_ensemble.min())     

		score_this_run = roc_auc_score(y_test, y_test_ensemble)
		score_this_run_train = roc_auc_score(y_train, y_train_ensemble)
		scores.append(score_this_run)
		scores_all[i_cv].append(score_this_run)
		scores_all_train[i_cv].append(score_this_run_train)		
		print "run %i combined score: %f" % (i_cv, scores[-1])
		print "       test: ", scores_all[i_cv]
		print "       train: ", scores_all_train[i_cv]
		print 
	print 
		
	scores_all = np.array(scores_all)
	scores_all_train = np.array(scores_all_train)
	t1 = time.time() # time it
	time_taken = (t1-t00)/60	
	print run_name,  ' finished in ', time_taken, ' minutes'
	print 'combined score: mean', (np.mean(scores), 'std:', np.std(scores))		
	print 'test set:', np.mean(scores_all, axis = 0)
	print 'train set:', np.mean(scores_all_train, axis = 0)
	return predictions_test, cv

def eval_ensemble_average_cv(predictions_test, df, cv, yname, weight, predictions_train = None):
	'''
	INPUT:
	- weight: n_fold * n_sub_models 
	- predictions_test: [i][j]: numpy array of [n, 1] : pred_proba for fold i and submodel j of class = 1
	OUTPUT:
	- roc_test: numpy array: n_fold * (n_submodels + 1)
	- roc_train
	'''
	y_tests_all = [] 
	roc_train = []
	for train, test in cv:
		y_tests_all.append(df.iloc[test][yname])
	y_tests_all = np.array(y_tests_all)
	roc_test = eval_ensemble_kfold(predictions_test, y_test_all, weight)

	if predictions_train is not None:
		y_train_all = [] 
		for train, test in cv:
			y_train_all.append(df.iloc[train][yname])
		y_train_all = np.array(y_train_all)
		roc_train = eval_ensemble_kfold(predictions_test, y_test_all, weight)	
	return roc_test, roc_train	

def eval_ensemble_kfold(y_preds_nfold, y_actual_nfold, weight):
	'''
	INPUT:
		- y_preds_nfold: [i][j]: numpy array of [n, 1] : pred_proba for fold i and submodel j of class = 1
		- y_actual_nfold: [i]: acutal of fold i
	'''
	roc = []
	for i_fold in xrange(y_preds_nfold.shape[0]):
		roc.append([])
		y_actual = y_actual_nfold[i_fold]
		y_preds = np.array([p[:,1] for p in y_preds_nfold[i_fold]])	
		w = np.array(weights)
		y_pred_ensemble = (y_preds.T).dot(w.T)
		y_pred_ensemble = (y_pred_ensemble - y_pred_ensemble.min())/(y_pred_ensemble.max() - \
		y_pred_ensemble.min())   		
		roc[i_fold] = [roc_auc_score(y_actual,y_pred ) for y_pred in y_preds]
		roc[i_fold].append(roc_auc_score(y_actual, y_pred_ensemble))
		print roc[i_fold][-1],'         ', roc[i_fold]
	roc = np.array(roc)
	print np.mean(roc, axis = 0)
	return roc

def find_weight_for_ensemble_cv(predictions_test, df, cv, yname):
	'''
	INPUT: predictions_test: [i][j]: numpy array of [n, 1] : pred_proba for fold i and submodel j of class = 1
	'''
	weights = []
	for i, (train, test) in enumerate(cv):
		y_test = df.iloc[test][yname]
		y_test_preds = predictions_test[i]
		weights.append(find_weight_for_ensemble(y_test_preds, y_test))
	weights  = np.array(weights)
	print weights
	avg_weight  = np.mean(weights, axis = 0)
	print 'average weights to use:'
	return avg_weight, weights

def find_weight_for_ensemble(predictions, y_test):
    '''
    fing optimal weight to average predcitions
    INPUT:
    - predictions: n 
    - y_test
    OUTPUT:
    - list of k
    '''
	# source: https://www.kaggle.com/hsperr/otto-group-product-classification-challenge/finding-ensamble-weights
    #the algorithms need a starting value, right now we chose 0.5 for all weights
    #its better to choose many random starting points and run minimize a few times
    starting_values = [0.5]*len(predictions)

    def log_loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
                final_prediction += weight*prediction
        return log_loss(y_test, final_prediction)

    #adding constraints  and a different solver as suggested by user 16universe
    #https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    #our weights are bound between 0 and 1
    bounds = [(0,1)]*len(predictions)

    res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))
    return list(res['x'])

def run_ensemble_models():
	models = init_all_models()
	eclf = EnsembleClassifier(clfs=models,
		voting='hard', verbose=1
		, weights=[1,1,1,1])
	eclf = eclf.fit(train_X, train_y)
	test_y_pred_e, test_y_pred_all, clf_names = eclf.predict_all_clfs(test_X)
	print 'emsemble: ', accuracy_score(test_y, test_y_pred_e ) 
	for i, test_y_pred in enumerate(test_y_pred_all) :
		print clf_names[i], accuracy_score(test_y, test_y_pred )

def run_test_classifier(df, xnames, yname): #, flag_test, n_samples):
	imp = Imputer(strategy='median', axis=1)
	df[xnames] = imp.fit_transform(df[xnames])
	test_classifiers_by_auc(df, xnames, yname)

def apply_fitted_models(weight, clfs, model_names = None):
	#w = [1.0/5]*5
	w = weight
	df = read_all_data('data/credit-training.csv')	 
	df, xnames, yname = pre_process(df)
	df, scaler = scale_X(df, xnames)

	df_test = read_all_data('data/credit-test.csv')
	df_test, xname, yname = pre_process(df_test)
	df_test = scale_X_transform(df_test, xnames, scaler)
	print df_test.shape

	models = clfs
	if model_names is None:
		model_names = [clf.__class__.__name__ for clf in models]

	preds = []
	for i, clf in enumerate(models):
		model_name = model_names[i]
		#if 'Classifier' in clf.__class__.__name__:
		if 'Classifier' in model_name:
			probs = clf.predict_proba(df_test[xnames])
		else:
			probs = np.zeros((df_test[xnames].shape[0],2))			
			probs[:,1] = clf.predict(df_test[xnames]).T
		preds.append(probs)
		#write_test(probs[:, 1], "test_prediction_model_%d.csv" % i,
		#	df_test)
	y_test_preds = (np.array([p[:,1] for p in preds]))	
	w = np.array(w)
	y_test_ensemble = (y_test_preds.T).dot(w.T)
	y_test_ensemble = (y_test_ensemble - y_test_ensemble.min())/(y_test_ensemble.max() - \
		y_test_ensemble.min())
	write_test(y_test_ensemble, "test_prediction_combined.csv", df_test)

def apply_models(weight, models = None, model_names = None):
	'''
	apply the models: train using all the labelled data and apply to testset for submission
	- INPUT:
		weight: list of n_submodels, i.e.  [1.0/5]*5
	'''
	print 'apply models and make submissions for %d sub-models' % (len(models))
	w = weight
	#df = read_all_data('data/credit-training_sample.csv')
	df = read_all_data('data/credit-training.csv')	 
	df, xnames, yname = pre_process(df)
	df, scaler = scale_X(df, xnames)

	df_test = read_all_data('data/credit-test.csv')
	df_test, xname, yname = pre_process(df_test)
	df_test = scale_X_transform(df_test, xnames, scaler)
	print df_test.shape

	if models is None:
		models = init_all_models()
	if model_names is None:
		model_names = [clf.__class__.__name__ for clf in models]

	t00 = time.time()
	preds = []
	for i, clf in enumerate(models):
		t0 = time.time()
		clf.fit(df[xnames], df[yname])
		if 'Classifier' in model_names[i]:
			probs = clf.predict_proba(df_test[xnames])
		else:
			probs = np.zeros((df_test[xnames].shape[0],2))			
			probs[:,1] = clf.predict(df_test[xnames]).T
		preds.append(probs)
		write_test(probs[:, 1], "test_prediction_model_%d.csv" % i,
			df_test)
		with open(model_names[i]+'_clf.pkl', 'w' ) as fh_out:
			pickle.dump(clf,fh_out)
		t1 = time.time()
		print "model %d %s taken %.2f " % (i, model_names[i], (t1-t0)/60)

	y_test_preds = np.array([p[:,1] for p in preds])	
	w = np.array(w)
	y_test_ensemble = (y_test_preds.T).dot(w.T)
	y_test_ensemble = (y_test_ensemble - y_test_ensemble.min())/(y_test_ensemble.max() - \
		y_test_ensemble.min())
	write_test(y_test_ensemble, "test_prediction_combined.csv", df_test)
	t1 = time.time()
	print "total time taken taken %.2f minutes " % ( (t1-t00)/60)

def main_run_it_all():
	cv = 5
	fname = 'data/credit-training.csv'

	flag_test, n_samples = 0, 0
	if len(sys.argv) > 1:
		flag_test = sys.argv[1]
		n_samples = int(sys.argv[1])
	df, xnames, yname = get_data(fname, flag_test, n_samples)
	t0 = time.time()	
	df = feature_enginerring(df)
	df = fill_na(df, xnames, v = -1)
	t1 = time.time() # time it
	time_taken = (t1-t0)/60	
	print 'feature engineering', ' finished in ', time_taken, ' minutes'

	print df.shape
	print df[yname].mean()
	xnames = df.columns.difference([yname, 'id'])
	print yname
	print xnames
	print df.describe().T
	print df.dtypes

	#baseline_model_gbc(df, xnames, yname)
	#baseline_model_rf(df, xnames, yname)
	#optimize_rf(df, xnames, yname)

	#grid_search = randomized_search(df, xnames, yname)
	grid_search = randomized_search_gbc(df, xnames, yname)


def calibrate_prob(y_true, y_score, bins=10, normalize=False):
	'''
	modified from http://jmetzen.github.io/2014-08-16/reliability-diagram.html

	returns two arrays which encode a mapping from predicted probability to empirical probability.
	For this, the predicted probabilities are partitioned into equally sized
	bins and the mean predicted probability and the mean empirical probabilties
	in the bins are computed. For perfectly calibrated predictions, both
	quantities whould be approximately equal (for sufficiently many test
	samples).

	Note: this implementation is restricted to binary classification.

	Parameters
	----------
	y_true : array, shape = [n_samples]
	    True binary labels (0 or 1).
	y_score : array, shape = [n_samples]
	    Target scores, can either be probability estimates of the positive
	    class or confidence values. If normalize is False, y_score must be in
	    the interval [0, 1]

	bins : int, optional, default=10
	    The number of bins into which the y_scores are partitioned.
	    Note: n_samples should be considerably larger than bins such that
	          there is sufficient data in each bin to get a reliable estimate
	          of the reliability
	normalize : bool, optional, default=False
	    Whether y_score needs to be normalized into the bin [0, 1]. If True,
	    the smallest value in y_score is mapped onto 0 and the largest one
	    onto 1.

	-------
	y_score_bin_mean : array, shape = [bins]
	    The mean predicted y_score in the respective bins.

	empirical_prob_pos : array, shape = [bins]
		The empirical probability (frequency) of the positive class (+1) in the
		respective bins.
	'''    
	if normalize:  # Normalize scores into bin [0, 1]
		y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

	bin_width = 1.0 / bins
	bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

	y_score_bin_mean = np.empty(bins)
	empirical_prob_pos = np.empty(bins)
	for i, threshold in enumerate(bin_centers):
		# determine all samples where y_score falls into the i-th bin
		bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
			y_score <= threshold + bin_width / 2)
		# Store mean y_score and mean empirical probability of positive class
		y_score_bin_mean[i] = y_score[bin_idx].mean()
		empirical_prob_pos[i] = y_true[bin_idx].mean()
	return y_score_bin_mean, empirical_prob_pos

def try_calibrate():
	n_folds = 5
	df, cv, predictions_test = read_pred_cv(n_folds = n_folds)
	yname = 'dlqin2yrs' 

	print df.shape
	print df[yname].mean()
	xnames = df.columns.difference([yname, 'id'])
	print yname
	print xnames
	print df.describe().T
	print df.dtypes

	scores_all = []
	for i, (train, test) in enumerate(cv):
		preds_calibrated = np.zeros((len(test),5))
		scores_all.append([])
		y_test = df.iloc[test][yname]
		y_test_preds = predictions_test[i]
		for j, y_test_pred in enumerate(y_test_preds):
			score_this_clf = roc_auc_score(y_test, y_test_pred[:, 1])
			print("testset score: %f" % score_this_clf)
			scores_all[i].append(score_this_clf)	
			y_pred_scaled = (y_test_pred[:, 1] - y_test_pred[:, 1].min(axis=0)) \
			  / (y_test_pred[:, 1].max(axis=0) - y_test_pred[:, 1].min(axis=0))
					
			score_bin, empirical_prob = calibrate_prob(y_test, y_pred_scaled\
				, bins=200, normalize=False)
			preds_calibrated[:, j] = (sp.interp(y_pred_scaled, score_bin, empirical_prob)).T
			print roc_auc_score(y_test, preds_calibrated[:, j] )
		pred_ensemble = np.mean(preds_calibrated[:, [0,1,3,4]], axis=1)
		score_this_clf = roc_auc_score(y_test, pred_ensemble)
		scores_all[i].append(score_this_clf)
		print("testset score: %f" % score_this_clf)		
	print np.array(scores_all).mean(axis = 0)

def eval_calibrated_submodles(flagtest = 0):
	'''
	evaluate calibrated submodels using 3 weight scheme: a) same weight; b) top 3; c) optimized weights
	'''
	df, xnames, yname = load_df_from_pkl()
	if flagtest == 1:
		df, xnames, yname = read_process_samplingdata(fname)

	models, model_names = init_calibrated_submodels()
	if flagtest == 1:
		predictions_test, cv = build_ensemble_models(df, xnames, yname, models, model_names, n_folds = 5, cv_pkl=False, warmstart = False, w = None, flag_print = False)
	else:
		predictions_test, cv = build_ensemble_models(df, xnames, yname, models, model_names, n_folds = 5, cv_pkl=True, warmstart = False, w = None)
	
	weights = [0.333,0.0,0.333,0.333]
	eval_ensemble_average_cv(predictions_test, df, cv, yname, weight)

	weight, weights = find_weight_for_ensemble_cv(predictions_test, df, cv, yname)
	print weights
	print weight
	weight = list(weight)
	eval_ensemble_average_cv(predictions_test, df, cv, yname, weight)

if __name__ == '__main__':
	pd.set_option('max_colwidth',500)
	pd.set_option('display.expand_frame_repr', False)
	sys.stdout.flush()
	fname = 'data/credit-training.csv'

	#df, xnames, yname = load_df_from_pkl()
	#df, xnames, yname = read_process_samplingdata(fname)

	#baseline_sgdc(df, xnames, yname)
	#optimize_sgdc()

	eval_calibrated_submodles(flagtest = 1)

	weight =  [0.38696696,  0.00785644,  0.21191936,  0.39325724]
	#apply_models(weight, models = None, model_names = None)

	#read_fitted_calibrated_model()
	#apply_fitted_models(weight, models, model_names)
