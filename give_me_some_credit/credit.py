'''
ipython credit.py  400  # test run on a sample of 400 data row 
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

def optimize_rf(df, xnames, yname):
	optmizer = ClassifierOptimizer('rf')
	params = [{"clf__n_estimators": [100, 300]
		, 'clf__max_depth':[5, 10, 20, 30]
		, 'clf__min_samples_leaf':[5]
		, 'clf__class_weight':[None, {0:1, 1:1}, {0:1,1:5}, {0:1,1:10}, 'auto'] }]
	optmizer.set_params(params)
	optmizer.optimize(df[xnames], df[yname], cv = 5, scoring = 'roc_auc')
	print optmizer.get_score_gridsearchcv()

def baseline_model_gbc(df, xnames, yname):
	# 0.84
	clf = GradientBoostingClassifier()

	m = cross_val_score(clf, df[xnames], \
	           df[yname], scoring = 'roc_auc', cv = 5)
	print m.mean(), m

#def test_classifiers_by_auc(df, flag_test, xnames, yname, n_samples=0):
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

def build_gbc():
	GradientBoostingClassifier(learning_rate=0.05, n_estimators=200 \
		, max_features = 6, min_samples_split = 15, max_depth = 5, min_samples_leaf = 16)
	return clf

def build_rfc():
	d = {'min_samples_leaf': 7, 'min_samples_split': 3, 'criterion': 'entropy', 'max_features': 9, 'max_depth': 11, 'class_weight': {0: 1, 1: 1} }

	clf = RandomForestClassifier(max_depth = 15, n_estimators = 250, min_samples_leaf =5 )
	return clf

def build_rfregressor():
	clf = RandomForestRegressor(random_state=0, n_estimators=100\
		, max_depth = 15, min_samples_leaf =5 )
	return clf

def init_all_models():
	clf1 = build_rfc()
	clf2 = build_rfregressor()
	clf3 = linear_model.SGDClassifier(loss='log') 
	clf4 = AdaBoostClassifier(learning_rate=0.1, n_estimators=500)
	clf5 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
	models = [clf1, clf2, clf3, clf4, clf5]
	return models

def build_ensemble_models(df, xnames, yname, n_folds = 5, warmstart = False):
	models = init_all_models()
	n_models = len(models)
	run_name = 'eval_ensemble_models'

	cv_fname = 'data/kfold_cv.pkl'
	if warmstart:
		cv = pickle.load(open(cv_fname))
	else:
		cv = KFold(df.shape[0], n_folds= cv) 
		pickle.dump(cv,open(cv_fname, 'wb'))

	print ','.join(c.__class__.__name__ for c in models)
	scores, scores_all, scores_all_train =  [], [], []
	t00 = time.time() # time it		
	t0 = t00
	i = 0
	for train, test in cv:
		if len(train) + len(test) != df.shape[0]:
			print 'ERROR: K fold and dataset mis-match'
			return 			
		probs_sum, probs_sum_train = np.zeros((len(test), 2)), np.zeros((len(train), 2))
		scores_all.append([])
		scores_all_train.append([])		
		for clf in models:
			X_train, y_train = df.iloc[train][xnames], df.iloc[train][yname]
			X_test, y_test = df.iloc[test][xnames], df.iloc[test][yname]

			probs, probs_train = get_predicted_prob(clf, i, X_train, y_train, X_test, y_test, warmstart = warmstart)
			
			score_this_clf = roc_auc_score(y_test, probs[:, 1])
			print("testset score: %f" % score_this_clf)
			scores_all[i].append(score_this_clf)
			scores_all_train[i].append(roc_auc_score(y_train, probs_train[:,1]))
			probs_sum += probs
			probs_sum_train += probs_train

			t1 = time.time() # time it
			time_taken = (t1-t0)/60	
			print 'run %d %s time taken %.2fmin' % (i, clf.__class__.__name__, time_taken)
			t0 = t1
		probs_avg = probs_sum / (1.0 * n_models)
		probs_avg_train = probs_sum_train / (1.0 * n_models)       
		score_this_run = roc_auc_score(y_test, probs_avg[:, 1])
		score_this_run_train = roc_auc_score(y_train, probs_avg_train[:, 1])
		scores.append(score_this_run)
		scores_all[i].append(score_this_run)
		scores_all_train[i].append(score_this_run_train)		
		print "run %i combined score: %f" % (i, scores[-1])
		print "       test: ", scores_all[i]
		print "       train: ", scores_all_train[i]
		print 
		i = i + 1
	print 
		
	scores_all = np.array(scores_all)
	scores_all_train = np.array(scores_all_train)
	t1 = time.time() # time it
	time_taken = (t1-t00)/60	
	print run_name,  ' finished in ', time_taken, ' minutes'
	print 'combined score: mean', (np.mean(scores), 'std:', np.std(scores))		
	print 'test set:', np.mean(scores_all, axis = 0)
	print 'train set:', np.mean(scores_all_train, axis = 0)

def read_pred_cv(n_folds = 5):
	cv_fname = 'data/kfold_cv.pkl'
	cv = pickle.load(open(cv_fname))
	models = init_all_models()
	print ','.join(c.__class__.__name__ for c in models)
	t00 = time.time() # time it		
	t0 = t00
	#i = 0
	X_train, y_train, X_test, y_test = [ None ] * 4
	predictions_test = []
	for i, (train, test) in enumerate(cv):
		predictions_test.append([])
		if len(train) + len(test) != df.shape[0]:
			print 'ERROR: K fold and dataset mis-match'
			return 				
		for clf in models:
			probs, probs_train = get_predicted_prob(clf, i, X_train, y_train, X_test, y_test, warmstart = True)
			predictions_test[i].append(probs)
			#i = i + 1			
	return cv, predictions_test

def eval_ensemble_average_cv(predictions_test, df, cv, yname, weights):
	y_tests_all = [] 
	for train, test in cv:
		y_tests_all.append(df.iloc[test][yname])
	y_tests_all = np.array(y_tests_all)

	roc = []
	for i_fold in xrange(y_tests_all.shape[0]):
		roc.append([])
		y_test = y_tests_all[i_fold]
		y_test_preds = np.array([p[:,1] for p in predictions_test[i_fold]])	
		w = np.array(weights[i_fold]).T
		y_test_ensemble = y_test_preds.T.dot(w)
		for y_test_pred in y_test_preds:
			roc[i_fold].append(roc_auc_score(y_test,y_test_pred ))
		roc[i_fold].append(roc_auc_score(y_test, y_test_ensemble))
		print roc[i_fold][-1],'         ', roc[i_fold]
	roc = np.array(roc)
	print np.mean(roc, axis = 0)

def find_weight_for_ensemble_cv(predictions_test, df, cv, yname):
	weights = []
	for i, (train, test) in enumerate(cv):
		y_test = df.iloc[test][yname]
		y_test_preds = predictions_test[i]
		weights.append(find_weight_for_ensemble(y_test_preds, y_test))
	return weights

def find_weight_for_ensemble(predictions, y_test):
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

def apply_fitted_models(weight, clfs):
	#w = [1.0/5]*5
	w = weight

	df_test = read_all_data('data/credit-test.csv')
	df_test, xnames = pre_process(df_test)

	models = clfs
	probs_common = np.zeros((df_test.shape[0], 2))
	preds = []
	for i, clf in enumerate(models):
		if 'Classifier' in clf.__class__.__name__:
			probs = clf.predict_proba(df_test[xnames])
		else:
			probs = np.zeros((df_test[xnames].shape[0],2))			
			probs[:,1] = clf.predict(df_test[xnames]).T
		preds.append(probs)
		write_test(probs[:, 1], "test_prediction_model_%d.csv" % i,
			df_test)
	y_test_preds = (np.array([p[:,1] for p in preds])).T	
	w = np.array(w).T
	y_test_ensemble = y_test_preds.dot(w)
	y_test_ensemble = (y_test_ensemble - y_test_ensemble.min())/(y_test_ensemble.max() - \
		y_test_ensemble.min())
	write_test(y_test_ensemble, "test_prediction_combined.csv", df_test)

def apply_models(weight):
	#w = [1.0/5]*5
	w = weight
	#df = read_all_data('data/credit-training_sample.csv')
	df = read_all_data('data/credit-training.csv')	
	df, xnames = pre_process(df)

	df_test = read_all_data('data/credit-test.csv')
	df_test, xname = pre_process(df_test)

	models = init_all_models()
	probs_common = np.zeros((df_test.shape[0], 2))
	preds = []
	for i, clf in enumerate(models):
		clf.fit(df[xnames], df[yname])
		if 'Classifier' in clf.__class__.__name__:
			probs = clf.predict_proba(df_test[xnames])
		else:
			probs = np.zeros((df_test[xnames].shape[0],2))			
			probs[:,1] = clf.predict(df_test[xnames]).T
		preds.append(probs)
		#print("score: %f" % auc_score(labels_test, probs[:, 1]))
		#probs_common += probs
		write_test(probs[:, 1], "test_prediction_model_%d.csv" % i,
			df_test)
	#probs_common /= 1.0 * len(models)
	#score = auc_score(labels_test, probs_common[:, 1])
	#print("combined score: %f" % score)
	y_test_preds = (np.array([p[:,1] for p in preds])).T	
	w = np.array(w).T
	y_test_ensemble = y_test_preds.dot(w)
	y_test_ensemble = (y_test_ensemble - y_test_ensemble.min())/(y_test_ensemble.max() - \
		y_test_ensemble.min())
	write_test(y_test_ensemble, "test_prediction_combined.csv", df_test)

def read_pred_submodels():
	y_test_preds = np.zeros((101503))
	fnames = ["test_prediction_model_%d.csv" % i for i in xrange(5)]
	y_test_preds = pd.concat([ pd.read_csv(fname) for fname in fnames], axis = 1).values
	i = [1,3,5,7,9]
	return y_test_preds[:, i]

def combine_preds(y_test_preds, w):
	w = np.array(w).T
	y_test_ensemble = y_test_preds.dot(w)
	write_test(y_test_ensemble, "test_prediction_combined.csv")	

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

if __name__ == '__main__':
	pd.set_option('max_colwidth',500)
	pd.set_option('display.expand_frame_repr', False)
	sys.stdout.flush()
	cv = 5
	fname = 'data/credit-training.csv'

	flag_test, n_samples = 0, 0
	if len(sys.argv) > 1:
		flag_test = sys.argv[1]
		n_samples = int(sys.argv[1])

	#df, icol_y, icols_x = read_data(fname)	

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

	'''
	df = scale_X(df, xnames)
	#run_test_classifier(df, xnames, yname) #, flag_test, n_samples)
	'''
