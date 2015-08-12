import math
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from configobj import ConfigObj
config = ConfigObj('config')

ml_home = config.get(
    'ml_home', './')
import sys
sys.path.append('../util')
sys.path.append(ml_home)
sys.path.append(ml_home+ 'features')
sys.path.append(ml_home+ 'eda')

import myutil
import ml_util  
import features

def read_all_data(datafname):
    df = pd.read_csv(datafname)
    df.columns = [myutil.camel_to_snake(c) for c in df.columns]
    print 'original column names:'
    print df.columns
    return df

def rename_columns(df):   
    df.rename(columns={'unnamed: 0':'id'}, inplace=True)
    df.columns = [u'id', u'dlqin2yrs', u'revolv_util_clines', u'age', u'n_30-59d_late', \
                  u'dti', u'monthly_income', u'n_loans_creditlines', \
                  u'n_90d_late', u'n_house_loans', u'n_60-89d_late', u'n_dependents']
    cols_reordered = [u'id', u'dlqin2yrs', u'age', u'revolv_util_clines',  u'dti'\
                      , u'monthly_income', u'n_loans_creditlines', u'n_house_loans',  u'n_dependents',\
                      u'n_30-59d_late',u'n_60-89d_late', u'n_90d_late']
    df = df[cols_reordered]
    yname = df.columns[1]
    xnames = df.columns[2:]
    return df, xnames, yname

def pre_process(df):
	df, xnames, yname = rename_columns(df)
	df = feature_enginerring(df)
	xnames = df.columns.difference([yname, 'id'])
	df = fill_na(df, xnames, v = -1)
	return df, xnames 

def scale_X(df, xnames):
	cols_binary = '''
	debt_is_0       
	mon_income_valid         
	mon_income_is_1         
	mon_income_is_0             
	mon_income_is_k     
	dti_is_int                      
	dti_is_0                         
	dti_gt_33                        
	dti_gt_40                                                        
	revolv_util_is0                  
	revolv_util_is1                                   
	n_90d_late_missing                             
	n_30-59d_late_missing                         
	n_60-89d_late_missing                  
	'''.split()
	xnames_toscale = xnames.difference(cols_binary)
	scaler = StandardScaler()
	df[xnames_toscale] = scaler.fit_transform(df[xnames_toscale])
	return df, scaler

def get_data(fname, flag_test =0, n_samples = None):
	'''
	return df with cleaned column name
	read in either the raw data or the sample dataset 
	'''
	if flag_test:
		fname_sample = fname.replace('.csv', '_sample.csv')
		if os.path.exists(fname_sample):		
			df = read_all_data(fname_sample)
		else:
			df = read_all_data(fname)
			k = min(1000, n_samples)
			lst_i = random.sample(xrange(df.shape[0]), k)
			df = df.iloc[lst_i,:]	
			df.to_csv(fname_sample, index = False)
	else: 
		df = read_all_data(fname)
	return rename_columns(df)

def feature_enginerring(df_orig):
	df = df_orig
	print 'read in columns:'
	print df_orig.columns
	df_orig['age'] = df_orig['age'].apply(lambda x: x if x > 16 else df_orig.age.median())
	df_orig['n_dependents'] = df_orig['n_dependents'].fillna(0)

	df_orig['debt'] = df_orig['monthly_income'] * df_orig['dti']
	cond_dti_is_debt = np.all([df_orig['dti'] > 0\
	                             ,df_orig['monthly_income'] == 0], axis=0 )
	df_orig['debt'] = np.where(cond_dti_is_debt, df_orig['dti'], df_orig['debt'])

	cond_dti_is_debt = np.all([df_orig['dti'] > 0\
	                             ,df_orig['monthly_income'].isnull()], axis=0 )
	df_orig['debt'] = np.where(cond_dti_is_debt, df_orig['dti'], df_orig['debt'])
	df_orig['debt'] = df_orig['debt'].fillna(-1)

	df_orig['debt_is_0'] = df_orig['debt'].apply(lambda x: 1 if x == 0 else 0)

	monthly_income_min_val = 99
	dt_max_val = 30
	# monthly income 0 or 1: 2239 rows 2-99: 40
	df_orig['mon_income_valid'] = df_orig['monthly_income'].apply(lambda x: 1 if x > monthly_income_min_val else 0)
	df_orig['mon_income_valid'] = df_orig['mon_income_valid'].fillna(0)
	df_orig['mon_income_is_1'] = df_orig['monthly_income'].apply(lambda x: 1 if  x == 1 else 0)
	df_orig['mon_income_is_0'] = df_orig['monthly_income'].apply(lambda x: 1 if  x == 0 else 0)

	# monthly_income_fixed: -1 code for invalid
	df_orig['monthly_income_fixed'] = df_orig['monthly_income'].apply(lambda x: x if x > monthly_income_min_val else -1)
	df_orig['monthly_income_fixed'] = df_orig['monthly_income_fixed'].fillna(-1)
	df_orig['mon_income_is_k'] = df_orig['monthly_income_fixed'].apply(lambda x: 1 if x % 1000 ==0 else 0)

	df_orig['n_creditline'] = df_orig.n_loans_creditlines - df_orig.n_house_loans

	df_orig['dti_is_int'] = df_orig['dti'].apply(lambda x: 1 if (x > 0) & (int(x) - x == 0)  else 0)
	# dti_fixed: -1 invalid
	df_orig['dti_fixed'] = np.where(df_orig['dti_is_int'], -1, df_orig['dti'])

	df_orig['dti_is_0'] = df_orig['dti'].apply(lambda x: 1 if x == 0 else 0)
	df_orig['dti_gt_33'] = df_orig['dti_fixed'].apply(lambda x: 1 if x > 0.33 else 0)
	df_orig['dti_gt_40'] = df_orig['dti_fixed'].apply(lambda x: 1 if x > 0.40 else 0)

	age_max = 75
	revolv_util_clines_max = 3
	cols_tocap = ['age'
	,'revolv_util_clines'
	,'dti_fixed'
	,'monthly_income_fixed' 
	,'n_loans_creditlines'
	,'n_house_loans'
	,'n_dependents'
	]
	df_x_capped = features.cap_df(df, cols_tocap, pct = 0.99)

	df2 = pd.concat([df, df_x_capped], axis = 1)

	df2['age_cap_manual'] = df2['age'].apply(lambda x: x if x <=age_max else age_max)
	df2['revolv_util_cap_manual'] = df2['revolv_util_clines'].apply(lambda x: x if x <=revolv_util_clines_max else revolv_util_clines_max)
	df2['revolv_util_is0']= df2['revolv_util_clines'].apply(lambda x: 1 if x == 0 else 1)
	df2['revolv_util_is1']= df2['revolv_util_clines'].apply(lambda x: 1 if abs(x-1.0)<0.00001 else 0)

	colsnames = ['n_90d_late','n_30-59d_late','n_60-89d_late']
	cols_mod_names = [x + '_mod' for x in colsnames]
	for col in colsnames:
		col_mod = col + '_mod'
		col_missing = col + '_missing'
		df2[col_mod] = df[col].apply(lambda x: -1 if x > 90 else x) # missing value coded as -1; won't work for linear model
		df2[col_missing] = df[col].apply(lambda x: 1 if x > 90 else 0)
	df2['total_late_days'] = df2[cols_mod_names].apply(lambda x: np.sum(x), axis = 1)
	df2['total_late_days'] = df2['total_late_days'].apply(lambda x: -1 if x < 0 else x)

	'''
	df2['total_late_per_line'] = df2[['total_late_days','n_loans_creditlines_capped']].apply(lambda x:\
		1.0 * x[0] / x[1] if x[1]>0 else 0, axis = 0)
	df2['total_late_per_line'].fillna(0)
	df2['total_late_per_line'].apply(lambda x: -1 if x < 0 else x)
	'''

	df2['monthly_income_log'] = df2['monthly_income_fixed'].apply(lambda x: np.log(x+2))
	df2['dti_log'] = df2['dti'].apply(lambda x: np.log(x+1))
	df2['revolv_util_clines_log'] = df2['revolv_util_clines_capped'].apply(lambda x: np.log(x+1))

	df2['income_per_person'] =  df2[['monthly_income_fixed','n_dependents_capped']].apply(lambda x:\
	                        x[0]/(1.0+x[1]), axis = 1)
	df2['income_per_person'] = df2['income_per_person'].apply(lambda x: -1 if x < 0 else x)
	df2['income_per_person'] = df2['income_per_person'].fillna(-1)	
	df2['monthly_disposable'] = df2[['monthly_income_fixed','debt']].apply(lambda x:\
	                    x[0] - x[1], axis = 1)
	df2['monthly_disposable'] = np.where(np.any([df2['monthly_income_fixed']==-1, df2['debt']==-1],axis=0), -1, df2['monthly_disposable'] )

	return df2

def fill_na(df, xnames, v = 0):
	for c in xnames:
		df[c].fillna(v, inplace=True)
	return df 

def get_predicted_prob(clf, i, X_train, y_train, X_test, y_test, warmstart = False):
	fname = 'data/' + clf.__class__.__name__ + '_' + str(i)+'.pkl'
	fname_train = 'data/'+ clf.__class__.__name__ + '_' + str(i)+'_train.pkl'	
	if warmstart is False:
		clf.fit(X_train, y_train)
		if 'Classifier' in clf.__class__.__name__:
			probs = clf.predict_proba(X_test)
			probs_train = clf.predict_proba(X_train)
		else:
			probs = np.zeros((len(y_test),2))
			probs_train = np.zeros((len(y_train),2))				
			probs[:,1] = clf.predict(X_test).T
			probs_train[:,1] = clf.predict(X_train).T
		pickle.dump(probs, open(fname, 'w'))	
		pickle.dump(probs_train, open(fname_train, 'w'))			
	else:
		probs = pickle.load(open(fname))
		probs_train = pickle.load(open(fname_train))
	return probs, probs_train


def write_test(probs, fname_out, df = None):
	with open(fname_out, 'w') as fh_out:
		fh_out.write("Id,Probability\n")
		#for i, prob in zip(df.id, probs):
		for i, prob in enumerate(probs):
			fh_out.write("%d," % (i+1 ))
			fh_out.write("%f" % prob)
			fh_out.write('\n')

def get_X_y(df, flag_test, xnames, yname, n_samples = 0):
	#features = ['revolving_utilization_of_unsecured_lines', 'debt_ratio',
	#            'number_of_times90_days_late', 'number_real_estate_loans_or_lines']
	features = xnames
	X = df[features].values
	#y = df.iloc[:, icol_y]
	y = df[yname]
	print X[1]
	if flag_test:
		if n_samples < 100: n_samples = 200 
		X, y = random_sample(X, y, k = n_samples)
	print X[1]
	print len(X), len(X[0])
	return X, y

