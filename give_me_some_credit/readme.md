## give me some creidt - kaggle dataset

##### The task
The task is to predict the probability somebody will experience loan delinquency in the next two years ([detailed description...](https://www.kaggle.com/c/GiveMeSomeCredit))

The training set has 150k data rows, 6.7% positives (default- serious_dlqin2yrs = 1). There are 10 numerical features including age, income, debt-to income ratio, number of days late. Two features have missing value: 19% for monthly income and 2% for number of independents. Models are evaluated by AUC.

##### Things that worked (some moderately)
1. Quick testing of commonly used classifiers showed tree based methods and boosting had the best results. 

	```
		                model   roc_auc      time
		5                 GBC  0.830638  0.257272
		3  AdaBoostClassifier  0.827332  0.102047
		4                  RF  0.714638  0.067388
		0                 KNN  0.689130  0.072305
		1  LogisticRegression  0.635908  0.050604
		2             SGDC_lr  0.471594  0.002476
	```

2. Coding missing values: -1 for missing income and 0 for missing number of independent. This allows inclusion of all the data and improved AUC of basic random forest model to 0.863 from 0.831.  
3. Hyper parameter search for GradientBoostingClassifier improved AUC from 0.863 to 0.8661 and for RandomForestClassifier from 0.83 to 0.8682.
4. Simple avearge of predictions from the three best methods (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier) had moderate improvement to 0.8685. 
5. Probability calibration using calibration.CalibratedClassifierCV and optmizing weights of submodels by mimimize logloss resulted in minor improvment of AUC to 0.8686.

Bechmark method has AUC of 0.8642; the 1st place 0.8696 and 50th place 0.8681. My final submission had score of 0.8686.

##### Things that did not work (yet?)
1. Additional feature engineering did not result in large change in AUC for the best classifiers (GradientBoostingClassifier and RandomForestClassifier), probably because these methods were able to learn complex decision boundary. It imporved performace of logistic regression. 
2. Adjusting weigths by class to account for un-balanced data did not help with the performance.  


##### Other things to try
1. Neural Network and SVM
2. Improve ensembling of predictions from different methods 

##### Methods and Tools
1. sklearn was used to build all models. Models were evaluated by comparing AUC using 5-fold cross-valiation. Hyper parameters were tuned using grid search and randomized search. 
2. [ML helper functions](https://github.com/joyce-duan/ml_helper) for model tuning, evaluation, and plots.


##### Results of modeling
Gradient Boosting Classification gave the highest AUC.    

![alt text](/give_me_some_credit/images/roc_gbc.png "ROC")

n_estimators vs. score 

![alt text](/give_me_some_credit/images/plot_staged_score_gbc.png "optimzing learning rate")

##### Data insights 
1. feature importance 

	![alt text](/give_me_some_credit/images/feature_importance_gbc.png "feature importance")

2. Delinquecy rate: 
	- increases with higher revoling utilization of credit line. 
	- decreases with age for age over ~25. 
	- tends to be large for people with 0 credit line or too many credit lines (>20). 
	- tends to be higher for people who have been more than 60 days late on payment. 

	![plot](/give_me_some_credit/images/uni1.png ) 

	![plot](/give_me_some_credit/images/uni3.png ) 








