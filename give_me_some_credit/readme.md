## Kaggle give me some creidt

#### The task
The task is to predict the probability that somebody will experience financial distress in the next two years.   

The training set has 150k data rows,  6.7% positives (default- serious_dlqin2yrs = 1). There are 10 numerical features (age, income, debt-to income ratio, number of days late). Two features have missing value: 19% for Monthly income and 2% for number of independents.

### Things that worked
1. Coding missing values -1 for income and 0 for number of independent allows inclusion of all the data and improved AUC of basic random forest model to 0.863 from 0.831.
2. hyper parameter search:  0.863 to 0.864

### Things that did not work
1. feature engineerings
2. change weigths by class to adjust models for un-balanced data 
3. quick ensemble predictions from different methods

### Next things to try
1. Neural Network and SVM
2. improve ensembling of predictions from different methods 

#### Data insights
1. feature importance
2. single variate
3. interactions.

### Methods
sklearn was used to build all models. Models were evaluated by comparing AUC using 5-fold cross-valiation. Hyper parameters were tued using grid search and randomized search.

### Results of Modeling
Gradient Boosting Classification gave the highest AUC of 0.8### for a single method. 


n_estimator vs. score


ROC curve


Partial Dependence Plot


