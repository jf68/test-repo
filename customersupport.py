"""
customersupport.py

This module runs models in order to determine
which attributes will have the most profound effect
on predicting positive supportive behavior
amongst customers.

"""





import numpy as np
import pandas as pd
import savReaderWriter as srw
import re
import pdb
import rpy2.robjects as ro
import pandas.rpy.common as com

from rpy2.robjects import pandas2ri
pandas2ri.activate()


import sklearn as skl
from sklearn import linear_model as lm
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA



# return spss dataset as a pandas dataframe
def readspss(filename):
	'''
	This file readers data sets (.sav) from SPSS
	and imports them into Python.
	'''

	mylist = []
	with srw.SavReader(filename, returnHeader=True) as reader:
	    header = reader.next()
	    for line in reader:
	        mylist.append(line)

	data = srw.SavReader(filename)
	data.close()

	df2 = pd.DataFrame(mylist)

	ind = mylist[0]
	arr = np.array(df2.iloc[1:,:])
	rawfile = pd.DataFrame(arr, columns = ind)

	# Change None strings/objects and Empty Strings into NaN values
	rawfile = rawfile.replace('None', np.nan)
	rawfile = rawfile.replace('', np.nan)
	rawfile.fillna(value = np.nan, inplace=True)

	return rawfile, ind



if __name__ == "__main__":
	print 'running program...'

	filepath = 'C:\\Users\\jfeng\\Desktop\\Processing\\Regression\\Allstate Customers-Cumu. 2016 Dec - SB.sav'

	alldata, varnames = readspss(filepath)

	alldata = alldata.replace(98, np.NaN)
	alldata = alldata.replace(99, np.NaN)

	attributes = ["Q515_1","Pulse_G_reported",
	"Q320_1r_G_reported",
	"Q320_2r_G_reported",
	"Q320_3r_G_reported",
	"Q320_4r_G_reported",
	"Q320_5r_G_reported",
	"Q320_6r_G_reported",
	"Q320_7r_G_reported",
	"Q320_8r_G_reported",
	"Q320_10r_G_reported",
	"Q320_12r_G_reported",
	"Q320_14r_G_reported",
	"Q320_15r_G_reported",
	"Q320_16r_G_reported",
	"Q320_18r_G_reported",
	"Q320_19r_G_reported",
	"Q320_20r_G_reported",
	"Q320_21r_G_reported",
	"Q320_23r_G_reported",
	"Q320_24r_G_reported",
	"Q320_25r_G_reported",
	"Q320_26r_G_reported",
	"Q320_27r_G_reported",
	"Month_Reporting",
	"Rating"]

	alldata = alldata[attributes]

	# Create Allstate dataframe
	Allstate = alldata[alldata['Rating'] == 1]


	# Create Allstate 2014 dataframe
	Allstate14 = Allstate[Allstate['Month_Reporting'] <= 60]
	Allstate14 = Allstate14[Allstate14['Month_Reporting'] >= 49]
	#Allstate14 = pd.DataFrame(Allstate15.ix[:, 9:102])
	#Allstate14 = pd.DataFrame(Allstate15.ix[:, 0:102])
	Allstate14 = pd.DataFrame(Allstate14.ix[:, 0:24])
	Allstate14 = Allstate14.reset_index(drop=True)

	#print Allstate14
	#pdb.set_trace()
	# Create Allstate 2015 dataframe
	Allstate15 = Allstate[Allstate['Month_Reporting'] <= 72]
	Allstate15 = Allstate15[Allstate15['Month_Reporting'] >= 61]
	#Allstate15 = pd.DataFrame(Allstate15.ix[:, 9:102])
	#Allstate15 = pd.DataFrame(Allstate15.ix[:, 0:102])
	Allstate15 = pd.DataFrame(Allstate15.ix[:, 0:24])
	Allstate15 = Allstate15.reset_index(drop=True)

	#print Allstate15
	
	# Create Allstate 2016 dataframe
	Allstate16 = Allstate[Allstate['Month_Reporting'] >= 73]
	Allstate16 = Allstate16[Allstate16['Month_Reporting'] <= 84]
	#Allstate16 = pd.DataFrame(Allstate16.ix[:, 9:102])
	#Allstate16 = pd.DataFrame(Allstate16.ix[:, 0:102])
	Allstate16 = pd.DataFrame(Allstate16.ix[:, 0:24])
	Allstate16 = Allstate16.reset_index(drop=True)
	
	#print Allstate16
	#print Allstate16.describe(include='all')
	#print Allstate16.isnull().sum()

	# Calls R package
	ro.r('library(randomForest)')


	######################################################
	############# 2016 ###################################
	######################################################

	# Converts pandas df to R df
	Allstate16 = Allstate16.dropna(axis=0, subset=['Q515_1'])
	rAllstate16 = com.convert_to_r_dataframe(Allstate16)

	# Passes R df into R environment
	ro.globalenv['AS16'] = rAllstate16
	#print(ro.r('AS16$S1006_1'))

	# Create response variable: Say something positive
	Allstate16['ySB2'] = 0
	Allstate16.ix[Allstate16.Q515_1 >= 6,'ySB2'] = 1

	ySB2_16 = pd.DataFrame(Allstate16['ySB2'])
	ySB2r_16 = com.convert_to_r_dataframe(ySB2_16)
	ro.globalenv['ySB2_16'] = ySB2r_16
	ro.r('ySB2_16 <- as.numeric(unlist(ySB2_16))')
	#print(ro.r('AS16[,c(10:102)]'))
	#print(ro.r('ySB2_16'))

	#ro.r('newdf16 = rfImpute(x = AS16[,c(10:102)], y = ySB2_16, iter=5, ntree=300)')
	ro.r('newdf16 = rfImpute(x = AS16[,c(2:24)], y = ySB2_16, iter=5, ntree=300)')
	#print(ro.r('newdf16'))

	x16df = com.load_data('newdf16')
	x16df = x16df.ix[:,1:]
	ySB2_16 = np.ravel(np.array(ySB2_16))
	
	#print x16df
	# x16df is predictor variables
	# ySB2_16 is response variable
	# Use these two dataframes for all modeling purposes

	#####################################################
	#####################################################




	######################################################
	############# 2015 ###################################
	######################################################

	# Converts pandas df to R df
	Allstate15 = Allstate15.dropna(axis=0, subset=['Q515_1'])
	rAllstate15 = com.convert_to_r_dataframe(Allstate15)

	# Passes R df into R environment
	ro.globalenv['AS15'] = rAllstate15


	# Create response variable: Say something positive
	Allstate15['ySB2'] = 0
	Allstate15.ix[Allstate15.Q515_1 >= 6,'ySB2'] = 1

	ySB2_15 = pd.DataFrame(Allstate15['ySB2'])
	ySB2r_15 = com.convert_to_r_dataframe(ySB2_15)
	ro.globalenv['ySB2_15'] = ySB2r_15
	ro.r('ySB2_15 <- as.numeric(unlist(ySB2_15))')
	#print(ro.r('AS16[,c(10:102)]'))
	#print(ro.r('ySB2_16'))

	#ro.r('newdf15 = rfImpute(x = AS15[,c(10:102)], y = ySB2_15, iter=5, ntree=300)')
	ro.r('newdf15 = rfImpute(x = AS15[,c(2:24)], y = ySB2_15, iter=5, ntree=300)')
	#print(ro.r('newdf16'))

	x15df = com.load_data('newdf15')
	x15df = x15df.ix[:,1:]
	ySB2_15 = np.ravel(np.array(ySB2_15))
	
	print x15df
	#####################################################
	#####################################################


	######################################################
	############# 2014 ###################################
	######################################################

	# Converts pandas df to R df
	Allstate14 = Allstate14.dropna(axis=0, subset=['Q515_1'])
	rAllstate14 = com.convert_to_r_dataframe(Allstate14)

	# Passes R df into R environment
	ro.globalenv['AS14'] = rAllstate14


	# Create response variable: Say something positive
	Allstate14['ySB2'] = 0
	Allstate14.ix[Allstate14.Q515_1 >= 6,'ySB2'] = 1

	ySB2_14 = pd.DataFrame(Allstate14['ySB2'])
	ySB2r_14 = com.convert_to_r_dataframe(ySB2_14)
	ro.globalenv['ySB2_14'] = ySB2r_14
	ro.r('ySB2_14 <- as.numeric(unlist(ySB2_14))')
	#print(ro.r('AS16[,c(10:102)]'))
	#print(ro.r('ySB2_16'))

	#ro.r('newdf14 = rfImpute(x = AS14[,c(10:102)], y = ySB2_14, iter=5, ntree=300)')
	ro.r('newdf14 = rfImpute(x = AS14[,c(2:24)], y = ySB2_14, iter=5, ntree=300)')
	#print(ro.r('newdf16'))

	x14df = com.load_data('newdf14')
	x14df = x14df.ix[:,1:]
	ySB2_14 = np.ravel(np.array(ySB2_14))
	
	#print x14df
	#####################################################
	#####################################################
	#x15m = np.matrix(x15df).T

	#pca = PCA(n_components=6)
	#pca.fit(x15m)
	#print pd.DataFrame(np.matrix(pca.components_).T)
	#print pca.explained_variance_
	#print pca.mean_
	#print pca.n_components_
	#print pca.explained_variance_ratio_
	#pdb.set_trace()
	

	##### 2015 Predictions
	print "\n2015 predictions"

	# Ridge Logistic Regression with L2 penalty
	print 'L2 penalty'
	logregr16 = lm.LogisticRegressionCV(Cs=[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,1e4,1e5], cv = 10, penalty='l2')
	logregr16.fit(X=x16df, y=ySB2_16)
	print logregr16.coef_
	ynew15 = logregr16.predict(x15df)
	print skl.metrics.accuracy_score(y_true=ySB2_15, y_pred=ynew15)
	print skl.metrics.confusion_matrix(y_true=ySB2_15, y_pred=ynew15)

	# LASSO Logistic Regression with L1 penalty
	print '\n\n\nL1 penalty'

	logregl16 = lm.LogisticRegressionCV(Cs=[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,1e4,1e5], cv = 10, penalty='l1', solver='liblinear')
	logregl16.fit(X=x16df, y=ySB2_16)
	print logregl16.coef_
	ynew15 = logregl16.predict(x15df)
	print skl.metrics.accuracy_score(y_true=ySB2_15, y_pred=ynew15)
	print skl.metrics.confusion_matrix(y_true=ySB2_15, y_pred=ynew15)


	# KNN
	print '\n\n\nKNN'
	
	parameters = {'n_neighbors':[1,2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]}
	knnd16 = skl.neighbors.KNeighborsClassifier()
	knn16 = skl.model_selection.GridSearchCV(knnd16, parameters, cv=10)
	knn16.fit(X=x16df, y=ySB2_16)
	ynew15 = knn16.predict(x15df)
	print skl.metrics.accuracy_score(y_true=ySB2_15, y_pred=ynew15)
	print skl.metrics.confusion_matrix(y_true=ySB2_15, y_pred=ynew15)


	# Multilayer Perceptron Classifier
	print '\n\n\nMLP'

	mlp16 = MLPClassifier(alpha=1)
	mlp16.fit(X=x16df, y=ySB2_16)
	ynew15 = mlp16.predict(x15df)
	print skl.metrics.accuracy_score(y_true=ySB2_15, y_pred=ynew15)
	print skl.metrics.confusion_matrix(y_true=ySB2_15, y_pred=ynew15)



	# Naive Bayes
	print '\n\n\nNaive Bayes'

	#parameters = {'max_depth':[5,10], 'min_samples_split':[2,3,5]}
	#parameters = {'max_features':('sqrt','log2')}
	nb16 = GaussianNB(priors=None)
	#rf16 = skl.model_selection.GridSearchCV(rfd16, parameters)
	nb16.fit(X=x16df, y=ySB2_16)
	ynew15 = nb16.predict(x15df)
	print skl.metrics.accuracy_score(y_true=ySB2_15, y_pred=ynew15)
	print skl.metrics.confusion_matrix(y_true=ySB2_15, y_pred=ynew15)





	# Support Vector Machines
	print '\n\n\nSVM'
	
	#parameters = {'kernel':('rbf','linear'), 'C':[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,1e4,1e5]}
	#parameters = {'C':[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,1e4,1e5]}
	#parameters = {'C':[1e-4,1e-3,1e-2,1e-1,1,10,100]}
	parameters = {'C':[1,2,3]}
	svmd16 = skl.svm.SVC(kernel='linear')
	svm16 = skl.model_selection.GridSearchCV(svmd16, parameters, cv = 3)
	svm16.fit(X=x16df, y=ySB2_16)
	ynew15 = svm16.predict(x15df)
	print skl.metrics.accuracy_score(y_true=ySB2_15, y_pred=ynew15)
	print skl.metrics.confusion_matrix(y_true=ySB2_15, y_pred=ynew15)
	



	# Linear Discriminant Analysis
	print '\n\n\nLinear Discriminant Analysis'

	#parameters = {'max_depth':[5,10], 'min_samples_split':[2,3,5]}
	#parameters = {'max_features':('sqrt','log2')}
	lda16 = LinearDiscriminantAnalysis()
	#rf16 = skl.model_selection.GridSearchCV(rfd16, parameters)
	lda16.fit(X=x16df, y=ySB2_16)
	ynew15 = lda16.predict(x15df)
	print skl.metrics.accuracy_score(y_true=ySB2_15, y_pred=ynew15)
	print skl.metrics.confusion_matrix(y_true=ySB2_15, y_pred=ynew15)


	
	# Quadratic Discriminant Analysis
	print '\n\n\nQuadratic Discriminant Analysis'

	#parameters = {'max_depth':[5,10], 'min_samples_split':[2,3,5]}
	#parameters = {'max_features':('sqrt','log2')}
	qda16 = QuadraticDiscriminantAnalysis()
	#rf16 = skl.model_selection.GridSearchCV(rfd16, parameters)
	qda16.fit(X=x16df, y=ySB2_16)
	ynew15 = qda16.predict(x15df)
	print skl.metrics.accuracy_score(y_true=ySB2_15, y_pred=ynew15)
	print skl.metrics.confusion_matrix(y_true=ySB2_15, y_pred=ynew15)
	


	# Random Forest
	print '\n\n\nRandom Forest'

	#parameters = {'max_depth':[5,10], 'min_samples_split':[2,3,5]}
	#parameters = {'max_features':('sqrt','log2')}
	rf16 = ensemble.RandomForestClassifier(n_estimators = 1000, n_jobs = -1, max_features='sqrt')
	#rf16 = skl.model_selection.GridSearchCV(rfd16, parameters)
	rf16.fit(X=x16df, y=ySB2_16)
	ynew15 = rf16.predict(x15df)
	print skl.metrics.accuracy_score(y_true=ySB2_15, y_pred=ynew15)
	print skl.metrics.confusion_matrix(y_true=ySB2_15, y_pred=ynew15)


	#####################################################################
	#####################################################################
	#####################################################################


	##### 2014 Predictions
	print "\n2014 predictions"

	# Ridge Logistic Regression with L2 penalty
	print 'L2 penalty'
	logregr16 = lm.LogisticRegressionCV(Cs=[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,1e4,1e5], cv = 10, penalty='l2')
	logregr16.fit(X=x16df, y=ySB2_16)
	print logregr16.coef_
	ynew14 = logregr16.predict(x14df)
	print skl.metrics.accuracy_score(y_true=ySB2_14, y_pred=ynew14)
	print skl.metrics.confusion_matrix(y_true=ySB2_14, y_pred=ynew14)

	# LASSO Logistic Regression with L1 penalty
	print '\n\n\nL1 penalty'

	logregl16 = lm.LogisticRegressionCV(Cs=[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,1e4,1e5], cv = 10, penalty='l1', solver='liblinear')
	logregl16.fit(X=x16df, y=ySB2_16)
	print logregl16.coef_
	ynew14 = logregl16.predict(x14df)
	print skl.metrics.accuracy_score(y_true=ySB2_14, y_pred=ynew14)
	print skl.metrics.confusion_matrix(y_true=ySB2_14, y_pred=ynew14)


	# KNN
	print '\n\n\nKNN'
	
	parameters = {'n_neighbors':[1,2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]}
	knnd16 = skl.neighbors.KNeighborsClassifier()
	knn16 = skl.model_selection.GridSearchCV(knnd16, parameters, cv=10)
	knn16.fit(X=x16df, y=ySB2_16)
	ynew14 = knn16.predict(x14df)
	print skl.metrics.accuracy_score(y_true=ySB2_14, y_pred=ynew14)
	print skl.metrics.confusion_matrix(y_true=ySB2_14, y_pred=ynew14)


	# Multilayer Perceptron Classifier
	print '\n\n\nMLP'

	mlp16 = MLPClassifier(alpha=1)
	mlp16.fit(X=x16df, y=ySB2_16)
	ynew14 = mlp16.predict(x14df)
	print skl.metrics.accuracy_score(y_true=ySB2_14, y_pred=ynew14)
	print skl.metrics.confusion_matrix(y_true=ySB2_14, y_pred=ynew14)



	# Naive Bayes
	print '\n\n\nNaive Bayes'

	#parameters = {'max_depth':[5,10], 'min_samples_split':[2,3,5]}
	#parameters = {'max_features':('sqrt','log2')}
	nb16 = GaussianNB(priors=None)
	#rf16 = skl.model_selection.GridSearchCV(rfd16, parameters)
	nb16.fit(X=x16df, y=ySB2_16)
	ynew14 = nb16.predict(x14df)
	print skl.metrics.accuracy_score(y_true=ySB2_14, y_pred=ynew14)
	print skl.metrics.confusion_matrix(y_true=ySB2_14, y_pred=ynew14)





	# Support Vector Machines
	print '\n\n\nSVM'
	
	#parameters = {'kernel':('rbf','linear'), 'C':[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,1e4,1e5]}
	#parameters = {'C':[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,1e4,1e5]}
	#parameters = {'C':[1e-4,1e-3,1e-2,1e-1,1,10,100]}
	parameters = {'C':[1]}
	svmd16 = skl.svm.SVC(kernel='linear')
	svm16 = skl.model_selection.GridSearchCV(svmd16, parameters, cv = 3)
	svm16.fit(X=x16df, y=ySB2_16)
	ynew14 = svm16.predict(x14df)
	print skl.metrics.accuracy_score(y_true=ySB2_14, y_pred=ynew14)
	print skl.metrics.confusion_matrix(y_true=ySB2_14, y_pred=ynew14)
	



	# Linear Discriminant Analysis
	print '\n\n\nLinear Discriminant Analysis'

	#parameters = {'max_depth':[5,10], 'min_samples_split':[2,3,5]}
	#parameters = {'max_features':('sqrt','log2')}
	lda16 = LinearDiscriminantAnalysis()
	#rf16 = skl.model_selection.GridSearchCV(rfd16, parameters)
	lda16.fit(X=x16df, y=ySB2_16)
	ynew14 = lda16.predict(x14df)
	print skl.metrics.accuracy_score(y_true=ySB2_14, y_pred=ynew14)
	print skl.metrics.confusion_matrix(y_true=ySB2_14, y_pred=ynew14)


	
	# Quadratic Discriminant Analysis
	print '\n\n\nQuadratic Discriminant Analysis'

	#parameters = {'max_depth':[5,10], 'min_samples_split':[2,3,5]}
	#parameters = {'max_features':('sqrt','log2')}
	qda16 = QuadraticDiscriminantAnalysis()
	#rf16 = skl.model_selection.GridSearchCV(rfd16, parameters)
	qda16.fit(X=x16df, y=ySB2_16)
	ynew14 = qda16.predict(x14df)
	print skl.metrics.accuracy_score(y_true=ySB2_14, y_pred=ynew14)
	print skl.metrics.confusion_matrix(y_true=ySB2_14, y_pred=ynew14)
	


	# Random Forest
	print '\n\n\nRandom Forest'

	#parameters = {'max_depth':[5,10], 'min_samples_split':[2,3,5]}
	#parameters = {'max_features':('sqrt','log2')}
	rf16 = ensemble.RandomForestClassifier(n_estimators = 1000, n_jobs = -1, max_features='sqrt')
	#rf16 = skl.model_selection.GridSearchCV(rfd16, parameters)
	rf16.fit(X=x16df, y=ySB2_16)
	ynew14 = rf16.predict(x14df)
	print skl.metrics.accuracy_score(y_true=ySB2_14, y_pred=ynew14)
	print skl.metrics.confusion_matrix(y_true=ySB2_14, y_pred=ynew14)





