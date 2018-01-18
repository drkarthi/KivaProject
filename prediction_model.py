import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as sklm
import sklearn.preprocessing as skp
import sklearn.metrics as skmetric
import sklearn.feature_selection as skfs
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from imblearn.over_sampling import RandomOverSampler
from fancyimpute import SoftImpute
from fancyimpute import SimpleFill
from fancyimpute import KNN
from fancyimpute import MICE
from fancyimpute import MatrixFactorization
import pdb

def preprocess(df):
	top20_countries = ['PH', 'KE', 'PE', 'KH', 'SV', 'NI', 'UG', 'TJ', 'PK', 'EC', 'BO', 'CO',
						'GH', 'PY', 'MX', 'VN', 'NG', 'TG', 'TZ', 'RW']
	top20_partners = [145, 133, 71, 123, 81, 126, 125, 164, 58, 119, 136, 128, 177, 100, 204, 163, 138, 9, 167, 144]
	df.ix[~df.country_code.isin(top20_countries), 'country_code'] = 'OT'
	df.ix[~df.partner_id.isin(top20_partners), 'partner_id'] = 0			
	return df

def train_lr(X_train, y_train):
	print("-----------Training--------------")
	model = sklm.LogisticRegression()
	model.fit(X_train, y_train)
	return model

def predict_lr(model, X_test, y_test):
	print("-----------Predicting--------------")
	pred = model.predict(X_test)
	probs = model.predict_proba(X_test)
	unfunded_prob = probs[:, 1]
	return pred, unfunded_prob

def get_auc_score(y, unfunded_prob):
	print("-----------Evaluating--------------")
	fpr, tpr, thresholds = skmetric.roc_curve(y, unfunded_prob,)
	plt.plot(fpr, tpr, 'k-o')
	plt.ylim(0,1.05)
	plt.yticks(np.arange(0,1.05,0.2))
	plt.title('ROC Curve')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()
	auc_score = skmetric.roc_auc_score(y_true=y==1, y_score=unfunded_prob,)
	return auc_score

def main():
	# predictors = ['lgAmount', 'repayment_term', 'repayment_interval', 'country_code', 'partner_id', 'language', 'description_length', 
	# 'use_length', 'sector', 'video_present', 'sim_desc_classMotivation1', 'sim_desc_classMotivation2', 'sim_desc_classMotivation3', 
	# 'sim_desc_classMotivation4', 'sim_desc_classMotivation5', 'sim_desc_classMotivation6', 'sim_desc_classMotivation7', 
	# 'sim_desc_classMotivation8', 'sim_desc_classMotivation9', 'sim_desc_classMotivation10']
	# predictors = predictors + ['posemo', 'negemo', 'i'] + ['funded_or_not']

	df_cols = pd.read_csv("AllTrain.csv", nrows = 0)
	predictors = df_cols.columns.tolist()

	print("-----------Reading the data--------------")
	# get the training set
	df_train_all = pd.read_csv('AllTrain.csv', encoding='cp1252')
	df_train_incomplete = df_train_all[predictors]
	df_train_incomplete_float = df_train_incomplete.select_dtypes(include=['float'])
	df_train_incomplete_object = df_train_incomplete.select_dtypes(include=['object'])
	df_train_incomplete_other = df_train_incomplete.select_dtypes(exclude=['float', 'object'])
	pdb.set_trace()

	print("------------Filling the missing values------------")
	arr_train_complete_float = SimpleFill().complete(df_train_incomplete_float)
	df_train_float = pd.DataFrame(arr_train_complete_float, columns = df_train_incomplete_float.columns)

	# drop date fields and the filename field
	df_train_incomplete_object = df_train_incomplete_object.drop(['funded_date', 'posted_date'], axis = 1)
	# convert na into a category for string fields
	df_train_object = df_train_incomplete_object.fillna("Missing Value")
	pdb.set_trace()
	
	print("------------Concatenating the dataframes------------")
	df = pd.concat([df_train_float, df_train_object, df_train_incomplete_other], axis = 1)
	# TODO: Add features from the posted date
	X_train = df.drop(['funded_or_not', 'use', 'activity', 'lender_count', 'status', 'num_images'], axis=1)
	X_train = preprocess(X_train)
	X_train = pd.get_dummies(X_train, columns=['repayment_interval', 'country_code', 'partner_id', 'language', 'sector'], sparse=True)
	# X_train_scaled = skp.scale(X_train)
	y_train = df['funded_or_not']
	pdb.set_trace()
	ros = RandomOverSampler(random_state=42)
	# X_train, y_train= ros.fit_sample(X_train, y_train)
	pdb.set_trace()
	# means = [np.mean(X_train[:,i]) for i in range(len(X_train[0]))]
	# std_devs = [np.std(X_train[:,i]) for i in range(len(X_train[0]))]

	# get the validation set
	df_test_all = pd.read_csv('AllValidation.csv', encoding='cp1252')
	df_test = df_test_all[predictors].dropna()
	X_test = df_test[predictors].drop(['funded_or_not', 'use', 'activity', 'lender_count', 'status', 'num_images', 'funded_date', 'posted_date'], axis=1)
	X_test = preprocess(X_test)
	X_test = pd.get_dummies(X_test, columns=['repayment_interval', 'country_code', 'partner_id', 'language', 'sector'], sparse=True)
	# X_test_scaled = skp.scale(X_test)
	y_test = df_test[predictors[-1]]
	y_list = y_test.tolist()
	n_test = len(y_list)
	pdb.set_trace()

	# get the results and evaluation
	beta = 2
	model = train_lr(X_train, y_train)
	pred, unfunded_prob = predict_lr(model, X_test, y_test)
	auc_score = get_auc_score(y_test, unfunded_prob)
	print('Predictors: ', predictors)
	print('AUC Score: ', auc_score)
	tp = len([x for x in range(n_test) if pred[x]==1 and y_list[x]==1])
	tn = len([x for x in range(n_test) if pred[x]==0 and y_list[x]==0])
	fp = len([x for x in range(n_test) if pred[x]==1 and y_list[x]==0])
	fn = len([x for x in range(n_test) if pred[x]==0 and y_list[x]==1])
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	f1 = 2*precision*recall/(precision+recall)
	f1_beta = (1+beta*beta) * (precision*recall) / (beta*beta*precision + recall)
	print("True Positive: ", tp)
	print("False Positive: ", fp)
	print("True Negative: ", tn)
	print("False Negative: ", fn)
	print("Precision: ", precision)
	print("Recall: ", recall)
	print("F1 Score: ", f1)
	print("F1 Beta Score: ", f1_beta)

	"""
	Feature Importance
	"""
	# cols = X_test.columns
	# coefs = model.coef_[0]
	# econ_significance = [[cols[i], coefs[i]*std_devs[i]] for i in range(len(cols))]
	# pd.DataFrame(econ_significance).to_csv('economic_significance.csv')
	# feat_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=1)
	# feat_selector.fit(X_test, y_test)
	# pdb.set_trace()
	# mi = mutual_info_classif(X_test, y_test)
	# df_summary = pd.DataFrame([X_test.columns, model.coef_[0], mi])
	# df_summary.transpose().to_csv('summary.csv')
	# scores, pvalues = skfs.chi2(X_train, y_train)
	# df_pval = pd.DataFrame([[cols[i], pvalues[i]] for i in range(len(cols))])
	# df_pval.to_csv("stat_significance.csv")
	pdb.set_trace()

main()