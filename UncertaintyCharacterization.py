"""
Filename: UncertaintyCharacterization.py
Author: Siya Kunde
Date: 2024-08-13
Description:
	Uncertainty Characterization method based on clustering
"""

import numpy as np
import pandas as pd

# Use a function to define the mask to create a subset of the data frame
def mask(df, key, value):
	return df[df[key] == value]

# Use a function to define the mask to create a subset of the data frame
def unmask(df, key, value):
	return df[df[key] != value]

pd.DataFrame.mask = mask
pd.DataFrame.unmask = unmask

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

INTERVAL_COLOR = 'silver'
VALIDATION_COLOR = 'mediumaquamarine'
TEST_COLOR = 'lightcoral'
TEST_GROUND_TRUTH = 'teal'

class UncertaintyCharacterization:
	"""
	Method used to characterize uncertainty in regression tasks. 
	Attributes
	----------
	
	Methods
	-------
	fit(X, y)
		Fits the Random Forest Regressor with the given training data.
		Fits the Agglomerative Clustering with the given validation data (and errors computed from Random Forest Regression prediction).
		Fits the Random Forest Classifier on the validation data and its cluster labels.
	
	predict(X)
		Uses the Random Forest Classifier to compute the predicted classes for test data.
	
	score(X, y, alpha=0.05)
		Uses the predictions on test data to provide uncertainty intervals and other metrics as second degree of uncertainty.
	"""
	
	def __init__(self, n_clusters=12, linkage='average', feature_names_in=[], ml_model=None, random_state=42,
			  regression_params={'n_estimators':100}, clustering_params={'n_clusters':12, 'linkage':'average'}, classification_params={'n_estimators':100},
			  X_train=None, X_val=None, X_test=None, y_train=None, y_val=None, y_test=None, 
			  y_pred_train=None, y_pred_val=None, y_pred_test=None, errors_train=None, errors_val=None, errors_test=None):
		"""
		Parameters
		----------
		n_clusters : integer
			Number of clusters that the method will detect
		linkage : string
			The linkage parameter for the Agglomerative Clustering algorithm
		feature_names_in : array of shape (num_features,)
			The feature names for the dataset
		ml_model : Regressor model object
			Pre-trained regression model
		random_state : integer, optional
			Parameter used to ensure reproducibility on the random processes
		"""

		self.fit_complete = False
		self.predict_complete = False
		
		self.regression_params = regression_params
		self.clustering_params = clustering_params
		self.classification_params = classification_params
		self.feature_names_in = feature_names_in
		self.random_state = random_state

		self.ml_model = ml_model
		self.regressor = ml_model
		self.clusterer = None
		self.classifier = None

		self.X_train = X_train
		self.X_val = X_val
		self.X_test = X_test

		self.y_train = y_train
		self.y_val = y_val
		self.y_test = y_test

		self.y_pred_train = y_pred_train
		self.y_pred_val = y_pred_val
		self.y_pred_test = y_pred_test

		self.errors_train = errors_train
		self.errors_val = errors_val
		self.errors_test = errors_test

		self.labels_val = None
		self.labels_test = None

		self.output_data = None

		self.rg_r2_val = None
		self.rg_accuracy_val = None
		self.rg_r2_test = None
		self.rg_accuracy_test = None
		self.clr_ch_index = None
		self.clr_db_score = None
		self.clr_silhouette_score = None
		
		self.lb = None
		self.ub = None
		self.interval = None
		self.marker = None
		self.inside_p = None
		self.outside_p = None
		self.si_a = None
		self.si_wta = None
		self.rsi_a = None
		self.rsi_wta = None

	def fit(self, X, y):
		"""
		Fits the Regressor with the training data.
		Fits the Clustering with the given validation data (and errors computed from Regression prediction).
		Fits the Classifier on the validation data and its cluster labels.
		
		Parameters
		----------
		X : array of shape (num_samples, num_features)
			Input data for fitting the Clustering
		y : array of shape (num_samples,)
			Target values for each sample in the input data
		
		Returns
		-------
		self : object
			Fitted estimator.
		"""
		if(self.X_train is None and self.y_train is None):
			self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
		else:
			self.X_val = X
			self.y_val = y

		if(self.regressor is None):
			print("Fitting regression model on training data.")
			self.regressor = self.__fit_regressor(self.X_train, self.y_train)
			print("Obtaining regression predictions for training data.")
			self.y_pred_train, self.errors_train = self.__predict_regressor(self.X_train, self.y_train)
			print("Obtaining regression predictions for validation data.")
			self.y_pred_val, self.errors_val = self.__predict_regressor(self.X_val, self.y_val)
		else:
			print("Model provided. Regressor not fit.")

		print("Fitting clustering model on validation data.")
		self.clusterer, self.labels_val = self.__fit_agglomerative_clustering(self.X_val, self.errors_val)
		print("Fitting classification model on validation data.")
		self.classifier = self.__fit_classifier(self.X_val, self.labels_val)

		self.fit_complete = True

		return self
	
	def __fit_regressor(self, X, y):
		'''
		Fits the Random Forest Regressor with the given training data. 
		Private method.
		'''
		regressor = RandomForestRegressor(**self.regression_params, random_state=self.random_state)
		regressor.fit(X, y)
		
		return regressor
	
	def __predict_regressor(self, X, y):
		
		y_pred = self.regressor.predict(X)
		errors = abs(y_pred - y)

		return y_pred, errors
	
	def __score_regressor(self, X, y, y_pred):
		if(self.ml_model is None):
			# Performance metrics
			errors = abs(y_pred - y)

			# print('Average absolute error:', round(np.mean(errors), 4), 'degrees.')

			# Calculate mean absolute percentage error (MAPE)
			mape = 100 * (errors / y)

			# Calculate and display accuracy
			accuracy = 100 - np.mean(mape)
			# print('Accuracy:', round(accuracy, 2), '%.')

			r2 = self.regressor.score(X, y)
			# print('R^2:', r2)
		else:
			accuracy = np.nan
			r2 = np.nan

		return r2, accuracy
	
	def __fit_agglomerative_clustering(self, X, errors):
		'''
		Fits the Agglomerative Clustering with the given validation data (and errors computed from Random Forest Regression prediction).
		Private method.
		'''
		adjacency_matrix = self.__get_graph(errors)
		clusterer = AgglomerativeClustering(**self.clustering_params, connectivity=adjacency_matrix).fit(X)
		labels = clusterer.labels_

		return clusterer, labels
	
	def __score_clusterer(self, X, labels):
		return metrics.calinski_harabasz_score(X, labels), metrics.davies_bouldin_score(X, labels), metrics.silhouette_score(X, labels)

	def __get_graph(self, errors, round_to = 5):
		'''
		Creates a graph from validation data's regression errors.
		Private method.
		'''
		dataDF = pd.DataFrame(dict(errors=errors, idx=list(range(0, len(errors)))))
		# Create a bubble linear graph
		dataDF = dataDF.round({'errors': round_to})
		vals = dataDF['errors'].unique()
		vals = sorted(vals)
		G = nx.Graph()
		G.add_nodes_from(dataDF['idx'].values)
		prev_nodes = []
		for v in vals:
			if(len(dataDF.mask('errors', v)) > 0):
				nodes = list(dataDF.mask('errors', v)['idx'].values)
				if(len(nodes) > 1):
					H = nx.complete_graph(nodes)
					G = nx.compose(G, H)
				if(len(prev_nodes) > 0):
					all_nodes = nodes + prev_nodes
					H = nx.complete_graph(all_nodes)
					G = nx.compose(G, H)

				prev_nodes = nodes
		
		adjacency_matrix = nx.adjacency_matrix(G).A

		return adjacency_matrix
	
	def __fit_classifier(self, X, labels):
		'''
		Fits the Random Forest Classifier on the validation data and its cluster labels.
		Private method.
		'''
		classifier = RandomForestClassifier(**self.classification_params, random_state=self.random_state)
		classifier = classifier.fit(X, labels)

		return classifier
	
	def __predict_classifier(self, X):
		'''
		Uses the Random Forest Classifier to compute the predicted classes for test data.
		Private method.
		'''
		return self.classifier.predict(X)
	
	def predict(self, X):
		'''
		Uses the Random Forest Classifier to compute the predicted classes for test data.
		
		Parameters
		----------
		X : array of shape (num_samples, num_features)
			Input data for making the predictions
		
		Returns
		----------
		labels : array of shape (num_samples,)
			The predicted class for each sample in X
		'''
		if(not self.fit_complete):
			self.fit(self.X_val, self.y_val)
		
		if(self.X_test is None):
			self.X_test = X
		print("Obtaining classification predictions for test data.")
		self.labels_test = self.__predict_classifier(self.X_test)

		self.predict_complete = True

		return self.labels_test
	
	def score(self, X, y, alpha=0.05):
		'''
		Uses the predictions on test data to provide uncertainty intervals and other metrics as second degree of uncertainty.
		
		Parameters
		----------
		X : array of shape (num_samples, num_features)
			Input data for computing features
		y : array of shape (num_samples,)
			Target values for each sample in the input data
		alpha : float, optional
            The maximum rate of incorrect predictions made by the method
		
		Returns
		----------
		rg_r2_val : float
			R^2 score of regression for validation data.
		rg_accuracy_val : float
			Accuracy score of regression for validation data.
		rg_r2_test : float
			R^2 score of regression for test data.
		rg_accuracy_test : float
			Accuracy score of regression for test data.
		clr_ch_index : float
			CH index of clustering.
		clr_db_score : float
			DB score of clustering.
		clr_silhouette_score : float
			Silhouette score of clustering.
		lb : array of shape (num_samples,)
			Lower bound of the uncertainty interval for each sample in the input data.
		ub : array of shape (num_samples,)
			Upper bound of the uncertainty interval for each sample in the input data.
		interval : array of shape (num_samples,)
			The uncertainty interval for each sample in the input data.
		inside_p : array of shape (num_samples,)
			% of samples within confidence level.
		outside_p : array of shape (num_samples,)
			% of samples outside of confidence level.
		si_a : array of shape (num_samples,)
			Size of interval average.
		si_wta : array of shape (num_samples,)
			Size of interval weighted average.
		rsi_a : array of shape (num_samples,)
			Relative size of interval average.
		rsi_wta : array of shape (num_samples,)
			Relative size of interval weighted average.
		'''

		self.alpha = alpha

		if(not self.predict_complete):
			self.predict(self.X_test)
		
		if(self.y_test is None):
			self.y_test = y
			self.y_pred_test, self.errors_test = self.__predict_regressor(self.X_test, self.y_test)

		self.__generate_prediction_uncertainties()
		dataDF = self.get_all_data()

		val_all = dataDF.mask('error_type', 'val').reset_index()
		test_all = dataDF.mask('error_type', 'test').reset_index()

		num_items_test_all = len(test_all)

		self.marker = dataDF['UC_P'].values

		self.inside_p = len(test_all.mask('UC_P', 'o').values)/num_items_test_all

		self.outside_p = len(test_all.mask('UC_P', '*').values)/num_items_test_all

		# Regression scores with validation data
		self.rg_r2_val, self.rg_accuracy_val = self.__score_regressor(self.X_val, self.y_val, self.y_pred_val)

		# Regression scores with test data
		self.rg_r2_test, self.rg_accuracy_test = self.__score_regressor(self.X_test, self.y_test, self.y_pred_test)

		# Clustering score with validation data
		self.clr_ch_index, self.clr_db_score, self.clr_silhouette_score = self.__score_clusterer(self.X_val, self.labels_val)

		iy = np.percentile(val_all['ground_truth'].values, 95) - np.percentile(val_all['ground_truth'].values, 5)
		print(np.percentile(val_all['ground_truth'].values, 95), np.percentile(val_all['ground_truth'].values, 5), iy)
		# Size of Interval Average
		self.si_a = sum(test_all['PInterval'])/num_items_test_all

		# Size of Interval Weighted Average
		test_all_grp = test_all.groupby('labels', as_index=False).agg({
			'PInterval' : ['sum', 'count']
		})
		test_all_grp.columns = ['_'.join(i).rstrip('_')
			for i in test_all_grp.columns.values]
		self.si_wta = sum(test_all_grp['PInterval_sum']/test_all_grp['PInterval_count'])

		# Relative Size of Interval Average
		self.rsi_a = self.si_a/iy

		# Relative Size of Interval Weighted Average
		self.rsi_wta = self.si_wta/iy

		return self.rg_r2_val, self.rg_accuracy_val, self.rg_r2_test, self.rg_accuracy_test, self.clr_ch_index, self.clr_db_score, self.clr_silhouette_score, self.lb, self.ub, self.interval, self.inside_p, self.outside_p, self.si_a, self.si_wta, self.rsi_a, self.rsi_wta
	
	def __generate_prediction_uncertainties(self):
		dataDF = pd.DataFrame(dict(
			data_id=np.concatenate([np.arange(0, len(self.X_val), 1), np.arange(0, len(self.X_test), 1)]), 
			ground_truth=np.concatenate([self.y_val, self.y_test]), 
			prediction=np.concatenate([self.y_pred_val, self.y_pred_test]), 
			errors=np.concatenate([self.errors_val, self.errors_test]), 
			error_type=(['val'] * len(self.errors_val))+(['test'] * len(self.errors_test)),
			labels=np.concatenate([self.labels_val,self.labels_test])))
		
		p = {}
		
		for i in range(0, self.clustering_params['n_clusters']):
			if(i in dataDF.mask('error_type', 'val')['labels'].values):
				validation_errors = dataDF.mask('error_type', 'val').mask('labels',i)['errors'].values
				p[i] = np.percentile(validation_errors, 100-self.alpha)

		def mark(x, lb, ub):
			lower_bound = x[lb]
			upper_bound = x[ub]
			if x['ground_truth'] > (upper_bound) or x['ground_truth'] < (lower_bound):
				return '*'
			else:
				return 'o'
		
		def getPLB(x, p):
			plb = x['prediction']-p[x['labels']]
			return plb
		
		def getPUB(x, p):
			pub = x['prediction']+p[x['labels']]
			return pub

		dataDF['PLB'] = dataDF.apply(lambda x: getPLB(x, p), axis=1)
		dataDF['PUB'] = dataDF.apply(lambda x: getPUB(x, p), axis=1)
		dataDF['PInterval'] = dataDF['PUB'] - dataDF['PLB']

		dataDF['UC_P'] = dataDF.apply(lambda x: mark(x, 'PLB', 'PUB'), axis=1)

		test_all = dataDF.mask('error_type', 'test').reset_index()

		self.lb = test_all['PLB'].values
		self.ub = test_all['PUB'].values
		self.interval = test_all['PInterval'].values

		self.output_data = dataDF

	def get_all_data(self):
		return self.output_data
	
	def get_clustering_classification_image(self, out_path=None):
		df_temp = self.output_data
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

		sns.boxplot(data=df_temp.mask('error_type', 'val'), x='labels', y='errors', boxprops={'facecolor':'None'}, ax=ax1)
		ax1.set_title("Validation data clustering output")
		ax1.set_xlabel('Clusters')
		ax1.set_ylabel('Errors')

		sns.boxplot(data=df_temp.mask('error_type', 'test'), x='labels', y='errors', boxprops={'facecolor':'None'}, ax=ax2)
		ax2.set_title("Test data classification output")
		ax2.set_xlabel('Clusters')
		ax2.set_ylabel('Errors')
		if(out_path is not None):
			plt.savefig(out_path)

	def get_regression_intervals_image(self, target, out_path=None):
		ax_min = min(min(self.y_val), min(self.y_pred_val), min(self.y_test), min(self.y_pred_test))
		ax_max = max(max(self.y_val), max(self.y_pred_val), max(self.y_test), max(self.y_pred_test))

		dataDF = self.get_all_data()

		val_all = dataDF.mask('error_type', 'val').reset_index()
		test_all = dataDF.mask('error_type', 'test').reset_index()

		val_in = val_all.mask('UC_P', 'o')
		val_out = val_all.mask('UC_P', '*')
		
		test_in = test_all.mask('UC_P', 'o')
		test_out = test_all.mask('UC_P', '*')

		fig = plt.subplots(figsize=(24, 8))

		plt.subplot(1, 2, 1)
		plt.scatter(self.y_val, # Ground Truth
				self.y_pred_val, # Prediction
				marker='o', color=VALIDATION_COLOR, alpha=0.2, label='validation')
		plt.scatter(self.y_test, # Ground Truth
				self.y_pred_test, # Prediction
				marker='o', color=TEST_COLOR, alpha=0.2, label='test')
		plt.axline([0,0],[1,1], color='k', ls=':')
		plt.title("Validation and test data for "+target)
		plt.xlabel('Ground Truth')
		plt.ylabel('Prediction')
		plt.xlim(ax_min,ax_max)
		plt.ylim(ax_min,ax_max)
		plt.legend()

		df_temp = pd.DataFrame(dict(prediction=self.y_pred_test, ground_truth=self.y_test, error=self.errors_test, interval=self.interval, lb=self.lb, ub=self.ub, marker=test_all['UC_P'].values))
		df_temp['lb_gt'] = df_temp['lb'] - df_temp['prediction'] + df_temp['ground_truth']
		df_temp['ub_gt'] = df_temp['ub'] - df_temp['prediction'] + df_temp['ground_truth']
		df_temp = df_temp.sort_values(by=['ground_truth']).reset_index()
		test_in = df_temp.mask('marker', 'o')
		test_out = df_temp.mask('marker', '*')
		plt.subplot(1, 2, 2)
		plt.scatter(df_temp.index.values, 
				df_temp['ground_truth'].values, # Ground Truth 
				marker='o', color=TEST_GROUND_TRUTH, alpha=0.2, label='test (ground truth)')
		plt.scatter(test_in.index.values, 
				test_in['prediction'].values, 
				marker='o', color=TEST_COLOR, alpha=0.2, label='test')
		plt.scatter(test_out.index.values, 
				test_out['prediction'].values, 
				marker='*', color='k', alpha=0.2, label='test outside threshold')
		plt.fill_between(df_temp.index.values, 
				df_temp['lb_gt'].values, # Lower bound values
				df_temp['ub_gt'].values, # Upper bound values
				alpha=0.2, label="Prediction Intervals", color=INTERVAL_COLOR)
		plt.title("Uncertainty analysis for test data (using 95th percentile of validation data)")
		plt.xlabel('Test Item Number (ascending order of ground truth)')
		plt.ylabel(target)
		plt.ylim(ax_min,ax_max)
		plt.legend()

		if(out_path is not None):
			plt.savefig(out_path)

	def get_intervals_image(self, target, out_path=None):
		ax_min = min(min(self.y_val), min(self.y_pred_val), min(self.y_test), min(self.y_pred_test))
		ax_max = max(max(self.y_val), max(self.y_pred_val), max(self.y_test), max(self.y_pred_test))

		dataDF = self.get_all_data()

		test_all = dataDF.mask('error_type', 'test').reset_index()
		
		test_in = test_all.mask('UC_P', 'o')
		test_out = test_all.mask('UC_P', '*')

		fig = plt.figure(figsize=(12, 8))

		df_temp = pd.DataFrame(dict(prediction=self.y_pred_test, ground_truth=self.y_test, error=self.errors_test, interval=self.interval, lb=self.lb, ub=self.ub, marker=test_all['UC_P'].values))
		df_temp['lb_gt'] = df_temp['lb'] - df_temp['prediction'] + df_temp['ground_truth']
		df_temp['ub_gt'] = df_temp['ub'] - df_temp['prediction'] + df_temp['ground_truth']
		df_temp = df_temp.sort_values(by=['ground_truth']).reset_index()
		test_in = df_temp.mask('marker', 'o')
		test_out = df_temp.mask('marker', '*')

		plt.scatter(df_temp.index.values, 
				df_temp['ground_truth'].values, # Ground Truth 
				marker='o', color=TEST_GROUND_TRUTH, alpha=0.2, label='test (ground truth)')
		plt.scatter(test_in.index.values, 
				test_in['prediction'].values, 
				marker='o', color=TEST_COLOR, alpha=0.2, label='test')
		plt.scatter(test_out.index.values, 
				test_out['prediction'].values, 
				marker='*', color='k', alpha=0.2, label='test outside threshold')
		plt.fill_between(df_temp.index.values, 
				df_temp['lb_gt'].values, # Lower bound values
				df_temp['ub_gt'].values, # Upper bound values
				alpha=0.2, label="Prediction Intervals", color=INTERVAL_COLOR)
		plt.title("Uncertainty analysis for test data (using 95th percentile of validation data)")
		plt.xlabel('Test Item Number (ascending order of ground truth)')
		plt.ylabel(target)
		plt.ylim(ax_min,ax_max)
		plt.legend()

		if(out_path is not None):
			plt.savefig(out_path)
