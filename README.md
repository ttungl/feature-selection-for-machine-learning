# Feature Selection for Machine Learning

**Udemy** <br/>
[Soledad Galli](https://www.udemy.com/user/soledad-galli/)

## Section 1: Intro

### Feature selection methods:

#### Filter methods
1. *Basics*
- Constant
- Quasi-constant
- Duplicated

2. *Correlation*

3. *Statiscal measures*
- Fisher score
- Univariate methods
- Mutual information

4. Alternative filter methods

#### Wrapper methods
1. Step forward selection
2. Step backward selection
3. Exhaustive search
4. Feature Shuffling

#### Embedded methods
1. LASSO
2. Decision tree derived importance
3. Regression coefficients

#### Hybrid methods
1. Recursive feature elimination

### Course requirements

#### Machine Learning
1. Linear and Logistic regression
2. Random Forest Trees
3. Gradient Boosted Trees (XGB)
4. Diagnostics: ROC-AUC, mse

#### References
1. kaggle
2. kdd.org/kdd-cup

#### Useful links:
1. [Best resources to learn machine learning](https://www.trainindata.com/post/best-resources-to-learn-machine-learning)
2. [Best resources to learn python for data science](https://www.trainindata.com/post/best-resources-to-learn-python-for-data-science)
3. [Harvard CS109A-2018](https://harvard-iacs.github.io/2018-CS109A/category/lectures.html)
4. [Harvard CS109A-2019](https://harvard-iacs.github.io/2019-CS109A/category/lectures.html)
5. [Harvard CS109B-2018](https://harvard-iacs.github.io/2018-CS109B/category/lectures.html)
6. [Harvard CS109B-2019](https://harvard-iacs.github.io/2019-CS109B/category/lectures.html)

## Section 2: Feature Selection

### Definition
**Feature selection** is the process of selecting a subset of relevant features (variables, predictors) for use in machine learning model building.

**Why should we select features?**
- Simple models are easier to interpret.
- Shorter training times.
- Enhanced generalization by reducing overfitting.
- Easier to implement by software developer.
- Reduced risk of data error during model use.
- Variable redundancy.
	- As the features within dataset are highly correlated, which means that they have essentially the same information, so they are redundant. We can keep one and remove the rest without losing information.

- Bad learning behaviour in high dimensional spaces.
	- ML algorithms, specifically tree-based algorithms, are favored by reduced feature spaces. This means, high dimension causes worst performance in tree-based methods and so reducing the feature space helps build more robust and predictive models.

**Procedure**
- A feature selection algorithm can be seen as the combination of a search technique for proposing new feature subsets, along with an evaluation measure which scores the different feature subsets.
	- Computationally expensive.
		- The feature selection method will search through all the possible subsets of feature combinations that can be obtined from a given dataset, and find the feature combination that produces the best ML model performance.
	- Different feature subsets render optimal performance for different ML algorithms.
		- There is not only one subset of features but many subset of optimal features depending on the machine learning algorithms that we intend to use.
		- Different methods of feature selection have been developed to try and accomodate as many limitations.
- **Feature selection algorithm** can be divided into three main categories, including filter methods, wrapper methods, embedded methods.
	- *Filter methods*:
		- Rely on the characteristics of the data (feature characteristics)
		- Do not use ML algorithms.
			- These methods only evaluate the features, and make a selection based on the feature characteristics.
		- Model agnostic
		- Tend to be less computationally expensive.
		- Usually give lower prediction performance than wrapper or embedded methods.
		- Are very well-suited for a quick screen and removal of irrelevant features.
		- **Methods**:
			- Variance
			- Correlation
			- Univariate selection:
				- Two-step procedure:
					- Rank features according to a certain criteria
						- Each feature is ranked independently of the feature space.
							- Feature scores on various statistical tests:
								- Chi-square or Fisher score
								- Univariate parametric tests (anova)
								- Mutual information
								- Variance
									- Constant features
									- Quasi-constant features
									
					- Select the highest ranking features.
						- May select redundant variables because they don't consider the relationships between features. 
			- Multivariate selection:
				- Handle redundant feature
				- Scanning for duplicated features
				- Correlated features
				- *Notes:* simple yet powerful methods to quickly screen and remove irrelevant and redundant features. First step in feature selection procedures/pipelines.


	- *Wrapper methods*:
		- Use  predictive ML models to score the feature subset.
		- Train a new model on each feature subset, then select the subset of variables that produces the highest performing algorithm. Therefore, it will build simpler ML models at each round of feature selection.
			- Downside: 
				- Tend to be very computationally expensive.
				- They may not produce the best feature combination for a different ML model.
			- Upside: Usually provide the best performing feature subset for a give ML algorithm.
		- In practice, if we used tree-based/derived methods to select the feature with wrapper methods, ie. gradient boost trees to select features, the optimal set of features selected with this algorithm most likely will produce a good performance for other tree-based algorithms like random forest. However, this set of features may not provide the best performance for logistic regression. Therefore, we need to keep in mind that when using wrapper methods, we also intend to choose which models we'll build with the selected features.
		- **Methods**:
			- Forward selection
			- Backward selection
			- Exhausive search

		- **Details**:
			- Evaluate the quality of features in the light of a specific ML algorithm.
			- Evaluate subsets of variables simultaneously.
			- *Advantages:*
				- Detect interactions between variables.
				- Find the optimal feature subset for the desired classifier.
		- Procedure:
			- Search for a subset of features
			- Build a machine learning model on the selected subset of features.
			- Evaluate model performance.
			- Repeat
		- Follow-up:
			- How to search for the subset of features?
				- Forward feature selection: adds 1 feature at a time in each iteration, the feature which best improves the ML model performance until a predefined criterion is met.
				
				- Backward feature elimination: removes 1 feature at a time. We start with all features, and remove the least significant feature at each iteration until a criteria is met.

				- Exhaustive feature search: searches across all possible feature combinations. It aims to find the best performing features subset. It builds all possible combinations of features from 1 to `n`, where `n` is the number of features, and it creates a ML model for each of these combinations, and finally, it selects the combination of features that performs the best.
					- Greedy algorithms
						- Computationally expensive.
						- Impractical. 

			- How to stop the search?
				- Performance not increase
				- Performance not decrease
				- Predefined number of features is reached.
				- *Notes:* stopping criteria are somewhat arbitrary and to be determined by user.

		- Summary:
			- Better predictive accuracy than filter methods.
			- Best performing feature subset for the predefined classifier
			- Computationally expensive
			- Stopping criterion is relatively arbitrary.

	- *Embedded methods*:
		- Performance feature selection as part of the model construction process or during the modeling algorithm's execution.
			- By combining feature selection with classifier or regressor construction, these methods have the advantages of wrapper methods.
		- Consider/detect the interactions between features and models.
		- They are less computationally expensive and faster than wrapper methods, because they fit the ML model only once. These methods also are more accurate than filter methods.
		- Find the feature subset for the algorithm being trained.

		- **Methods**:
			- LASSO for linear regression model.
			- Tree importance 
			- Regression coefficients

		- **Procedure**:
			- Train a ML algorithm
			- Derive the feature importance
			- Remove non-important features.

		- Summary:
			- Better predictive accuracy than filter methods.
			- Faster than wrapper methods
			- Render generally good feature subsets for the used algorithm
			- Downside:
				- Constrained to the limitations of the algorithm.

## Section 3: Filter methods | Basics

### Constant, quasi-constant, duplicated features - Intro

- This is the first step for any ML modeling practice. They provide quick and easy sanity checks for the variables that will immediately allow you to reduce the feature space and get rid of unuseful features. Note that the duplicated features may arise after one-hot encoding of categorical variables. There are many datasets that present constant, quasi-constant, duplicated features and if we remove them, it will make the ML modeling much simpler. 

#### Constant features 
- Constant features are those that show only one value for all the observations in the dataset (same value for that variable).
	- Using variance threshold from sklearn.
		- Simple baseline approach to feature selection where it removes all features which variance doesn't meet some threshold. By default, it removes all zero-variance features, i.e. features that have the same value in all observations.
		- VarianceThreshold(threshold=0) 
		```python
		from sklearn.feature_selection import VarianceThreshold
		sel = VarianceThreshold(threshold=0)
		# fit finds the features with zero variance.
		sel.fit(X_train) 
		# get_support() method returns which features are retained.
		retained_features = sel.get_support()
		```
	- Coding ourselves.
		- Basically check if standard deviation of the feature values is zero.
	- Removing constant features for categorical variables.
		- Check if unique value of the feature values == 1.

#### Quasi-constant features
- Quasi-constant features are those where a single value is shared by the major observations in the dataset. It's varied but typically, more than 95-99 percent of the observations will present the same value in the dataset. It's up to you to decide the cutoff to call the feature quasi-constant.
	- Using variance threshold from sklearn. If threshold is 0.01, meaning the method will drop the feature if 99% of the observations represent the same value in the dataset.  

#### Duplicated features
- Duplicated features are those that in essence are the same. When two features in the dataset show the same value for all the observations, they are in essence the same feature. It introduces duplicated features after performing one-hot encoding of categorical  variables when using several highly cardinal variables. So the information of one in two is redundant. Keep in mind that duplicated features may arise after some process that generates new features from existing one like one-hot encoding, these variables can end up with several identical binary features. Therefore, checking duplicated features provide a good way to get rid of them.
	- **Small dataset**
		- Pandas has the function `duplicated` that evaluates if the dataframe contains duplicated rows. So we can use this function for checking duplicated columns if we transpose the dataframe where the columns are now rows, and leverage this function to identify those duplicated rows, which actually are duplicated columns. 
			- ```python
				# transposed dataframe
				data_t = X_train.T
				# get duplicated dataframe
				duplicated_features = data_t[data_t.duplicated()]
				# get duplicated columns name
				duplicateFeatsIndex = duplicated_features.index.values
				# get unique dataframe without duplication and transpose back to the variables as the columns, 
				# keep first of the sets of duplicated variables.
				data_unique = data_t.drop_duplicates(keep='first').T
				```
	- **Big dataset**
		- Transposing a big dataframe is memory-intensive, therefore, we use the alternative loop to find duplicated columns in big datasets.
		- This procedure takes O(n^2). 	
			```python
			for each column C[i] of data columns
				for each column C[i+1:] until the end.
					if values are equal in comparable columns C[i] and C[i+1] 
						then duplicated columns are found. 
			```

## Section 4: Filter methods | Correlation

### Correlation

#### Definition
- Correlation is a measure of the linear relationship of two or more variables. 
- Through correlation, we can predict one variable from the other. Good variables are highly correlated with the target.
- Correlated predictor variables provide redundant information. 
	- Variables should be correlated with the target but uncorrelated among themselves.

#### Correlation feature selection
- The central hypothesis is that good feature sets contain features that are highly correlated with the target, yet uncorrelated with each other.

#### Correlation and machine learning
- Correlated features do not necessarily affect model accuracy by itself, but high dimensionality does.
- If two features are highly correlated, the second one will add little information over the previous one. So, removing it will help reduce dimension.
- Correlation affects model interpretability: linear models.
	- If two variables are correlated, the linear models will fit coefficients to both variables that somehow capture the correlation, therefore, these will be misleading on the true importance of each individual features.
	- This is also true for ensemble tree models, if two features are correlated, tree methods will assign roughly the same importance to both but half of the importance they would assign if we had only one of the correlated features in the dataset.
	- Therefore, removing correlated features improves both the machine learning models by making them simpler with less variables and yet similar predictive performance. It also helps the interpretability of the model as it preserves the relationship of the feature with the target by removing the interaction with the correlated variable. 

- Different classifiers show different sensitivity to correlation.
	- Typically, linear models are more sensitive and tree methods are quite robust to correlate the features.

#### How to determine correlation

- **Pearson's correlation coefficient**
	```python
	sum((x1 -x1.mean) * (x2 - x2.mean) * (xn - xn.mean)) / var(x1) * var(x2) * var(xn)
	```  
	- Pearson's coefficient values vary between [-1,1]
		- `1` is highly correlated: the more of variable x1, the more of x2.
		- `-1` is highly anti-correlated: the more of variable x1, the less of x2. 

- **Procedure for correlation selection**
	- First approach is a Brute force function, that finds correlated features without any further insight.  
	- Second approach finds groups of correlated features. Often, more than two features are correlated with each other. We can find groups of 3, 4, or more features that are correlated. By identifying these groups, we can select from each group, which feature we want to keep or remove.

	- Code review
		- In practice, feature selection should be done after data preprocessing, so ideally, all the categorical variables are encoded into numbers, and you can assess whether they are correlated with other. So, filtering all numerical variables columns.
		```python
		numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
		numerical_vars = list(data.select_dtypes(include=numerics).columns)
		numerical_data = data[numerical_vars]
		```

		- Visualize correlated features.
			- Correlation matrix, which examines the correlation of all features (for all possible feature combinations and then visualize the correlation matrix).
			```python
			corrmat = X_train.corr()
			# plot
			fig, ax = plt.subplots()
			fig.set_size_inches(11,11)
			sns.heatmap(corrmat)
			```
			- Notes: the white squares are associated with highly correlated features (> 0.8). The diagonal represents the correlation of the feature with itself.


	- **Brute Force approach**: we can select highly correlated features by removing the first feature that is correlated to anything else. 

		```python
		def correlation(df, threshold):
			col_corr = set()
			corrmat = df.corr()
			for i in range(len(corrmat.columns)):
				for j in range(i):
					# interested in abs coefficient values
					if abs(corrmat.iloc[i, j]) > threshold:
						col_corr.add(corrmat.columns[i]) 
			return col_corr
		```
		
		- Then the set of correlated features in `col_corr` are highly correlated with other features in the training set. Removing these features will drop very little of your ML models performance.

		```python
		corr_feats = correlation(X_train, 0.8)
		X_train.drop(labels=corr_feats, axis=1, inplace=True)
		X_test.drop(labels=corr_feats, axis=1, inplace=True)
		```

	- **Second approach**: identifies groups of highly correlated features. Then we can make further investigation within these groups to decide which feature we can keep or remove.
		- Build a dataframe with the correlation between features. Note that the absolute value of the correlation coefficient is important and not the sign.
		```python
		corrmat = X_train.corr()
		corrmat = corrmat.abs().unstack() # absolute value of corr coef. # series.
		corrmat = corrmat.sort_values(ascending=False)
		corrmat = corrmat[(corrmat >=0.8) & (corrmat < 1)]
		corrmat = pd.DataFrame(corrmat).reset_index()
		corrmat.columns = ['feat1', 'feat2', 'corr']
		```
		- Then we find groups of correlated features.
		```python
		grouped_feat = []
		corr_groups = []
		for feat in corrmat.feat1.unique():
			if feat not in grouped_feat:
				# find all feats correlated to a single feat.
				corr_block = corrmat[corrmat.feat1 == feat]
				grouped_feat += list(corr_block.feat2.unique()) + [feat]
				# append the block of feats to the list
				corr_groups.append(corr_block)
		
		# found 32 correlated groups [corr_groups]
		# out of 112 total feats in X_train.
		```

		- Alternatively, we can build a ML algorithm using all the features from the corr_groups and select the predictive one.
		```python
		from sklearn.ensemble import RandomForestClassifier
		group = corr_groups[2]
		feats = list(group.feat2.unique()) + ['v17']
		rf_clf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)
		rf_clf.fit(X_train[feats].fillna(0), y_train)
		``` 
		- We get the feature importance attributed by the RF model.
		```python
		importance = pd.concat([pd.Series(feats),
								pd.Series(rf_clf.feature_importances_)], axis=1)
		importance.columns = ['feature', 'importance']
		importance.sort_values(by='importance', ascending=False)
		
		# output:
		feature	importance
		2	v48	0.173981
		3	v93	0.154484
		6	v101	0.129764
		1	v64	0.118110
		7	v17	0.117571
		4	v106	0.113958
		0	v76	0.108071
		5	v44	0.084062
		```
		- As a result, `v48` appears to the top of this correlated group according to the random forests model, so we should select `v48` and remove the rest of features in this group from the dataset. 






### Basic methods plus correlation pipeline



## Section 5: Filter methods | Statistical measures

## Section 6: Wrapper methods

## Section 7: Embedded methods | Lasso regulization

## Section 8: Embedded methods | Linear models

## Section 9: Embedded methods | Trees

## Section 10: Reading resources

## Section 11: Hybrid feature selection methods

## Section 12: Final section and next steps
