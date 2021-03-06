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
		importance = pd.concat([pd.Series(feats), pd.Series(rf_clf.feature_importances_)], axis=1)
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

	- Notes:
		- None of two approaches for removing correlated features are perfect, it's necessary to double check after the processes.


## Section 5: Filter methods | Statistical measures

### Statistical and ranking methods

1. Information Gain
2. Fisher score
3. Univariate tests
4. Univariate ROC-AUC/RMSE

#### Two steps:
- Rank features based on certain criteria/metrics.
	- This is the statistical test, ranks the features based on certain criteria. Each feature is ranked independently of the other features based on their interaction or relationship with the target.
- Select features with highest rankings.
	- The feature with the highest rankings are chosen to be in the classification or regression models. 
	- How many of the highest ranking features to select is arbitrary and usually be limited by the user.
- Take-aways:
	- Pros: Fast
	- Cons: Doesn't foresee feature redundancy
		- You would have to screen for duplicated and correlated features in previous steps. 
		- Also, these selection procedures do not foresee feature interaction. This means, if one feature in isolation is `not a good predictor`, but it's when combined with the second feature, these filter methods will not see that and will remove features on an individual assessment basis.
		- Each feature is assessed against the target, individually, therefore, it doesn't foresee feature redundancy. So, we see the importance of using these methods in combination with the others to evaluate features all together.

### Mutual information (Information Gain)
- Definition:
	- Mutual information measures how much information the presence/absence of a feature contributes to making the correct prediction on Y.
- Measures the mutual dependence of two variables.
- Determines how similar the joint distribution p(X,Y) is to the products of individual distributions p(X)p(Y).
- If X and Y are independent, their mutual information is zero.
- If X is deterministic of Y, the mutual information is the uncertainty contained in Y or X alone. In other words, this is the entropy of Y or the entropy of X. 
	```python
	mutual information = sum{i,y} P(xi, yj) * log(P(xi,yj)/P(xi)*P(yj))
	```

- **How to select features** based on mutual information using `sklearn.feature_selection` on a regression `mutual_info_regression` and classification problem `mutual_info_classif`, along with `SelectKBest, SelectPercentile`. 
	- In practice, feature selection should be done after data preprocessing. Ideally, all the categorical variables are encoded into numbers, and you can assess how deterministic they are of the target.
		```python
		numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
		numerical_vars = list(data.select_dtypes(include=numerics).columns)
		num_data = data[numerical_vars]
		```
	- Calculate the mutual information between the variables and the target, this returns the mutual information value of each feature. The smaller the value, the less information the feature has about the target.
		```python
		## used bnp-paribas dataset.
		# fill null values
		mi = mutual_info_classif(X_train.fillna(0), y_train)
		# add var names and order the features
		mi = pd.Series(mi)
		mi.index = X_train.columns # add var names
		mi.sort_values(ascending=False) # sorting
		# top k=10 features
		sel = SelectKBest(mutual_info_classif, k=10).fit(X_train.fillna(0), y_train)
		X_train.columns[sel.get_support()]
		### output: ['v10', 'v12', 'v14', 'v21', 'v33', 'v34', 'v50', 'v62', 'v114', 'v129']
		```

		- *Notes*: not use mutual information in any projects but there is some value in the method for reference. 

### Fisher Score (chi-square implementation)
- Measures the dependence of two variables.
- Suited for `categorical variables`.
- `Target` should be `binary`.
- Variable values should be `non-negative`, and typically `boolean`, `frequencies`, `counts`.
- It compares observed distribution of class among the different labels against the expected one, would there be no labels.

- Fisher score computes chi-squared stats between each non-negative feature and class. 
	- This score should be used to evaluate categorical variables in a classification problem.
	- It compares the observed distribution of the different classes of target Y among the different categories of the feature, against the expected distribution of the target classes, regardless of the feature categories. 

- **How it works**:
	```python
	## use titanic dataset
	# convert categorical variables in `Embarked`, `Sex` into numbers.
	data['Sex'] = np.where(data.Sex=='male',1,0)
	ordinal_label = {k : i for i, k in enumerate(data['Embarked'].unique(), 0)}
	data['Embarked'] = data['Embarked'].map(ordinal_label)

	## calculate the chi-square (chi2) p_value between each of the variables and the target.
	## It returns two arrays, 
	## one contains the F-Scores which are then evaluated against 
	## the chi-square (chi2) distribution to obtain the p-value.
	## the second array are p-values.
	f_score = chi2(X_train.fillna(0), y_train)
	p_values = pd.Series(f_score[1])
	p_values.index = X_train.columns
	p_values.sort_values(ascending=False)
	### output
		Embarked    0.000580
		Pclass      0.000003
		Sex              NaN
	```
	- For Fisher score, the smaller the p-value, the more significant the feature is to predict the target (Survival) in titanic dataset.
	Therefore, from above result, `Sex` is the most important feature, then `Pclass`, and then `Embarked`.

- **Notes**:
	- When using Fisher score or univariate selection methods in very big datasets, most of the features will show a small p-value, and therefore, it looks like they are highly predictive. This is in fact an effect of the sample size, so `care` should be taken when selecting features using these procedures. An ultra tiny p_value does not highlight an *ultra-important feature*, it rather *indicates that the dataset contains too many samples*. 


### Univariate tests
- Measures the dependence of two variables => ANOVA
	- ANOVA compares the distribution of the variable when the target is `1` versus the distribution of the variable where the target is `0`. 
	- ANOVA assumes:
		- `linear relationship` between variable and target
		- `variables` are `normally distributed`.
		- `Sensitive` to sample size, and in big datasets, most of the features will end up being significant, this is, with p values below 0.05. These small p-values indicate that the distribution differ where the target is 1 or 0. However, the difference between these distributions might be trivial. Therefore, the important thing when using this method is to compare the p-values among different features rather than pay attention to the p-value itself. This means, the method may be a good indication to compare the relative importance of the different features but not the intrinsic importance of each feature respect to the target, because these will be inflated by the sample size.  
- Suited for `continuous variables`.
- Requires a `binary target`
	- Sklearn extends the test to continuous targets with a correlation trick.

- Univariate feature selection works by selecting the best features based on univariate statistical tests (ANOVA). The methods based on F-test estimate the degree of linear dependency between two random variables. They assume that:
	- A `linear relationship` between the `feature` and the `target`. 
	- The variables follow a `Gaussian distribution` (aka normal distribution).

- Note that these assumptions may not always be the case for the variables in your dataset, so if looking to implement these procedure, you'd need to corroborate these assumptions.

- Reminder, in practice, feature selection should be done after data pre-processing, meaning all the categorical variables are encoded into numbers, and then you can assess how deterministic they are of the target.

- **Important**: It's a good practice to select the features by examining on the training set to avoid overfit.

- Use BNP dataset for classification demo.

	```python
	X_train, X_test, y_train, y_test = train_test_split(
		data.drop(labels=['target','ID'], axis=1),
		data['target'],
		test_size=0.3,
		random_state=0)

	# calculate the univariate statistical measure between each of 
	# the variables and the target, it's similar to chi-square, 
	# the output is the array of `f-scores` and 
	# an array of `p-values` which are the ones we will compare.

	univariate = f_classif(X_train.fillna(0), y_train)
	univariate = pd.Series(univariate[1]) # p-value
	univariate.index = X_train.columns
	univariate.sort_values(ascending=False, inplace=True)

	``` 
	- Reminder, the lower the p-value, the most predictive the feature is in principle. There are a few features that do not seem to have predictive power according to the tests, which are those values on the left with p-values above 0.05 (rejected). Those features with p-value > 0.05 are not important. However, `keep in mind that this test assumes a linear relationship, so it might also be the case that the feature is related to the target bit not in a linear manner`.

	- In big datasets, it's not unusual that the p-values of the different features are really small. This does not indicate much about the relevance of the feature to the target though. Mostly, it indicates that it's a big dataset.

	```python
	# select top 10th percentile or top 10, 20 features by using ANOVA 
	# in combination with `SelectKBest` or `SelectPercentile` from sklearn.

	sel = SelectKBest(f_classif, k=10).fit(X_train.fillna(0), y_train)
	X_train.columns[sel.get_support()]
	X_train = sel.transform(X_train.fillna(0))
	```

- Use HousePrice dataset for regression demo.
	```python
	# split train test
	# get numerical variables
	univariate = f_regression(X_train.fillna(0), y_train)
	univariate = pd.Series(univariate[1]) # p-value
	univariate.index = X_train.columns
	univariate.sort_values(ascending=False, inplace=True)
	```

	- **Observations**:
		- A lot of features appear to the left with `p-values above 0.05`, which are candidates to be rejected. This means that `those features do not statistically significantly discriminate the target.`

	```python
	# select top 10 percentile
	sel = SelectPercentile(f_regression, percentile=10).fit(X_train.fillna(0), y_train)
	X_train.columns[sel.get_support()]
	X_train = sel.transform(X_train.fillna(0))
	```
**Take-away notes**: Rarely use these methods to select features. Do use them when investigating the relationship of specific variables with the target in custom problems that do not necessarily come in machine learning model building.



### Univariate ROC-AUC/RMSE
- Measures the dependence of two variables => using Machine Learning to evaluate the dependency.
- Suited for all types of variables and targets.
- Makes no assumption on the distribution of the variables. 
- **Procedure**
	- Builds decision tree using a single variable and the target.
	- Ranks the features according to the model roc-auc or rmse.
	- Selects the features with the highest machine learning metrics.
		
	- Notes: It is perhaps the most powerful but it also has the weakness that it does not foresee feature redundancy. In an extreme example, duplicated features will show the same roc-auc and therefore both will be kept. Where should we put the threshold to select or remove features when using this method? roc-auc = 0.5 means random, therefore, all features with roc-auc that equal to 0.5 could be removed. We also want to remove those features with a low roc-auc the values for example 0.55. For the RMSE or MSE, you could select the cut off above the mean cut off of all the features. Sklean implements the default cutoff for feature selection.


- **How it works**:
	- First, it builds one decision tree per feature, to predict the target.
	- Second, it makes predictions using the decision tree and the feature.
	- Third, it ranks the features according to the ML metric (ROC-AUC or mse).
	- Select the highest ranked features.

	- **Implementation**:
		- Note that it's good practice to select the features by examining only the training set in order to avoid overfit.

		- *Classification*:
		```python
		# use bnp-paribas dataset
		roc_vals = []
		for feat in X_train.columns:
			clf = DecisionTreeClassifier()
			clf.fit(X_train[feat].fillna(0).to_frame(), y_train)
			y_scored = clf.predict_proba(X_test[feat].fillna(0).to_frame())
			roc_vals.append(roc_auc_score(y_test, y_scored[:,1]))
		```
		```python
		rocvals = pd.Series(roc_vals)
		rocvals.index = X_train.columns
		rocvals.sort_values(ascending=False)
		# number of features shows a roc-auc value higher than random.
		len(rocvals[rocvals>0.5])
		```
		- Output: 98 out of 112 features show a predictive performance higher than 0.5. That means we can remove 14 feats from 112 feats. Using cross validation with sklearn to get a more accurate measure of the roc-auc per feature.

		- *Regression*: do the same with `DecisionTreeRegressor()` and mse metric `mean_squared_error(y_test, y_scored)` and append to the list `mse_vals`. Remember for regression, the smaller the mse, the better the model performance is. Therefore, the top of features will be in an increasing order of the sorted list. Also considering the cutoff is depending on how many features you would like to end up with.

- **Take-away notes**: Use this method in the projects, particularly when we have an enormous amount of features and need to reduce the feature space quickly. [A usecase at pydata London](https://www.youtube.com/watch?v=UHtAjLYgDQ4). 

- **Filter methods**:
	```python
	# libraries
	from sklearn.model_selection import train_test_split
	from sklearn.feature_selection import VarianceThreshold
	from sklearn.preprocessing import StandardScaler
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.metrics import roc_auc_score
	#
	import pandas as pd
	import numpy as np
	import seaborn as sns
	import matplotlib.pyplot as plt
	%matplotlib inline
	
	# 1. get data
	data = pd.read_csv(pathfile)
	
	# 2. split data
	X_train, X_test, y_train, y_test = train_test_split(
		data.drop(labels=['target'], axis=1),
		data['target'], test_size=0.3, random_state=0)
	X_train_org, X_test_org = X_train.copy(), X_test.copy()

	#### Basic filter
	# 3. Remove constant 
	constant_feats = [feat for feat in X_train.columns if X_train[feat].std()==0]
	X_train.drop(labels=constant_feats, axis=1, inplace=True)
	X_test.drop(labels=constant_feats, axis=1, inplace=True)

	# 4. Remove quasi-constant 
	sel = VarianceThreshold(threshold=0.01) 
	sel.fit(X_train)
	# remove quasi-constant features
	X_train = sel.transform(X_train)
	X_test = sel.transform(X_test)

	# 5. Remove duplicated features
	duplifeats = []
	for i in range(len(X_train.columns)):
		if i%10==0: print(i)
		col_1 = X_train.columns[i]
		for col_2 in X_train.columns[i+1:]:
			if X_train[col_1].equals(X_train[col_2]):
				duplifeats.append(col_2)
	X_train.drop(labels=duplifeats, axis=1, inplace=True)
	X_test.drop(labels=duplifeats, axis=1, inplace=True)
	
	X_train_basic, X_test_basic = X_train.copy(), X_test.copy()

	# 6. Remove correlated features
	def correlation(data, threshold):
		col_corr = set() # all correlated columns.
		corr_mat = data.corr()
		for i in range(len(corr_mat.columns)):
			for j in range(i):
				if abs(corr_mat.iloc[i,j])> threshold:	
					colname = corr_matrix.columns[i]
					col_corr.add(colname)
		return col_corr
	corr_feats = correlation(X_train, 0.8)
	X_train.drop(labels=corr_feats, axis=1, inplace=True)
	X_test.drop(labels=corr_feats, axis=1, inplace=True)
	X_train_corr, X_test_corr = X_train.copy(), X_test.copy()

	# 7. Remove features using univariate ROC_AUC.
	## loop to build a tree, make predictions and get the ROC-AUC 
	## for each feature of the trainset.
	roc_vals = []
	for feat in X_train.columns:
		clf = DecisionTreeClassifier()
		clf.fit(X_train[feat].fillna(0).to_frame(), y_train)
		y_scored = clf.predict_proba(X_test[feat].fillna(0).to_frame())
		roc_vals.append(roc_auc_score(y_test, y_scored[:,1]))
	roc_vals = pd.Series(roc_vals)
	roc_vals.index = X_train.columns
	selected_feats = roc_vals[roc_vals>0.5]

	# 8. Compare the performance in machine learning algorithms
	def run_randomForests(X_train, X_test, y_train, y_test):
		clf_rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)
		clf_rf.fit(X_train, y_train)
		pred = clf_rf.predict_proba(X_train)
		print("Train RF roc-auc: ", roc_auc_score(y_train, pred[:,1]))
		pred = clf_rf.predict_proba(X_test)
		print("Test RF roc-auc: ", roc_auc_score(y_test, pred[:,1]))

	run_randomForests(X_train_org.drop(labels=['ID'], axis=1),
					X_test_org.drop(labels=['ID'], axis=1),
					y_train, y_test)
		Train set
		Random Forests roc-auc: 0.8012314741948454
		Test set
		Random Forests roc-auc: 0.7900499757912425

	run_randomForests(X_train_basic.drop(labels=['ID'], axis=1),
					X_test_basic.drop(labels=['ID'], axis=1),
					y_train, y_test)
		Train set
		Random Forests roc-auc: 0.8016577097093865
		Test set
		Random Forests roc-auc: 0.791033019265853

	run_randomForests(X_train_corr.drop(labels=['ID'], axis=1),
					X_test_corr.drop(labels=['ID'], axis=1),
					y_train, y_test)
		Train set
		Random Forests roc-auc: 0.8073914001626228
		Test set
		Random Forests roc-auc: 0.7937667747098247

	run_randomForests(X_train[selected_feats.index],
					X_test_corr[selected_feats.index],
					y_train, y_test)
		Train set
		Random Forests roc-auc: 0.8105671870819526
		Test set
		Random Forests roc-auc: 0.7985492537265694
	```
	- **Observation**: After removing constant, quasi-constant, duplicated, correlated features, now univariate roc-auc = `0.5`. We enhanced the performance after reducing the feature space from 371 to 90.

	- **Regression**
	```python
	def run_logistic(X_train, X_test, y_train, y_test):
		logit = LogisticRegression(random_state=44)
		logit.fit(X_train, y_train)
		pred = logit.predict_proba(X_train)
		print("Train Logistic roc-auc: ", roc_auc_score(y_train, pred[:,1]))
		pred = logit.predict_proba(X_test)
		print("Test Logistic roc-auc: ", roc_auc_score(y_test, pred[:,1]))

	scaler = StandardScaler().fit(X_train_org.drop(labels=['ID'],axis=1))
	
	run_logistic(scaler.transform(X_train_org.drop['ID'], axis=1),\
				scaler.transform(X_test_org.drop['ID'], axis=1),
				y_train, y_test)

		Train set
		Logistic Regression roc-auc: 0.8068058675730494
		Test set
		Logistic Regression roc-auc: 0.7948755847784289

	run_logistic(scaler.transform(X_train_basic.drop(labels=['ID'], axis=1)), \
	scaler.transform(X_test_basic.drop(labels=['ID'], axis=1)), y_train, y_test)
		Train set
		Logistic Regression roc-auc: 0.8057765181239191
		Test set
		Logistic Regression roc-auc: 0.7951930873302437

	run_logistic(scaler.transform(X_train_corr.drop(labels=['ID'], axis=1)), \
		scaler.transform(X_test_corr.drop(labels=['ID'], axis=1)),
				y_train, y_test)

		Train set
		Logistic Regression roc-auc: 0.7966160581765025
		Test set
		Logistic Regression roc-auc: 0.7931114290523482

	run_logistic(scaler.transform(X_train[selected_feats.index]),\

				scaler.transform(X_test[selected_feats.index]),
				y_train, y_test)

		Train set
		Logistic Regression roc-auc: 0.7930113686469875
		Test set
		Logistic Regression roc-auc: 0.7947741568815442

	```
	- **Observation:**
		- univariate roc-auc helped improving the performance of logistic regression.	
		

- **Bonus: Method used in KDD'09 competition**:
	- The task is to predict churn based on a dataset with a huge number of features.
	- This is an aggressive non-parametric feature selection procedure, which is based in contemplating the relationship between the feature and the target as a filter methods.

	- **Procedure**:
		- For each categorical variable:
			1. Separate into train test.
			2. Determine the `mean value` of the `target` `within each label` of the `categorical variable` using the `train` set.
			3. Use that `mean target value per label as the prediction` in the `test` set and `calculate the roc-auc`.

		- For each numerical variable:
			1. Separate into train test.
			2. Divide the variable into 100 quantiles.
			3. Calculate the mean target within each quantile using the train set.
			4. Use the mean target value (bin) as the prediction on the test set and calculate the roc-auc.

	- **Advantages of the method**:
		- Speed: computing mean and quantiles is direct and efficient.
		- Stability respect to scale: extreme values for continuous variables do not skew the predictions.
		- Comparable between categorical and numerical variables.
		- Accommodation of non-linearities.

	- **Implementation**:
		
		- **Categorical variables**:
		
		```python
		# 1. load titanic dataset
		# 2. variable preprocessing
		## cabin feature contains missing value, replacing by "Missing"
		## to narrow down the different cabins by selecting first letter
		## which represents where the cabin was located.
		data['Cabin'].fillna("Missing", inplace=True)
		data['Cabin'] = data['Cabin'].str[0]
		## output: ['M', 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T']
		# 3. split data into train test
		# 4. feature selection on categorical variables
		## cat_feats = ['Sex','Pclass','Cabin','Embarked']
		```

		- Calculate the mean of target 'Survival' (equivalent to the probability of survival) of the passenger, within each label of a categorical variable. Using a dictionary and train set that maps each label of the train set variable, to a probability of survival.

		- Then the function replaces the label in both train and test sets by the probability of survival. It's like making a prediction on the outcome, by using only the label of the variable. This way, the function replaces the original strings by probabilities.

		- **Take-away**:
			- We use just the label of the variable to estimate the probability of survival of the passenger.
			- It's like "Tell me which one was your cabin, I will tell you your probability of survival."

		- If the labels of a categorical variable are good predictors, then, we should obtain a roc-auc above 0.5 for that variable, when we evaluate those probabilities with the real outcome, which is whether the passenger is survived or not.

		```python
		def mean_encoding(dftrain, dftest, cat_cols=[]):
			dftrain_tem, dftest_tem = dftrain.copy(), dftest.copy()
			for col in cat_cols:
				# dictionary mapping labels/categories to the 
				# mean target of that label.
				risk_dict = dftrain.groupby([col])['Survived'].mean().to_dict()
				# re-map the labels
				dftrain_tem[col] = dftrain[col].map(risk_dict)
				dftest_tem[col] = dftest[col].map(risk_dict)
			
			# drop the target
			dftrain_tem.drop(['Survived'], axis=1, inplace=True)
			dftest_tem.drop(['Survived'], axis=1, inplace=True)
			return dftrain_tem, dftest_tem
		## X_train_enc, X_test_enc = mean_encoding(X_train, X_test, cat_cols=['Sex', 'Cabin', 'Embarked', 'Cabin'])
		```   

		- Now, calculate a roc-auc value, using the probabilities that replace the labels in the encoded test set, and comparing it with the true target.

		```python
		cat_cols=['Sex', 'Cabin', 'Embarked', 'Cabin']
		roc_vals = []
		for feat in cat_cols:
			roc_vals.append(roc_auc_score(y_test, X_test_enc[feat]))
		
		# result of roc_vals
			Sex         0.771667
			Cabin       0.641637
			Cabin       0.641637
			Embarked    0.577500
		```
		- **Observation**:
			- All the features are important because roc-auc > 0.5.
			- Feature 'Sex' seems to be the most important feature to predict the target ('Survival').

		- **Numerical variables**:
			- The procedure is the same as for categorical variables, but it requires an additional first step which is to divide the continuous variables into bins.
			- In this work, the authors divided into 100 quantiles, meaning 100 bins.
			- The numerical variables in titanic dataset are `Age` and `Fare`.
			
			```python
			# 1. get data, split data into train test.
			
			# 2. use the `qcut` (quantile cut) function from pandas 
			# with 9 cutting points, meaning 10 bins.
			# retbins=True indicates we want to capture the limits of 
			# each interval for use to cut the test set.

			# 3. create 10 labels, one for each quantile
			# instead of having the quantile limits, the new variable will
			# have labels in its bins.

			labels = ['Q' + str(i+1) for i in range(10)]
			X_train['Age_binned'], intervals = pd.qcut(
					X_train['Age'],
					10,
					labels=labels,
					retbins=True,
					precision=3,
					duplicates='drop')
			# notes: Since `Age` contains missing values, its length is 11.

			# output of X_train.Age_binned.unique()
				[Q10, Q9, Q1, NaN, Q4, ..., Q6, Q2, Q7, Q5, Q8]

			# these are the cutting points of the intervals.
				[  0.67,  13.1 ,  19.  ,  22.  ,  25.4 ,  29.  ,  32.  ,
				36.  , 41.  ,  49.  ,  80.  ]

			# now use the boundaries calculated in the previous cell to bin the testing set.
				X_test['Age_binned'] = pd.cut(x = X_test['Age'],
												bins=intervals,
												labels=labels)
			# we see the NAN values in both X_train and X_test, 
			# replace the NAN values by a new category called "Missing"
			# first, recast variables as objects, then replacing missing
			# values with a new category.

			for df in [X_train, X_test]:
				df['Age_binned'] = df['Age_binned'].astype('O')
				df['Age_binned'].fillna('Missing', inplace=True)

			# Now create a dict that maps the bins to the mean of target.
			risk_dict = X_train.groupby(['Age_binned'])['Survived'].mean().to_dict()

			# re-map the labels, replace the bins by the probabilities of survival.
			for df in [X_train, X_test]:
				df['Age_binned'] = df['Age_binned'].map(risk_dict)

			# finally, calculate the roc-auc value, using the probabilities
			# that replace the labels, and compare it with the true target.
			roc_auc_score(y_test, X_test['Age_binned'])
			
			# output: 
				0.57238095238095243

			# This is higher than 0.5, in principle, `Age` has some 
			# predictive power even though it seems worse than 
			# any of categorical variables we evaluated.

			```
			
			- Feature `Fare` can be done the same way.
			```python
			labels = ['Q' + str(i) for i in range(10)]
			# train
			X_train['Fare_binned'], intervals = pd.qcut(
				X_train.Fare,
				10,
				labels=labels,
				retbins=True,
				precision=3,
				duplicates='drop')
			# test
			X_test['Fare_binned'] = pd.cut(
				x = X_test.Fare,
				bins=intervals,
				labels=labels)
			# check if X_test, X_train has missing values, if so, add dtype=object.
			X_test['Fare_binned'].isnull().sum(), X_train['Fare_binned'].isnull().sum()

			# parse as categorical values.
			for df in [X_train, X_test]:
				df['Fare_binned'] = df['Fare_binned'].astype('O')
			
			# create dictionary that maps the bins to the mean of target.
			risk_dict = X_train.groupby(['Fare_binned'])['Survived'].mean().to_dict()

			# then re-map the labels, replace the bins by the probabilities of survival.
			for df in [X_train, X_test]:
				df['Fare_binned'] = df['Fare_binned'].map(risk_dict)

			# output of X_train['Fare_binned'].head()
				857    0.492063
				52     0.533333
				386    0.354839
				124    0.730159
				578    0.396825

			# calculate a roc-auc value, using the probabilities that
			# we used to replace the labels, and compare it to the 
			# true target.

			# estimate all missing values in X_test to be zero.
			X_test['Fare_binned'].fillna(0, inplace=True)

			# calculate roc-auc
			roc_auc_score(y_test, X_test['Fare_binned'])

			# output:
				0.72538690476190471

			```
			- The output indicates that `Fare` feature is a much better predictor of Survival.

			- **Notes**:
				- Keep in mind that the categorical variables may or may not (typically will not) show the same percentage of observations per label.
				- However, when we divide a numerical variable into quantile bins, we guarantee that each bin shows the same percentage of observations.
				- Alternatively, instead of binning into quantiles, we can bin into equal-distance bins by calculating the max value - min value range and divide that distance into the amount of bins we want to construct. That would determine the cut-points for the bins.

## Section 6: Wrapper methods

### Step forward feature selection
- Sequential feature selection algorithms are a family of greedy search algorithms that are used to reduce an initial d-dimensional feature space to a k-dimensional feature subspace where `k < d`.

- This method starts by evaluating all features individually and selects the one that generates the best performing algorithm, according to a pre-set evaluation criteria. In the second step, it evaluates all possible combinations of the selected feature and a second feature, then selects the pair that produces the best performance.

- The pre-set criteria can be the roc_auc for classification and the r-squared for regression for example. 

- This method is called greedy due to evaluating all possible features combinations. So it's quite expensive in terms of time and space complexity.

- Use a special package `mlxtend` that implements this type of feature selection.
	- The stopping criteria is an arbitrarily set number of features. So the search will finish when we reached this upper bound.
- We use the step forward feature selection method from `mlxtend` library.
	- For classification problems:

	```python
	from mlxtend.feature_selection import SequentialFeatureSelector as SFS
	from sklearn.metrics import roc_auc_score
	## Notes: feature selection should be done after data preprocessing.
	# 0. convert categorical vars into numbers, get all numerical variables.
	# 1. split data.
	# 2. collect correlated features from `.corr()` with a `threshold`(=0.8).
	# 3. drop correlated features.
	# 4. depend on which type of problems in order to apply the model properly.
	# 5. fit the model.
	clf_sfs = SFS(RandomForestClassifier(n_jobs=4), 
					k_features=10, 
					forward=True,
					floating=False,
					verbose=2,
					scoring='roc_auc', 
					cv=3 )
	clf_sfs = clf_sfs.fit(np.array(X_train.fillna(0)), y_train)
	
	selected_feats = X_train.columns[list(clf_sfs.k_feature_idx_)]

	rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)
	rf.fit(X_train, y_train)
	print('Train set')
    pred = rf.predict_proba(X_train)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    print('Test set')
    pred = rf.predict_proba(X_test)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
    ## output: 
    ## Train set
	## Random Forests roc-auc: 0.716244551855826
	## Test set
	## Random Forests roc-auc: 0.7011551996870431
	``` 

	- For regression problems:
		- Basically, the process will be the same as in classification method, but we will change the scoring method in SFS to be `r2`.

### Step backward feature selection
- This method starts by fitting a model using all features. Then it removes one feature that produces the highest performing algorithm for a certain evaluation criteria. In the second step, it will remove a second feature that also produces the best performing algorithm, the process continues until a certain criteria is met.
- The selection procedure is called greedy, because it evaluates all `n` possible features.
- The procedure are the same as in the step forward feature selection, except we replace by `forward=False` in the `SFS`.

### Exhaustive feature selection
- In an exhaustive feature selection, the best subset of features is selected by optimizing a specified performance metric for a certain ML algorithm. For example, if the classifier is a logistic regression, and the dataset has 4 features, the algorithm will evaluate all 15 feature combinations as follows:
	- All combinations of 1 feat.
	- All combinations of 2 feats.
	- All combinations of 3 feats.
	- All 4 feats.
- Then select the one that results in the best performance (i.e. accuracy) of the logistic regression classifier.

- This is quite computationally expensive and unfeasible if feature space is big.

- There is a special package for python that implements this type of feature selection: `mlxtend`. This method is the exhaustive feature selection, the stopping criteria is an arbitrarily set number of features. So the search will finish when we reach the desired number of selected features.

- **Notes**: it's computationally expensive. Not use unless you run it on a cloud cluster.

### Additional resources
1. Articles
	- [Least angle and l1 penalised regression: A review](https://projecteuclid.org/download/pdfview_1/euclid.ssu/1211317636)
	- [Penalised feature selection and classification in bioinformatics](https://www.ncbi.nlm.nih.gov/pubmed/18562478)
	- [Feature selection for classification: A review](https://web.archive.org/web/20160314145552/http://www.public.asu.edu/~jtang20/publication/feature_selection_for_classification.pdf)

2. Blog: [Machine Learning Explained: Regularization](https://www.r-bloggers.com/machine-learning-explained-regularization/)

3. [Online course](https://see.stanford.edu/materials/aimlcs229/cs229-notes5.pdf).


## Section 7: Embedded methods | Lasso regulization
- Lasso regularization
	- Adding a penalty to the model to reduce overfitting. In linear model regularization, the penalty is applied over the coefficients that multiply each of the predictors. 
	- Lasso (a.k.a `l1`-penalty with beta-absolute/norm) has the property that is able to shrink coefficients to zeros, therefore, some features can be removed from the model. This helps Lasso to be fit for feature selection. 
	```python
		
		L1-penalty = (y-∑xβ)^2 + λ∑|βi|
		
		y^2- 2xyβ + x^2 β^2 + λβ = 0

		Partial derivatives w.r.t β:

			-2xy + 2x^2β + λ=0

			2β(x^2) = 2xy - λ

			β = (2xy - λ) / 2x^2 

		Finally,

			β = (y - λ/2x) / x
	```

	- When `lambda` increases, more features coefficients will be zeros and therefore it reduces the variance but increases the bias. When it's small, it tends to be a high variance. In theory, when `lambda`=inf, all coefficients are zeros, and therefore all predictors are dropped.
	- In Ridge regularization (with a `l2`-penalty term beta-squared), it can only shrink coefficients close to zeros, but not zeros. This is because when we take the derivatives w.r.t. beta in the ridge regression, the `lambda` will be in denominator [(reference-ridge)](https://stats.stackexchange.com/questions/176599/why-will-ridge-regression-not-shrink-some-coefficients-to-zero-like-lasso). Therefore the value of `beta` will be as low as possible but not zero.
	```python
		
		L2-penalty = (y-∑xβ)^2 + λ∑βi^2
		
		y^2- 2xyβ + x^2 β^2 + λβ^2 = 0

		Partial derivatives w.r.t β:

			-2xy + 2x^2β + 2βλ=0

			2β(x^2+λ) = 2xy

			β = 2xy / 2(x^2 + λ)

		Finally,

			β = xy / (x^2 + λ)
	```


## Section 8: Embedded methods | Linear models

- Regression coefficients
	```
	y = β0 + β1X1 + β2X2 + ... + βnXn
	```
	where β is the coefficient of the feature X. 
		  βs are directly proportional to how much
		  that feature contributes to the final value y.
	
	- Note that, coeffients in linear models depend on a few assumptions.
		- Linear relationship between predictor (X) and outcome (Y).
		- Xs are independent.
		- Xs are not correlated to each other (non-multi-collinearity).
		- Xs are normally distributed (Gaussian distribution).
		- Homoscedasticity (variance should be the same).
		- For direct coefficient comparison Xs should be in the same scale, i.e. [0,1] or [-1,1].
		- The magnitude of the coefficients is influenced by regulization and the scale of the features. Therefore, all features are within the same scale to compare coefficients across features using normalization.
	- Another note is that Linear Regression model is fitted by matrix multiplication, not by gradient descent.

- Logistic regression coefficients.
	```
	sel_ = SelectFromModel(LogisticRegression(C=1000, penalty='l2')) 
	```
	where C=1000 is set to obtain as much as possible the real relationship between the features and the target.
	```
	sel_.get_support()
	```
	to get the selected features. 
	The `SelectFromModel()` will select those variables that their absolute coefficient values are greater than the mean coefficient value of all the variables.


## Section 9: Embedded methods | Trees
- Feature selection by tree derived variable importance.
	- Decision trees:
		- Most popular machine learning algorithms.
		- High accurate.
		- Good generalization (low overfitting).
		- Interpretability.
		- Importance:
			- Top layer: Highest impurity (all classes are mixed).
			- Second layer: Impurity decreases.
			- Third layer: Impurity continues to decrease, so on. 
		- For classification, the impurity measurement is either to give any or the information gain or entropy.
		- For regression, the impurity measurement is the variance.
		- Therefore, when training a tree, it's possible to compute how much each feature decreases the impurity, in other words, how good the feature is at separating the classes. The more a feature decreases the impurity, the more important the feature is. Feature selectors at higher nodes lead to the greater gains and therefore the most important ones.

	- Random Forests:
		- Consist of several hundreds of individual decision trees.
		- The impurity decreases for each feature is averaged across trees.
		- Limitations:
			- Correlated features show equal or similar importance.
			- Correlated features importance is lower than the real importance, determined when tree is built in absence of correlated couterparts.
			- Highly cardinal variables show greater importance (trees are biased to this type of variables).

		- How it works:
			- Build a random forest
			- Determine feature importance
			- Select the features with highest importance
			- Use sklearn-enabled.

		- Recursive feature elimination
			- Build random forests
			- Calculate feature importance
			- Remove least important feature
			- Repeat until a condition is met

		- Note: If the feature removed is correlated to another feature in the dataset, then by removing the correlated feature, the true importance of the other feature will be verified by its incremental importance value (i.e. info gain).	
	- Gradient Boosted trees feature importance:
		- Feature importance calculated in the same way
		- Biased to highly cardinal features.
		- Importance is susceptible to correlated features.
		- Interpretability of feature importance is not straight-forward.
			- Later trees fit to the errors of the first trees, therefore the feature importance is not necessarily proportional on the influence of the feature on the outcome, rather than that on the low performance of previous trees.
			- Averaging across trees may not add much information on true relation between feature and target.


## Section 10: Reading resources

- [Feature Selection for Classification: A Review, Tang et al](https://web.archive.org/web/20160314145552/http://www.public.asu.edu/~jtang20/publication/feature_selection_for_classification.pdf)
- [An Introduction to Variable and Feature Selection, Guyon and Elisseeff, 2003.](http://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf)
- [A review of feature selection methods with applications, Jovic et al](https://pdfs.semanticscholar.org/3130/5b131a69fc2a8980698e2ccae5a701d9cae8.pdf)
- [Least angle and ℓ 1 penalized regression: A review, Hesterberg et al.](https://projecteuclid.org/download/pdfview_1/euclid.ssu/1211317636)
- [Chapter 7: Feature Selection](https://www.cs.cmu.edu/~kdeng/thesis/feature.pdf)
- [Correlation Based Feature Selection for Machine Learning. Thesis](https://www.cs.waikato.ac.nz/~mhall/thesis.pdf)
- [Data Preprocessing for Supervised Learning. Kotsiantis et al.](https://www.researchgate.net/publication/228084519_Data_Preprocessing_for_Supervised_Learning?enrichId=rgreq-4a7b75a2b9198bae2d92c556e25c08eb-XXX&enrichSource=Y292ZXJQYWdlOzIyODA4NDUxOTtBUzoxMDQwMTY1NDk3Nzc0MDlAMTQwMTgxMDg4NjgxMQ%3D%3D&el=1_x_3&_esc=publicationCoverPdf)
- [The 2009 Knowledge Discovery in Data Competition (KDD Cup 2009) Challenges in Machine Learning.](http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf)
- [Blog: Machine Learning Explained: Regularisation](https://www.r-bloggers.com/machine-learning-explained-regularization/)

## Section 11: Hybrid feature selection methods












