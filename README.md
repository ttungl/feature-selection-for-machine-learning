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
			- Univariate selection
				- Two-step procedure:
					- Rank features according to a certain criteria.
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

	- *Embedded methods*:
		- Performance feature selection as part of the model construction process.
			- By combining feature selection with classifier or regressor construction, these methods have the advantages of wrapper methods.
		- Consider the interaction between features and models.
		- They are less computationally expensive than wrapper methods, because they fit the ML model only once.
		- **Methods**:
			- LASSO for linear regression model.
			- Tree importance 

## Section 3: Filter methods | Basics

## Section 4: Filter methods | Correlation

## Section 5: Filter methods | Statistical measures

## Section 6: Wrapper methods

## Section 7: Embedded methods | Lasso regulization

## Section 8: Embedded methods | Linear models

## Section 9: Embedded methods | Trees

## Section 10: Reading resources

## Section 11: Hybrid feature selection methods

## Section 12: Final section and next steps
