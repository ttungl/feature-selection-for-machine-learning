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
*Feature selection* is the process of selecting a subset of relevant features (variables, predictors) for use in machine learning model building.

*Why should we select features?*
- Simple models are easier to interpret.
- Shorter training times.
- Enhanced generalization by reducing overfitting.
- Easier to implement by software developer.
- Reduced risk of data error during model use.
- Variable redundancy.
	- As the features within dataset are highly correlated, which means that they have essentially the same information, so they are redundant. We can keep one and remove the rest without losing information.

- Bad learning behaviour in high dimensional spaces.
	- ML algorithms, specifically tree-based algorithms, are favored by reduced feature spaces. This means, high dimension causes worst performance in tree-based methods and so reducing the feature space helps build more robust and predictive models.

*Procedure*
- A feature selection algorithm can be seen as the combination of a search technique for proposing new feature subsets, along with an evaluation measure which scores the different feature subsets.
	- Computationally expensive.
	- Different feature subsets render optimal performance for different ML algorithms.
		- Different methods of feature selection.


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
