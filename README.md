# Udacity MLE Nanodegree Capstone Project

## Table of Contents

 1. [Motivation](https://github.com/bubekaro/MLE-Capstone#Motivation)
 2. [Libraries](https://github.com/bubekaro/MLE-Capstone#Libraries)
 3. [Files](https://github.com/bubekaro/MLE-Capstone#Files)
 4. [Results](https://github.com/bubekaro/MLE-Capstone#Results)
 5. [Acknowledgements](https://github.com/bubekaro/MLE-Capstone#Acknowledgements)

## Motivation
This project was chosen mainly for two reasons. First, because it tackles a real-life problem, the task of designing a targeted advertising campaign based on data from known customers, more specifically, based on how the customers differ from the general population at large. Second, because it uniquely combines a number of Machine Learning Engineering techniques covered during the course, namely, dimensionality reduction, segmentation (clustering), and classification. The combination of a task that can be easily generalized to a wide scope of applications, together with the multiplicity of moving parts involved in this project, makes for a rich experience for someone who is getting started with Data Science in general, and Machine Learning in particular. As an added bonus, this project offers a substantial opportunity to learn or practice Data Exploration and Preprocessing techniques. The amount of missing data, the disparity with wich such data gaps appear, the high level of correlated features, the subtly hidden outliers, and the combination of numerical with categorical data, makes for an experience in data wrangling that is truly second to none.

More concretely, the project consists of analyzing demographics data for customers of a mail-order sales company in Germany, comparing it against demographics information for the general population of that country. Aside from the customer and general population datasets, a separate pair of datasets was made available for training and testing a classifier; these datasets were obtained from a recent mailout campaign and have the same type of data features and quality.

The sequential flow of the project is as follows: First, the customer and general population datasets are preprocessed and visualized in preparation to be used with unsupervised learning models. Customer segmentation is then carried out in order to spot customers in the population, or rather, in order to identify which features of the data find overrepresentation with customers, and which are underrepresented. After this, the datasets of the recent marketing campaign are used to model and to predict which individuals are most likely to become customers for the company. Finally, the predictions are tested and scored in a [Kaggle Competition](https://www.kaggle.com/c/udacity-arvato-identify-customers/submissions).

## Libraries
* [Python 3.*](https://docs.python.org/3/)
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Jupyter Widgets](https://ipywidgets.readthedocs.io/en/latest/)

## Files
• `images` : Folder containing all the figures included in the write-up.
• `predictions` : Folder containing a few of the submissions made at the Kaggle cmptetition in the form of CSV files.
• `Capstone Arvato.ipynb` : All parts of the capstone project in an Interactive Python Jupyter Notebook.
• `Capstone Proposal.pdf` : Write-up to propose taking on this project.
• `data_exploration.py` : Assorted collection of Python routines for the Data Exploration and Preprocessing piece of the project.


## Results
During the Data Exploration and Preprocessing part of the project, it was ascertained through visualizations of the datasets' distributions, that the customers of the mail-order company tended to be older, sedentary, of versatile consumption type, heavily into saving or investing money, suburban, with multiple cars in the household, mostly top earners of advanced age, and with a slightly heavier male presence. After implementing the unsupervised piece of this project, similar conclusions were drawn, with some degree of measurement attached to them. Among other things, customers were found to be mostly single, high income earners, or top earners of advanced age, interested in investing, low in mobility, mostly male, dominant and feisty. The supervised learning part of the project consisted of choosing a metric for testing a few classifiers in order to select and fine-tune one of them to generate predictions. Given the nature of the problem (marketing) and the imbalance of the data, the ROC AUC metric was chosen. Using ROC AUC the GradientBosstingClassifier was selected and seven of its parameter fine-tuned, namely, the number of trees (n_estimators), tree depth (max_depth), learning rate (learning_rate), as well as max_features, subsample, min_samples_leaf, and min_samples_split. Predictions from the fine-tuned Gradient Boost classifier were submitted to the Kaggle competition to achieve a score of 79.61%. Further attempts at fine tuning and combining PCA with classification failed to improve on the score.

## Acknowledgements
* [Udacity](https://www.udacity.com) : Machine Learning Nanodegree - Capstone Project.
* [Arvato bertelsmann](https://www.bertelsmann.com/divisions/arvato/#st-1) : Datasets.
