# Udacity MLE Nanodegree Capstone Project

## Table of Contents

 1. [Motivation](https://github.com/bubekaro/MLE-Capstone#Motivation)
 2. [Libraries](https://github.com/bubekaro/MLE-Capstone#Libraries)
 3. [Files](https://github.com/bubekaro/MLE-Capstone#Files)
 4. [Results](https://github.com/bubekaro/MLE-Capstone#Results)
 5. [Acknowledgements](https://github.com/bubekaro/MLE-Capstone#Acknowledgements)

## Motivation
The project was chosen for two reasons: First, it tackles a real-life problem, that of designing a targeted advertising campaign based on demographic data. Second, it combines several Machine Learning techniques covered during the course, namely, dimensionality reduction, clustering, classification, and tuning. The combination of a task that can be easily generalized to a wide scope of applications, together with the multiplicity of moving parts involved, makes for a rich experience for someone who is getting started with Data Science in general, and Machine Learning in particular. The project also offers multiple challenges and opportunities to practice Data Exploration and Preprocessing. The amount of missing data, the nature of the missingness, the high level of correlated features, and the variety of outliers, makes for a truuly rich experience in data wrangling.

More concretely, the project consists of analyzing two demographics datasets, one for the customers of a mail-order sales company in Germany, and another for the general population of that country. A separate pair of datasets obtained from a recent mailout campaign was also made available for training and testing a classifier.

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
* `references` : Folder containing some of the literature referenced in the write-up.
* `images` : Folder containing all the figures included in the write-up.
* `predictions` : Folder containing a few of the submissions made at the Kaggle cmptetition in the form of CSV files.
* `Capstone.ipynb` : All parts of the capstone project in an Interactive Python Jupyter Notebook.
* `Capstone Project Report.pdf` : Write-up about the project implementation steps and concluding remarks.
* `data_exploration.py` : Assorted collection of Python routines for the Data Exploration and Preprocessing piece of the project.
* `unsupervised_ML.py` : Python code to facilitate the PCA analysis, K-Means Segmentation, and Clusters dissection.


## Results
After exploring and cleaning the data, but before any ML techniques were applied, the datasets were compared to spot any marked differences in feature distributions. From this analysis, a broad customer profile emerged, describing customers, among other things, as wealthy and sedentary nature lovers, mostly West German old males living without children at home.

With unsupervised ML techniques, the complexity of the problem was made more tractable. First the number of variables was reduced from 366 to 195 by means of PCA. After this, using K-Means clustering, the near 1M observations were collapsed into four clusters that crystallized what made customers stand out from the general population. Two segments in particular were heavily representative of customers, namely, wealthy older West German males, and money-savvy wealthy males. Here a caveat was inserted to remark that these conclusions may have been heavily tainted by the manner in which a particular feature - "year of birth", was imputed. Further, the incomplete description of features that plague the data was also noted as reason for concern, in that it leads to conclusions that conceal true segments.

The supervised learning part of the project presented its own challenge; the data available for training was heavily imbalanced. To deal with this, an appropriate metric was picked based on considerations pertaining to the problem at hand, i.e., a targeted marketing strategy. Using ROC AUC as the metric of choice, several classifiers were tried and the Gradient Boosting Machine selected for tuning and predicting. After tuning some parameters, predictions were made and submitted to the Kaggle competition to achieve a score of 79.61%.

## Acknowledgements
* [Udacity](https://www.udacity.com) : Machine Learning Nanodegree - Capstone Project.
* [Arvato Bertelsmann](https://www.bertelsmann.com/divisions/arvato/#st-1) : Datasets.
