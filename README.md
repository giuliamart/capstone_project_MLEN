# capstone_project_MLEN
### Capstone Project for the Machine Learning Engineer Nanodegree Program <br>
### Predicting Dengue Disease Spread <br>
In 2019 I had the chance of visiting an astonishing country, Sri Lanka. During our stay there, although we were already aware of it and had taken the necessary precautions, we had a few conversations with local people about the phenomenon of Dengue and the risk that one can run every day by being bitten by a particular species of mosquitoes, _Aedes_ mosquitos, that carry this disease. From there my curiosity about this phenomenon grew and I began to wonder if it was possible in some way to use statistical and / or machine learning techniques to analyze and predict it. This project of mine, therefore, is based on a competition held by the site DrivenData.org, called **_"DengAI: Predicting Disease Spread"_** (https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/), which focuses on finding a way of predicting the next dengue fever local epidemic in San Juan, Puerto Rico and Iquitos, Peru.

## Python Requirements
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from numpy import array
from math import sqrt
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import statsmodels.api as sm
from datetime import *
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
%matplotlib inline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
# xgboost
!pip install xgboost
import xgboost as xgb
print("xgboost", xgb.__version__)
from xgboost import XGBRegressor
from xgboost import plot_importance
# LSTM
!pip install tensorflow
!pip install keras==2.3.1
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional
%load_ext autoreload
%autoreload 2

