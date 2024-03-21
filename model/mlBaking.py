
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

class BakingRegression:
    def __init__(self):
        self.dt = None
        self.logreg = None
        self.X_train = None
        self.y_train = None
        self.encoder = None
        self.initBaking()
        
    def initBaking(self):
        baking_data = pd.read_csv('datasets/recipesDetail.csv')
        self.td = baking_data
        self.td.drop(['id','ingredients_raw_str','serving_size','servings'], axis=1, inplace=True)
        self.td.dropna(inplace=True)
        