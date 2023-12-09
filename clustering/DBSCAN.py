import numpy as np
import pandas as pd
from sklearn import preprocessing

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

file = '../data/obesity_dataset_clean.csv'
obesity = pd.read_csv(file)
obesity = obesity.iloc[:, 1:17]
