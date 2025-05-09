import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split


def calculate_percent(dataset, number, type="HeartDisease"):
    total_count = dataset[type].shape[0]
    zeros_count_heart_disease = (dataset[type] == number).sum()
    zeros_percentage_heart_disease = (zeros_count_heart_disease / total_count) * 100
    print(f"Percentage of {number}s in {type} column: {zeros_percentage_heart_disease}%")


dataset = pd.read_csv('Datasets/heart.csv')
#Dataset previewing
print(dataset.head())
print(dataset.info())

calculate_percent(dataset, 0)
calculate_percent(dataset, 1)

"""dataset_G = dataset.groupby("Age")
print(dataset_G["HeartDisease"].mean().sort_values())
#Commented out because fills up output
"""


one_hot_headings = ["Sex", "ChestPainType", "RestingECG", "ST_Slope", "ExerciseAngina"]
dataset = pd.get_dummies(dataset, columns=one_hot_headings, dtype=int)
print(dataset.head())

train_features = dataset.drop(["HeartDisease"], axis=1)
test_features = dataset["HeartDisease"]
print(train_features.info())
X = torch.tensor(train_features.values, dtype=torch.float32)
y = torch.tensor(test_features.values, dtype=torch.float32).view(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, test_size=.2, random_state=42)