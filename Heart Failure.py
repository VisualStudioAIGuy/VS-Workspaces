import pandas as pd
import numpy as np

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

object_headings = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
print(dataset[object_headings].head())