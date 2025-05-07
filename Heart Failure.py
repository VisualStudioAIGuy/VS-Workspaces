import pandas as pd
import numpy as np

def calculate_percent(dataset, number):
    total_count = dataset['HeartDisease'].shape[0]
    zeros_count_heart_disease = (dataset['HeartDisease'] == number).sum()
    zeros_percentage_heart_disease = (zeros_count_heart_disease / total_count) * 100
    print(f"Percentage of {number}s in 'HeartDisease' column: {zeros_percentage_heart_disease}%")


dataset = pd.read_csv('Datasets/heart.csv')
#Dataset previewing
print(dataset.head())
print(dataset.info())

calculate_percent(dataset, 0)
calculate_percent(dataset, 1)