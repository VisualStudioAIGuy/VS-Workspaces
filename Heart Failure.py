import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def calculate_percent(dataset, number, type="HeartDisease"):
    total_count = dataset[type].shape[0]
    zeros_count_heart_disease = (dataset[type] == number).sum()
    zeros_percentage_heart_disease = (zeros_count_heart_disease / total_count) * 100
    print(f"Percentage of {number}s in {type} column: {zeros_percentage_heart_disease}%")

dataset = pd.read_csv('Datasets/heart.csv')

'''print(dataset.head())
print(dataset.info())'''

calculate_percent(dataset, 0)
calculate_percent(dataset, 1)

"""dataset_G = dataset.groupby("Age")
print(dataset_G["HeartDisease"].mean().sort_values())
#Commented out because fills up output
"""

one_hot_headings = ["Sex", "ChestPainType", "RestingECG", "ST_Slope", "ExerciseAngina"]
dataset = pd.get_dummies(dataset, columns=one_hot_headings, dtype=int)
#print(dataset.head())

train_features = dataset.drop(["HeartDisease"], axis=1)
test_features = dataset["HeartDisease"]
#print(train_features.info())

X = torch.tensor(train_features.values, dtype=torch.float32)
y = torch.tensor(test_features.values, dtype=torch.float32).view(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, test_size=.2, random_state=42)

torch.manual_seed(42)
binary_model = nn.Sequential(
    nn.Linear(20,64),
    nn.ReLU(),
    nn.Linear(64,36),
    nn.ReLU(),
    nn.Linear(36, 1),
    nn.Sigmoid()
)

loss = nn.BCELoss()
optimizer = optim.Adam(binary_model.parameters(), lr=0.005)

#val = []

num_epochs = 300

"""for epoch in range(num_epochs):
    predictions = binary_model(X_train)
    BCE = loss(predictions, y_train)
    BCE.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 100 == 0:
        predicted_labels = (predictions >= 0.5).int()
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}], BCELoss: {BCE.item():.4f}, Accuracy: {accuracy:.4f}')
        '''predictions_val = binary_model(X_test)
        predicted_labels_val = (predictions_val >= 0.5).int()
        accuracy_val = accuracy_score(y_test, predicted_labels_val)
        print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy_val:.4f}')
        val.append(accuracy_val)
test = 0
for item in val:
    if item >= test:
        test = item
print(test)
'''"""

#torch.save(binary_model, 'models/heart_model900.pth')
binary_model = torch.load('models/heart_model900.pth', weights_only=False)

binary_model.eval()
with torch.no_grad():
    test_predictions = binary_model(X_test)
    test_predicted_labels = (test_predictions>=.5).int()

print("Accuracy: " + str(accuracy_score(y_test, test_predicted_labels)))
print("--------")
print("Classification: " + str(classification_report(y_test, test_predicted_labels)))
print("Confusion Matrix: " + str(confusion_matrix(y_test, test_predicted_labels)))