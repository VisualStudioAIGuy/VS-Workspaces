import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim


bee_production = pd.read_csv("Datasets/US_honey_dataset_updated.csv")
bee_production = bee_production.drop(["id", "value_of_production", "year"], axis=1)
bee_production = pd.get_dummies(
    bee_production,
    columns=["state"],
    dtype=int
)
bee_production = bee_production.astype("float")

X = bee_production.drop(["average_price"], axis=1)
y = bee_production["average_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, test_size=.2, random_state=2)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
y_train_tesor = torch.tensor(y_train.values, dtype=torch.float).view(-1,1)

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
y_test_tesor = torch.tensor(y_test.values, dtype=torch.float).view(-1,1)

torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(48, 56),
    nn.ReLU(),
    nn.Linear(56, 50),
    nn.ReLU(),
    nn.Linear(50, 25),
    nn.ReLU(),
    nn.Linear(25, 1)
)

loss = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
"""
num_epochs = 6000
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    MAE = loss(outputs, y_train_tesor)
    MAE.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1)% 500 == 0:
        print(f"Epoch [{epoch +1 }/{num_epochs}], MAE Loss: {MAE.item()}")
torch.save(model, "models/Bee_production_model6000.pth")"""

model3000 = torch.load("models/Bee_production_model3000.pth", weights_only=False)
model4000 = torch.load("models/Bee_production_model4000.pth", weights_only=False)
model5000 = torch.load("models/Bee_production_model5000.pth", weights_only=False)
model6000 = torch.load("models/Bee_production_model6000.pth", weights_only=False)

model3000.eval()
model4000.eval()
model5000.eval()
model6000.eval()

with torch.no_grad():
    predictions3 = model3000(X_test_tensor)
    predictions4 = model4000(X_test_tensor)
    predictions5 = model5000(X_test_tensor)
    predictions6 = model6000(X_test_tensor)
    test_loss3 = loss(predictions3, y_test_tesor)
    test_loss4 = loss(predictions4, y_test_tesor)
    test_loss5 = loss(predictions5, y_test_tesor)
    test_loss6 = loss(predictions6, y_test_tesor)


print("Neural Network 3000 epochs - Test Set MAE: ", test_loss3.item())
print("Neural Network 4000 epochs - Test Set MAE: ", test_loss4.item())
print("Neural Network 5000 epochs - Test Set MAE: ", test_loss5.item())
print("Neural Network 6000 epochs - Test Set MAE: ", test_loss6.item())