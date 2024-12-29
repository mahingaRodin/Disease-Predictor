from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader, TensorDataset
from utils import define_model, train_model

#fetching the dataset
heart_disease = fetch_ucirepo(id=45)

#extracting features and targets
x = heart_disease.data.features
y = heart_disease.data.targets

#display the metadata to understand the datasert
print(heart_disease.metadata)
print(heart_disease.variables)

#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#convert data to pytorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

#prepare dataloader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#training the models

from torch.optim import Adam
from torch.nn import CrossEntropyLoss


#defineing the model
model = define_model(input_size=X_train.shape[1] , output_size=len(y.unique()))
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

#training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "../models/health_model.pth")


