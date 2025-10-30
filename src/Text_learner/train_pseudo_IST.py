# train TNet for IST dataset

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from TNet_text_IST import TNet
import torch.nn as nn

# Add device configuration near the top of the file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load data
df_pseudo = pd.read_csv("dataset/IST/IST_tabular/pseudo_outcomes.csv")
# load representations
df_rep = pd.read_csv("dataset/IST/IST_ALL_embeddings.csv")
df = pd.read_csv("dataset/IST/IST_tabular/IST_syn.csv")

X = df_rep
print(X.shape)
# normalize X
X = (X - X.mean()) / X.std()

Y = df_pseudo
# normalize Y
Y = (Y - Y.mean()) / Y.std()

Y0 = df['Y0']
print(Y0.shape)
Y1 = df['Y1']
print(Y1.shape)

# Split data
X_train = X.iloc[:int(0.8*len(X))]
X_test = X.iloc[int(0.8*len(X)):]
Y_train = Y.iloc[:int(0.8*len(Y))]
Y_test = Y.iloc[int(0.8*len(Y))]



Y0_train = Y0.iloc[:int(0.8*len(Y0))]
Y0_test = Y0.iloc[int(0.8*len(Y0)):]
Y1_train = Y1.iloc[:int(0.8*len(Y1))]
Y1_test = Y1.iloc[int(0.8*len(Y1)):]


# Convert to tensors
X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)

Y_train = torch.FloatTensor(Y_train.values).reshape(-1, 1)
Y_test = torch.FloatTensor(Y_test.values).reshape(-1, 1)

Y0_train = torch.FloatTensor(Y0_train.values).reshape(-1, 1)
Y0_test = torch.FloatTensor(Y0_test.values).reshape(-1, 1)
Y1_train = torch.FloatTensor(Y1_train.values).reshape(-1, 1)
Y1_test = torch.FloatTensor(Y1_test.values).reshape(-1, 1)

#  Create datasets and dataloaders
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)



class RegressionNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(RegressionNet, self).__init__()
        self.single_net = self._build_network(input_dim, hidden_dims, dropout_rate)

    def _build_network(self, input_dim, hidden_dims, dropout_rate):
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
        
    
    def forward(self, x):
        return self.single_net(x)


# Initialize model, optimizer, and loss function
input_dim = X_train.shape[1]
model = RegressionNet(input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
criterion = nn.MSELoss()

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    
    for X_batch, Y_batch in train_loader:
        # Move batch data to device
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    # Move test data to device
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    Y0_test = Y0_test.to(device)
    Y1_test = Y1_test.to(device)
    cate_test = Y1_test - Y0_test
    
    Y_pred_test = model(X_test)
    test_loss = criterion(Y_pred_test, cate_test)
    
    # Calculate predicted CATE
    cate_pred = model(X_test)
    
    # Calculate true ITE (Y1 - Y0)
    true_ite = Y1_test - Y0_test
    
    # Calculate MSE and MAE for CATE estimation
    cate_mse = nn.MSELoss()(cate_pred, true_ite)
    cate_mae = nn.L1Loss()(cate_pred, true_ite)
    
    # Calculate PEHE
    pehe = torch.sqrt(torch.mean((cate_pred - true_ite)**2))
    
print(f'Test Loss: {test_loss.item():.4f}')
print(f'Average CATE: {torch.mean(cate_pred):.4f}')
print(f'CATE MSE: {cate_mse.item():.4f}')
print(f'CATE MAE: {cate_mae.item():.4f}')
print(f'PEHE: {pehe.item():.4f}')




