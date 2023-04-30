#!/usr/bin/env python
# coding: utf-8

# In[90]:


from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import copy

device = "mps" if torch.backends.mps.is_available() else "cpu"


# In[91]:


x = torch.rand(size=(3, 4)).to(device)


# In[92]:


df = pd.read_csv('./SeoulBikeData.csv', encoding= 'unicode_escape').drop(["Date", "Seasons", "Holiday"], axis=1)


# In[93]:


dataset_cols = []
for col in df.columns:
    col_name = col
    col_name.replace(" ", "_")
    col_name = re.sub('[^A-Za-z0-9_]+', '', col_name)
    dataset_cols.append(col_name)

df.columns = dataset_cols


# In[94]:


df["FunctioningDay"] = (df["FunctioningDay"] == "Yes").astype(int)
df = df[df["Hour"] == 12]
df = df.drop(["Hour","Windspeedms", "Visibility10m", "FunctioningDay"], axis=1)


# In[95]:


train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])


# In[104]:


batch_size = 1000

train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val, batch_size=batch_size, shuffle=True)


train_dl


# In[97]:


class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# In[98]:


learning_rate=0.001
num_epochs=1000

model = RegressionModel(input_size=6, hidden_size=16, output_size=1)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


# In[99]:


for epoch in range(num_epochs):
    for xb, yb in train_dl:
        # Move data to device
        xb, yb = xb.to(device), yb.to(device)
        
        # Forward pass
        pred = model(xb)
        loss = criterion(pred, yb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Compute validation loss
    with torch.no_grad():
        valid_loss = sum(criterion(model(xb.to(device)), yb.to(device)) for xb, yb in valid_dl)
        
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {valid_loss.item():.4f}")
    
    # Examine the dataloader
    for xb, yb in train_dl:
        print(xb, yb)
        break


# In[ ]:

