
import torch as t
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split as tts
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import preprocessing 
from torch.utils.tensorboard import SummaryWriter
import ModelBuilding as MB
writer = SummaryWriter('runs')
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
data_train = None
data_test =  None
data_validation = None



num_classes = 1 #True or false
input_size = 33
hidden_size = 64
num_epochs = 10
batch_size = 8
learning_rate = 0.001
weigth_decay = 1e-5
drop_out_rate = 0.3
gamma = 0.5
step_size=3

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_test['loan_status'] = -1
data_train, _ = MB.preprocessing_data(data_train)
data_test, _ = MB.preprocessing_data(data_test)






data_train, data_validation = tts(data_train, stratify=data_train["loan_status"], test_size=0.1)

target = data_train['loan_status'].values


data_train_features, data_train_target = data_train.drop("loan_status", axis=1), data_train['loan_status']
data_validation_features, data_validation_target = data_validation.drop("loan_status", axis=1), data_validation['loan_status']
data_test_features, data_test_target = data_test.drop("loan_status", axis=1), data_test['loan_status']


sc = StandardScaler()
data_train_features = sc.fit_transform(data_train_features)
data_validation_features = sc.transform(data_validation_features)
data_test_features = sc.transform(data_test_features)

data_train = MB.LoanDataset(data_train_features, data_train_target)
data_validation = MB.LoanDataset(data_validation_features, data_validation_target)
data_test = MB.LoanDataset(data_test_features, data_test_target)

# Data loaders
train_loader = t.utils.data.DataLoader(dataset=data_train,
                                           batch_size=batch_size,
                                           shuffle=True)

validation_loader = t.utils.data.DataLoader(dataset=data_validation,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = t.utils.data.DataLoader(dataset=data_test, 
                                          batch_size=batch_size, 
                                          shuffle=False)


model = MB.LoanNet(input_size, hidden_size, num_classes, drop_out_rate)

optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


model = MB.model_training(model=model, num_epochs=num_epochs, loader=train_loader,
                          criterion=criterion, scheduler=scheduler, optimizer=optimizer, device=device)

MB.eval_model(model=model, loader=validation_loader, device=device)

MB.create_test_file(model, device,test_loader)