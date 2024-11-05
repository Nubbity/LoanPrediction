import torch as t
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split as tts
import numpy as np
import pandas as pd
import torchmetrics
from sklearn import preprocessing 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import yeojohnson
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs')


#Dataset wrapper for loading the data and preprossing
class LoanDataset(Dataset):
    def __init__(self, features, targets):
        self.features = t.tensor(np.array(features), dtype=t.float32)
        self.targets = t.tensor(np.array(targets), dtype=t.float32).unsqueeze(1)
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]



class LoanNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, drop_out_rate):
        super(LoanNet, self).__init__()
        self.input_size = input_size

        l1_size = int(hidden_size)
        l2_size = int(hidden_size/2)
        l3_size = int(hidden_size/4)

        self.l1 = nn.Linear(input_size, l1_size)
        self.bn1 = nn.BatchNorm1d(l1_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=drop_out_rate)

        self.l2 = nn.Linear(l1_size, l2_size)
        self.bn2 = nn.BatchNorm1d(l2_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=drop_out_rate)

        self.l3 = nn.Linear(l2_size, l3_size)
        self.bn3 = nn.BatchNorm1d(l3_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=drop_out_rate)

        self.l4 = nn.Linear(l3_size, num_classes)
    def forward(self, x):
        #layer 1
        out = self.l1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        #layer 2
        out = self.l2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        #Layer 3
        out = self.l3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)       
        # Forward pass through output layer
        out = self.l4(out)

        return out

def plot_roc(fpr, tpr, roc_auc):
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.5f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('(ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()


def preprocessing_data(data):
        data.drop("id", axis=1, inplace=True)
        data['income_to_age'] = data['person_income'] / data['person_age']
        data['loan_to_income'] = data['loan_amnt'] / data['person_income']
        data['rate_to_loan'] = data['loan_int_rate'] / data['loan_amnt']
        data['age_credit_history_interaction'] = data['person_age'] * data['cb_person_cred_hist_length']
        data['loan_to_employment'] = data['loan_amnt'] / (data['person_emp_length'] + 0.01)
        data['age_to_credit_history'] = data['person_age'] / (data['cb_person_cred_hist_length'] + 0.01)
        data['log_income_to_loan'] = data['person_income'] / data['loan_amnt']
        data['age_interest_interaction'] = data['person_age'] * data['loan_int_rate']
        data['age_to_employment'] = data['person_age'] / (data['person_emp_length']+ 0.01)
        data["DI_Ratio"] = (data['loan_amnt']*data['loan_int_rate'])/(data["person_income"]+ 0.01)
        data["DI_Ratio_3"] = (data['loan_amnt']*data['loan_int_rate'])/(data["person_income"]+ 0.01)**3
        data = pd.get_dummies(data,
                              columns=data.select_dtypes(include=['object']).columns, drop_first=True)
        _, num_columns = data.shape
        return data, num_columns

def model_training(num_epochs, loader, model, criterion,scheduler, optimizer, device):
    model = model.to(device)

    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(loader):  
            optimizer.zero_grad()

            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}]Loss: {loss.item():.4f}, Learning rate: {optimizer.param_groups[0]['lr']}")
        scheduler.step()

    return model


def eval_model(model, loader, device):
    model = model.to(device)
    model.eval()
    accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)
    precision_metric = torchmetrics.Precision(task="binary").to(device)
    recall_metric = torchmetrics.Recall(task="binary").to(device)
    all_labels = []
    all_probabilities = []
    with t.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device).int()

             # Forward pass
            outputs = model(features)  # Model outputs logits directly
            probabilities = t.sigmoid(outputs)  # Converts logits to probabilities for binary classification
            # Generate predictions (binary)
            predictions = (probabilities >= 0.49).int()
            
            # Update metrics
            all_labels.append(labels.cpu())
            all_probabilities.append(probabilities.cpu())
            accuracy_metric.update(predictions, labels)
            precision_metric.update(predictions, labels)
            recall_metric.update(predictions, labels)
    accuracy = accuracy_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    all_labels = t.cat(all_labels).numpy()
    all_probabilities = t.cat(all_probabilities).numpy()
    fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities)
    roc_auc = auc(fpr, tpr)
    plot_roc(fpr, tpr, roc_auc)
    print(f"Accuracy:{accuracy} | Precision: {precision} | recall:{recall}")
    return (accuracy, precision, recall)


def create_test_file(model, device,test_loader, input_path="sample_submission.csv"):
    model = model.to(device)
    predictions = []
    with t.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            probabilities = t.sigmoid(outputs)
            batch_predictions = (probabilities >= 0.49).float().cpu()  # Converts logits to probabilities
            predictions.extend(batch_predictions.squeeze().tolist())
    data = pd.read_csv(input_path)
    data['loan_status'] = pd.Series(predictions)
    data.to_csv("Final_sub.csv", index=False)