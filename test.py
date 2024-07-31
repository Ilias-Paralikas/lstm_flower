import flwr as fl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
# Set logging level to INFO
logging.basicConfig(level=logging.INFO)

class LSTMModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_layer_size,
                 num_layers , 
                 output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size,num_layers , batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)


    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
     
        return predictions




def train_lstm(model,epochs,train_dataloader,lr,criterion=None,optimizer=None,device='cpu'):
    if criterion is None:
        criterion = nn.MSELoss()  # or any other loss function based on your task
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)  # or any other optimizer

    losses = []
    # Training loop
    for epoch in range(epochs):  # number of epochs
        accumulative_loss = 0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            accumulative_loss += loss.item()
        losses.append(accumulative_loss)
        if epoch %10 ==0:
            logging.info(f"Epoch {epoch}, loss: {accumulative_loss / len(train_dataloader)}")
    logging.info('Finished Training')
    
    return model
    

def test(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = mse_loss(outputs, targets)
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss


def preprocess(df,FILL_NAN):
    # fill nan values
    df_ffill = df.ffill(limit=FILL_NAN)
    df_bfill = df.bfill(limit=FILL_NAN)
    df = (df_ffill + df_bfill) / 2
    # scaler
    scaler = StandardScaler()
    df  = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df

def train_test_split(df,split):
    train_split = int(len(df)*split)
    test_split=  len(df) -train_split
    return train_split,test_split

class LoadPredictionDataset(Dataset):
    def __init__(self,df,
                 start_index,
                 population,                 
                 time_step,
                 column,
                 device):
        previous_overflow=  max(start_index-time_step,0)
        df = df[column]
        self.df = df.iloc[previous_overflow:start_index+population]
        self.column = column
        self.time_step = time_step
        self.device = device
        self.length = len(self.df)-self.time_step -1
        
    def __getitem__(self, index):
        previous_values = self.df.iloc[index:index+self.time_step].values
        previous_values = torch.tensor(previous_values).unsqueeze(0)
        previous_values = previous_values.float().to(self.device)
        target_values = self.df.iloc[index+self.time_step]
        target_values = torch.tensor(target_values).float().to(self.device)
        target_values = target_values.unsqueeze(0)
        return previous_values, target_values
    
    def __len__(self):
        return self.length





TRAIN_TEST_SPLiT = 0.2
TIME_STEP = 10
COLUMN = 'cpu_consumption'
BATCH_SIZE = 64
FILL_NAN = 10
ROOT_FOLDER = '.'
HIDDEN_LAYER_SIZE =10
EPOCHS = 10
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


csv_file=  os.path.join(ROOT_FOLDER,'results_cpu_memory_eco-efficiency.csv')
df = pd.read_csv(csv_file)
df = preprocess(df,FILL_NAN)

train_split,test_split = train_test_split(df,TRAIN_TEST_SPLiT)
train_dataset = LoadPredictionDataset(df,time_step=TIME_STEP,column=COLUMN,start_index=0,population=train_split,device=device)
test_dataset  = LoadPredictionDataset(df,time_step=TIME_STEP,column=COLUMN,start_index=train_split,population=test_split,device=device)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



model = LSTMModel(input_size=TIME_STEP,hidden_layer_size=HIDDEN_LAYER_SIZE,num_layers=10,output_size=1).to(device)
train_lstm(model,epochs=EPOCHS,train_dataloader=train_dataloader,lr=lr)
loss = test(model, test_dataloader,device)


print(loss)
