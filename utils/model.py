import torch
import torch.nn as nn
import torch.optim as optim
import logging

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