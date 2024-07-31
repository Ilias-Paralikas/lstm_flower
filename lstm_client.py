import flwr as fl
import os 
import pandas  as pd
import torch
from torch.utils.data import DataLoader
from utils import LSTMModel, train_lstm , test
from utils import LoadPredictionDataset,TRAIN_TEST_SPLIT, preprocess
from utils import get_hyperparameters
import logging

logging.basicConfig(level=logging.INFO)


hyperparameters = get_hyperparameters()

BATCH_SIZE = hyperparameters['BATCH_SIZE']
FILL_NAN= hyperparameters['FILL_NAN']
TIME_STEP =hyperparameters['TIME_STEP']
COLUMN = hyperparameters['COLUMN'] # 'cpu_consumption
EPOCHS = hyperparameters['EPOCHS']
lr  = hyperparameters['lr'] 
HIDDEN_LAYER_SIZE =     hyperparameters['HIDDEN_LAYER_SIZE']
NUM_LAYERS= hyperparameters['NUM_LAYERS']
TRAIN_TEST_SPLIT=   hyperparameters['TRAIN_TEST_SPLIT']
ROOT_FOLDER =hyperparameters['ROOT_FOLDER']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


csv_file=  os.path.join(ROOT_FOLDER,'results_cpu_memory_eco-efficiency.csv')
df = pd.read_csv(csv_file)
df = preprocess(df,FILL_NAN)

train_split,test_split = TRAIN_TEST_SPLIT(df,TRAIN_TEST_SPLIT)
train_dataset = LoadPredictionDataset(df,time_step=TIME_STEP,column=COLUMN,start_index=0,population=train_split,device=device)
test_dataset  = LoadPredictionDataset(df,time_step=TIME_STEP,column=COLUMN,start_index=train_split,population=test_split,device=device)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMModel(input_size=TIME_STEP,hidden_layer_size=HIDDEN_LAYER_SIZE,num_layers=NUM_LAYERS,output_size=1).to(device)
# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().detach().numpy() for val in model.parameters()]

    def set_parameters(self, parameters):
        for val, param in zip(parameters, model.parameters()):
            param.data = torch.tensor(val)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_lstm(model,epochs=EPOCHS,train_dataloader=train_dataloader,lr=lr)
        return self.get_parameters(config), len(train_dataset), {}

    def evaluate(self, parameters):
        self.set_parameters(parameters)
        loss = test(model, test_dataloader,device)
        logging.info(f"Evaluation Mean Squared Error: {loss}")
        return float(loss), len(test_dataset) ,{}

# Start Flower client with the new method
fl.client.start_client(
    server_address="192.168.2.63:65432",
    client=FlowerClient().to_client()
)