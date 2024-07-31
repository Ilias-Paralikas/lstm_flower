import os 
import pandas  as pd
import torch
from torch.utils.data import DataLoader
from utils import LSTMModel, train_lstm , test
from utils import LoadPredictionDataset,train_test_split, preprocess
from utils import get_hyperparameters

hyperparameters = get_hyperparameters()

BATCH_SIZE = hyperparameters['BATCH_SIZE']
FILL_NAN= hyperparameters['FILL_NAN']
TIME_STEP =hyperparameters['TIME_STEP']
COLUMN = hyperparameters['COLUMN'] # 'cpu_consumption
EPOCHS = hyperparameters['EPOCHS']
lr  = hyperparameters['lr'] 
HIDDEN_LAYER_SIZE =     hyperparameters['HIDDEN_LAYER_SIZE']
NUM_LAYERS= hyperparameters['NUM_LAYERS']
TRAIN_TEST_SPLOT=   hyperparameters['TRAIN_TEST_SPLOT']
ROOT_FOLDER =hyperparameters['ROOT_FOLDER']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


csv_file=  os.path.join(ROOT_FOLDER,'results_cpu_memory_eco-efficiency.csv')
df = pd.read_csv(csv_file)
df = preprocess(df,FILL_NAN)

train_split,test_split = train_test_split(df,TRAIN_TEST_SPLOT)
train_dataset = LoadPredictionDataset(df,time_step=TIME_STEP,column=COLUMN,start_index=0,population=train_split,device=device)
test_dataset  = LoadPredictionDataset(df,time_step=TIME_STEP,column=COLUMN,start_index=train_split,population=test_split,device=device)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMModel(input_size=TIME_STEP,hidden_layer_size=HIDDEN_LAYER_SIZE,num_layers=NUM_LAYERS,output_size=1).to(device)
train_lstm(model,epochs=EPOCHS,train_dataloader=train_dataloader,lr=lr)

# Example usage
mse = test(model, test_dataloader, device)
print(f'Mean Squared Error: {mse}')