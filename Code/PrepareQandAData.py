import pandas as pd
from transformers import BertTokenizer
import torch

#################################################################################################################
### Other Functions

def LoadCSVData(file_path):
    Q_and_A_dataframe = pd.read_csv(file_path)
    return Q_and_A_dataframe

def CreateTokenizedData(Q_and_A_dataframe,tokenizer,tokenized_Q_and_A_data_filepath):
    questions = Q_and_A_dataframe["Questions"].tolist()
    answers = Q_and_A_dataframe["Answers"].tolist()
    tokenized_data = tokenizer(questions,answers,return_tensors="pt",padding=True,truncation=True,)
    torch.save(tokenized_data,tokenized_Q_and_A_data_filepath)
    return tokenized_data

def CreateOrLoadTokenizedData(Q_and_A_dataframe,tokenizer,tokenized_Q_and_A_data_filepath,create_or_load_string='load'):
    if create_or_load_string in ['Create','create']:
        tokenized_data = CreateTokenizedData(Q_and_A_dataframe,tokenizer,tokenized_Q_and_A_data_filepath)
    else:
        tokenized_data = torch.load(tokenized_Q_and_A_data_filepath)
    return tokenized_data
    
#################################################################################################################
### Main Functions

if __name__ == "__main__":

    initial_Q_and_A_data_filepath = 'Initial_Data/QandA_Dataset.csv'
    create_or_load_string = 'Create'
    tokenized_Q_and_A_data_filepath = 'Additional_Data/Prepared_QandA_Dataset.csv'
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    Q_and_A_dataframe = LoadCSVData(tokenized_Q_and_A_data_filepath)
    tokenized_data = CreateOrLoadTokenizedData(Q_and_A_dataframe,tokenizer,tokenized_Q_and_A_data_filepath,create_or_load_string)

