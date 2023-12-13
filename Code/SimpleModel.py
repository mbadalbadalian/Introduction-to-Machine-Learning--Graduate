import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import pickle
import os

#################################################################################################################
### Other Functions

def LoadDF(DF_path):
    dataframe = pd.read_csv(DF_path)
    return dataframe

def SavePKL(data,data_filename):
    with open(data_filename,'wb') as file:
        pickle.dump(data,file)
    return

def LoadPKL(data_filename):
    with open(data_filename,'rb') as file:
        data = pickle.load(file)
    return data

class BibleDataset(Dataset):
    def __init__(self, text_tokens, tokenizer):
        self.text_tokens = text_tokens
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text_tokens)

    def __getitem__(self, idx):
        return {"input_ids": self.text_tokens[idx,:]}

def CreateTrainAndTestLoader(ESV_Bible_DF,train_loader_filepath,test_loader_filepath,tokenizer_filepath):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    text_tokens = tokenizer.encode("\n".join(ESV_Bible_DF["Text"]),return_tensors="pt",max_length=1024,truncation=True)
    text_tokens = text_tokens[0].tolist()
    train_tokens,test_tokens = train_test_split(text_tokens,test_size=0.3,random_state=20777980)
    train_dataset = BibleDataset(train_tokens,tokenizer)
    test_dataset = BibleDataset(test_tokens,tokenizer)
    train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=4,shuffle=False)
    torch.save(train_loader,train_loader_filepath)
    torch.save(test_loader,test_loader_filepath)
    tokenizer.save_pretrained(tokenizer_filepath)
    return train_loader,test_loader,tokenizer

def LoadTrainAndTestLoader(train_loader_filepath,test_loader_filepath,tokenizer_filepath):
    train_loader = torch.load(train_loader_filepath)
    test_loader = torch.load(test_loader_filepath)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_filepath)
    return train_loader,test_loader,tokenizer

def CreateOrLoadTrainAndTestLoader(ESV_Bible_DF,train_loader_filepath,test_loader_filepath,tokenizer_filepath,create_or_load_string_train_and_test_loader='load'):
    if create_or_load_string_train_and_test_loader in ['Create','create']:
        train_loader,test_loader,tokenizer = CreateTrainAndTestLoader(ESV_Bible_DF,train_loader_filepath,tokenizer_filepath,test_loader_filepath)
    else:
        train_loader,test_loader,tokenizer = LoadTrainAndTestLoader(train_loader_filepath,test_loader_filepath,tokenizer_filepath)
    return train_loader,test_loader,tokenizer

def CreateTrainedNLPModel(train_loader,test_loader,NLP_model_filepath,average_train_loss_filename,average_test_loss_filename,checkpoint_directory):
    num_epochs = 3

    NLP_model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NLP_model.to(device)
    optimizer = torch.optim.AdamW(NLP_model.parameters(),lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()
    average_train_loss_list = []
    average_test_loss_list = []
    os.makedirs(checkpoint_directory,exist_ok=True)
    existing_checkpoints = [f for f in os.listdir(checkpoint_directory) if f.startswith("checkpoint_epoch_")]
    if existing_checkpoints:
        existing_checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
        last_checkpoint = existing_checkpoints[-1]
        checkpoint_filename = os.path.join(checkpoint_directory,last_checkpoint)
        checkpoint = torch.load(checkpoint_filename)
        NLP_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        average_train_loss_list = checkpoint['average_train_loss_list']
        average_test_loss_list = checkpoint['average_test_loss_list']
    else:
        start_epoch = 0

    for epoch in range(start_epoch,num_epochs):
        NLP_model.train()
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            labels[:, :-1] = -100
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = NLP_model(input_ids, labels=labels)
            loss = criterion(outputs.logits[:,:-1,:].contiguous().view(-1,outputs.logits.size(-1)),labels[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        NLP_model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="test"):
                input_ids = batch["input_ids"].to(device)
                labels = input_ids.clone()
                labels[:, :-1] = -100
                labels = labels.to(device)
                outputs = NLP_model(input_ids, labels=labels)
                loss = criterion(outputs.logits[:,:-1,:].contiguous().view(-1,outputs.logits.size(-1)),labels[:, 1:].contiguous().view(-1))
                total_test_loss += loss.item()

        checkpoint_filename = os.path.join(checkpoint_directory, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': NLP_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': total_test_loss/len(test_loader),
            'average_train_loss_list': average_train_loss_list,
            'average_test_loss_list': average_test_loss_list,
        }, checkpoint_filename)

        print(f"Epoch {epoch + 1}/{num_epochs}, test Loss: {total_test_loss/len(test_loader):.4f}, Checkpoint saved at {checkpoint_filename}")

        average_train_loss_list.append(total_train_loss/len(train_loader))
        average_test_loss_list.append(total_test_loss/len(test_loader))

    NLP_model.save_pretrained(NLP_model_filepath)
    SavePKL(average_train_loss_list, average_train_loss_filename)
    SavePKL(average_test_loss_list, average_test_loss_filename)
    return NLP_model,average_train_loss_list,average_test_loss_list

#################################################################################################################
### Main Functions

#The main function is the driver for the code
if __name__ == "__main__":
    
    #Variables
    ESV_Bible_DF_filepath = 'Additional_Data/ESV_Bible_DF.csv'
    ESV_Bible_DF_tokenized_filepath = 'Simple_Method_Data/ESV_Bible_DF_tokenized_Simple_Method.pt'
    train_loader_filepath = 'Simple_Method_Data/NLP_train_loader_Simple_Method.pth'
    test_loader_filepath = 'Simple_Method_Data/NLP_test_loader_Simple_Method.pth'
    tokenizer_filepath = 'Simple_Method_Data/NLP_tokenizer_Simple_Method.pth'
    NLP_model_filepath = 'Simple_Method_Data/NLP_model_trained_Simple_Method'
    average_train_loss_filename = 'Simple_Method_Data/NLP_training_loss_Simple_Method.pkl'
    average_test_loss_filename = 'Simple_Method_Data/NLP_test_loss_Simple_Method.pkl'
    checkpoint_directory = 'Simple_Method_Data/NLP_model_checkpoints_Simple_Method'
    
    create_or_load_string_train_and_test_loader = 'Create'
    
    #Main Code
    ESV_Bible_DF = LoadDF(ESV_Bible_DF_filepath)
    train_loader,test_loader,tokenizer = CreateOrLoadTrainAndTestLoader(ESV_Bible_DF,train_loader_filepath,test_loader_filepath,tokenizer_filepath,create_or_load_string_train_and_test_loader)
    NLP_model,average_train_loss,average_test_loss = CreateTrainedNLPModel(train_loader,test_loader,NLP_model_filepath,average_train_loss_filename,average_test_loss_filename,checkpoint_directory)
    
    
    