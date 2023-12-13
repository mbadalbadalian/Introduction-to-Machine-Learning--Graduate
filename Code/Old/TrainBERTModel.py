import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#################################################################################################################
### Other Functions

def LoadDF(DF_path):
    dataframe = pd.read_csv(DF_path)
    return dataframe

def CreateBibleDatasetDF(prepared_Q_and_A_DF,ESV_Bible_DF,Bible_dataset_DF_filepath):    
    Bible_dataset_DF = pd.concat([ESV_Bible_DF[['Text']], prepared_Q_and_A_DF[['Questions', 'Answers']]], axis=0)
    Bible_dataset_DF.index = ['Text']*len(ESV_Bible_DF) + ['Questions', 'Answers']*len(prepared_Q_and_A_DF)
    Bible_dataset_DF.to_csv(Bible_dataset_DF_filepath)
    return Bible_dataset_DF

def CreateOrLoadBibleDatasetDF(Q_and_A_DF,ESV_Bible_DF,Bible_dataset_DF_filepath,create_or_load_string_Bible_dataset_DF='load'):
    if create_or_load_string_Bible_dataset_DF in ['Create','create']:
        Bible_dataset_DF = CreateBibleDatasetDF(Q_and_A_DF,ESV_Bible_DF,Bible_dataset_DF_filepath)
    else:
        Bible_dataset_DF = LoadDF(Bible_dataset_DF_filepath)
    return Bible_dataset_DF

class BibleQADataset(Dataset):
    def __init__(self, texts, questions, answers):
        self.texts = texts
        self.questions = questions
        self.answers = answers

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts.iloc[idx],
            'questions': self.questions.iloc[idx],
            'answers': self.answers.iloc[idx]
        }
    
def CreateTrainAndTestLoader(Bible_dataset_DF,train_loader_filepath,test_loader_filepath):
    train_dataset,test_dataset = train_test_split(Bible_dataset_DF,test_size=0.3,random_state=20777980)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = BibleDataset(train_dataset, tokenizer)
    test_dataset = BibleDataset(test_dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    torch.save(train_dataset,train_loader_filepath)
    torch.save(test_dataset,test_loader_filepath)
    return train_loader,test_loader

def LoadTrainAndTestLoader(train_loader_filepath,test_loader_filepath):
    train_loader = torch.load(train_loader_filepath)
    test_loader = torch.load(test_loader_filepath)
    return train_loader,test_loader

def CreateOrLoadTrainAndTestLoader(Bible_dataset_DF,train_loader_filepath,test_loader_filepath,create_or_load_string_train_and_test_loader='load'):
    if create_or_load_string_train_and_test_loader in ['Create','create']:
        train_loader,test_loader = CreateTrainAndTestLoader(Bible_dataset_DF,train_loader_filepath,test_loader_filepath)
    else:
        train_loader,test_loader = LoadTrainAndTestLoader(train_loader_filepath,test_loader_filepath)
    return train_loader,test_loader
    
def CreateBERTModel(train_loader,test_loader,BERT_model_filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BERT_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(BERT_model.parameters(),lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=len(train_loader))
    num_epochs = 10
    BERT_model.to(device)
    for epoch in range(num_epochs):
        BERT_model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs = {key: val.to(device) for key, val in batch.items()}
            labels = inputs.pop("labels")
            outputs = BERT_model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss for Epoch {epoch + 1}: {avg_train_loss}")

        BERT_model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Test - Epoch {epoch + 1}"):
                inputs = {key: val.to(device) for key, val in batch.items()}
                labels = inputs.pop("labels")
                outputs = BERT_model(**inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Average test loss for Epoch {epoch + 1}: {avg_test_loss}")
    BERT_model.save_pretrained(BERT_model_filepath)
    return BERT_model

def LoadBERTModel(BERT_model_filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BERT_model = BertForQuestionAnswering.from_pretrained(BERT_model_filepath)
    return BERT_model

def CreateOrLoadBERTModel(train_loader,test_loader,BERT_model_filepath,create_or_load_string_BERT_model):
    if create_or_load_string_BERT_model in ['Create','create']:
        BERT_model = CreateBERTModel(train_loader,test_loader,BERT_model_filepath)
    else:
        BERT_model = LoadBERTModel(BERT_model_filepath)
    return BERT_model

#################################################################################################################
### Main Functions

# Sample usage
if __name__ == "__main__":
    
    #Variables
    prepared_Q_and_A_DF_filepath = 'Additional_Data/Prepared_QandA_Dataset.csv'
    ESV_Bible_DF_filepath = 'Additional_Data/ESV_Bible_DF.csv'
    Bible_dataset_DF_filepath = 'Additional_Data/Bible_dataset_DF.csv'
    train_loader_filepath = 'Additional_Data/BERT_train_dataset.pth'
    test_loader_filepath = 'Additional_Data/BERT_test_dataset.pth'
    BERT_model_filepath = 'Additional_Data/BERT_model'
    
    create_or_load_string_Bible_dataset_DF = 'Load'
    create_or_load_string_train_and_test_loader = 'Load'
    create_or_load_string_BERT_model = 'Create'
    
    #Main Code
    prepared_Q_and_A_DF = LoadDF(prepared_Q_and_A_DF_filepath)
    ESV_Bible_DF = LoadDF(ESV_Bible_DF_filepath)
    Bible_dataset_DF = CreateBibleDatasetDF(prepared_Q_and_A_DF,ESV_Bible_DF,Bible_dataset_DF_filepath)
    train_loader,test_loader = CreateOrLoadTrainAndTestLoader(Bible_dataset_DF,train_loader_filepath,test_loader_filepath,create_or_load_string_train_and_test_loader)
    BERT_model = CreateOrLoadBERTModel(train_loader,test_loader,BERT_model_filepath,create_or_load_string_BERT_model)