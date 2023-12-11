from transformers import BertForQuestionAnswering, BertTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
import torch

#################################################################################################################
### Other Functions

def TokenizeData(fine_tuning_data, save_path=None):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_data = tokenizer(
        [(entry["question"], entry["answer"]) for entry in fine_tuning_data],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    
    if save_path:
        torch.save(tokenized_data, save_path)
    
    return tokenized_data

def LoadTokenizedData(path):
    return torch.load(path)

class BibleDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_data["input_ids"][idx],
            "attention_mask": self.tokenized_data["attention_mask"][idx],
            "start_positions": self.tokenized_data["start_positions"][idx],
            "end_positions": self.tokenized_data["end_positions"][idx],
        }

def CreateDataLoader(dataset, batch_size=2, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def FineTuneModel(model, dataloader, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                start_positions=batch["start_positions"],
                end_positions=batch["end_positions"],
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

def SaveModel(model, path):
    model.save_pretrained(path)

def LoadModel(path):
    return BertForQuestionAnswering.from_pretrained(path)

def AnswerQuestion(model, tokenizer, user_question, passage):
    inputs = tokenizer(user_question, passage, return_tensors="pt")
    outputs = model(**inputs)
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1
    answer_tokens = inputs["input_ids"][0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer

#################################################################################################################
### Main Functions

#The main function is the driver for the code
if __name__ == "__main__":
    
    #Variables
    fine_tuning_data = [
    {"question": "What did God create?", "answer": "the heavens and the earth."},
    {"question": "Who were the twelve disciples?", "answer": "Peter, James, John, etc."},
    {"question": "Describe the parable of the prodigal son.", "answer": "A son squanders his inheritance and returns home to a welcoming father."},
    ]
    tokenized_data_path = "tokenized_data.pt"
    tokenized_data = TokenizeData(fine_tuning_data,save_path=tokenized_data_path)

    loaded_tokenized_data = LoadTokenizedData(tokenized_data_path)

    dataset = BibleDataset(loaded_tokenized_data)
    dataloader = CreateDataLoader(dataset)

    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

    optimizer = AdamW(model.parameters(), lr=5e-5)
    FineTuneModel(model,dataloader,optimizer)

    SaveModel(model,"fine_tuned_bert_model")

    loaded_model = LoadModel("fine_tuned_bert_model")

    user_question = "What is the parable of the prodigal son?"
    passage = "The parable of the prodigal son is about a son who squanders his inheritance..."
    answer = AnswerQuestion(loaded_model,tokenizer,user_question, passage)

    print(f"User Question: {user_question}")
    print(f"Answer: {answer}")
