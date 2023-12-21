import torch
from transformers import BertForQuestionAnswering, BertTokenizer
import pandas as pd

def LoadFineTunedBERTModel(model_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained(model_path)
    return model, tokenizer

def LoadContext(ESV_Bible_DF_filepath):
    bible_df = pd.read_csv(ESV_Bible_DF_filepath)
    context = bible_df['Text'][bible_df['b'] == 8].str.cat(sep=' ')
    #context = bible_df['Text'].str.cat(sep=' ')
    return context

def PredictAnswer(model, tokenizer, context, question, chunk_size=512, overlap=256):
    answers = []

    start_idx = 0
    max_confidence = float('-inf')
    best_answer = ""

    while start_idx < len(context):
        end_idx = start_idx + chunk_size
        if end_idx > len(context):
            end_idx = len(context)

        start_idx = max(0, end_idx - overlap)

        chunk = context[start_idx:end_idx]
        encoding = tokenizer(
            text=question,
            text_pair=chunk,
            return_tensors='pt',
            truncation=True,
            padding="longest",
            max_length=chunk_size,
        )

        with torch.no_grad():
            outputs = model(**encoding)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_index = torch.argmax(start_logits, dim=1).item()
        end_index = torch.argmax(end_logits, dim=1).item()
        answer = tokenizer.decode(encoding['input_ids'][0][start_index:end_index + 1])
        answers.append(answer)

        # Update the best answer based on confidence scores
        confidence = start_logits[0][start_index].item() + end_logits[0][end_index].item()
        if confidence > max_confidence:
            max_confidence = confidence
            best_answer = answer

        start_idx += chunk_size - overlap

    return best_answer

def PredictAnswerNoChunks(model, tokenizer, context, question):
    encoding = tokenizer(
        text=question,
        text_pair=context,
        return_tensors='pt',
        truncation=True,
        padding="longest",
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**encoding)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits).item()
    end_index = torch.argmax(end_logits).item()
    answer = tokenizer.decode(encoding['input_ids'][0][start_index:end_index + 1])

    return answer

if __name__ == "__main__":
    
    #Variables
    model_path = 'Models/DistilBERT_model_fine_tuned'
    ESV_Bible_DF_filepath = 'Additional_Data/ESV_Bible_DF.csv'
    
    #Main Code
    fine_tuned_model,tokenizer = LoadFineTunedBERTModel(model_path)
    context = LoadContext(ESV_Bible_DF_filepath)
    questions = ["Who is the mother of Jesus?","Who is the son of Judah?"]
    
    for question in questions:
        answer = PredictAnswerNoChunks(fine_tuned_model,tokenizer,context,question)
        print("Question:", question)
        print("Answer:", answer)