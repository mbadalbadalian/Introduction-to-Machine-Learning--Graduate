from transformers import pipeline
from transformers import DistilBertForQuestionAnswering,DistilBertTokenizer
import pandas as pd

def LoadFineTunedDistilBERTModel(model_path):
    tokenizer = DistilBertTokenizer.from_pretrained('bert-base-uncased')
    model = DistilBertForQuestionAnswering.from_pretrained(model_path)
    return model,tokenizer

def LoadContext(ESV_Bible_DF_filepath):
    bible_df = pd.read_csv(ESV_Bible_DF_filepath)
    context = bible_df['Text'][bible_df['b'] == 1].str.cat(sep=' ')
    #context = bible_df['Text'].str.cat(sep=' ')
    return context

def AnswerQuestionSsingFineTunedModel(model,tokenizer,question,context):
    question_answering_pipeline = pipeline('question-answering',model=model,tokenizer=tokenizer)
    result = question_answering_pipeline(context=context, question=question)
    return result['answer']

if __name__ == "__main__":
    
    #Variables
    model_path = 'Models/DistilBERT_model_fine_tuned'
    ESV_Bible_DF_filepath = 'Additional_Data/ESV_Bible_DF.csv'
    
    fine_tuned_model,tokenizer = LoadFineTunedDistilBERTModel(model_path) 
    context = LoadContext(ESV_Bible_DF_filepath)
    questions = ["Who is the mother of Jesus?","Who is the son of Judah?"]

    for question in questions:
        answer = AnswerQuestionSsingFineTunedModel(fine_tuned_model,tokenizer,question,context)
        print("Question:", question)
        print("Answer:", answer)
        
    