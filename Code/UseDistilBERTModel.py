from transformers import pipeline
from transformers import DistilBertForQuestionAnswering,DistilBertTokenizer
import pandas as pd

def LoadFineTunedBERTModel(model_path):
    tokenizer = DistilBertTokenizer.from_pretrained('bert-base-uncased')
    model = DistilBertForQuestionAnswering.from_pretrained(model_path)
    return model, tokenizer

def LoadContext(ESV_Bible_DF_filepath):
    bible_df = pd.read_csv(ESV_Bible_DF_filepath)
    context = bible_df['Text'][bible_df['b'] == 1].str.cat(sep=' ')
    #context = bible_df['Text'].str.cat(sep=' ')
    return context

def answer_question_using_fine_tuned_model(model, tokenizer, question, context):
    question_answering_pipeline = pipeline('question-answering',model=model,tokenizer=tokenizer)
    result = question_answering_pipeline(context=context, question=question)
    return result['answer']

if __name__ == "__main__":
    
    #Variables
    model_path = 'Models/BERT_model_fine_tuned'
    ESV_Bible_DF_filepath = 'Additional_Data/ESV_Bible_DF.csv'
    
    context = "Your context goes here..."
    question = "Your question goes here..."

    answer = answer_question_using_fine_tuned_model(fine_tuned_model, tokenizer, question, context)
    print("Question:", question)
    print("Answer:", answer)