from transformers import pipeline, BertForQuestionAnswering, BertTokenizer, DistilBertForQuestionAnswering, DistilBertTokenizer
import torch
import pandas as pd


# ... (previous code)

def AskQuestion(context, question, fine_tuned_model, tokenizer, max_seq_length=512):
    # Tokenize the input text
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, max_length=max_seq_length)

    # Perform inference
    with torch.no_grad():
        outputs = fine_tuned_model(**inputs)

    # Extract answer
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_index = torch.argmax(start_logits, dim=1).item()
    end_index = torch.argmax(end_logits, dim=1).item()

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_index:end_index+1]))

    return answer

def LoadContext(ESV_Bible_DF_filepath):
    bible_df = pd.read_csv(ESV_Bible_DF_filepath)
    context = bible_df['Text'][bible_df['b'] == 40][bible_df['c'] == 1].str.cat(sep=' ')
    #context = bible_df['Text'].str.cat(sep=' ')
    return context

def answer_question(context, question):
    question_answering_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='distilbert-base-cased-distilled-squad')
    result = question_answering_pipeline(context=context, question=question)
    return result['answer']

if __name__ == "__main__":
    ESV_Bible_DF_filepath = 'Additional_Data/ESV_Bible_DF.csv'
    ditilBERT_model_fine_tuned_filepath = 'Models/DistilBERT_model_fine_tuned'
    BERT_model_fine_tuned_filepath = 'Models/BERT_model_fine_tuned'
    
    print("**********************Using DistilBERT************************")
    fine_tuned_model = DistilBertForQuestionAnswering.from_pretrained(ditilBERT_model_fine_tuned_filepath)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

    context = LoadContext(ESV_Bible_DF_filepath)
    #context = 'This is the genealogy of Jesus the Messiah the son of David, the son of Abraham: Abraham was the father of Isaac, Isaac the father of Jacob, Jacob the father of Judah and his brothers, Judah the father of Perez and Zerah, whose mother was Tamar, Perez the father of Hezron, Hezron the father of Ram, Ram the father of Amminadab, Amminadab the father of Nahshon, Nahshon the father of Salmon, Salmon the father of Boaz, whose mother was Rahab, Boaz the father of Obed, whose mother was Ruth, Obed the father of Jesse, and Jesse the father of King David. David was the father of Solomon, whose mother had been Uriah’s wife, Solomon the father of Rehoboam, Rehoboam the father of Abijah, Abijah the father of Asa, Asa the father of Jehoshaphat, Jehoshaphat the father of Jehoram, Jehoram the father of Uzziah, Uzziah the father of Jotham, Jotham the father of Ahaz, Ahaz the father of Hezekiah, Hezekiah the father of Manasseh, Manasseh the father of Amon, Amon the father of Josiah,'
    question = "Who is Jesus?"

    #answer_context = answer_question_with_context(context, question)
    answer = answer_question(context, question)
    print("Question:", question)
    print("Answer:", answer)
    
    ################################################################
    print("**********************Using Bert************************")
    fine_tuned_model = BertForQuestionAnswering.from_pretrained(BERT_model_fine_tuned_filepath)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    #answer_context = answer_question_with_context(context, question)
    answer = answer_question(context, question)
    print("Question:", question)
    print("Answer:", answer)
    
    
    