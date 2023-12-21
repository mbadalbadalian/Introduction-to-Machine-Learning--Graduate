import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

def LoadDF(DF_path):
    dataframe = pd.read_csv(DF_path)
    return dataframe

def LoadBertModelAndTokenizer(BERT_most_relevant_verse_model_fine_tuned_filepath):
    model = BertModel.from_pretrained(BERT_most_relevant_verse_model_fine_tuned_filepath)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
def LoadBibleDFAndText(ESV_ESV_Bible_DF_filepath):
    ESV_Bible_DF = pd.read_csv(ESV_ESV_Bible_DF_filepath)
    ESV_Bible_DF = ESV_Bible_DF['Text'][ESV_Bible_DF['b'] == 1][ESV_Bible_DF['c'] == 1].tolist()
    ESV_Bible_text = ESV_Bible_DF['Text'].tolist()
    return ESV_Bible_DF,ESV_Bible_text

def CalculateBibleEmbeddings(ESV_Bible_text, model, tokenizer):
    encoded_bible = tokenizer(ESV_Bible_text, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        bible_embeddings = model(**encoded_bible).last_hidden_state[:, 0, :]
    
    return bible_embeddings

def save_and_load_embeddings(bible_embeddings, file_path):
    torch.save(bible_embeddings, file_path)
    loaded_embeddings = torch.load(file_path)
    return loaded_embeddings

def ProcessQuestion(question, model, tokenizer, bible_embeddings, ESV_Bible_DF, bible_books):
    encoded_question = tokenizer(question, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        question_embedding = model(**encoded_question).last_hidden_state[:, 0, :]

    similarities = cosine_similarity(question_embedding, bible_embeddings)

    most_similar_verse_index = similarities.argmax().item()

    metadata = ESV_Bible_DF.iloc[most_similar_verse_index][['b', 'c', 'v']]
    book_id, chapter, verse = metadata['b'], metadata['c'], metadata['v']

    book_name = bible_books[bible_books['id'] == book_id]['b'].values[0]

    answer = f"{ESV_Bible_DF.iloc[most_similar_verse_index]['Text']} ({book_name} {chapter}:{verse})"
    return f"Q: {question}\nA: {answer}"

if __name__ == "__main__":
    #Variables
    ESV_ESV_Bible_DF_filepath = 'Additional_Data/ESV_ESV_Bible_DF.csv'
    ESV_Bible_Book_id_filepath = 'Additional_Data/ESV_Bible_Book_id_DF.csv'
    BERT_most_relevant_verse_model_fine_tuned_filepath = 'Models/BERT_most_relevant_verse_model_fine_tuned'

    #Main Code
    ESV_Bible_DF = LoadDF(ESV_ESV_Bible_DF_filepath)
    bible_books = LoadDF(ESV_Bible_Book_id_filepath)
    
    model,tokenizer = LoadBertModelAndTokenizer(BERT_most_relevant_verse_model_fine_tuned_filepath)

    # Calculate and save Bible embeddings
    ESV_Bible_DF,ESV_Bible_text = LoadBibleDFAndText(ESV_ESV_Bible_DF_filepath)
    bible_embeddings = CalculateBibleEmbeddings(ESV_Bible_text,model,tokenizer)
    save_and_load_embeddings(bible_embeddings, 'bible_embeddings_original.pt')

    # Example questions
    questions = ["Who is the mother of Jesus?","Who is the son of Judah?"]

    # Process each question
    for question in questions:
        answer = ProcessQuestion(question,model,tokenizer,bible_embeddings,ESV_Bible_DF,bible_books)
        print(answer)