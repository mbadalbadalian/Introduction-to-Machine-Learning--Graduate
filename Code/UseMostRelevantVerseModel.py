import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

def LoadDF(DF_path):
    dataframe = pd.read_csv(DF_path)
    return dataframe

def load_bible_books(ESV_Bible_Book_id_filepath):
    # Load the Bible books dataset from the CSV file or another format
    # Adjust the file path accordingly
    return pd.read_csv(ESV_Bible_Book_id_filepath)

def load_bert_model_and_tokenizer(BERT_most_relevant_verse_model_fine_tuned_filepath):
    # Load pre-trained BERT tokenizer and model
    model = BertModel.from_pretrained(BERT_most_relevant_verse_model_fine_tuned_filepath)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

def calculate_bible_embeddings(bible_text, model, tokenizer):
    # Tokenize and encode the Bible verses
    encoded_bible = tokenizer(bible_text, return_tensors='pt', padding=True, truncation=True)

    # Get BERT embeddings for the Bible verses
    with torch.no_grad():
        bible_embeddings = model(**encoded_bible).last_hidden_state[:, 0, :]
    
    return bible_embeddings

def save_and_load_embeddings(bible_embeddings, file_path):
    # Save the embeddings
    torch.save(bible_embeddings, file_path)

    # Load the embeddings
    loaded_embeddings = torch.load(file_path)
    return loaded_embeddings

def process_question(question, model, tokenizer, bible_embeddings, bible_data, bible_books):
    # Tokenize and encode the question
    encoded_question = tokenizer(question, return_tensors='pt', padding=True, truncation=True)

    # Get BERT embeddings for the question
    with torch.no_grad():
        question_embedding = model(**encoded_question).last_hidden_state[:, 0, :]

    # Calculate cosine similarity
    similarities = cosine_similarity(question_embedding, bible_embeddings)

    # Get the index of the most similar verse
    most_similar_verse_index = similarities.argmax().item()

    # Extract metadata for the most similar verse
    metadata = bible_data.iloc[most_similar_verse_index][['b', 'c', 'v']]
    book_id, chapter, verse = metadata['b'], metadata['c'], metadata['v']

    # Map book_id to book name
    book_name = bible_books[bible_books['id'] == book_id]['b'].values[0]

    # Formulate the answer
    answer = f"{bible_data.iloc[most_similar_verse_index]['Text']} ({book_name} {chapter}:{verse})"
    return f"Q: {question}\nA: {answer}"

if __name__ == "__main__":
    # Load Bible data
    ESV_Bible_DF_filepath = 'Additional_Data/ESV_Bible_DF.csv'
    ESV_Bible_Book_id_filepath = 'Additional_Data/ESV_Bible_Book_id_DF.csv'
    BERT_most_relevant_verse_model_fine_tuned_filepath = 'Models/BERT_most_relevant_verse_model_fine_tuned'

    bible_data = LoadDF(ESV_Bible_DF_filepath)
    bible_books = load_bible_books(ESV_Bible_Book_id_filepath)

    # Extract text and metadata columns
    #bible_text = bible_data['Text'][bible_data['b'] == 1][bible_data['c'] == 1].tolist()
    bible_text = bible_data['Text'].tolist()

    # Load BERT model and tokenizer
    model, tokenizer = load_bert_model_and_tokenizer(BERT_most_relevant_verse_model_fine_tuned_filepath)

    # Calculate and save Bible embeddings
    bible_embeddings = calculate_bible_embeddings(bible_text, model, tokenizer)
    save_and_load_embeddings(bible_embeddings, 'bible_embeddings_original.pt')

    # Example questions
    questions = ["Who is the mother of Jesus?","Who is the son of Judah?"]

    # Process each question
    for question in questions:
        answer = process_question(question, model, tokenizer, bible_embeddings, bible_data, bible_books)
        print(answer)