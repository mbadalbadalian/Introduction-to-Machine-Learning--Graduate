# load the sentence-bert model from the HuggingFace model hub
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
import warnings
import spacy

# To suppress all warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Reconstruct the processed text
    processed_text = ' '.join(tokens)
    
    return processed_text

def imp(question):

    # Preprocessing and tokenization (you might need to add more preprocessing steps)
    # Here, 'preprocess_text' is a function that preprocesses text
    preprocessed_question = preprocess_text(question)

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the dataset to build the vocabulary and transform the question
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_question])

    # Get feature names (words) and their corresponding TF-IDF scores
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Combine feature names and their scores into a dictionary
    word_scores = dict(zip(feature_names, tfidf_scores))

    # Sort the words based on TF-IDF scores (higher scores mean more important words)
    important_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    # Print top N important words
    top_n = 3
    print(f"Top {top_n} important words in the question:")
    for word, score in important_words[:top_n]:
        print(f"Word: {word}, TF-IDF Score: {score}")
    return important_words[:top_n]

def Tokenizer(sentence):
    # List of lines to compare
    # Load the SpaCy English language model
    nlp = spacy.load("en_core_web_sm")
    # Process the text with SpaCy
    lemmatized_text = ' '.join([token.lemma_  for token in nlp(sentence)])
     # Process the text with SpaCy
    doc = nlp(lemmatized_text)
    # Extract nouns and verbs from the tagged words
    nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN","VERB"]]
    return nouns

def zero_shot_classification(sentence,labels):
    tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
    model = AutoModel.from_pretrained('deepset/sentence_bert')

    

    # run inputs through model and mean-pool over the sequence
    # dimension to get sequence-level representations
    inputs = tokenizer.batch_encode_plus([sentence] + labels,
                                         return_tensors='pt',
                                         pad_to_max_length=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    output = model(input_ids, attention_mask=attention_mask)[0]
    sentence_rep = output[:1].mean(dim=1)
    label_reps = output[1:].mean(dim=1)

    # now find the labels with the highest cosine similarities to
    # the sentence
    similarities = F.cosine_similarity(sentence_rep, label_reps)
    closest = similarities.argsort(descending=True)
    for ind in closest:
        print(f'label: {labels[ind]} \t similarity: {similarities[ind]}')
    return similarities

def responses(responses,scores):
    if max(scores)<0.4:
        print('Again')
    else:
        print(responses[scores.argmax()])

if __name__ == "__main__":  
    labels = ['greeting', 'creator', 'functions', 'What you do','architecture', 'How you work','weakness']
    sen = 'Who is the mother of Jesus?'
    simil = [0] * len(labels)
    n = Tokenizer(sen)
    words = imp(sen)
    for i,s in words:
        print(i)
        sim = zero_shot_classification(i, labels)
        simil = [x + y for x, y in zip(simil, sim)]
    for ind in range(len(labels)):
        print(f'label: {labels[ind]} \t similarity: {simil[ind]}')
    