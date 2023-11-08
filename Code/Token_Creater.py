import spacy
import PrepareBibleData

def tokenizer(sentence):
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

def Bible_Tokens():
    Bible = PrepareBibleData.GetBibleLines()
    Tokens = {}
    for i in Bible.keys(): #Books
        Book = {}
        for j in Bible[i].keys(): #Chapters
            Chapter = {}
            print(j)
            for k in Bible[i][j].keys():
                word =  tokenizer(Bible[i][j][k])
                Chapter[k] = word
            Book[j] = Chapter
        Tokens[i] = Book
    print(Tokens)
        
#The main function is the driver for the code
if __name__ == "__main__":     
    Bible_Tokens()