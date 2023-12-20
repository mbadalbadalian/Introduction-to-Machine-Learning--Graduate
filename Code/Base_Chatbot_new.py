# load the sentence-bert model from the HuggingFace model hub
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
import warnings
import spacy
import classification
import numpy as np
# To suppress all warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer



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
    #for ind in closest:
    #    print(f'label: {labels[ind]} \t similarity: {similarities[ind]}')
    return similarities

def responses(responses,scores):
    if max(scores)<0.4:
        print('Again')
    else:
        print(responses[scores.argmax()])

if __name__ == "__main__":  
    Bible_Tokens = classification.Get_Chapter_Tokens()
    labels = Bible_Tokens['Matthew']
    sen = 'When did Jesus meet the devil?'
    simil = zero_shot_classification(sen,labels)
    # Assuming 'simil' is a tensor
    simil = simil.detach().numpy()  # Detach the tensor and convert to NumPy array

    argmax_index = np.argmax(simil)
    for i in range(len(simil)):
        print(f'label: {i} \t similarity: {simil[i]}')
