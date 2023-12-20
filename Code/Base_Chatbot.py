# load the sentence-bert model from the HuggingFace model hub
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
import warnings

# To suppress all warnings
warnings.filterwarnings("ignore")


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
    return labels,similarities

def responses(responses,scores):
    score = scores.argmax()
    if scores[score]<0.4:
        print('Sorry!! Can you be more specific.')
        print('If your question is about the bible, please let me know')
    else:
        print(responses[score])
    return score

if __name__ == "__main__":  
    labels = ['greeting', 'creator', 'functions', 'What you do','architecture', 'How you work','weakness']
    zero_shot_classification('How do you do, dear sir?', labels)
