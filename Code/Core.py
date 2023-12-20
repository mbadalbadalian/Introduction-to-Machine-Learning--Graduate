import Base_Chatbot
import Code.Sentiment_Analysis as Sentiment
import warnings

# To suppress all warnings
warnings.filterwarnings("ignore")

def Bible_Questions(model):
    user_input = input('What is the question you wanted to ask?')
    
    user_input = input("Are you satisfied with the anser\n")
    opinion = Sentiment.model_use(user_input, model)
    if opinion == 'neg':
        print('I am sorry to hear that. Can you please repeat the question more concisely?')
        Bible_Questions(model)
    else:
        print('Thats Lovely! Returning to general chatbot now')
    return


model = Sentiment.sentiment()
responses = ['Hello! It is a pleasure meeting you',
              'I was made by Chris Verghese and Matthew Badal-Badalion',
              'I am supposed to answer verses from the bible in response to user questions',
              'I am supposed to answer verses from the bible in response to user questions',
              'I work using BERT and Zero Shot Classification',
              'I work using BERT and Zero Shot Classification',
              'I am unable to properly function right now',
              'Sure thing \nloading in the ML model now!! \n',
               'Thank you for using our chatbot' ]
labels1 = ['greeting', 'creator', 'functions', 'What you do','architecture', 'How you work','weakness', 'want to ask a question about the Bible','No']
print('If you want to ask questions about the Bible, let me know first')
while(True):
# Accepting string input from the user
    user_input = input("Any Questions: (Enter 'No' to leave)\n")
    labels, scores = Base_Chatbot.zero_shot_classification(user_input,labels1)
    label_ind = Base_Chatbot.responses(responses,scores)
    if label_ind == 7:
        Bible_Questions(model)
    if label_ind == 8:
        break

