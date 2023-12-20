import Base_Chatbot
import Sentiment_Analysis as Sentiment
import warnings

# To suppress all warnings
warnings.filterwarnings("ignore")

def Religious_Questions(model,book):
    while(True):
        if book == 0:
            user_input = input('What is the question you wanted to ask about the Bible ESV edition?\n')
            print(user_input)
        elif book == 1:
            user_input = input('What is the question you wanted to ask about the Bhagwad Gita translated by Swami Sivananda?\n')
            print(user_input)
        user_satisfaction = input("Are you satisfied with the answer\n")
        opinion = Sentiment.model_use(user_satisfaction, model)
        if opinion == 'neg':
            print('I am sorry to hear that. Can you please repeat the question more concisely?')
        else:
            print('That\'s Great! Returning to general chatbot now')
            break
    return

if __name__ == "__main__":  
    model = Sentiment.sentiment()
    responses = ['Hello! It is a pleasure meeting you',
                  'I was made by Chris Verghese and Matthew Badal-Badalion',
                  'I am supposed to answer verses from the bible in response to user questions',
                  'I am supposed to answer verses from the bible in response to user questions',
                  'I work using BERT and Zero Shot Classification',
                  'I work using BERT and Zero Shot Classification',
                  'I am unable to properly function right now',
                  'Sure thing \nloading in the ML model now!! \n',
                   'Thank you for using our chatbot' ,
                   'Sure thing \nloading in the ML model now!! \n']
    labels1 = ['greeting', 'creator', 'functions', 'What you do','architecture', 'How you work','weakness', 'want to ask a question about the Bible','No', 'want to ask a question about the Bhagwad Gita']
    print('If you want to ask questions about the Bible or Bhagwad Gita, let me know first')
    while(True):
    # Accepting string input from the user
        user_input = input("Any Questions: (Enter 'No' to leave)\n")
        labels, scores = Base_Chatbot.zero_shot_classification(user_input,labels1)
        label_ind = Base_Chatbot.responses(responses,scores)
        if label_ind == 7:
            Religious_Questions(model,0)
        if label_ind == 9:
            Religious_Questions(model,1)
        if label_ind == 8:
            break


