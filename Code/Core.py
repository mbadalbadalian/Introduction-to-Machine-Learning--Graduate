import Base_Chatbot
import tester

def Bible_Questions():
    print('What is the question you wanted to ask?')
    print('Are you satisfied with the answer?')


responses = ['Hello! It is a pleasure meeting you',
              'I was made by Chris Verghese and Matthew Badal-Badalion',
              'I am supposed to answer verses from the bible in response to user questions',
              'I am supposed to answer verses from the bible in response to user questions',
              'I work using BERT and Zero Shot Classification',
              'I work using BERT and Zero Shot Classification',
              'I am unable to properly function right now',
              'Sure thing \nloading in the ML model now!! \n' ]
labels1 = ['greeting', 'creator', 'functions', 'What you do','architecture', 'How you work','weakness', 'want to ask a question about the Bible']
print('If you want to ask questions about the Bible, let me know first')
while(True):
# Accepting string input from the user
    user_input = input("Any Questions: \n")
    if user_input != '0':
        labels, scores = Base_Chatbot.zero_shot_classification(user_input,labels1)
        label_ind = Base_Chatbot.responses(responses,scores)
        if label_ind == 7:
            Bible_Questions()
    else:
        break
