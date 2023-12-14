import Base_Chatbot
responses = ['Hello! It is a pleasure meeting you',
              'I was made by Chris Verghese and Matthew Badal-Badalion',
              'I am supposed to answer verses from the bible in response to user questions',
              'I am supposed to answer verses from the bible in response to user questions',
              'I work using BERT and Zero Shot Classification',
              'I work using BERT and Zero Shot Classification',
              'I am unable to properly function right now']
labels1 = ['greeting', 'creator', 'functions', 'What you do','architecture', 'How you work','weakness']
labels2 = ['greeting', 'creator', 'functions', 'What you do','architecture', 'How you work','weakness']
while(True):
# Accepting string input from the user
    user_input = input("Enter a string: ")
    if user_input != '0':
        labels, scores = Base_Chatbot.zero_shot_classification(user_input,labels1)
        Base_Chatbot.responses(responses,scores)
    else:
        break