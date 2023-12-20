import Base_Chatbot
import pandas as pd

def LoadContext(ESV_Bible_DF_filepath):
    bible_df = pd.read_csv(ESV_Bible_DF_filepath)
    context = []
    context = bible_df['Text'][(bible_df['b'] == 40) & (bible_df['c'] == 1)].tolist()
    #context = bible_df['Text'].str.cat(sep=' ')
    return context


ESV_Bible_DF_filepath = 'Additional_Data/ESV_Bible_DF.csv'
#context = LoadContext(ESV_Bible_DF_filepath)

context = ["Abraham was the father of Isaac,"," Isaac the father of Jacob,"," Jacob the father of Judah and"," Judah the father of Perez ","and Zerah, whose mother was Tamar,"," Perez the father of Hezron, ","Hezron the father of Ram,"]
question = "Who is the father of Perez?"

#while(True):
#user_input = input("Enter a string: ")
user_input = question
labels_list = ['father','Perez']
scores_list = []
for i in context:
    labels, scores = Base_Chatbot.zero_shot_classification(i,labels_list)
    scores_list.append(scores)
print(context[scores_list.index(max(scores_list))])
print(max(scores_list))


