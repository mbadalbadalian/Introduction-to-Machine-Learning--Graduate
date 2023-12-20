import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def LoadDF(DF_path):
    dataframe = pd.read_csv(DF_path)
    return dataframe

# importing required modules
import PyPDF2
 
# creating a pdf file object
pdfFileObj = open('Additional_Data/the-4-vedas.pdf', 'rb')
 
# creating a pdf reader object
pdfReader = PyPDF2.PdfReader(pdfFileObj)
 
# creating a page object
pageObj = pdfReader.pages[222]
text = pageObj.extract_text().split('\n')
lines = []
for i in range(len(text)):
    if text[i].split(None, 1)[0].isdigit():
        if text[i+1].split(None, 1)[0].isdigit():
            lines.append(text[i])
        else:
            lines.append(text[i]+ " " + text[i + 1])
            i+=1
mandela = []
mno = 0
for i in range(len(lines)):
    if lines[i][0] == '1':
        mandela.append(mantras)
        mantras = []
    if 'mantras'     not in locals():
        mantras = []
    mantras.append(lines[i][1:])
    

# extracting text from page
print(mandela)
 
# closing the pdf file object
pdfFileObj.close()
'''
ESV_Bible_DF_filename = "Additional_Data/ESV_Bible_DF.csv"
ESV_Bible = LoadDF(ESV_Bible_DF_filename)
# Extract 'id' and 'Text' columns into lists
id_list = ESV_Bible['id'].tolist()
text_list = ESV_Bible['Text'].tolist()

print("ID List:")
print(id_list[:10])  # Print the first 10 elements of the ID list
print("\nText List:")
print(text_list[:10])  # Print the first 10 elements of the Text list
'''