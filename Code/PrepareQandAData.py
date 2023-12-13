import pandas as pd

#################################################################################################################
### Other Functions

def LoadDF(DF_path):
    dataframe = pd.read_csv(DF_path)
    return dataframe

def CreateBookMappingDictionary(prepared_Q_and_A_DF,ESV_Bible_Book_id_DF):
    prepared_Q_and_A_DF_unique_books = prepared_Q_and_A_DF['b'].unique()
    ESV_Bible_Book_id_DF_unique_books = ESV_Bible_Book_id_DF['b'].unique()
    book_mapping_dict = dict(zip(prepared_Q_and_A_DF_unique_books,ESV_Bible_Book_id_DF_unique_books))
    return book_mapping_dict

def CreatePreparedQAndAData(Q_and_A_DF,ESV_Bible_Book_id_DF,prepared_Q_and_A_DF_filepath):
    prepared_Q_and_A_DF = pd.DataFrame(columns=['Questions','Answers','b','c','v'])
    prepared_Q_and_A_DF['Questions'] = Q_and_A_DF['Questions'] 
    prepared_Q_and_A_DF['Answers'] = Q_and_A_DF['Answers']
    prepared_Q_and_A_DF[['b','cv']] = prepared_Q_and_A_DF['Answers'].str.extract(r'(\w+) (\d+:\d+)')
    prepared_Q_and_A_DF[['c','v']] = prepared_Q_and_A_DF['cv'].str.split(':', expand=True)
    prepared_Q_and_A_DF = prepared_Q_and_A_DF.drop('cv',axis=1)
    prepared_Q_and_A_DF['Questions'] = prepared_Q_and_A_DF['Questions'].str.replace(r'^\d+\.\s', '', regex=True)
    prepared_Q_and_A_DF['Answers'] = prepared_Q_and_A_DF['Answers'].str.replace(r'^\d+\.\s', '', regex=True)
    prepared_Q_and_A_DF['Answers'] = prepared_Q_and_A_DF['Answers'].str.replace(r'\s\(.+?\)', '', regex=True)
    book_mapping_dict = CreateBookMappingDictionary(prepared_Q_and_A_DF,ESV_Bible_Book_id_DF)
    prepared_Q_and_A_DF['b'] = prepared_Q_and_A_DF['b'].replace(book_mapping_dict)
    ESV_Bible_Book_id_dict = ESV_Bible_Book_id_DF.set_index('b')['id'].to_dict()
    prepared_Q_and_A_DF['b'] = prepared_Q_and_A_DF['b'].replace(ESV_Bible_Book_id_dict)
    prepared_Q_and_A_DF.to_csv(prepared_Q_and_A_DF_filepath,index=False)
    return prepared_Q_and_A_DF

def CreateOrLoadTokenizedData(Q_and_A_DF,ESV_Bible_Book_id_DF,prepared_Q_and_A_DF_filepath,create_or_load_string='load'):
    if create_or_load_string in ['Create','create']:
        prepared_Q_and_A_DF = CreatePreparedQAndAData(Q_and_A_DF,ESV_Bible_Book_id_DF,prepared_Q_and_A_DF_filepath)
    else:
        prepared_Q_and_A_DF = LoadDF(prepared_Q_and_A_DF_filepath)
    return prepared_Q_and_A_DF
    
#################################################################################################################
### Main Functions

if __name__ == "__main__":

    #Variables
    initial_Q_and_A_DF_filepath = 'Initial_Data/QandA_Dataset.csv'
    ESV_Bible_Book_id_DF_filepath = 'Additional_Data/ESV_Bible_Book_id_DF.csv'
    prepared_Q_and_A_DF_filepath = 'Additional_Data/Prepared_QandA_Dataset.csv'
    create_or_load_string = 'Create'
    
    #Main Code
    Q_and_A_DF = LoadDF(initial_Q_and_A_DF_filepath)
    ESV_Bible_Book_id_DF = LoadDF(ESV_Bible_Book_id_DF_filepath)
    prepared_Q_and_A_DF = CreateOrLoadTokenizedData(Q_and_A_DF,ESV_Bible_Book_id_DF,prepared_Q_and_A_DF_filepath,create_or_load_string)
    
    