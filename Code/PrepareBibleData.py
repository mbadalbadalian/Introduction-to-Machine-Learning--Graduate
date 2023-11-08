import numpy as np
import xml.etree.ElementTree as ET
import json
from collections import OrderedDict

#################################################################################################################
### Other Functions

#SaveJSONData Function
def SaveJSONData(json_data,json_filename):
    #Saving data as JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(json_data, json_file)
    return

#LoadJSONData Function
def LoadJSONData(json_filename):
    #Loading JSON file into variable
    with open(json_filename, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

#SaveTxtFile Function
def SaveTxtFile(txt_data,txt_filename):
    #Saving txt file
    with open(txt_filename, 'w') as txt_file:
        txt_file.write(txt_data)
    return

#LoadTxtData Function
def LoadTxtData(txt_filename):
    #Loading txt file
    with open(txt_filename,'r') as txt_file:
        json_data = txt_file.read()
    return json_data

#CreateAndSaveTxtFileFromList Function
def CreateAndSaveTxtFileFromList(list_data,txt_filename):
    #Writine a txt file from a list
    with open(txt_filename, 'w', encoding='utf-8') as txt_file:
        for item in list_data:
            txt_file.write(item + '\n')
    return txt_file

#CreateBibleDictionary Function
def CreateBibleDictionary(xml_filename,dictionary_filename):
    #Parse XML content
    xml_parsed_data = ET.parse(xml_filename)
    xml_parsed_data_root = xml_parsed_data.getroot()

    #Initialize dictionary
    ESV_Bible_dict = {}

    #Loop through each book, chapter and verse to create dictionary
    for book in xml_parsed_data_root.findall("b"):
        book_name = book.get("n")
        ESV_Bible_dict[book_name] = {}
        for chapter in book.findall("c"):
            chapter_number = chapter.get("n")
            ESV_Bible_dict[book_name][chapter_number] = {}
            for verse in chapter.findall("v"):
                verse_number = verse.get("n")
                verse_text = verse.text
                ESV_Bible_dict[book_name][chapter_number][verse_number] = verse_text

    #Save dictionary
    SaveJSONData(ESV_Bible_dict,dictionary_filename)
    return ESV_Bible_dict

#ProcessXML Function
def ProcessXML(xml_parsed_data_root):
    output = OrderedDict()
    
    #Loop through each book, chapter and verse
    for book in xml_parsed_data_root.findall('.//b'):
        book_number = book.get('n')
        for chapter in book.findall('.//c'):
            chapter_number = chapter.get('n')
            for verse in chapter.findall('.//v'):
                verse_number = verse.get('n')
                verse_text = verse.text
                output[f'"{book_number} {chapter_number}:{verse_number}"'] = verse_text
    return output

#CreateBibleText Function
def CreateBibleText(xml_filename,txt_filename):
    ESV_Bible_xml = LoadTxtData(xml_filename)
    ESV_Bible_parsed_xml_root = ET.fromstring(ESV_Bible_xml)
    ESV_Bible_list = ProcessXML(ESV_Bible_parsed_xml_root)
    ESV_Bible_txt = CreateAndSaveTxtFileFromList(ESV_Bible_list,txt_filename)
    return ESV_Bible_txt

#CreateOrLoad Function
def CreateOrLoad(Bible_xml_filename,dictionary_filename,txt_filename,create_or_load_string = 'load'):
    if create_or_load_string in ['Create','create']:
        ESV_Bible_dict = CreateBibleDictionary(Bible_xml_filename,Bible_dictionary_filename)
        ESV_Bible_txt = CreateBibleText(Bible_xml_filename,Bible_txt_filename)
    else:
        ESV_Bible_dict = LoadJSONData(dictionary_filename)
        ESV_Bible_txt = LoadTxtData(txt_filename)
    return ESV_Bible_dict,ESV_Bible_txt

#################################################################################################################
### Main Functions

#################################################################################################################
### Variables

Bible_xml_filename = "Initial_Data\\ESVBible_Database.xml"
Bible_dictionary_filename = "Additional_Data\\ESV_Bible_Dictionary.json"
Bible_txt_filename = "Additional_Data\\ESV_Bible_Text.txt"
Bible_list_filename = "Additional_Data\\ESV_Bible_List.json"
create_or_load_string = 'Create'

#################################################################################################################
### Main Code

ESV_Bible_dict,ESV_Bible_txt = CreateOrLoad(Bible_xml_filename,Bible_dictionary_filename,Bible_txt_filename,create_or_load_string)