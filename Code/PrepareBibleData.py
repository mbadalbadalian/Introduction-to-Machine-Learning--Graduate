import numpy as np
import xml.etree.ElementTree as ET
import json
from collections import OrderedDict

#################################################################################################################
### Other Functions

def SaveJSONData(json_data,json_filename):
    with open(json_filename, 'w') as json_file:
        json.dump(json_data, json_file)
    return

def LoadJSONData(json_filename):
    with open(json_filename, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def SaveTxtFile(txt_data,txt_filename):
    with open(txt_filename, 'w') as txt_file:
        txt_file.write(txt_data)
    return

def LoadTxtData(txt_filename):
    with open(txt_filename,'r') as txt_file:
        json_data = txt_file.read()
    return json_data

def CreateAndSaveTxtFileFromList(list_data,txt_filename):
    with open(txt_filename, 'w', encoding='utf-8') as txt_file:
        for item in list_data:
            txt_file.write(item + '\n')
    return txt_file

def CreateBibleDictionary(xml_filename,dictionary_filename):
    tree = ET.parse(xml_filename)
    root = tree.getroot()

    ESV_Bible_dict = {}

    for book in root.findall("b"):
        book_name = book.get("n")
        ESV_Bible_dict[book_name] = {}
        for chapter in book.findall("c"):
            chapter_number = chapter.get("n")
            ESV_Bible_dict[book_name][chapter_number] = {}
            for verse in chapter.findall("v"):
                verse_number = verse.get("n")
                verse_text = verse.text
                ESV_Bible_dict[book_name][chapter_number][verse_number] = verse_text

    for book in root.findall("b"):
        book_name = book.get("n")
        ESV_Bible_dict[book_name] = {}
        for chapter in book.findall("c"):
            chapter_number = chapter.get("n")
            ESV_Bible_dict[book_name][chapter_number] = {}
            for verse in chapter.findall("v"):
                verse_number = verse.get("n")
                verse_text = verse.text
                ESV_Bible_dict[book_name][chapter_number][verse_number] = verse_text

    SaveJSONData(ESV_Bible_dict,dictionary_filename)
    return ESV_Bible_dict

def process_xml(root):
    output = OrderedDict()
    for b in root.findall('.//b'):
        b_value = b.get('n')
        for c in b.findall('.//c'):
            c_value = c.get('n')
            for v in c.findall('.//v'):
                v_number = v.get('n')
                v_text = v.text
                output[f'"{b_value} {c_value}:{v_number}"'] = v_text
    return output

def CreateBibleText(xml_filename,txt_filename):
    ESV_Bible_xml = LoadTxtData(xml_filename)
    root = ET.fromstring(ESV_Bible_xml)
    ESV_Bible_list = process_xml(root)
    ESV_Bible_txt = CreateAndSaveTxtFileFromList(ESV_Bible_list,txt_filename)
    return ESV_Bible_txt

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
print(ESV_Bible_dict['Genesis']['1']['1'])

