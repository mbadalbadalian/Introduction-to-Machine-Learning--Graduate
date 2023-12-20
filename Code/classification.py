import json
#LoadJSONData Function
def LoadJSONData(json_filename):
    #Loading JSON file into variable
    with open(json_filename, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def Get_Chapter_Tokens():
    ESV_Bible_Tokens = "Additional_Data\\ESV_Bible_Tokens.json"
    ESV_Bible_dict = LoadJSONData(ESV_Bible_Tokens)
    unique_values_dict = {}
    for first_key, second_dict in ESV_Bible_dict.items():
        unique_values_dict[first_key] = []
        for second_key, third_dict in second_dict.items():
            line = set()
            for third_key, values in third_dict.items():
                line.update(values)
            unique_values_dict[first_key].append(' '.join(list(line)))

    return unique_values_dict
if __name__ == "__main__": 
    unique_values_dict = Get_Chapter_Tokens()
    print(unique_values_dict['Genesis'])
