import csv

def extract_columns(input_file, output_file):
    columns_to_extract = [0, 1, 6]  # 1st, 2nd, and 7th columns (0-indexed)

    with open(input_file, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        data = [row for row in reader]

    extracted_data = [[row[i] for i in columns_to_extract] for row in data]

    with open(output_file, 'w', newline='', encoding='utf-8') as new_csv_file:
        writer = csv.writer(new_csv_file)
        writer.writerows(extracted_data)

        writer.writerows(extracted_data)

# Example usage:
input_filename = 'Additional_Data/Bhagwad_Gita_Verses_English.csv'  # Replace with your input CSV filename
output_filename = 'Additional_Data/Gita_DF.csv'  # Replace with your desired output CSV filename

extract_columns(input_filename, output_filename)