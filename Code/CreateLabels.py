import json

with open('Additional_Data\ESV_Bible_Tokens.json', 'r') as file:
    bible_data = json.load(file)

unique_tokens = []

for book, chapters in bible_data.items():
    for chapter, verses in chapters.items():
        for verse, tokens in verses.items():
            unique_tokens.extend(tokens)

unique_tokens = list(set(unique_tokens))

print(unique_tokens)