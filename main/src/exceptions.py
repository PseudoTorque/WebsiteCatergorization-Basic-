from csv import reader

with open('data/WebsiteClassification(Small).csv', mode='r', encoding='utf-8') as file:

    temp_data = reader(file)

    data = [row for row in temp_data]

data = [{"website_url": row[0], "cleaned_text": row[2], "category": row[3]} for row in data[1:]]

class_mapping = {label: i for i, label in enumerate(set(row["category"] for row in data))}

print(class_mapping)
