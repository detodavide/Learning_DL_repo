import requests
from bs4 import BeautifulSoup

url = "https://python.langchain.com/docs/use_cases/question_answering/quickstart"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

text_data = soup.get_text()

file_path = "extracted_text.txt"

with open(file_path, "w", encoding="utf-8") as file:
    file.write(text_data)

print(f"Text data saved to '{file_path}' successfully.")
