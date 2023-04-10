from ebooklib import epub
from bs4 import BeautifulSoup
from transformers import pipeline
from datasets import Dataset
import re

book = epub.read_epub('Alices Adventures in Wonderland.epub')

texts = []

for item in book.get_items():
    if item.get_type() == 9:
        chapter = item.get_content()
        aux = BeautifulSoup(chapter, 'html.parser')
        texts.append(aux.get_text())

translator = pipeline(
    "translation",  model="Helsinki-NLP/opus-mt-en-es", device=0)
text_to_trans = []

for text in texts:
    seamless_text = re.sub("\n+", "\n", text)
    for paragraph in seamless_text.split("\n"):
        if "" == paragraph:
            continue
        text_to_trans.append(paragraph)

"".join(text_to_trans).split(".")

dataset = Dataset.from_dict({"en": " ".join(text_to_trans).split(".")})

translated_dataset = dataset.map(lambda batch: {"es": translator(batch["en"],max_length=400) },remove_columns=["en"], batched=True,batch_size=12)

print(translated_dataset)

for i in translated_dataset:
    print(i)

translated_dataset.save_to_disk("translated_dataset")

