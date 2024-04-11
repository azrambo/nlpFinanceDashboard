# This script is to summarize the top 5 finance articles.

from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import pandas as pd
pd.set_option('display.max_colwidth', None)

summarizer = pipeline(
    model="falba/t5-small-finetuned-news-ep1",
    tokenizer = "google-t5/t5-small",
    min_length=100, max_length=500
)

url = "https://www.cnbc.com/world/?region=world"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

headlines = {}
headline_elements = soup.find_all('a', class_='LatestNews-headline')
for headline_element in headline_elements:
    headlines[headline_element.text.strip()] = headline_element['href']

df = pd.DataFrame({'headline': headlines.keys(),
                   'url': headlines.values()})

first_5_articles = df.head()
first_5_articles = first_5_articles.assign(text='')
first_5_articles = first_5_articles.assign(summary='')

for index, article in first_5_articles.iterrows():
    response = requests.get(article['url'])
    sub_soup = BeautifulSoup(response.content, 'html.parser')
    article_body_element = sub_soup.find('div', class_='ArticleBody-articleBody') # ArticleBody-articleBody
    article_text = article_body_element.get_text()  # Extracting only the text
    print(article_text)
    article['text'] = article_text
    article['summary'] = summarizer(article['text'])[0]['generated_text']

print(first_5_articles['summary'])





