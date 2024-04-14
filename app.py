# Gradio Application Interface

import gradio as gr
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import pandas as pd
import gensim
import re 
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import os

def summarizer_func():
  return pipeline(
      model="Majon911/pegasus_multi_news_ep1",
      tokenizer = "google/pegasus-xsum",
      min_length=100, max_length=200,
      truncation = True
  )

def sentiment_func():
   return pipeline("text-classification", 
                   model="kbaumgartner/DeBERTa_Finetuned_Financial_News", 
                   tokenizer = "microsoft/deberta-v3-base")

def source_outlet(choise):
    if choise == 'CNBC':
      url = "https://www.cnbc.com/finance/"
      response = requests.get(url)
      soup = BeautifulSoup(response.content, 'html.parser')

      headlines = {}
      headline_elements = soup.find_all('a', class_='Card-title')
      for headline_element in headline_elements:
          headlines[headline_element.text.strip()] = headline_element['href']
    elif choise == "Reuters":
      pass

    df = pd.DataFrame({'headline': headlines.keys(),
                    'url': headlines.values()})

    first_5_articles = df.head()
    first_5_articles = first_5_articles.assign(text='')
    first_5_articles = first_5_articles.assign(summary='')
    first_5_articles = first_5_articles.assign(sentiment='')
    first_5_articles = first_5_articles.assign(topic='')
    return first_5_articles

def sentiment_translation(curr_sentiment):
  if curr_sentiment == "LABEL_0":
    trans_lbl = "NEGATIVE"
  elif curr_sentiment == "LABEL_1":
    trans_lbl = "NEUTRAL"
  elif curr_sentiment == "LABEL_2":
    trans_lbl = "POSITIVE"
  return trans_lbl

def preprocess(text):
    # Remove special characters and digits
    text = text.lower()
    text = re.sub("(\\d|\\W)+", " ", text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.lower().split() if word not in stop_words and len(word) > 3]
    return tokens

def lda_topic_modeling(text):
    lda_model = gensim.models.LdaModel.load("lda_gensim_5t/lda_model5.gensim")
    dictionary = gensim.corpora.Dictionary.load("lda_gensim_5t/dictionary5.gensim")

    processed_text = preprocess(text)
    bow = dictionary.doc2bow(processed_text)
    topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0.0)
    topic_distribution = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
    
    topic_names = {
    '0': "Corporate Valuation & Performance",
    '1': "Quarterly Financial Reports",
    '2': "Stock Market & Investment Funds",
    '3': "Corporate Affairs & Products",
    '4': "Investment Research"
    }

    # Extract the most probable topic and its probability
    if topic_distribution:
        dominant_topic, probability = topic_distribution[0]
        topic_name = topic_names.get(str(dominant_topic), "Unknown Topic")
        return (topic_name, probability)
    else:
        # If no topic is found, return a placeholder and zero probability
        return ("No Topic Found", 0.0)

def gradio_stocknews(source_ch, art_number):

  # Defining the summarizer
  summarizer = summarizer_func()
  # Defining the semtiment analysis
  pipe_sentiment = sentiment_func()

  # Identyfying the Articles
  first_5_articles = source_outlet(source_ch)

  # Scraping text for the chosen article
  response = requests.get(first_5_articles.loc[art_number-1, 'url'])
  sub_soup = BeautifulSoup(response.content, 'html.parser')
  article_body_element = sub_soup.find('div', class_='ArticleBody-articleBody') # ArticleBody-articleBody
  article_text = article_body_element.get_text()  # Extracting only the text
  first_5_articles.loc[art_number-1, 'text']  = article_text
  first_5_articles.loc[art_number-1, 'summary']  = summarizer(article_text)[0]['generated_text']

  label_sentiment = pipe_sentiment(article_text)[0]['label']
  first_5_articles.loc[art_number-1, 'sentiment'] = sentiment_translation(label_sentiment)
  
  # Get the human-readable topic name using the topic names mapping
  first_5_articles.loc[art_number-1, 'topic'] = lda_topic_modeling(article_text)[0]


  return first_5_articles.loc[art_number-1, 'headline'], first_5_articles.loc[art_number-1, 'url'], first_5_articles.loc[art_number-1, 'summary'], first_5_articles.loc[art_number-1, 'sentiment'], first_5_articles.loc[art_number-1, 'topic']

def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # nltk.download('stopwords')
    # nltk.download('wordnet')

    #print(gradio_stocknews("CNBC", 1))

    iface = gr.Interface(fn=gradio_stocknews,
                        inputs=[gr.Dropdown(choices=["CNBC"], label="Select Source"), gr.Dropdown(choices=[1, 2, 3, 4, 5], label="Select Article Number")],
                        outputs=[gr.Textbox(lines=1, label="Article Title"), gr.Textbox(lines=1, label="Article Link"), gr.Textbox(lines=1, label="Article Summary"), gr.Textbox(lines=1, label="Article Sentiment"), gr.Textbox(lines=1, label="Article Topic")],   # Add this line for topic
                        title="Latest 5 Stock News Dashboard",
                        description="Click the button to refresh the news summary.")

    iface.launch()

if __name__ == "__main__":
    main()
