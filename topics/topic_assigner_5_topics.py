# Import necessary libraries
import re
import random
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from datasets import load_dataset

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# This function will get an assigned topic for a text based on the keywords in the topics
def format_topics_sentences(ldamodel, corpus, texts):
    data = []
    for i, row_list in enumerate(ldamodel[corpus]):
        row = sorted(row_list, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # Dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                topic_name = topic_names.get(topic_num, "Not Assigned")
                data.append([int(topic_num), round(prop_topic, 4), topic_keywords, topic_name, texts[i]])
                break
    return pd.DataFrame(data, columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Topic_Name', 'Text'])

# Displays the random text with its assigned topic
def display_random_text_with_topic(df):
    random_index = random.randint(0, len(df) - 1)
    random_entry = df.iloc[random_index]
    print("Random Text: ", random_entry['Text'])
    print("Assigned Topic: ", random_entry['Topic_Name'])
    print("Topic Keywords: ", random_entry['Topic_Keywords'])
    print("Topic Contribution: ", random_entry['Perc_Contribution'])

def main():
    # Load the dataset
    dataset = load_dataset('edarchimbaud/news-stocks', split='train')
    dataset = dataset.remove_columns(['publisher', 'publish_time', 'url', 'uuid', 'symbol'])

    # Combine title and body into a single text
    dataset = dataset.map(lambda example: {'combined_text': example['title'] + ". " + example['body']})
    dataset = dataset.remove_columns(['body', 'title'])

    # Preprocess the combined text
    def preprocess(text):
        text = re.sub("(\\d|\\W)+", " ", text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in text.lower().split() if word not in stop_words and len(word) > 3]
        return tokens

    texts = [preprocess(example['combined_text']) for example in dataset]

    # Create a dictionary and corpus for LDA
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train LDA model
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15, random_state=42)

    # Display topics
    for idx, topic in lda_model.print_topics(-1):
        print(f'Topic: {idx} \nWords: {topic}')

    # Define topic names
    topic_names = {
        0: "Corporate Valuation & Performance",
        1: "Quarterly Financial Reports",
        2: "Stock Market & Investment Funds",
        3: "Corporate Services & Products",
        4: "Investment Research, Analysis & Stock Rankings",
    }

    # Format topics and sentences
    df_topic_sents_keywords = format_topics_sentences(lda_model, corpus, [doc['combined_text'] for doc in dataset])
    print(df_topic_sents_keywords.head())

    # Display a random text with its topic
    display_random_text_with_topic(df_topic_sents_keywords)


if __name__ == "__main__":
    main()
