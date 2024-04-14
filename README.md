# nlpFinanceDashboard

Project by: Abdullah Ghosheh (aghosheh.ieu2021@student.ie.edu), Ali Rammal (arammal.ieu2021@student.ie.edu), Francisco Alba (falba.ieu2021@student.ie.edu), Jan Maciejowski (jmaciejowski.ieu2021@student.ie.edu), Kevin Baumgartner (kbaumgartner.ieu2021@student.ie.edu), Nathanel Zeidel (nzeidel.ieu2021@student.ie.edu)

## **Description:**
This project provides a financial dashboard that scraps articles from CNBC and provides article sentiment analysys, company stock tags (using Latent Dirichlet allocation) and custom article summarization, providing the user with most relevant insight into most important chnages in the market. 
Below is an overview of the project's structure and contents.

**/nlpFinanceDashboard**

 **/sentiment**
 
    all-data.csv             # Dataset for sentiment analysis.
    
    nlp_group_sentiment.py     # Script for sentiment analysis.

**/summaries**

    cnbc_article_summaries.csv # Summarized CNBC articles.
    
    cnbc_summarizer.py         # Script to summarize CNBC articles.
    
    reuters_summarizer.py      # Script to summarize Reuters articles.
    
    tryingPegasus.ipynb        # Notebook for summarization with Pegasus model.

**/topics**

    topic_assigner_5_topics.py # Script for assigning topics to documents.
    
    GradioInterfaceIntegration.ipynb # Notebook for topic modeling with a Gradio interface.
    

**/lda_gensim_5t**                

    Outputs and models for a 5-topic LDA model.

**/lda_gensim_9t**    

    Outputs and models for a 9-topic LDA model.




## **Scripts and Notebooks**
### **Sentiment Analysis**

**nlp_group_sentiment.py:** This script serves as a pipeline for sentiment analysis on a dataset, performing text preprocessing, training a Deberta V2-based model, and evaluating the results. It uses a fine-tuning approach on pre-trained language models to accurately predict sentiment labels for text inputs.

### **Summarization**

**cnbc_summarizer.py:** A tool designed to condense CNBC news articles into brief summaries, utilizing extractive summarization techniques to retain the most significant information from the original content.

**reuters_summarizer.py:** Similar to the CNBC summarizer, this script focuses on Reuters articles, applying NLP summarization methods to create concise abstracts of lengthy news stories.

**tryingPegasus.ipynb:** This Jupyter notebook demonstrates the use of the Pegasus model, a powerful abstractive summarization algorithm, to generate summaries of large text corpora with context coherence and fluency.

### **Topic Modeling**

**topic_assigner_5_topics.py:** Implements a topic modeling algorithm that assigns one of five predetermined topics to a given document. It uses Latent Dirichlet Allocation (LDA) to discover abstract topics within a collection of texts.

**GradioInterfaceIntegration.ipynb:** Integrates the output of topic modeling with Gradio, providing an interactive web interface for users to input text and receive topic classification results in real-time.

### **LDA Models**

**/lda_gensim_5t & /lda_gensim_9t:** These directories are intended to store the trained models, corpus, and dictionary files used for the 5-topic and 9-topic LDA models created with Gensim. The models help in uncovering latent topics in a document corpus for thematic analysis.

## **Usage**

Navigate to each directory to execute the respective Python scripts or Jupyter notebooks. Ensure that you have the necessary Python environment and dependencies installed.

## **Dependencies**

Make sure to install the following dependencies to work with the toolkit:

Python 3.8+

pandas

numpy

torch

nltk

transformers

datasets

sklearn

gensim

gradio


**Pip Install command:**

!pip install pandas numpy torch nltk transformers datasets sklearn gensim gradio





