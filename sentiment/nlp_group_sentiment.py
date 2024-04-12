import pandas as pd
import numpy as np
import torch
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
import random

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.dataset = None

    def load_and_preprocess_data(self):
        """
        Description: [MODIFY]
        Parameters: [MODIFY]
        Result: [MODIFY]
        """
        # Load the dataset
        self.df = pd.read_csv(self.file_path, encoding='latin1')

        # Rename the columns for clarity
        self.df.rename(columns={
            'neutral': 'label',  # Assuming 'neutral' is your original label column
            'According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .': 'text'
        }, inplace=True)

        # Map the labels to numerical values
        label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.df['label'] = self.df['label'].map(label_mapping)

        # Shuffle the dataframe
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

    def split_data(self):
        """
        Description: [MODIFY]
        Parameters: [MODIFY]
        Result: [MODIFY]
        """
        # Calculate the sizes for train, validation, and test sets
        train_size = int(len(self.df) * 0.6)
        val_size = int(len(self.df) * 0.2)

        # Split the data
        train_df = self.df[:train_size]
        val_df = self.df[train_size:train_size + val_size]
        test_df = self.df[train_size + val_size:]

        # Convert to Hugging Face's datasets
        self.dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df),
            'test': Dataset.from_pandas(test_df)
        })

class TextClassifier:
    def __init__(self, model_name='microsoft/deberta-v3-base', num_labels=3, seed=42):
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.seed = seed
        self.setup_seed()

    def setup_seed(self):
        """
        Description: [MODIFY]
        Parameters: [MODIFY]
        Result: [MODIFY]
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def preprocess_text(self, examples):
        """
        Description: [MODIFY]
        Parameters: [MODIFY]
        Result: [MODIFY]
        """
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        processed_texts = []
        for text in examples['text']:
            text = text.lower().translate(str.maketrans('', '', string.punctuation))
            tokens = word_tokenize(text)
            lemmatized = [lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in tokens]
            stemmed = [stemmer.stem(w) for w in lemmatized]
            processed_texts.append(' '.join(stemmed))
        return {'text': processed_texts}

    def get_wordnet_pos(self, word):
        """
        Description: [MODIFY]
        Parameters: [MODIFY]
        Result: [MODIFY]
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def tokenize(self, examples):
        """
        Description: [MODIFY]
        Parameters: [MODIFY]
        Result: [MODIFY]
        """
        return self.tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    def compute_metrics(self, pred):
        """
        Description: [MODIFY]
        Parameters: [MODIFY]
        Result: [MODIFY]
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {'accuracy': accuracy_score(labels, preds), 'f1': f1_score(labels, preds, average='macro')}

    def train(self, dataset):
        """
        Description: [MODIFY]
        Parameters: [MODIFY]
        Result: [MODIFY]
        """
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            do_eval=True,
            evaluation_strategy="epoch",
            fp16=torch.cuda.is_available(),
            learning_rate=5e-05,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test']
        )
        self.trainer.train()
        return self.trainer.evaluate()

    def export_results(self, dataset, file_name='results.csv'):
        """
        Description: [MODIFY]
        Parameters: [MODIFY]
        Result: [MODIFY]
        """
        predictions = self.trainer.predict(dataset['test'])
        predicted_labels = np.argmax(predictions.predictions, axis=-1)
        results_df = pd.DataFrame({'Actual': dataset['test']['label'], 'Predicted': predicted_labels})
        results_df.to_csv(file_name, index=False)
        print(f'Classification results exported to {file_name}')

# Usage example
if __name__ == '__main__':
    processor = TextDataProcessor('/content/all-data.csv')
    processor.load_and_preprocess_data()
    processor.split_data()

    classifier = TextClassifier()
    reduced_dataset = processor.dataset.map(classifier.preprocess_text, batched=True)
    dataset_encoded = reduced_dataset.map(classifier.tokenize, batched=True)
    dataset_encoded.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    results = classifier.train(dataset_encoded)
    print(results)
    classifier.export_results(dataset_encoded)

#5 epochs: .8355
#5 epochs: .8355

import pandas as pd
import numpy as np
import torch
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
import random

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.dataset = None


    def load_and_preprocess_data(self):
        """
        Loads data from a CSV file specified by the file_path attribute, renames columns for consistency,
        maps string labels to numerical values, and shuffles the dataset for randomness.

        Parameters:
        None

        Result:
        The DataFrame attribute 'df' is populated with processed data, with labels mapped to numerical
        values and columns renamed for clarity.
        """
        # Load the dataset
        self.df = pd.read_csv(self.file_path, encoding='latin1')

        # Rename the columns for clarity
        self.df.rename(columns={
            'neutral': 'label',  # Assuming 'neutral' is your original label column
            'According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .': 'text'
        }, inplace=True)

        # Map the labels to numerical values
        label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.df['label'] = self.df['label'].map(label_mapping)

        # Shuffle the dataframe
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)


    def split_data(self):
        """
        Splits the data into training, validation, and test sets in a 60/20/20 proportion and converts
        them into Hugging Face's Dataset format for compatibility with the transformers library.

        Parameters:
        None

        Result:
        The 'dataset' attribute is populated with a DatasetDict containing 'train', 'validation',
        and 'test' datasets.
        """
        # Calculate the sizes for train, validation, and test sets
        train_size = int(len(self.df) * 0.6)
        val_size = int(len(self.df) * 0.2)

        # Split the data
        train_df = self.df[:train_size]
        val_df = self.df[train_size:train_size + val_size]
        test_df = self.df[train_size + val_size:]

        # Convert to Hugging Face's datasets
        self.dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df),
            'test': Dataset.from_pandas(test_df)
        })

class TextClassifier:
    def __init__(self, model_name='microsoft/deberta-v3-base', num_labels=3, seed=42):
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.seed = seed
        self.setup_seed()

    def setup_seed(self):
        """
        Sets the seed for all necessary libraries to ensure reproducibility.

        Parameters:
        None

        Result:
        The random seed is set for numpy, random, torch, and torch.cuda (if available).
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def preprocess_text(self, examples):
        """
        Preprocesses the text data by converting to lowercase, removing punctuation, tokenizing,
        lemmatizing, and stemming the text.

        Parameters:
        examples: A dictionary with a key 'text' containing a list of strings to be processed.

        Result:
        A dictionary with a key 'text' containing the list of processed texts.
        """
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        processed_texts = []
        for text in examples['text']:
            text = text.lower().translate(str.maketrans('', '', string.punctuation))
            tokens = word_tokenize(text)
            lemmatized = [lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in tokens]
            stemmed = [stemmer.stem(w) for w in lemmatized]
            processed_texts.append(' '.join(stemmed))
        return {'text': processed_texts}


    def get_wordnet_pos(self, word):
        """
        Maps the part of speech tag to the format recognized by WordNet lemmatizer.

        Parameters:
        word: A single word for which the part of speech tag is to be determined.

        Result:
        The part of speech tag corresponding to the given word in WordNet's format.
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def tokenize(self, examples):
        """
        Tokenizes the text examples using the DebertaV2 tokenizer, padding and truncating them
        to a maximum length.

        Parameters:
        examples: A dictionary with a key 'text' containing a list of strings to be tokenized.

        Result:
        A dictionary with tokenized outputs including 'input_ids' and 'attention_mask'.
        """
        return self.tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    def compute_metrics(self, pred):
        """
        Computes the accuracy and weighted F1 score of the predictions.

        Parameters:
        pred: A 'NamedTuple' with 'predictions' and 'label_ids' fields containing the model's predictions
        and the true labels respectively.

        Result:
        A dictionary with keys 'accuracy' and 'f1' mapping to the calculated metric scores.
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {'accuracy': accuracy_score(labels, preds), 'f1': f1_score(labels, preds, average='macro')}

    def train(self, dataset):
        """
        Trains the classifier using the specified training arguments and dataset, evaluates it on the test set.

        Parameters:
        dataset: A 'DatasetDict' containing 'train' and 'test' datasets for training and evaluation.

        Result:
        A dictionary containing evaluation results on the test set.
        """
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            do_eval=True,
            evaluation_strategy="epoch",
            fp16=torch.cuda.is_available(),
            learning_rate=5e-05,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test']
        )
        self.trainer.train()
        return self.trainer.evaluate()


    def export_results(self, dataset, file_name='results.csv'):
        """
        Predicts the labels for the test set and exports the actual and predicted labels to a CSV file.

        Parameters:
        dataset: A 'DatasetDict' containing a 'test' dataset used for making predictions.
        file_name: The name of the CSV file to which the results will be exported.

        Result:
        A CSV file with the actual and predicted labels for the test dataset.
        """
        predictions = self.trainer.predict(dataset['test'])
        predicted_labels = np.argmax(predictions.predictions, axis=-1)
        results_df = pd.DataFrame({'Actual': dataset['test']['label'], 'Predicted': predicted_labels})
        results_df.to_csv(file_name, index=False)
        print(f'Classification results exported to {file_name}')

# Usage example
if __name__ == '__main__':
    processor = TextDataProcessor('/content/all-data.csv')
    processor.load_and_preprocess_data()
    processor.split_data()

    classifier = TextClassifier()
    reduced_dataset = processor.dataset.map(classifier.preprocess_text, batched=True)
    dataset_encoded = reduced_dataset.map(classifier.tokenize, batched=True)
    dataset_encoded.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    results = classifier.train(dataset_encoded)
    print(results)
    classifier.export_results(dataset_encoded)

# epoch 5: .8355
# epoch 5: .
