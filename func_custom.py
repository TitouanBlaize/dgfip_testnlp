import re
import string
import unicodedata
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

def replace_punctuation_with_space(text):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(translator)

def preprocess_text_print(text):
    # Convert text to lowercase
    text = text.lower()
    print(text)

    # Replace punctuation with spaces
    # Note : je préfère faire cela avant la tokenisation, lorsqu'un signe de ponctuation
    # est placé juste après un mot inconnu le tokenizer n'arrive pas à le séparer sinon
    text_punct = replace_punctuation_with_space(text)

    # Tokenization
    tokens = word_tokenize(text_punct, language = "french")
    print(tokens)

    # Remove punctuation
    # tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stopwords
    stop_words = (stopwords.words('french'))
    anonymisation_stopword = ["xxxxx", "0000", "00000", "0000"]
    stop_words.extend(anonymisation_stopword)
    tokens = [token for token in tokens if token not in stop_words]
    print(tokens)
    # Remove single caracter (molecules often)
    # tokens = [token for token in tokens if len(token) > 1]

    # Lemmatisation ou racinisation
    stemmer = SnowballStemmer('french')
    tokens = [stemmer.stem(mot) for mot in tokens]
    print(tokens)   
    # Join tokens back into a single string
    processed_text = ' '.join(tokens)
    
    return processed_text

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Replace punctuation with spaces
    # Note : je préfère faire cela avant la tokenisation, lorsqu'un signe de ponctuation
    # est placé juste après un mot inconnu le tokenizer n'arrive pas à le séparer sinon
    modified_text = replace_punctuation_with_space(text)

    # Remove digits
    modified_text = re.sub(r'\d', '', modified_text)

    # Tokenization
    tokens = word_tokenize(modified_text, language = "french")
    
    # Remove stopwords
    stop_words = (stopwords.words('french'))
    anonymisation_stopword = ["xxxxxé", "xxxxxer", "xxxxx", "xx", "0000", "00000", "0000"]
    stop_words.extend(anonymisation_stopword)
    tokens = [token for token in tokens if token not in stop_words]

    # Remove single caracter
    tokens = [token for token in tokens if len(token) > 1]

    # Lemmatisation ou racinisation
    stemmer = SnowballStemmer('french')
    tokens = [stemmer.stem(mot) for mot in tokens]
    
    # Join tokens back into a single string
    processed_text = ' '.join(tokens)
    
    return processed_text

def preprocess_text_save(text):
    # Convert text to lowercase
    text = text.lower()

    punctuation_pattern = r'[^\w\s]'
    # Replace punctuation with spaces using sub() function
    modified_text = re.sub(punctuation_pattern, ' ', text)

    # Remove digits
    modified_text = re.sub(r'\d', '', modified_text)
    
    # Tokenization
    tokens = word_tokenize(modified_text, language = "french")
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stopwords
    stop_words = set(stopwords.words('french'))
    tokens = [token for token in tokens if token not in stop_words]

    # Remove single caracter (molecules often)
    # tokens = [token for token in tokens if len(token) > 1]

    # Lemmatisation ou racinisation
    stemmer = SnowballStemmer('french')
    tokens = [stemmer.stem(mot) for mot in tokens]
    
    # Join tokens back into a single string
    processed_text = ' '.join(tokens)
    
    return processed_text

def count_words(sentence: str) -> int:
    """
    Count the number of words in a sentence.

    Args:
        sentence (str): The input sentence.

    Returns:
        int: The number of words in the sentence.
    """
    # Split the sentence into words using whitespace as delimiter
    words = sentence.split()
    # Return the number of words
    return len(words)