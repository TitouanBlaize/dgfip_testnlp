import re
import string
import unicodedata
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')

def impurity(text, min_len = 10):
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text))/len(text)

def replace_punctuation_with_space(text):
    translator = str.maketrans(string.punctuation, 
                               ' ' * len(string.punctuation))
    return text.translate(translator)

def preprocess_stopwords(text, stopwords):
    # Convert text to lowercase
    text = text.lower()

    # Uniformisation unicode
    modified_text = unicodedata.normalize("NFKC", text)

    # Replace punctuation with spaces
    # Note : je préfère faire cela avant la tokenisation, lorsqu'un signe de ponctuation
    # est placé juste après un mot inconnu le tokenizer n'arrive pas à le séparer sinon
    modified_text = replace_punctuation_with_space(modified_text)

    # Remove digits
    modified_text = re.sub(r'\d', '', modified_text)

    # Tokenization
    tokens = word_tokenize(modified_text, language = "french")

    # Remove single caracter
    tokens = [token for token in tokens if len(token) > 1]
    
    # Remove defined stopwords
    tokens = [token for token in tokens if token not in stopwords]
    
    return tokens

def preprocess_text(text, stopwords):
    # Convert text to lowercase
    text = text.lower()

    # Uniformisation unicode
    modified_text = unicodedata.normalize("NFKC", text)

    # Replace punctuation with spaces
    # Note : je préfère faire cela avant la tokenisation, lorsqu'un signe de ponctuation
    # est placé juste après un mot inconnu le tokenizer n'arrive pas à le séparer sinon
    modified_text = replace_punctuation_with_space(modified_text)

    # Remove digits
    modified_text = re.sub(r'\d', '', modified_text)

    # Tokenization
    tokens = word_tokenize(modified_text, language = "french")

    # Remove single caracter
    tokens = [token for token in tokens if len(token) > 1]
    
    # Remove defined stopwords
    tokens = [token for token in tokens if token not in stopwords]

    # Lemmatisation ou racinisation
    # stemmer = SnowballStemmer('french')
    # tokens = [stemmer.stem(mot) for mot in tokens]
    
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
    # words = sentence.split()
    
    words = word_tokenize(sentence, language = "french")
    # Return the number of words
    return len(words)