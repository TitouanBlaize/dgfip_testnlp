import re
import string
import unicodedata
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

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

    # Ici utilisation du tokenizer directement
    words = word_tokenize(sentence, language = "french")

    return len(words)

def replace_punctuation_with_space(text: str) -> str:
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
    # Cf. exemple dans notebook 1. Preprocessing.ipynb
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

    # Stemmer
    # stemmer = SnowballStemmer('french')
    # tokens = [stemmer.stem(mot) for mot in tokens]

    # Ou Lemmetizer :
    nlp = spacy.load("fr_core_news_md")
    tokens = [token.lemma_ for token in nlp(" ".join(tokens))]

    # Join tokens back into a single string
    processed_text = ' '.join(tokens)
    
    return processed_text