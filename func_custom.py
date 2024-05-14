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