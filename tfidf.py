"""
An Introduction to TF-IDF
TF-IDF is a method of information retrieval that is used to rank the importance of words in a document.
It is based on the idea that words that appear in a document more often are more relevant to the document.

TF-IDF is the product of Term Frequency and Inverse Document Frequency.

Here’s the formula for TF-IDF calculation.
TF-IDF = Term Frequency (TF) * Inverse Document Frequency (IDF)

What are Term Frequency and Inverse Document Frequency you ask? let’s see what they actually are.
1. What is Term Frequency?
It is the measure of the frequency of words in a document. It is the ratio of the number of times the
word appears in a document compared to the total number of words in that document.

2. What is Inverse Document Frequency?It is the measure of how much information the word provides about the topic of the document.
It is the log of the ratio of the number of documents to the number of documents containing the word.
We take log of this ratio because when the corpus becomes large IDF values can get large causing it
to explode hence taking log will dampen this effect.

We cannot divide by 0, we smoothen the value by adding 1 to the denominator.
idf(t) = log(N/(df + 1))

"""

########################################################################################################################
# 1. Importing the libraries
########################################################################################################################
import numpy as np
from nltk.tokenize import word_tokenize

########################################################################################################################
# 2. Sample to consider for TF-IDF
########################################################################################################################
sample_text = ['Topic sentences are similar to mini thesis statements.'
               'Like a thesis statement, a topic sentence has a specific '
               'main point. Whereas the thesis is the main point of the essay',
               'the topic sentence is the main point of the paragraph.'
               'Like the thesis statement, a topic sentence has a unifying function. '
               'But a thesis statement or topic sentence alone doesn’t guarantee unity.',
               'An essay is unified if all the paragraphs relate to the thesis,'
               'whereas a paragraph is unified if all the sentences relate to the topic sentence.']

########################################################################################################################
# 3. Tokenizing the sample text and creating unique set of words
########################################################################################################################
sentences = []
word_set = []

for sent in sample_text:
    words = [word.lower() for word in word_tokenize(sent) if word.isalpha()]
    sentences.append(words)
    for word in words:
        if word not in word_set:
            word_set.append(word)

# Set of words
word_set = set(word_set)
# total documents in our corpus
total_docs = len(sample_text)
print('Total documents: ', total_docs)
print('Total words: ', len(word_set))

########################################################################################################################
# 4. Creating index for each word from vocabulary
########################################################################################################################
word_index = {}
for i, word in enumerate(word_set):
    word_index[word] = i


########################################################################################################################
# 5. Creating a dictionary for keeping count
########################################################################################################################
def count_dict(sentences):
    """
    Create a dictionary to keep the count of the number of documents containing the given word.
    """
    count_dict = {}
    for word in word_set:
        count_dict[word] = 0
    for sent in sentences:
        for word in sent:
            count_dict[word] += 1
    return count_dict

word_count = count_dict(sentences)
print(word_count)

########################################################################################################################
# 6. Now calculate term frequency
########################################################################################################################
def term_frequency(document, word):
    """
    Calculate the term frequency of each word in the corpus.
    """
    N = len(document)
    occurance = len([token for token in document if token == word])
    return occurance / N


########################################################################################################################
# 7. Now calculate inverse document frequency
########################################################################################################################
def inverse_document_frequency(word):
    """
    Calculate the inverse document frequency of each word in the corpus.
    """
    try:
        word_occurance = word_count[word] + 1
    except:
        word_occurance = 1
    return np.log(total_docs / word_occurance)


########################################################################################################################
# 8. Now calculate TF-IDF
########################################################################################################################
def tf_idf(sentence):
    """
    Calculate the TF-IDF of each sentence in the corpus.
    """
    vec = np.zeros((len(word_set),))
    for word in sentence:
        tf = term_frequency(sentence, word)
        idf = inverse_document_frequency(word)
        vec[word_index[word]] = tf * idf
    return vec


########################################################################################################################
# 9. Apply TF-IDF to our sample
########################################################################################################################
vectors = []
for sent in sentences:
    vectors.append(tf_idf(sent))

print(vectors[0])






