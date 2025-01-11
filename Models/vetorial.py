import copy
import math
import string
import numpy

def get_words_desc(desc:str) -> list[str]:
    desc_words = []

    # Removing punctuation
    desc = desc.translate(str.maketrans('', '', string.punctuation))

    for word in desc.split(" "):
        desc_words.append(word.lower())

    return desc_words

def get_vocab(doc_dict: dict) -> set[str]:
    vocab = []
    doc_desc_words = []

    # Get docs values: Id, Desc, Price
    for doc in doc_dict.values():
        doc_desc = doc.get("Nome") + " " + doc.get("Descricao")

        # Get separate words from doc_desc
        doc_desc_words += get_words_desc(doc_desc)

    # doc_desc_words.sort()

    vocab = set(doc_desc_words)

    # print(f"Vocabulario: {vocab}")

    return vocab

def get_freq(doc_dict:dict, vocab:set[str]):

    doc_freq = {}

    for doc, doc_content in doc_dict.items():

        doc_freq[doc] = {}

        doc_desc = doc_content.get("Nome") + " " + doc_content.get("Descricao")

        doc_desc_words = get_words_desc(doc_desc)

        for word in doc_desc_words:
            if word not in vocab:
                continue
            if word in doc_freq[doc].keys():
                doc_freq[doc][word] += 1
            else:
                doc_freq[doc][word] = 1

    # print(f"Doc_frequencia: {doc_freq}")

    return doc_freq


def get_TF(doc_freq:dict, idf: float):

    doc_freq_copy = copy.deepcopy(doc_freq)

    for doc in doc_freq_copy.keys():
        doc_content = doc_freq_copy[doc]

        for word in doc_content.keys():
            doc_content_word_freq = doc_content[word]

            # (1 + log(freq)) * log(n/ni)
            doc_content[word] = (1 + math.log(doc_content_word_freq, 10)) * idf

    print(f"TF: {doc_freq_copy}")

    return doc_freq_copy

def doc_normalization(doc_:dict):
    doc_copy = copy.deepcopy(doc_)
    doc_magnitude = {}

    # D1, D2, D3,...
    for doc in doc_copy.keys():
        doc_sum = []

        doc_content = doc_copy[doc]

        # camisa, azul, calca, ...
        for word in doc_content.keys():
            freq = doc_content[word]

            doc_sum.append(freq ** 2)

        # Sum of squared frequencies
        doc_sum_sum = sum(doc_sum)

        # Square root of the sum of squared frequencies
        magnitude = numpy.sqrt(doc_sum_sum)

        doc_magnitude[doc] = magnitude

        print(f"{doc} magnitude: {magnitude}")

    # D1, D2, D3,...
    for doc in doc_copy.keys():
        doc_content = doc_copy[doc]

        # camisa, azul, calca, ...
        for word in doc_content.keys():
            freq = doc_content[word]

            # Frequency = (frequency / Square root of the sum of squared frequencies (magnitude)(for each document))
            doc_content[word] = freq / doc_magnitude[doc]

    # print(f"Normalized doc: {doc_freq_copy}")

    return doc_magnitude, doc_copy


def modelo_vetorial(test_data:dict, query:str):

    n = len(test_data.keys())
    ni = 0
    ni_docs = []

    query_fixed = query.lower()
    query_word_list = query_fixed.split(" ")

    query_dict = {}

    for word in query_word_list:
        if word not in query_dict.keys():
            query_dict[word] = 1
        else:
            query_dict[word] += 1

    vocab = get_vocab(test_data)

    doc_freq = get_freq(test_data, vocab)

    # camisa, azul,...
    for word in query_dict.keys():

        # D1, D2, D3
        for doc in doc_freq.keys():

            doc_content = doc_freq[doc]

            # camisa, calca
            if word in doc_content.keys():
                if doc not in ni_docs:
                    ni += 1
                    ni_docs.append(doc)

    relevant_docs = {key: doc_freq[key] for key in ni_docs if key in doc_freq}

    # Remove words that are not on the query
    relevant_docs_words = {
        key: { inner_key: inner_value for inner_key, inner_value in value.items() if inner_key in query_word_list }
        for key, value in relevant_docs.items()
    }

    print(relevant_docs_words)

    idf = math.log(n/ni, 10)

    print(f"IDF: {idf}")

    doc_TF_IDF = get_TF(relevant_docs_words, idf)

    query_TF_IDF = {key: (1+math.log(value, 10)) * idf for key, value in query_dict.items()}

    print(query_TF_IDF)

    doc_magnitude, doc_normalized = doc_normalization(doc_TF_IDF)

    # for doc in doc_normalized:
    #     print(f"{doc}: {doc_normalized[doc]}")
    # print()

    similarity = {}

    for doc in doc_TF_IDF:
        sum = 0
        for word in doc_TF_IDF[doc].keys():
            sum += doc_TF_IDF[doc][word] * query_TF_IDF[word]

        similarity[doc] = sum / doc_magnitude[doc]

    print("\n")
    print(similarity)

    print(ni_docs)
