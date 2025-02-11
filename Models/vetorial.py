import copy
import math
from collections import Counter
from copy import deepcopy
from typing import Any

import numpy

from Models.Rocchio import rocchio


def get_vocab(test_data: dict, query_dict: dict[str:int]) -> dict[str:int]:
    """
    Gets all the words from all the documents, deduplicates them, and returns the set.

    Parameters
    ----------

    doc_dict: dict
        dictionary of documents.
    query_dict: dict
        dictionary of words and frequencies.

    Returns
    -------

    vocab_dict: dict
        vocabulary, with words and frequencies.
    doc_dict: dict
        dictionary of documents, with all the keys from the vocab and frequencies.

    """
    vocab = []
    doc_dict = deepcopy(test_data)

    # Add each Descricao to vocab
    for key, value in doc_dict.items():

        vocab.extend(value.get("Descricao"))

    # Add query to vocab
    vocab.extend(query_dict.keys())

    vocab = sorted(set(vocab))
    print("\n")
    print(vocab)
    print("\n")

    # Create a dictionary with all keys set to 0
    vocab_dict = dict.fromkeys(vocab, 0)

    # Update each document with all the keys
    for key, value in doc_dict.items():

        desc_dict = dict(Counter(value.get("Descricao")))

        doc_dict[key] = {
            vocab_key: desc_dict.get(vocab_key, 0) for vocab_key in vocab_dict
        }

        # print(f"{key}: {value}")

    return vocab_dict, doc_dict


# def get_freq(doc_dict: dict, vocab: set[str]) -> dict:
#     """
#     Gets the frequency of each word per document and returns a dictionary {word:freq}.
#
#     Parameters
#     --------
#
#     doc_dict: dict
#         dictionary of documents.
#
#     vocab: set[str]
#         set of words present in the documents.
#
#     """
#
#     doc_freq = {}
#
#     for doc, doc_content in doc_dict.items():
#
#         doc_freq[doc] = {}
#
#         doc_desc_words = doc_dict[doc].get("Descricao")
#
#         for word in doc_desc_words:
#             if word not in vocab:
#                 continue
#             if word in doc_freq[doc].keys():
#                 doc_freq[doc][word] += 1
#             else:
#                 doc_freq[doc][word] = 1
#
#     # print(f"Doc_frequencia: {doc_freq}")
#
#     return doc_freq


def get_TF_IDF(doc_freq: dict, idf: float):
    """
    Gets the term frequency for each word in document,
    using the formula (1 + log10(freq))
    and multiplies it by the inverse of document frequency log10(documents/relevant documents).

    Parameters
    ---------

    doc_freq: dict
        dictionary of documents, with each word in them and their frequency.

    idf: float
        inverse document frequency.

    """

    doc_freq_copy = copy.deepcopy(doc_freq)

    for doc in doc_freq_copy.keys():
        doc_content = doc_freq_copy[doc]

        for word in doc_content.keys():
            doc_content_word_freq = doc_content[word]

            if doc_content_word_freq != 0:
                # (1 + log(freq)) * log(n/ni)
                doc_content[word] = (1 + math.log(doc_content_word_freq, 10)) * idf

    print(f"TF: {doc_freq_copy}\n")

    return doc_freq_copy


def doc_normalization(doc_: dict):
    """
    Normalization of documents, so documents with more words don't perform better than the others.

    Parameters
    ---------

    doc_: dict
        dictionary of documents, and words with their frequency.

    """

    doc_copy = copy.deepcopy(doc_)
    doc_magnitude = {}

    # D1, D2, D3,...
    for doc in doc_copy.keys():
        doc_sum = []

        doc_content = doc_copy[doc]

        # camisa, azul, calca, ...
        for word in doc_content.keys():
            freq = doc_content[word]

            doc_sum.append(freq**2)

        # Sum of squared frequencies
        doc_sum_sum = sum(doc_sum)

        # Square root of the sum of squared frequencies
        magnitude = numpy.sqrt(doc_sum_sum)

        doc_magnitude[doc] = magnitude

        # print(f"{doc} magnitude: {magnitude}")
    # print("\n")

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


def modelo_vetorial(test_data: dict, query: list[str]) -> str | list[Any]:
    """
    RI model, using vectors of similarity between documents to determine which ones match the query.

    Parameter
    --------

    test_data: dict
        Data obtained from the json, documents with name, description, etc.

    query:list[str]
        List of words in the query that comes from the user.

    """

    ### Variables
    n = len(test_data.keys())  # if type(test_data="dict") else 1
    ni = 0
    ni_docs = []

    ### Query
    original_query_dict = dict(Counter(query))
    query_dict = original_query_dict.copy()

    ### Vocab
    vocab_dict, doc_freq = get_vocab(test_data, query_dict)

    # Update query_dict with all vocabulary keys
    query_dict = {vocab_key: query_dict.get(vocab_key, 0) for vocab_key in vocab_dict}

    ### Modified Query
    relevant_docs, n_relevant_docs = [], []

    for doc_id, doc in test_data.items():
        if doc["R"]:
            relevant_docs.append(doc_id)
        else:
            n_relevant_docs.append(doc_id)

    if relevant_docs and n_relevant_docs:
        modified_query = rocchio(
            query_dict,
            doc_freq,
            relevant_docs,
        )
    else:
        print(
            "Can't use Rocchio, because either there are no relevant documents, or no irrelevant documents\n"
        )

    ### Calculating ni (Relevant documents)
    # camisa, azul,...
    for word in original_query_dict.keys():

        print("Query " + word)

        # D1, D2, D3
        for doc in doc_freq.keys():

            doc_content = doc_freq[doc]

            # print(doc_content)

            # camisa, calca em query e doc e se e igual a 1
            if word in doc_content.keys() and doc_content[word] != 0:

                if doc not in ni_docs:
                    ni += 1
                    ni_docs.append(doc)

    if ni == 0:
        return "Couldn't find anything with that query"

    relevant_docs = {key: doc_freq[key] for key in ni_docs if key in doc_freq}

    # Remove words, that are not on the query, from relevant_docs
    relevant_docs_words = {
        key: {
            inner_key: inner_value
            for inner_key, inner_value in value.items()
            if inner_key in query
        }
        for key, value in relevant_docs.items()
    }

    print("Relevant words: " + f"{relevant_docs_words}\n")

    ### TF_IDF
    idf = math.log(n / ni, 10)

    print(f"IDF: {idf}\n")

    doc_TF_IDF = get_TF_IDF(relevant_docs_words, idf)

    query_TF_IDF = {
        key: (1 + math.log(value, 10)) * idf
        for key, value in original_query_dict.items()
    }

    print("Query TF_IDF: " + f"{query_TF_IDF}\n")

    ### Normalization
    doc_magnitude, doc_normalized = doc_normalization(doc_TF_IDF)

    # for doc in doc_normalized:
    #     print(f"{doc}: {doc_normalized[doc]}")

    ### Kmeans

    # kmeans(doc_TF_IDF)

    ### Similarity scores between documents and the query

    similarity = {}

    for doc in doc_TF_IDF:
        sum = 0
        for word in doc_TF_IDF[doc].keys():
            sum += doc_TF_IDF[doc][word] * query_TF_IDF[word]

        similarity[doc] = {"sim": float(sum / doc_magnitude[doc])}

    # Sorting documents based on similarity scores
    sorted_docs = sorted(
        similarity.keys(), key=lambda l_doc: similarity[l_doc]["sim"], reverse=True
    )

    print(sorted_docs)
    print("\n")

    # Adding ranking to each document
    for rank, doc_id in enumerate(sorted_docs, start=1):
        similarity[doc_id]["ranking"] = rank
        similarity[doc_id]["ID"] = str(int("".join([id_n for id_n in doc_id if id_n.isdigit()]))+1)

    ### This is for returning product details together with the score, not necessary for the API
    similar_products = {key: value for key, value in test_data.items() if key in similarity.keys()}

    for key in similar_products.keys():
        similar_products[key]["sim"] = similarity[key]["sim"]
        similar_products[key]["ranking"] = similarity[key]["ranking"]

    print(similar_products)
    print("\n")

    # Sorting the similar products by their rankings and creating a dictionary
    sorted_similarity = {
        doc: similarity[doc]
        for doc in sorted(
            similarity.keys(), key=lambda l_doc: similarity[l_doc]["ranking"]
        )
    }

    similar_products = list(sorted_similarity.values())

    return similar_products
