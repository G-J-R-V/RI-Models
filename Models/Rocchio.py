"""
# TODO

I need the query to have all vocabulary words, so it can compare to the other documents (applies to then too)

Currently vetorial.py doesn't do that, and I need full vocabulary for kmeans too :|

Probably a lot of others problems that I don't know of yet too :D
"""

import numpy as np


def rocchio(
    query: dict[str:int],
    doc_freq,
    relevant_docs,
    n_relevant_docs,
    alpha: float = 1,
    beta: float = 0.6,
    gama: float = 0.4,
) -> dict[str:int]:
    """

    Parameters
    ----------

    query: dict
        query with all vocabulary keys and frequencies.
    doc_freq: dict
        dictionary of documents with words and frequencies.
    relevant_docs:list
        list of relevant doc_ids
    n_relevant_docs:list
        ~relevant_docs
    alpha: float = 1
    beta: float = 0.6
    gama: float = 0.4

    Returns
    -------

    modified_query_dict: dict
        modified query
    """

    query_keys = [key for key in query.keys() if query[key] > 0]
    query_freqs = np.array(list(query.values()))

    relevant_words = []
    n_relevant_words = []

    for doc_id, doc in doc_freq.items():

        if doc_id in relevant_docs:
            # print(f"\nRelevant {doc_id} {doc}\n")
            relevant_words.append(np.array(list(doc.values())))
        else:
            # print(f"\nn_Relevant {doc_id} {doc}\n")
            n_relevant_words.append(np.array(list(doc.values())))

    # print(relevant_words)

    relevant_words_sum = np.sum(relevant_words, axis=0)
    n_relevant_words_sum = np.sum(n_relevant_words, axis=0)

    print(f"Query:{query_freqs}")
    print(f"relevant_ {relevant_words_sum}")
    print(f"n_relevant_ {n_relevant_words_sum}")

    # Rocchio
    modified_query = (
        alpha * query_freqs
        + (beta / len(relevant_docs)) * relevant_words_sum
        - (gama / len(n_relevant_docs)) * n_relevant_words_sum
    )

    print(f"New query_freqs: {modified_query}")

    modified_query_dict = {
        key: float(round(value, 3)) for key, value in zip(query, modified_query)
    }

    # Sort modified query by the word value, descending
    modified_query_dict_sorted = {
        key: modified_query_dict[key]
        for key in sorted(
            modified_query_dict.keys(),
            key=lambda l_doc: modified_query_dict[l_doc],
            reverse=True,
        )
    }

    # Get new recommended key
    new_key = [key for key in modified_query_dict_sorted if key not in query_keys][0]

    # Filter modified query to only the original keys and the new one
    modified_query_dict_filtered = {
        key: value
        for key, value in modified_query_dict.items()
        if key in query_keys or key == new_key
    }

    print("\n")
    print(f"New query: {modified_query_dict}\n")
    print(f"New query_filtered: {modified_query_dict_filtered}\n")

    return modified_query_dict_filtered


if __name__ == "__main__":
    test_query = {
        "alface": 0,
        "arroz": 1,
        "batata": 0,
        "caju": 0,
        "feijao": 5,
        "maca": 0,
    }
    doc_freq = {
        "Doc0": {
            "alface": 0,
            "arroz": 1,
            "batata": 3,
            "caju": 0,
            "feijao": 0,
            "maca": 0,
        },
        "Doc1": {
            "alface": 4,
            "arroz": 0,
            "batata": 0,
            "caju": 0,
            "feijao": 0,
            "maca": 3,
        },
        "Doc2": {
            "alface": 0,
            "arroz": 0,
            "batata": 0,
            "caju": 3,
            "feijao": 1,
            "maca": 0,
        },
    }

    modified_query = rocchio(test_query, doc_freq, ["Doc1"], ["Doc0", "Doc2"])
