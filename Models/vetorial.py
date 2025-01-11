import math
import string

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

    print(vocab)

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

    print(doc_freq)

    return doc_freq


def get_TF(doc_freq:dict):

    doc_freq_copy = doc_freq.copy()

    for doc in doc_freq_copy.keys():
        doc_content = doc_freq[doc]

        for word in doc_content.keys():
            doc_content_word_freq = doc_content[word]
            doc_content[word] = (1 + math.log(doc_content_word_freq, 10))

    print(doc_freq_copy)


# def modelo_vetorial():
