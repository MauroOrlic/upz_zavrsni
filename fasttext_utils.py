import fasttext
from typing import Collection, List
from scipy import spatial
from numpy import mean


def get_model(model_path: str, train_data_path: str):
    try:
        model = fasttext.load_model(model_path)
    except ValueError:
        model = fasttext.train_unsupervised(train_data_path, model='skipgram')
        model.save_model(model_path)
    return model


def _get_sentence_vectors(model: fasttext.FastText._FastText, doc_path: str):
    text = open(doc_path, 'r').read()
    vector = mean([model.get_sentence_vector(sentence) for sentence in text.split('\n')], axis=0)
    return vector

def get_similarity_matrix(model:fasttext.FastText._FastText, doc_paths: Collection[str]) -> List[List[float]]:
    result: List[List[float]] = []
    vectors = [_get_sentence_vectors(model, doc_path) for doc_path in doc_paths]
    for i in range(len(vectors)):
        result.append([])
        for j in range(len(vectors)):
            result[i].append(round(
                1 - spatial.distance.cosine(vectors[i], vectors[j]),
                2))

    return result

def pp_matrix(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))