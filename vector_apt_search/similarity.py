import numpy as np


def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two NumPy vectors.
    """
    dot_product = np.dot(vec2, vec1)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2, axis=1)

    return dot_product / (norm_vec1 * norm_vec2)
