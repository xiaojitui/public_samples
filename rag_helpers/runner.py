import pickle
import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.sparse import csr_matrix


def sparse_dict_to_vector(sparse_dict: Dict[int, float], dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for idx, val in sparse_dict.items():
        vec[idx] = val
    return vec

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return 1.0 - np.dot(a, b) / denom

def lexical_retrieval(
    df: pd.DataFrame,
    query_sparse: np.ndarray,
    top_k: int
) -> pd.DataFrame:
    scores = []

    for _, row in df.iterrows():
        doc_sparse = row["sparse_embedding"]

        # handle dict or vector
        if isinstance(doc_sparse, dict):
            doc_vec = sparse_dict_to_vector(doc_sparse, len(query_sparse))
        else:
            doc_vec = np.asarray(doc_sparse)

        # pgvector <#>  == negative inner product
        lexical_distance = -np.dot(query_sparse, doc_vec)

        lexical_score = lexical_distance - 1  # EXACT match to SQL

        scores.append(lexical_score)

    df_out = df.copy()
    df_out["lexical_score"] = scores
    df_out["cosine_similarity"] = None

    df_out = df_out.sort_values("lexical_score").head(top_k)

    return df_out[
        [
            "embedding_id",
            "article_id",
            "document",
            "lexical_score",
            "cosine_similarity",
            "metadata",
        ]
    ]


def semantic_retrieval(
    df: pd.DataFrame,
    query_dense: np.ndarray,
    top_k: int
) -> pd.DataFrame:
    distances = []

    for _, row in df.iterrows():
        doc_dense = np.asarray(row["dense_embedding"])
        dist = cosine_distance(doc_dense, query_dense)
        distances.append(dist)

    df_out = df.copy()
    df_out["cosine_similarity"] = distances
    df_out["lexical_score"] = None

    df_out = df_out.sort_values("cosine_similarity").head(top_k)

    return df_out[
        [
            "embedding_id",
            "article_id",
            "document",
            "lexical_score",
            "cosine_similarity",
            "metadata",
        ]
    ]

def hybrid_search(
    df: pd.DataFrame,
    query_sparse: np.ndarray,
    query_dense: np.ndarray,
    top_k_lexical: int,
    top_k_semantic: int,
) -> pd.DataFrame:
    lexical_df = lexical_retrieval(df, query_sparse, top_k_lexical)
    semantic_df = semantic_retrieval(df, query_dense, top_k_semantic)

    # UNION ALL (no dedup, no re-rank)
    return pd.concat([lexical_df, semantic_df], ignore_index=True)


# Load data
with open("embedded_data.pickle", "rb") as f:
    embedded_df = pickle.load(f)

# Example query vectors
query_sparse = np.random.rand(50000).astype(np.float32)  # match sparse dim
query_dense = np.random.rand(768).astype(np.float32)

results = hybrid_search(
    embedded_df,
    query_sparse=query_sparse,
    query_dense=query_dense,
    top_k_lexical=50,
    top_k_semantic=50,
)

print(results.head())
