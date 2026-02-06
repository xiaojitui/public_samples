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



import numpy as np
from scipy.sparse import csr_matrix
def parse_sparse_string_fast(s: str):
    sparse_part, dim_part = s.split("/")
    dim = int(dim_part)

    # remove braces
    body = sparse_part.strip("{}").strip()
    if not body:
        return {}, dim

    items = body.split(",")
    d = {}
    for item in items:
        k, v = item.split(":")
        d[int(k)] = float(v)

    return d, dim

def build_sparse_matrix(sparse_strings: pd.Series):
    rows = []
    cols = []
    data = []

    dim = None

    for row_idx, s in enumerate(sparse_strings):
        d, this_dim = parse_sparse_string_fast(s)

        if dim is None:
            dim = this_dim
        elif dim != this_dim:
            raise ValueError("Sparse dimension mismatch")

        for k, v in d.items():
            rows.append(row_idx)
            cols.append(k)
            data.append(v)

    mat = csr_matrix((data, (rows, cols)), shape=(len(sparse_strings), dim))
    return mat

def query_to_sparse_vector(query_sparse_str: str):
    d, dim = parse_sparse_string_fast(query_sparse_str)

    cols = list(d.keys())
    data = list(d.values())

    return csr_matrix((data, ([0] * len(cols), cols)), shape=(1, dim))

def lexical_retrieval_fast(
    df: pd.DataFrame,
    doc_sparse_matrix: csr_matrix,
    query_sparse_str: str,
    top_k: int
) -> pd.DataFrame:

    query_vec = query_to_sparse_vector(query_sparse_str)

    # pgvector <#> == negative inner product
    # result shape: (N, 1)
    scores = -(doc_sparse_matrix @ query_vec.T).toarray().ravel()

    lexical_scores = scores - 1  # EXACT SQL match

    idx = np.argsort(lexical_scores)[:top_k]

    df_out = df.iloc[idx].copy()
    df_out["lexical_score"] = lexical_scores[idx]
    df_out["cosine_similarity"] = None

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

# one-time cost
doc_sparse_matrix = build_sparse_matrix(df["sparse_embedding"])

# per query
results = lexical_retrieval_fast(
    df,
    doc_sparse_matrix,
    query_sparse_str,
    top_k=20
)

# import ast

# def parse_sparse_string(s: str):
#     """
#     Parses '{1: 0.1, 2: 0.2}/995300'
#     Returns: (dict, dimension)
#     """
#     sparse_part, dim_part = s.split("/")
#     sparse_dict = ast.literal_eval(sparse_part)
#     dimension = int(dim_part)
#     return sparse_dict, dimension

# def sparse_dot(q: dict, d: dict) -> float:
#     """
#     Efficient sparse dot product
#     """
#     # iterate over smaller dict
#     if len(q) > len(d):
#         q, d = d, q
#     return sum(v * d.get(k, 0.0) for k, v in q.items())

# def lexical_retrieval(
#     df: pd.DataFrame,
#     query_sparse_str: str,
#     top_k: int
# ) -> pd.DataFrame:

#     query_sparse, query_dim = parse_sparse_string(query_sparse_str)

#     scores = []

#     for _, row in df.iterrows():
#         doc_sparse_str = row["sparse_embedding"]
#         doc_sparse, doc_dim = parse_sparse_string(doc_sparse_str)

#         if doc_dim != query_dim:
#             raise ValueError("Sparse dimension mismatch")

#         # pgvector <#> == negative inner product
#         lexical_distance = -sparse_dot(query_sparse, doc_sparse)

#         # EXACT SQL match
#         lexical_score = lexical_distance - 1

#         scores.append(lexical_score)

#     df_out = df.copy()
#     df_out["lexical_score"] = scores
#     df_out["cosine_similarity"] = None

#     df_out = df_out.sort_values("lexical_score").head(top_k)

#     return df_out[
#         [
#             "embedding_id",
#             "article_id",
#             "document",
#             "lexical_score",
#             "cosine_similarity",
#             "metadata",
#         ]
#     ]


# def lexical_retrieval(
#     df: pd.DataFrame,
#     query_sparse: np.ndarray,
#     top_k: int
# ) -> pd.DataFrame:
#     scores = []

#     for _, row in df.iterrows():
#         doc_sparse = row["sparse_embedding"]

#         # handle dict or vector
#         if isinstance(doc_sparse, dict):
#             doc_vec = sparse_dict_to_vector(doc_sparse, len(query_sparse))
#         else:
#             doc_vec = np.asarray(doc_sparse)

#         # pgvector <#>  == negative inner product
#         lexical_distance = -np.dot(query_sparse, doc_vec)

#         lexical_score = lexical_distance - 1  # EXACT match to SQL

#         scores.append(lexical_score)

#     df_out = df.copy()
#     df_out["lexical_score"] = scores
#     df_out["cosine_similarity"] = None

#     df_out = df_out.sort_values("lexical_score").head(top_k)

#     return df_out[
#         [
#             "embedding_id",
#             "article_id",
#             "document",
#             "lexical_score",
#             "cosine_similarity",
#             "metadata",
#         ]
#     ]


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
