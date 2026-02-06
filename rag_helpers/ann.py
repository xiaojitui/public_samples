"""
Run this in SQL:
SELECT indexdef
FROM pg_indexes
WHERE tablename = 'embedded_data';

You’re looking for something like:
USING hnsw (dense_embedding vector_cosine_ops) WITH (m=32, ef_construction=200)

If you see:
m=32 → use M=32
nothing specified → default M=16

"""

import pickle
import numpy as np
import pandas as pd
import faiss


with open("embedded_data.pickle", "rb") as f:
    df = pickle.load(f)

# Ensure dense embeddings are float32 numpy arrays
dense_matrix = np.vstack(df["dense_embedding"].values).astype("float32")

# Normalize for cosine distance (CRITICAL)
faiss.normalize_L2(dense_matrix)

# 🔹 Build FAISS HNSW index (matches pgvector)
DIM = dense_matrix.shape[1]
M = 32  # MUST match pgvector index m
EF_SEARCH = 400

index = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)

index.hnsw.efSearch = EF_SEARCH
index.add(dense_matrix)


def semantic_retrieval_faiss(
    df: pd.DataFrame,
    index: faiss.Index,
    query_dense: np.ndarray,
    top_k: int
) -> pd.DataFrame:
    q = query_dense.astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)

    # Inner product similarity
    sims, idxs = index.search(q, top_k)

    rows = []
    for sim, idx in zip(sims[0], idxs[0]):
        row = df.iloc[idx]
        rows.append({
            "embedding_id": row["embedding_id"],
            "article_id": row["article_id"],
            "document": row["document"],
            "lexical_score": None,
            "cosine_similarity": 1 - sim,  # convert similarity → distance
            "metadata": row["metadata"],
        })

    return pd.DataFrame(rows)


def lexical_retrieval_exact(
    df: pd.DataFrame,
    query_sparse: np.ndarray,
    top_k: int
) -> pd.DataFrame:
    scores = []

    for _, row in df.iterrows():
        doc_sparse = np.asarray(row["sparse_embedding"])
        lexical_distance = -np.dot(query_sparse, doc_sparse)
        lexical_score = lexical_distance - 1
        scores.append(lexical_score)

    out = df.copy()
    out["lexical_score"] = scores
    out["cosine_similarity"] = None

    out = out.sort_values("lexical_score").head(top_k)

    return out[
        ["embedding_id", "article_id", "document",
         "lexical_score", "cosine_similarity", "metadata"]
    ]


def hybrid_search_faiss(
    df,
    index,
    query_sparse,
    query_dense,
    top_k_lexical,
    top_k_semantic,
):
    lexical = lexical_retrieval_exact(df, query_sparse, top_k_lexical)
    semantic = semantic_retrieval_faiss(df, index, query_dense, top_k_semantic)

    return pd.concat([lexical, semantic], ignore_index=True)


