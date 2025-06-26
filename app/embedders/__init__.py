"""Embedders package for text-to-vector conversion.

This package contains different embedding strategies to convert text
into numerical vectors for similarity search and retrieval.

Available embedders:
- tfidf_embedder: Word frequency based embeddings (simple, fast)
- sentence_transformer_embedder: Neural sentence embeddings (best quality)
- huggingface_embedder: Free transformer embeddings (BERT-based)
""" 