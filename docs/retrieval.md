# Retrieval Augmented Generation

Retrieval augmented generation combines search with a language model. The search system first retrieves relevant context, then the model answers using that context.

A simple RAG system needs document loading, chunking, embeddings, vector search, prompt construction, and answer generation. The search component is the foundation because poor retrieval usually produces poor answers.
