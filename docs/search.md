# Semantic Search

Semantic search finds documents by meaning rather than only exact words. A user can search for "customer churn warning signs" and still find passages about account cancellation risk, poor renewal signals, or declining product usage.

Most production systems encode documents into vectors, encode the query into the same vector space, and rank documents by similarity. The vector index can be refreshed when new documents are added.

Good semantic search products show the matching passage, the source document, and a confidence score. They also make it easy to inspect why a result was returned.
