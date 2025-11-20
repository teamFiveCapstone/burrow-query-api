# Project Overview
- In LlamaIndex Postgres.md there is a tutorial for using Postgres and pgvector with LlamaIndex
- For our RAG ETL project, we have an Aurora Serverless database that an ECS task uploads data to using the LlamaIndex PGVectorStore abstraction.
- My job is now to create an API that can query from that database and expose retrieval routes. Basic functionality would be to retrieve top-k vectors by cosine similarity, retrieve most relevant documents. More advanced would be hybrid search and reranking.
-Database info (it is a public Aurora database for development purposes for now)
DB_NAME=embeddings
DB_USER=burrow
DB_PASSWORD=capstone
DB_HOST=burrow-serverless-wilson.cluster-cwxgyacqyoae.us-east-1.rds.amazonaws.com
DB_PORT=5432

# Explanation Preferences (for learning)
- Explain design decisions as you implement
- Link to relevant documentation
- Highlight alternatives you considered
- Point out best practices vs anti-patterns
- Add comments for complex logic