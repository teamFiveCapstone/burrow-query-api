#For unit tests
```
uv run --extra test pytest test_main.py
```



# RAGline Query API

A FastAPI-based service for querying vector embeddings stored in PostgreSQL with pgvector. This API provides retrieval endpoints for RAG (Retrieval Augmented Generation) applications.

## Features

- **Vector Similarity Search**: Retrieve top-K most similar documents using cosine similarity
- **Metadata Filtering**: Filter results by document metadata
- **RAG Query Engine**: Synthesized responses using retrieved context
- **AWS Bedrock Integration**: Uses Amazon Titan embeddings
- **Health Monitoring**: Built-in health check endpoints

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Client    │─────▶│  Query API   │─────▶│  PostgreSQL +   │
│             │      │  (FastAPI)   │      │   pgvector      │
└─────────────┘      └──────────────┘      └─────────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ AWS Bedrock  │
                     │   (Titan)    │
                     └──────────────┘
```

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- AWS credentials configured for Bedrock access

### Local Development

#### Using uv (recommended - faster)

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Navigate to the project directory:
```bash
cd query_api
```

3. Install dependencies (uv automatically manages the virtual environment):
```bash
uv sync
```

4. Run the application:
```bash
uv run python main.py
```

#### Using pip

1. Clone the repository and navigate to the project directory:
```bash
cd query_api
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python main.py
```

### Configuration

Create a `.env` file (optional, defaults are in config.py):
```bash
DB_NAME=embeddings
DB_USER=burrow
DB_PASSWORD=capstone
DB_HOST=burrow-serverless-wilson.cluster-cwxgyacqyoae.us-east-1.rds.amazonaws.com
DB_PORT=5432
TABLE_NAME=data_burrow_table
EMBED_DIM=1024
AWS_REGION=us-east-1
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### GET /
Root endpoint with API information.

### GET /health
Health check endpoint showing service and database status.

**Response:**
```json
{
  "status": "healthy",
  "database_connected": true,
  "vector_store_initialized": true
}
```

### POST /retrieve
Retrieve top-K similar documents without synthesis.

**Request:**
```json
{
  "query": "What is machine learning?",
  "top_k": 5,
  "filters": {
    "filters": [
      {
        "key": "source",
        "value": "wikipedia",
        "operator": "=="
      }
    ],
    "condition": "and"
  }
}
```

**Response:**
```json
{
  "nodes": [
    {
      "node_id": "abc123",
      "text": "Machine learning is...",
      "score": 0.89,
      "metadata": {
        "source": "wikipedia",
        "date": "2024-01-15"
      }
    }
  ],
  "query": "What is machine learning?",
  "total_results": 5
}
```

### POST /query
RAG query with synthesis (requires LLM configured).

**Request:**
```json
{
  "query": "Explain machine learning",
  "top_k": 5,
  "filters": null
}
```

**Response:**
```json
{
  "response": "Machine learning is a branch of artificial intelligence...",
  "source_nodes": [...],
  "query": "Explain machine learning"
}
```

## Docker Deployment

### Build the image:
```bash
docker build -t query-api .
```

### Run the container:
```bash
docker run -p 8000:8000 \
  -e DB_PASSWORD=your_password \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  query-api
```

## AWS ECS Deployment

### 1. Build and push to ECR:
```bash
# Authenticate to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker build -t query-api .
docker tag query-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/query-api:latest

# Push
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/query-api:latest
```

### 2. Create ECS task definition with environment variables:
- DB credentials
- AWS region
- Table configuration

### 3. Deploy to ECS service

## Metadata Filtering

Supported operators:
- `==`: Equal
- `>`: Greater than
- `<`: Less than
- `>=`: Greater than or equal
- `<=`: Less than or equal
- `!=`: Not equal
- `in`: In list
- `nin`: Not in list

Example with multiple filters:
```json
{
  "query": "search term",
  "top_k": 10,
  "filters": {
    "filters": [
      {"key": "category", "value": "science", "operator": "=="},
      {"key": "year", "value": 2020, "operator": ">="}
    ],
    "condition": "and"
  }
}
```

## Configuration

All configuration is managed through `config.py` and can be overridden with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| DB_NAME | embeddings | Database name |
| DB_USER | burrow | Database user |
| DB_PASSWORD | capstone | Database password |
| DB_HOST | burrow-serverless-wilson... | Database host |
| DB_PORT | 5432 | Database port |
| TABLE_NAME | burrow_table | Vector store table name (PGVectorStore adds "data_" prefix) |
| EMBED_DIM | 1024 | Embedding dimensions |
| AWS_REGION | us-east-1 | AWS region |
| BEDROCK_MODEL_ID | amazon.titan-embed-text-v2:0 | Bedrock model |

## Development

### Interactive API docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Testing with curl:
```bash
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "top_k": 3
  }'
```

## Troubleshooting

### Database connection issues:
- Verify database credentials
- Check security group rules for Aurora
- Ensure pgvector extension is installed

### Vector store initialization fails or returns 0 results:
- Confirm table exists: `SELECT COUNT(*) FROM data_burrow_table;`
- Check embed_dim matches your Bedrock model output
- **Important**: If your table is named `data_X`, set `TABLE_NAME=X` (without the "data_" prefix) because PGVectorStore automatically adds it

### AWS Bedrock errors:
- Verify IAM permissions for Bedrock
- Check AWS credentials are configured
- Ensure model is available in your region

## License

MIT
