"""FastAPI application for querying vector database."""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config import settings
from vector_store import vector_store_manager
from models import (
    RetrieveRequest,
    RetrieveResponse,
    QueryRequest,
    QueryResponse,
    NodeResponse,
    HealthResponse,
    ErrorResponse,
)
from llama_index.core.vector_stores.types import (
    MetadataFilters as LlamaMetadataFilters,
    MetadataFilter as LlamaMetadataFilter,
    FilterOperator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting up Query API...")
    try:
        vector_store_manager.initialize()
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Query API...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_filters(filters):
    """Convert API filters to LlamaIndex filters."""
    if not filters:
        return None

    llama_filters = []
    for f in filters.filters:
        # Map string operators to FilterOperator enum
        operator_map = {
            "==": FilterOperator.EQ,
            ">": FilterOperator.GT,
            "<": FilterOperator.LT,
            ">=": FilterOperator.GTE,
            "<=": FilterOperator.LTE,
            "!=": FilterOperator.NE,
            "in": FilterOperator.IN,
            "nin": FilterOperator.NIN,
        }

        llama_filter = LlamaMetadataFilter(
            key=f.key,
            value=f.value,
            operator=operator_map.get(f.operator, FilterOperator.EQ),
        )
        llama_filters.append(llama_filter)

    return LlamaMetadataFilters(
        filters=llama_filters,
        condition=filters.condition,
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "RAGline Query API",
        "version": settings.api_version,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the status of the service and database connection.
    """
    try:
        is_initialized = vector_store_manager.index is not None
        db_connected = vector_store_manager.vector_store is not None

        return HealthResponse(
            status="healthy" if is_initialized and db_connected else "degraded",
            database_connected=db_connected,
            vector_store_initialized=is_initialized,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            database_connected=False,
            vector_store_initialized=False,
        )


@app.post("/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
async def retrieve(request: RetrieveRequest):
    """
    Retrieve top-K most similar documents.

    Supports three search modes:
    - **vector**: Pure vector similarity search (default)
    - **keyword**: Text/keyword search using PostgreSQL full-text search
    - **hybrid**: Combined vector + keyword search for best results

    Optional reranking with Cohere Rerank (via AWS Bedrock) for improved relevance.

    Args:
        request: RetrieveRequest with query, top_k, mode, rerank options, and filters

    Returns:
        RetrieveResponse with retrieved nodes
    """
    try:
        # Map search mode to internal mode
        mode_map = {"vector": "default", "keyword": "sparse", "hybrid": "hybrid"}
        internal_mode = mode_map.get(request.mode.value, "default")

        logger.info(f"Retrieve request: query='{request.query[:50]}...', top_k={request.top_k}, mode={request.mode.value}")

        # Convert filters
        llama_filters = convert_filters(request.filters)

        # When reranking, retrieve more candidates for better results
        retrieval_top_k = request.top_k * 3 if request.rerank else request.top_k

        # Get retriever with appropriate mode
        retriever = vector_store_manager.get_retriever(
            similarity_top_k=retrieval_top_k,
            filters=llama_filters,
            mode=internal_mode,
        )

        # Retrieve nodes
        nodes = retriever.retrieve(request.query)
        logger.info(f"Initial retrieval: {len(nodes)} nodes")

        # Apply reranking if requested
        if request.rerank and nodes:
            rerank_top_n = request.rerank_top_n or request.top_k
            logger.info(f"Applying reranking, top_n={rerank_top_n}")
            nodes = vector_store_manager.rerank_nodes(
                nodes=nodes,
                query=request.query,
                top_n=rerank_top_n,
            )
            logger.info(f"After reranking: {len(nodes)} nodes")

        # Convert to response format
        node_responses = [
            NodeResponse(
                node_id=node.node_id,
                text=node.get_content(),
                score=node.score or 0.0,
                metadata=node.metadata,
            )
            for node in nodes
        ]

        logger.info(f"Retrieved {len(node_responses)} nodes")

        return RetrieveResponse(
            nodes=node_responses,
            query=request.query,
            total_results=len(node_responses),
        )

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}",
        )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Query with RAG synthesis.

    This endpoint retrieves relevant documents and synthesizes
    a response using a language model.

    Args:
        request: QueryRequest with query, top_k, and optional filters

    Returns:
        QueryResponse with synthesized response and source nodes
    """
    try:
        logger.info(f"Query request: query='{request.query}', top_k={request.top_k}")

        # Convert filters
        llama_filters = convert_filters(request.filters)

        # Get query engine
        query_engine = vector_store_manager.get_query_engine(
            similarity_top_k=request.top_k,
            filters=llama_filters,
        )

        # Execute query
        response = query_engine.query(request.query)

        # Convert source nodes to response format
        source_nodes = [
            NodeResponse(
                node_id=node.node.node_id,
                text=node.node.get_content(),
                score=node.score or 0.0,
                metadata=node.node.metadata,
            )
            for node in response.source_nodes
        ]

        logger.info(f"Query completed with {len(source_nodes)} source nodes")

        return QueryResponse(
            response=str(response),
            source_nodes=source_nodes,
            query=request.query,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
