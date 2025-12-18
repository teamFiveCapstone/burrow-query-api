from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config import settings
from vector_store import vector_store_manager
from models import (
    RetrieveRequest,
    RetrieveResponse,
    QueryRequest,
    QueryResponse,
    NodeResponse,
    CountResponse,
    HealthResponse,
)
from llama_index.core.vector_stores.types import (
    MetadataFilters as LlamaMetadataFilters,
    MetadataFilter as LlamaMetadataFilter,
    FilterOperator,
)
from security import verify_api_token
from logger import log_info, log_exception
import time
import asyncpg


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_info("Starting up Query API")
    try:
        vector_store_manager.initialize()
        log_info("Vector store initialized successfully")
    except Exception:
        log_exception("Failed to initialize vector store")
        raise
    yield
    log_info("Shutting down Query API")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan,
    root_path="/query-service",
)


@app.middleware("http")
async def log_http_requests(request, call_next):
    start = time.time()

    response = await call_next(request)

    duration_ms = (time.time() - start) * 1000

    log_info(
        "HTTP request completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )

    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_filters(filters):
    if not filters:
        return None

    llama_filters = []
    for f in filters.filters:
        if not f.key or f.value is None:
            continue

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

    if not llama_filters:
        return None

    return LlamaMetadataFilters(
        filters=llama_filters,
        condition=filters.condition,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "RAGline Query API",
        "version": settings.api_version,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health_check():
    try:
        is_initialized = vector_store_manager.index is not None
        db_connected = vector_store_manager.vector_store is not None

        status_str = "healthy" if is_initialized and db_connected else "degraded"

        log_info(
            "Health check",
            status=status_str,
            database_connected=db_connected,
            vector_store_initialized=is_initialized,
        )

        return HealthResponse(
            status=status_str,
            database_connected=db_connected,
            vector_store_initialized=is_initialized,
        )
    except Exception:
        log_exception("Health check failed")
        return HealthResponse(
            status="unhealthy",
            database_connected=False,
            vector_store_initialized=False,
        )


@app.get("/count", response_model=CountResponse, tags=["Counts"], dependencies=[Depends(verify_api_token)])
async def get_document_count():
    try:
        log_info("Document count request received")

        conn = await asyncpg.connect(
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port,
        )

        query = f"""
            SELECT
                COUNT(DISTINCT (metadata_->>'doc_id')) as document_count,
                COUNT(*) as chunk_count
            FROM data_{settings.table_name}
        """

        row = await conn.fetchrow(query)
        doc_count = row['document_count'] if row and row['document_count'] is not None else 0
        chunk_count = row['chunk_count'] if row and row['chunk_count'] is not None else 0

        await conn.close()

        log_info("Document count query completed", total_documents=doc_count, total_chunks=chunk_count)

        return {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "status": "success",
        }

    except Exception as e:
        error_type = type(e).__name__

        if error_type == "UndefinedTableError":
            log_info("Table does not exist yet, returning count of 0")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "status": "table_not_initialized",
            }
        else:
            log_exception(
                "Document count failed",
                error_type=error_type,
                error_message=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve document count",
            )


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    tags=["Retrieval"],
    dependencies=[Depends(verify_api_token)],
)
async def retrieve(request: RetrieveRequest):
    try:
        mode_map = {"vector": "default", "keyword": "sparse", "hybrid": "hybrid"}
        internal_mode = mode_map.get(request.mode.value, "default")

        log_info(
            "Retrieve request received",
            mode=request.mode.value,
            internal_mode=internal_mode,
            top_k=request.top_k,
            rerank=request.rerank,
            query_preview=request.query[:100],
        )

        llama_filters = convert_filters(request.filters)

        retrieval_top_k = request.top_k * 3 if request.rerank else request.top_k

        retriever = vector_store_manager.get_retriever(
            similarity_top_k=retrieval_top_k,
            filters=llama_filters,
            mode=internal_mode,
        )

        nodes = retriever.retrieve(request.query)
        log_info("Initial retrieval completed", node_count=len(nodes))

        if request.rerank and nodes:
            rerank_top_n = request.rerank_top_n or request.top_k
            log_info("Applying reranking", rerank_top_n=rerank_top_n)
            nodes = vector_store_manager.rerank_nodes(
                nodes=nodes,
                query=request.query,
                top_n=rerank_top_n,
            )
            log_info("Reranking completed", node_count=len(nodes))

        node_responses = [
            NodeResponse(
                node_id=node.node_id,
                text=node.get_content(),
                score=node.score or 0.0,
                metadata=node.metadata,
            )
            for node in nodes
        ]

        log_info(
            "Retrieve request completed",
            total_results=len(node_responses),
        )

        return RetrieveResponse(
            nodes=node_responses,
            query=request.query,
            total_results=len(node_responses),
        )

    except ValueError as e:
        log_exception("Invalid request parameters", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        log_exception("Service error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        )
    except Exception as e:
        log_exception(
            "Retrieval failed",
            error_type=type(e).__name__,
            error_message=str(e),
            mode=request.mode.value,
            top_k=request.top_k,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {type(e).__name__}",
        )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Query"],
    dependencies=[Depends(verify_api_token)],
)
async def query(request: QueryRequest):
    try:
        log_info(
            "Query request received",
            top_k=request.top_k,
            query_preview=request.query[:100],
        )

        llama_filters = convert_filters(request.filters)

        query_engine = vector_store_manager.get_query_engine(
            similarity_top_k=request.top_k,
            filters=llama_filters,
        )

        response = query_engine.query(request.query)

        source_nodes = [
            NodeResponse(
                node_id=node.node.node_id,
                text=node.node.get_content(),
                score=node.score or 0.0,
                metadata=node.node.metadata,
            )
            for node in response.source_nodes
        ]

        log_info(
            "Query completed",
            source_node_count=len(source_nodes),
        )

        return QueryResponse(
            response=str(response),
            source_nodes=source_nodes,
            query=request.query,
        )

    except Exception:
        log_exception(
            "Query failed",
            top_k=request.top_k,
            query_preview=request.query[:100],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query failed",
        )
