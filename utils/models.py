"""Pydantic models for API request and response validation."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class SearchMode(str, Enum):
    VECTOR = "vector"  # Vector similarity search (default)
    KEYWORD = "keyword"  # Keyword/text search (sparse)
    HYBRID = "hybrid"  # Combined vector + keyword search


class MetadataFilterOperator(str, Enum):
    EQ = "=="
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    NE = "!="
    IN = "in"
    NIN = "nin"


class MetadataFilter(BaseModel):
    key: str = Field(..., description="Metadata key to filter on")
    value: Any = Field(..., description="Value to filter by")
    operator: MetadataFilterOperator = Field(
        default=MetadataFilterOperator.EQ,
        description="Comparison operator"
    )


class MetadataFilters(BaseModel):
    filters: List[MetadataFilter] = Field(default_factory=list)
    condition: str = Field(
        default="and",
        description="Logical condition: 'and' or 'or'"
    )


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Query text for similarity search")
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of top results to return"
    )
    mode: SearchMode = Field(
        default=SearchMode.VECTOR,
        description="Search mode: vector, keyword, or hybrid"
    )
    rerank: bool = Field(
        default=False,
        description="Whether to apply reranking using Cohere Rerank"
    )
    rerank_top_n: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of results after reranking (defaults to top_k)"
    )
    filters: Optional[MetadataFilters] = Field(
        default=None,
        description="Optional metadata filters"
    )


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of documents to retrieve"
    )
    filters: Optional[MetadataFilters] = Field(
        default=None,
        description="Optional metadata filters"
    )


class NodeResponse(BaseModel):
    node_id: str = Field(..., description="Unique node identifier")
    text: str = Field(..., description="Node text content")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Node metadata"
    )


class RetrieveResponse(BaseModel):
    nodes: List[NodeResponse] = Field(..., description="Retrieved nodes")
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Number of results returned")


class QueryResponse(BaseModel):
    response: str = Field(..., description="Synthesized response")
    source_nodes: List[NodeResponse] = Field(..., description="Source nodes used")
    query: str = Field(..., description="Original query")


class CountResponse(BaseModel):
    total_documents: int = Field(..., description="Total number of unique documents")
    total_chunks: int = Field(..., description="Total number of chunks/vectors")
    status: str = Field(..., description="Query status")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    database_connected: bool = Field(..., description="Database connection status")
    vector_store_initialized: bool = Field(
        ...,
        description="Vector store initialization status"
    )


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
