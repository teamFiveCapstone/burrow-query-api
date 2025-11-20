"""Pydantic models for API request and response validation."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class MetadataFilterOperator(str, Enum):
    """Supported metadata filter operators."""
    EQ = "=="
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    NE = "!="
    IN = "in"
    NIN = "nin"


class MetadataFilter(BaseModel):
    """Single metadata filter."""
    key: str = Field(..., description="Metadata key to filter on")
    value: Any = Field(..., description="Value to filter by")
    operator: MetadataFilterOperator = Field(
        default=MetadataFilterOperator.EQ,
        description="Comparison operator"
    )


class MetadataFilters(BaseModel):
    """Collection of metadata filters."""
    filters: List[MetadataFilter] = Field(default_factory=list)
    condition: str = Field(
        default="and",
        description="Logical condition: 'and' or 'or'"
    )


class RetrieveRequest(BaseModel):
    """Request model for document retrieval."""
    query: str = Field(..., description="Query text for similarity search")
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of top results to return"
    )
    filters: Optional[MetadataFilters] = Field(
        default=None,
        description="Optional metadata filters"
    )


class QueryRequest(BaseModel):
    """Request model for RAG query with synthesis."""
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
    """Single retrieved node/document."""
    node_id: str = Field(..., description="Unique node identifier")
    text: str = Field(..., description="Node text content")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Node metadata"
    )


class RetrieveResponse(BaseModel):
    """Response model for retrieval endpoint."""
    nodes: List[NodeResponse] = Field(..., description="Retrieved nodes")
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Number of results returned")


class QueryResponse(BaseModel):
    """Response model for query endpoint with synthesis."""
    response: str = Field(..., description="Synthesized response")
    source_nodes: List[NodeResponse] = Field(..., description="Source nodes used")
    query: str = Field(..., description="Original query")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    database_connected: bool = Field(..., description="Database connection status")
    vector_store_initialized: bool = Field(
        ...,
        description="Vector store initialization status"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
