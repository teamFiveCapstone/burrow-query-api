"""Vector store initialization and management."""
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import Settings as LlamaSettings
from config import settings
from reranker import BedrockCohereRerank
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store and index initialization."""

    def __init__(self):
        self.vector_store = None
        self.index = None
        self.embed_model = None
        self.reranker = None

    def initialize(self):
        """Initialize the vector store, embedding model, and index."""
        try:
            # Initialize Bedrock embedding model
            logger.info(f"Initializing Bedrock embedding model: {settings.bedrock_model_id}")
            self.embed_model = BedrockEmbedding(
                model=settings.bedrock_model_id,
                region_name=settings.aws_region,
            )

            # Set global embedding model for LlamaIndex
            LlamaSettings.embed_model = self.embed_model

            # Connect to existing vector store with hybrid search enabled
            logger.info(f"Connecting to vector store table: {settings.table_name}")
            self.vector_store = PGVectorStore.from_params(
                database=settings.db_name,
                host=settings.db_host,
                password=settings.db_password,
                port=settings.db_port,
                user=settings.db_user,
                table_name=settings.table_name,
                embed_dim=settings.embed_dim,
                schema_name="public",
                hybrid_search=True,
                text_search_config="english",
            )

            # Create index from existing vector store (no document loading needed)
            logger.info("Creating VectorStoreIndex from existing store")
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )

            # Initialize reranker
            logger.info("Initializing Bedrock Cohere Reranker")
            self.reranker = BedrockCohereRerank(
                region_name=settings.aws_region,
                top_n=10,  # Default, can be overridden per request
            )

            logger.info("Vector store initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def get_retriever(self, similarity_top_k: int = 5, filters=None, mode: str = "default"):
        """
        Get a retriever for querying the vector store.

        Args:
            similarity_top_k: Number of top similar results to return
            filters: Optional metadata filters
            mode: Search mode - "default" (vector), "hybrid", or "sparse" (keyword)

        Returns:
            VectorIndexRetriever instance
        """
        if not self.index:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        kwargs = {
            "similarity_top_k": similarity_top_k,
            "filters": filters,
        }

        if mode == "hybrid":
            kwargs["vector_store_query_mode"] = "hybrid"
            kwargs["sparse_top_k"] = similarity_top_k
        elif mode == "sparse":
            kwargs["vector_store_query_mode"] = "sparse"
            kwargs["sparse_top_k"] = similarity_top_k

        return self.index.as_retriever(**kwargs)

    def get_query_engine(self, similarity_top_k: int = 5, filters=None, mode: str = "default"):
        """
        Get a query engine for RAG-style queries.

        Args:
            similarity_top_k: Number of top similar results to return
            filters: Optional metadata filters
            mode: Search mode - "default" (vector), "hybrid", or "sparse" (keyword)

        Returns:
            Query engine instance
        """
        if not self.index:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        kwargs = {
            "similarity_top_k": similarity_top_k,
            "filters": filters,
        }

        if mode == "hybrid":
            kwargs["vector_store_query_mode"] = "hybrid"
            kwargs["sparse_top_k"] = similarity_top_k
        elif mode == "sparse":
            kwargs["vector_store_query_mode"] = "sparse"
            kwargs["sparse_top_k"] = similarity_top_k

        return self.index.as_query_engine(**kwargs)

    def rerank_nodes(self, nodes, query: str, top_n: int = 5):
        """
        Rerank retrieved nodes using Bedrock Cohere Rerank.

        Args:
            nodes: List of NodeWithScore from retrieval
            query: Original query string
            top_n: Number of top results to return after reranking

        Returns:
            Reranked list of NodeWithScore
        """
        if not self.reranker:
            raise RuntimeError("Reranker not initialized.")

        return self.reranker.rerank(query=query, nodes=nodes, top_n=top_n)


# Global vector store manager instance
vector_store_manager = VectorStoreManager()