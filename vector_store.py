"""Vector store initialization and management."""
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import Settings as LlamaSettings
from config import settings
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store and index initialization."""

    def __init__(self):
        self.vector_store = None
        self.index = None
        self.embed_model = None

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

            # Connect to existing vector store
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
            )

            # Create index from existing vector store (no document loading needed)
            logger.info("Creating VectorStoreIndex from existing store")
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )

            logger.info("Vector store initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def get_retriever(self, similarity_top_k: int = 5, filters=None):
        """
        Get a retriever for querying the vector store.

        Args:
            similarity_top_k: Number of top similar results to return
            filters: Optional metadata filters

        Returns:
            VectorIndexRetriever instance
        """
        if not self.index:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        return self.index.as_retriever(
            similarity_top_k=similarity_top_k,
            filters=filters,
        )

    def get_query_engine(self, similarity_top_k: int = 5, filters=None):
        """
        Get a query engine for RAG-style queries.

        Args:
            similarity_top_k: Number of top similar results to return
            filters: Optional metadata filters

        Returns:
            Query engine instance
        """
        if not self.index:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        return self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            filters=filters,
        )


# Global vector store manager instance
vector_store_manager = VectorStoreManager()
