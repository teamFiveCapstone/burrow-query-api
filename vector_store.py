from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import Settings as LlamaSettings
from llama_index.core.retrievers import QueryFusionRetriever
from config import settings
from reranker import BedrockCohereRerank
from logger import log_info, log_error, log_exception


class VectorStoreManager:
    def __init__(self):
        self.vector_store = None
        self.index = None
        self.embed_model = None
        self.reranker = None

    def initialize(self):
        try:
            log_info(
                "Initializing Bedrock embedding model",
                bedrock_model_id=settings.bedrock_model_id,
                aws_region=settings.aws_region,
            )

            self.embed_model = BedrockEmbedding(
                model=settings.bedrock_model_id,
                region_name=settings.aws_region,
            )

            LlamaSettings.embed_model = self.embed_model

            log_info(
                "Connecting to PGVectorStore",
                db_name=settings.db_name,
                db_host=settings.db_host,
                table_name=settings.table_name,
                embed_dim=settings.embed_dim,
            )

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

            log_info("Creating VectorStoreIndex from existing store")

            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )

            log_info(
                "Initializing Bedrock Cohere Reranker",
                aws_region=settings.aws_region,
                default_top_n=10,
            )

            self.reranker = BedrockCohereRerank(
                region_name=settings.aws_region,
                top_n=10,
            )

            log_info("Vector store initialization complete")

        except Exception:
            log_exception("Failed to initialize vector store")
            raise

    def get_retriever(
        self,
        similarity_top_k: int = 5,
        filters=None,
        mode: str = "default",
    ):
        if not self.index:
            log_error(
                "get_retriever called before initialization",
                similarity_top_k=similarity_top_k,
                mode=mode,
            )
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

        log_info(
            "Creating retriever",
            similarity_top_k=similarity_top_k,
            mode=mode,
        )

        return self.index.as_retriever(**kwargs)

    def get_query_engine(
        self,
        similarity_top_k: int = 5,
        filters=None,
        mode: str = "default",
    ):
        if not self.index:
            log_error(
                "get_query_engine called before initialization",
                similarity_top_k=similarity_top_k,
                mode=mode,
            )
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

        log_info(
            "Creating query engine",
            similarity_top_k=similarity_top_k,
            mode=mode,
        )

        return self.index.as_query_engine(**kwargs)

    def get_fusion_retriever(self, similarity_top_k: int = 5, filters=None):
        
        if not self.index:
            log_error(
                "get_fusion_retriever called before initialization",
                similarity_top_k=similarity_top_k,
            )
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        vector_retriever = self.index.as_retriever(
            vector_store_query_mode="default",
            similarity_top_k=similarity_top_k,
            filters=filters,
        )

        keyword_retriever = self.index.as_retriever(
            vector_store_query_mode="sparse",
            similarity_top_k=similarity_top_k,
            sparse_top_k=similarity_top_k,
            filters=filters,
        )

        log_info(
            "Creating QueryFusionRetriever for hybrid search",
            similarity_top_k=similarity_top_k,
        )

        return QueryFusionRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            similarity_top_k=similarity_top_k,
            num_queries=1,  
            mode="reciprocal_rerank",
            use_async=True,
        )

    def rerank_nodes(self, nodes, query: str, top_n: int = 5):
        if not self.reranker:
            log_error(
                "rerank_nodes called before reranker initialization",
                top_n=top_n,
            )
            raise RuntimeError("Reranker not initialized.")

        log_info(
            "Reranking nodes",
            node_count=len(nodes),
            top_n=top_n,
        )

        return self.reranker.rerank(query=query, nodes=nodes, top_n=top_n)


vector_store_manager = VectorStoreManager()
