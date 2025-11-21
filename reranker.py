"""Bedrock Cohere Reranker implementation."""
import json
import logging
from typing import List

import boto3
from llama_index.core.schema import NodeWithScore, TextNode

logger = logging.getLogger(__name__)


class BedrockCohereRerank:
    """Reranker using Cohere's rerank model via AWS Bedrock."""

    def __init__(
        self,
        model_id: str = "cohere.rerank-v3-5:0",
        region_name: str = "us-east-1",
        top_n: int = 5,
    ):
        self.model_id = model_id
        self.region_name = region_name
        self.top_n = top_n
        self._client = boto3.client("bedrock-runtime", region_name=region_name)

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_n: int = None,
    ) -> List[NodeWithScore]:
        """
        Rerank nodes using Bedrock Cohere model.

        Args:
            query: Query string
            nodes: List of NodeWithScore from retrieval
            top_n: Number of results to return (defaults to self.top_n)

        Returns:
            Reranked list of NodeWithScore
        """
        if not nodes:
            return nodes

        top_n = top_n or self.top_n
        documents = [node.get_content() for node in nodes]

        if not documents:
            return nodes

        try:
            # Prepare request for Bedrock Cohere Rerank
            request_body = {
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
            }

            logger.info(f"Reranking {len(documents)} documents with Bedrock Cohere")

            # Call Bedrock
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            # Parse response
            response_body = json.loads(response["body"].read())
            results = response_body.get("results", [])

            logger.info(f"Reranking complete, got {len(results)} results")

            # Reorder nodes based on rerank results
            reranked_nodes = []
            for result in results:
                idx = result["index"]
                relevance_score = result["relevance_score"]

                # Create new NodeWithScore with updated score
                original_node = nodes[idx]
                reranked_node = NodeWithScore(
                    node=original_node.node,
                    score=relevance_score,
                )
                reranked_nodes.append(reranked_node)

            return reranked_nodes

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original nodes (truncated) if reranking fails
            return nodes[:top_n]
