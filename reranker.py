import json
from typing import List
import boto3
from llama_index.core.schema import NodeWithScore
from logger import log_info, log_exception


class BedrockCohereRerank:
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

        log_info(
            "Initialized BedrockCohereRerank",
            model_id=self.model_id,
            region=self.region_name,
            default_top_n=self.top_n,
        )

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_n: int = None,
    ) -> List[NodeWithScore]:
        if not nodes:
            return nodes

        top_n = top_n or self.top_n
        documents = [node.get_content() for node in nodes]

        if not documents:
            return nodes

        try:
            request_body = {
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
            }

            log_info(
                "Reranking documents with Bedrock Cohere",
                model_id=self.model_id,
                document_count=len(documents),
                requested_top_n=top_n,
                effective_top_n=request_body["top_n"],
                query_preview=query[:100],
            )

            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            results = response_body.get("results", [])

            log_info(
                "Reranking completed",
                model_id=self.model_id,
                result_count=len(results),
            )

            reranked_nodes = []
            for result in results:
                idx = result["index"]
                relevance_score = result["relevance_score"]

                original_node = nodes[idx]
                reranked_node = NodeWithScore(
                    node=original_node.node,
                    score=relevance_score,
                )
                reranked_nodes.append(reranked_node)

            return reranked_nodes

        except Exception:
            log_exception(
                "Reranking failed",
                model_id=self.model_id,
                document_count=len(documents),
                top_n=top_n,
                query_preview=query[:100],
            )
            return nodes[:top_n]
