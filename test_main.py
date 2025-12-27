from fastapi.testclient import TestClient
import pytest
from main import app


# FIXTURES - Test Setup Code (Runs before tests)
@pytest.fixture
def client(mock_vector_store_manager):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_api_token():
    return "test-token-123"


@pytest.fixture
def mock_nodes(mocker):
    nodes = []
    for i in range(5):
        node = mocker.Mock()
        node.node_id = f"a0282698-b92c-4f61-b93f-b5f634a1c7b{i}"
        node.get_content.return_value = f"Document {i} about wildlife and ecosystems."
        node.score = 0.9 - (i * 0.1)

        node.metadata = {
            "file_name": f"document-{i}.pdf",
            "doc_id": f"doc-id-{i}",
            "source": f"s3://bucket/document-{i}.pdf",
            "page": i + 2,
            "content_type": "text",
            "chunk_index": i + 7,
            "total_chunks": 16,
        }

        nodes.append(node)

    return nodes


@pytest.fixture
def mock_vector_store_manager(mocker, mock_nodes):
    mock_vsm = mocker.patch("main.vector_store_manager")

    mock_retriever = mocker.Mock()
    mock_retriever.retrieve.return_value = mock_nodes

    mock_vsm.get_retriever.return_value = mock_retriever

    def fake_rerank(nodes, query, top_n):
        return nodes[:top_n]

    mock_vsm.rerank_nodes.side_effect = fake_rerank

    return mock_vsm


@pytest.fixture(autouse=True)
def mock_settings(mocker, valid_api_token):
    mock_settings = mocker.patch("services.security.settings")
    mock_settings.api_token = valid_api_token
    return mock_settings


# AUTHENTICATION TESTS (3 tests)
def test_retrieve_requires_authentication(client, mock_vector_store_manager):
    response = client.post("/retrieve", json={"query": "lion behavior"})
    assert response.status_code == 401
    assert "Invalid or missing API token" in response.json()["detail"]


def test_retrieve_invalid_token(client, mock_vector_store_manager):
    headers = {"x-api-token": "wrong-token-xyz"}

    response = client.post(
        "/retrieve", json={"query": "lion behavior"}, headers=headers
    )
    assert response.status_code == 401
    assert "Invalid or missing API token" in response.json()["detail"]


def test_retrieve_valid_token(client, mock_vector_store_manager, valid_api_token):
    headers = {"x-api-token": valid_api_token}

    response = client.post(
        "/retrieve", json={"query": "lion behavior"}, headers=headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert len(data["nodes"]) == 5


# SEARCH MODE TESTS (3 tests)
def test_retrieve_vector_mode(client, mock_vector_store_manager, valid_api_token):
    headers = {"x-api-token": valid_api_token}

    response = client.post(
        "/retrieve", json={"query": "lion behavior", "mode": "vector"}, headers=headers
    )
    assert response.status_code == 200
    mock_vector_store_manager.get_retriever.assert_called_once()
    call_kwargs = mock_vector_store_manager.get_retriever.call_args[1]
    assert call_kwargs["mode"] == "default"


def test_retrieve_keyword_mode(client, mock_vector_store_manager, valid_api_token):
    headers = {"x-api-token": valid_api_token}

    response = client.post(
        "/retrieve", json={"query": "lion behavior", "mode": "keyword"}, headers=headers
    )
    assert response.status_code == 200
    mock_vector_store_manager.get_retriever.assert_called_once()
    call_kwargs = mock_vector_store_manager.get_retriever.call_args[1]
    assert call_kwargs["mode"] == "sparse"


def test_retrieve_hybrid_mode(client, mock_vector_store_manager, valid_api_token):
    headers = {"x-api-token": valid_api_token}

    response = client.post(
        "/retrieve", json={"query": "lion behavior", "mode": "hybrid"}, headers=headers
    )
    assert response.status_code == 200
    mock_vector_store_manager.get_retriever.assert_called_once()
    call_kwargs = mock_vector_store_manager.get_retriever.call_args[1]
    assert call_kwargs["mode"] == "hybrid"


# TOP-K PARAMETER TESTS (2 tests)
def test_retrieve_custom_top_k(
    client, mock_vector_store_manager, valid_api_token, mocker
):
    custom_nodes = []
    for i in range(3):
        node = mocker.Mock()
        node.node_id = f"node-{i}"
        node.get_content.return_value = f"Content {i}"
        node.score = 0.9
        node.metadata = {"source": f"doc{i}.pdf"}
        custom_nodes.append(node)

    mock_retriever = mocker.Mock()
    mock_retriever.retrieve.return_value = custom_nodes
    mock_vector_store_manager.get_retriever.return_value = mock_retriever

    headers = {"x-api-token": valid_api_token}

    response = client.post(
        "/retrieve", json={"query": "test", "top_k": 3}, headers=headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 3
    assert len(data["nodes"]) == 3


def test_retrieve_default_top_k(client, mock_vector_store_manager, valid_api_token):
    headers = {"x-api-token": valid_api_token}
    response = client.post("/retrieve", json={"query": "test"}, headers=headers)
    assert response.status_code == 200
    mock_vector_store_manager.get_retriever.assert_called_once()
    call_kwargs = mock_vector_store_manager.get_retriever.call_args[1]
    assert call_kwargs["similarity_top_k"] == 5


# RERANKING TESTS (3 tests)
def test_retrieve_without_reranking(client, mock_vector_store_manager, valid_api_token):
    headers = {"x-api-token": valid_api_token}

    response = client.post(
        "/retrieve", json={"query": "test", "rerank": False}, headers=headers
    )
    assert response.status_code == 200
    mock_vector_store_manager.rerank_nodes.assert_not_called()


def test_retrieve_with_reranking(client, mock_vector_store_manager, valid_api_token):
    headers = {"x-api-token": valid_api_token}

    response = client.post(
        "/retrieve", json={"query": "test", "top_k": 5, "rerank": True}, headers=headers
    )
    assert response.status_code == 200
    mock_vector_store_manager.rerank_nodes.assert_called_once()


def test_reranking_multiplies_topk(client, mock_vector_store_manager, valid_api_token):
    headers = {"x-api-token": valid_api_token}

    response = client.post(
        "/retrieve",
        json={"query": "test", "top_k": 10, "rerank": True},
        headers=headers,
    )
    assert response.status_code == 200
    mock_vector_store_manager.get_retriever.assert_called_once()
    call_kwargs = mock_vector_store_manager.get_retriever.call_args[1]
    assert call_kwargs["similarity_top_k"] == 30


# METADATA FILTERING TESTS (3 tests)
def test_retrieve_single_filter(client, mock_vector_store_manager, valid_api_token):
    headers = {"x-api-token": valid_api_token}

    response = client.post(
        "/retrieve",
        json={
            "query": "test",
            "filters": {
                "filters": [{"key": "page", "value": 1, "operator": "=="}],
                "condition": "and",
            },
        },
        headers=headers,
    )
    assert response.status_code == 200
    mock_vector_store_manager.get_retriever.assert_called_once()
    call_kwargs = mock_vector_store_manager.get_retriever.call_args[1]
    assert call_kwargs["filters"] is not None


def test_retrieve_multiple_and(client, mock_vector_store_manager, valid_api_token):
    headers = {"x-api-token": valid_api_token}

    response = client.post(
        "/retrieve",
        json={
            "query": "test",
            "filters": {
                "filters": [
                    {"key": "page", "value": 1, "operator": "=="},
                    {"key": "content_type", "value": "text", "operator": "=="},
                ],
                "condition": "and",
            },
        },
        headers=headers,
    )
    assert response.status_code == 200


def test_retrieve_multiple_or(client, mock_vector_store_manager, valid_api_token):
    headers = {"x-api-token": valid_api_token}

    response = client.post(
        "/retrieve",
        json={
            "query": "test",
            "filters": {
                "filters": [
                    {"key": "page", "value": 1, "operator": "=="},
                    {"key": "page", "value": 2, "operator": "=="},
                ],
                "condition": "or",
            },
        },
        headers=headers,
    )
    assert response.status_code == 200


# ERROR HANDLING TESTS (3 tests)
def test_retrieve_value_error(
    client, mock_vector_store_manager, valid_api_token, mocker
):
    mock_retriever = mocker.Mock()
    mock_retriever.retrieve.side_effect = ValueError("Invalid parameter")
    mock_vector_store_manager.get_retriever.return_value = mock_retriever

    headers = {"x-api-token": valid_api_token}
    response = client.post("/retrieve", json={"query": "test"}, headers=headers)
    assert response.status_code == 400


def test_retrieve_runtime_error(
    client, mock_vector_store_manager, valid_api_token, mocker
):
    mock_retriever = mocker.Mock()
    mock_retriever.retrieve.side_effect = RuntimeError("Database connection failed")
    mock_vector_store_manager.get_retriever.return_value = mock_retriever

    headers = {"x-api-token": valid_api_token}
    response = client.post("/retrieve", json={"query": "test"}, headers=headers)
    assert response.status_code == 503


def test_retrieve_generic_error(
    client, mock_vector_store_manager, valid_api_token, mocker
):
    mock_retriever = mocker.Mock()
    mock_retriever.retrieve.side_effect = Exception("Unexpected error")
    mock_vector_store_manager.get_retriever.return_value = mock_retriever

    headers = {"x-api-token": valid_api_token}
    response = client.post("/retrieve", json={"query": "test"}, headers=headers)
    assert response.status_code == 500


# RESPONSE STRUCTURE TEST (1 test)
def test_retrieve_response_format(client, mock_vector_store_manager, valid_api_token):
    headers = {"x-api-token": valid_api_token}
    response = client.post(
        "/retrieve",
        json={
            "query": "lion behavior",
            "top_k": 5,
            "mode": "hybrid",
            "rerank": False,
            "filters": {
                "filters": [{"key": "page", "value": 1, "operator": "=="}],
                "condition": "and",
            },
        },
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "query" in data
    assert "total_results" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["query"], str)
    assert isinstance(data["total_results"], int)
    assert data["query"] == "lion behavior"
    assert data["total_results"] == len(data["nodes"])
    assert len(data["nodes"]) > 0
    node = data["nodes"][0]
    assert "node_id" in node
    assert "text" in node
    assert "score" in node
    assert "metadata" in node
    assert isinstance(node["node_id"], str)
    assert isinstance(node["text"], str)
    assert isinstance(node["score"], (int, float))
    assert isinstance(node["metadata"], dict)
    assert "source" in node["metadata"]
    assert isinstance(node["metadata"]["source"], str)
