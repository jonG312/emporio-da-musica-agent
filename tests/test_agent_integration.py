
"""Integration tests for the Empório da Música support agent."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src.agent.orchestrator import answer
from src.agent.tools import build_tools
from src.database.csv_repository import CSVRepository
from src.rag.retriever import PolicyRetriever


# --------------------------------------------------
# TEST CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
POLICY_PDF = DATA_DIR / "store_policies_source.pdf"


# --------------------------------------------------
# SHARED FIXTURES
# --------------------------------------------------

@pytest.fixture(scope="module")
def repository():
    """Load the operational CSV repository once per test module."""
    return CSVRepository(DATA_DIR)


@pytest.fixture(scope="module")
def policy_retriever(tmp_path_factory):
    """Create a temporary policy index for integration tests."""
    if not POLICY_PDF.is_file():
        pytest.fail(f"Policy PDF not found: {POLICY_PDF}")

    chroma_dir = tmp_path_factory.mktemp("test_chroma")

    return PolicyRetriever(
        pdf_path=POLICY_PDF,
        persist_directory=chroma_dir,
    )


@pytest.fixture(scope="module")
def agent_tools(repository, policy_retriever):
    """Build the same tools used by the production agent."""
    tools = build_tools(
        repository=repository,
        policy_retriever=policy_retriever,
    )

    return {
        tool.name: tool
        for tool in tools
    }


@pytest.fixture(scope="module")
def valid_order(repository):
    """Find an existing order and its associated customer email."""
    for _, order in repository.orders.iterrows():
        customer = repository.customers[
            repository.customers["customer_id"]
            == order["customer_id"]
        ]

        if customer.empty:
            continue

        email = str(customer.iloc[0]["email"]).strip()

        if not email:
            continue

        return {
            "order_id": int(order["order_id"]),
            "customer_email": email,
        }

    pytest.fail(
        "No valid order with an associated customer email was found."
    )


# --------------------------------------------------
# TOOL REGISTRATION TESTS
# --------------------------------------------------

def test_all_expected_tools_are_registered(agent_tools):
    """Verify that all eight expected agent tools are available."""
    expected_tools = {
        "search_products",
        "get_product_price",
        "check_stock",
        "get_active_promotions",
        "get_order_status",
        "get_customer_orders",
        "search_store_policies",
        "get_store_information",
    }

    assert set(agent_tools.keys()) == expected_tools


# --------------------------------------------------
# CSV TOOL INTEGRATION TESTS
# --------------------------------------------------

def test_search_products_tool_returns_catalog_results(agent_tools):
    """Verify that the product search tool retrieves CSV data."""
    results = agent_tools["search_products"].invoke({
        "query": "",
        "category": "electric guitars",
        "in_stock_only": True,
        "limit": 5,
    })

    assert isinstance(results, list)
    assert len(results) > 0

    for product in results:
        assert product["product_id"] is not None
        assert product["name"]
        assert product["price"]
        assert product["stock_quantity"] > 0
        assert product["category"] == "Electric guitars"


def test_search_products_tool_distinguishes_acoustic_guitars(
    agent_tools,
):
    """Verify that acoustic guitar searches exclude electric guitars."""
    results = agent_tools["search_products"].invoke({
        "category": "acoustic guitars",
        "in_stock_only": True,
        "limit": 5,
    })

    assert isinstance(results, list)
    assert len(results) > 0

    for product in results:
        assert product["category"] == "Acoustic guitars"


def test_get_product_price_tool_returns_product_details(agent_tools):
    """Verify that the product details tool returns catalog information."""
    results = agent_tools["search_products"].invoke({
        "category": "electric guitars",
        "in_stock_only": True,
        "limit": 1,
    })

    assert results

    product_name = results[0]["name"]

    result = agent_tools["get_product_price"].invoke({
        "product_name": product_name,
    })

    assert result["found"] is True
    assert result["name"] == product_name
    assert result["price"]
    assert "stock_quantity" in result
    assert "specifications" in result


def test_check_stock_tool_returns_availability(agent_tools):
    """Verify that the stock tool returns a valid availability result."""
    products = agent_tools["search_products"].invoke({
        "category": "electric guitars",
        "in_stock_only": True,
        "limit": 1,
    })

    assert products

    result = agent_tools["check_stock"].invoke({
        "product_name": products[0]["name"],
    })

    assert result["found"] is True
    assert result["product"] == products[0]["name"]
    assert result["stock_quantity"] > 0
    assert result["available_to_sell"] is True


def test_get_active_promotions_tool_returns_valid_results(agent_tools):
    """Verify that the promotions tool returns active promotions."""
    results = agent_tools["get_active_promotions"].invoke({
        "product_name": "",
    })

    assert isinstance(results, list)

    for promotion in results:
        assert promotion["product"]
        assert promotion["discount_percent"] > 0
        assert "regular_price" in promotion
        assert "available_for_purchase" in promotion


# --------------------------------------------------
# ORDER VERIFICATION TESTS
# --------------------------------------------------

def test_get_order_status_tool_verifies_customer_email(
    agent_tools,
    valid_order,
):
    """Verify that order information requires a matching customer email."""
    result = agent_tools["get_order_status"].invoke({
        "order_id": valid_order["order_id"],
        "customer_email": valid_order["customer_email"],
    })

    assert result["found"] is True
    assert result["order_id"] == valid_order["order_id"]
    assert result["status"]


def test_get_customer_orders_tool_returns_verified_order(
    agent_tools,
    valid_order,
):
    """Verify that the customer order tool requires valid credentials."""
    results = agent_tools["get_customer_orders"].invoke({
        "order_id": valid_order["order_id"],
        "customer_email": valid_order["customer_email"],
    })

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["order_id"] == valid_order["order_id"]


# --------------------------------------------------
# PDF AND RAG INTEGRATION TESTS
# --------------------------------------------------

def test_policy_retriever_returns_relevant_documents(
    policy_retriever,
):
    """Verify that the PDF retriever returns relevant policy documents."""
    results = policy_retriever.search(
        "política de devolução e prazo para reembolso",
        k=5,
    )

    assert isinstance(results, list)
    assert len(results) > 0

    for document in results:
        assert document.page_content.strip()
        assert "page" in document.metadata


def test_search_store_policies_tool_returns_policy_content(
    agent_tools,
):
    """Verify that the policy tool exposes retrieved PDF content."""
    results = agent_tools["search_store_policies"].invoke({
        "question": "política de devolução e reembolso",
    })

    assert isinstance(results, list)
    assert len(results) > 0

    for result in results:
        assert result["content"].strip()
        assert "page" in result
        assert "source" in result


def test_get_store_information_tool_returns_company_information(
    agent_tools,
):
    """Verify that the company information tool searches the policy PDF."""
    results = agent_tools["get_store_information"].invoke({})

    assert isinstance(results, list)
    assert len(results) > 0

    for result in results:
        assert result["content"].strip()
        assert "page" in result
        assert "source" in result


# --------------------------------------------------
# REAL LANGGRAPH + OLLAMA TEST
# --------------------------------------------------

@pytest.mark.skipif(
    os.getenv("RUN_LLM_TESTS") != "1",
    reason="Set RUN_LLM_TESTS=1 to execute the real Ollama agent test.",
)
def test_real_agent_responds_to_product_question():
    """Verify that the real LangGraph agent produces a response."""
    response = answer(
        message="Show me electric guitars available for purchase.",
    )

    assert isinstance(response, str)
    assert response.strip()
