
"""Unit tests for the CSVRepository module."""

import json

import pandas as pd
import pytest

from src.database.csv_repository import CSVRepository


@pytest.fixture
def repository(tmp_path):
    """Create a temporary CSV repository for testing."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Products
    pd.DataFrame([
        {
            "product_id": 101,
            "name": "Violão Acústico",
            "category_id": 5,
            "price_brl": 1200.0,
            "stock_quantity": 4,
            "status": "active",
            "description": "Violão de madeira para iniciantes",
            "specs": json.dumps({"strings": 6}),
        },
        {
            "product_id": 102,
            "name": "Guitarra Elétrica",
            "category_id": 1,
            "price_brl": 2500.0,
            "stock_quantity": 0,
            "status": "active",
            "description": "Guitarra elétrica",
            "specs": json.dumps({"pickups": "HSS"}),
        },
    ]).to_csv(data_dir / "products.csv", index=False)

    # Categories
    pd.DataFrame([
        {
            "category_id": 1,
            "name": "Guitarras",
            "description": "Guitarras elétricas",
        },
        {
            "category_id": 5,
            "name": "Violões",
            "description": "Violões acústicos",
        },
    ]).to_csv(data_dir / "categories.csv", index=False)

    # Customers
    pd.DataFrame([
        {
            "customer_id": 1,
            "name": "Test Customer",
            "email": "customer@example.com",
        }
    ]).to_csv(data_dir / "customers.csv", index=False)

    # Orders
    pd.DataFrame([
        {
            "order_id": 5001,
            "customer_id": 1,
            "order_date": "2026-01-10",
            "status": "shipped",
            "total_brl": 1200.0,
            "payment_method": "credit_card",
            "tracking_code": "TRACK123",
            "estimated_delivery": "2026-01-15",
            "notes": "",
        }
    ]).to_csv(data_dir / "orders.csv", index=False)

    # Order items
    pd.DataFrame([
        {
            "order_item_id": 1,
            "order_id": 5001,
            "product_id": 101,
            "quantity": 1,
            "unit_price_brl": 1200.0,
        }
    ]).to_csv(data_dir / "order_items.csv", index=False)

    # Promotions
    pd.DataFrame([
        {
            "promotion_id": 1,
            "product_id": 101,
            "discount_percent": 10,
            "description": "Test promotion",
            "is_active": True,
        }
    ]).to_csv(data_dir / "promotions.csv", index=False)

    return CSVRepository(data_dir)


def test_search_products_filters_by_category(repository):
    """Products should be filtered by category."""

    results = repository.search_products(category="acoustic guitars")

    assert len(results) == 1
    assert results[0]["product_id"] == 101


def test_search_products_applies_maximum_price(repository):
    """Products above the maximum price should be excluded."""

    results = repository.search_products(max_price_brl=1000)

    assert results == []


def test_get_product_returns_matching_product(repository):
    """The repository should retrieve a product by its name."""

    result = repository.get_product("Violão Acústico")

    assert result is not None
    assert result["product_id"] == 101
    assert result["price"] == "BRL 1,200.00"


def test_get_order_status_requires_matching_email(repository):
    """Order information must only be returned to the matching customer."""

    denied = repository.get_order_status(
        5001,
        "wrong@example.com",
    )

    accepted = repository.get_order_status(
        5001,
        "customer@example.com",
    )

    assert denied["found"] is False
    assert accepted["found"] is True
    assert accepted["tracking_code"] == "TRACK123"


def test_get_promotions_returns_active_promotions(repository):
    """Only active promotions should be returned."""

    results = repository.get_promotions()

    assert len(results) == 1
    assert results[0]["product"] == "Violão Acústico"
    assert results[0]["discount_percent"] == 10.0


def test_missing_csv_file_raises_clear_error(tmp_path):
    """Missing required CSV files should raise an explicit error."""

    with pytest.raises(
        FileNotFoundError,
        match="Required data file not found",
    ):
        CSVRepository(tmp_path)
