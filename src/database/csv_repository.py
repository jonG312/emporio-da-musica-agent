
"""Read-only repository for the supplied operational CSV files."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


class CSVRepository:
    """Load operational CSV files and expose deterministic business queries."""

    REQUIRED_COLUMNS = {
        "products.csv": {
            "product_id",
            "name",
            "category_id",
            "price_brl",
            "description",
            "stock_quantity",
            "status",
            "specs",
        },
        "categories.csv": {"category_id", "name", "description"},
        "customers.csv": {"customer_id", "email"},
        "orders.csv": {
            "order_id",
            "customer_id",
            "order_date",
            "status",
            "total_brl",
            "payment_method",
            "tracking_code",
            "estimated_delivery",
            "notes",
        },
        "order_items.csv": {"order_id", "quantity", "product_id"},
        "promotions.csv": {
            "promotion_id",
            "product_id",
            "discount_percent",
            "description",
            "is_active",
        },
    }

    CATEGORY_NAMES = {
        1: "Electric guitars",
        2: "Bass guitars",
        3: "Drums and percussion",
        4: "Keyboards and pianos",
        5: "Acoustic guitars",
        6: "Woodwind instruments",
        7: "Brass instruments",
        8: "Orchestral strings",
        9: "Ukuleles",
    }

    CATEGORY_SYNONYMS = {
        # Electric guitars
        "electric guitars": [1],
        "electric guitar": [1],
        "electric": [1],
        "guitarras electricas": [1],
        "guitarras elétricas": [1],
        "guitarra eletrica": [1],
        "guitarra elétrica": [1],

        # Acoustic guitars
        "acoustic guitars": [5],
        "acoustic guitar": [5],
        "acoustic": [5],
        "violao": [5],
        "violão": [5],
        "violoes": [5],
        "violões": [5],

        # General guitars: electric and acoustic
        "guitars": [1, 5],
        "guitar": [1, 5],
        "guitarras": [1, 5],
        "guitarra": [1, 5],

        # Bass guitars
        "bass guitars": [2],
        "bass guitar": [2],
        "bass": [2],
        "baixos": [2],
        "baixo": [2],

        # Drums and percussion
        "drums and percussion": [3],
        "drums": [3],
        "drum": [3],
        "percussion": [3],
        "bateria": [3],
        "baterias": [3],
        "percussao": [3],
        "percussão": [3],

        # Keyboards and pianos
        "keyboards and pianos": [4],
        "keyboards": [4],
        "keyboard": [4],
        "pianos": [4],
        "piano": [4],
        "teclados": [4],
        "teclado": [4],

        # Woodwind instruments
        "woodwind instruments": [6],
        "woodwind": [6],
        "madeiras": [6],
        "instrumentos de sopro (madeiras)": [6],

        # Brass instruments
        "brass instruments": [7],
        "brass": [7],
        "metais": [7],
        "instrumentos de sopro (metais)": [7],

        # Orchestral strings
        "orchestral strings": [8],
        "violin": [8],
        "cello": [8],
        "violino": [8],
        "violoncelo": [8],
        "cordas orquestrais": [8],

        # Ukuleles
        "ukuleles": [9],
        "ukulele": [9],
    }

    SEARCH_TRANSLATIONS = {
        "acoustic guitar": "violão",
        "acoustic": "violão",
        "electric guitar": "guitarra",
        "electric guitars": "guitarra",
        "guitar": "guitarra",
        "bass": "baixo",
        "drum": "bateria",
        "keyboard": "teclado",
        "piano": "piano",
        "violin": "violino",
        "trumpet": "trompete",
        "flute": "flauta",
    }

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)

        self.products = self._load("products.csv")
        self.categories = self._load("categories.csv")
        self.customers = self._load("customers.csv")
        self.orders = self._load("orders.csv")
        self.order_items = self._load("order_items.csv")
        self.promotions = self._load("promotions.csv")

        self._normalize_types()

    def _load(self, filename: str) -> pd.DataFrame:
        """Load a CSV file and validate its required columns."""
        path = self.data_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Required data file not found: {path}"
            )

        try:
            frame = pd.read_csv(path)
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            raise ValueError(
                f"Could not read CSV file '{path}': {error}"
            ) from error

        required = self.REQUIRED_COLUMNS[filename]
        missing = required.difference(frame.columns)

        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(
                f"CSV file '{filename}' is missing required columns: "
                f"{missing_columns}"
            )

        return frame

    def _normalize_types(self) -> None:
        """Normalize numeric identifiers, prices, stock and boolean values."""
        numeric_columns = {
            "products": [
                "product_id",
                "category_id",
                "price_brl",
                "stock_quantity",
            ],
            "categories": ["category_id"],
            "customers": ["customer_id"],
            "orders": ["order_id", "customer_id", "total_brl"],
            "order_items": ["order_id", "quantity", "product_id"],
            "promotions": [
                "promotion_id",
                "product_id",
                "discount_percent",
            ],
        }

        for frame_name, columns in numeric_columns.items():
            frame = getattr(self, frame_name)

            for column in columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

        self.promotions["is_active"] = self.promotions["is_active"].map(
            self._to_bool
        )

        # Normalize customer email addresses for matching.
        self.customers["email"] = (
            self.customers["email"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    @staticmethod
    def _to_bool(value: Any) -> bool:
        """Convert common CSV boolean representations to bool."""
        if pd.isna(value):
            return False

        return str(value).strip().casefold() in {
            "1",
            "true",
            "yes",
            "y",
            "sim",
            "s",
        }

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Convert a value to int without raising on missing or invalid data."""
        if pd.isna(value):
            return default

        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return default

    @staticmethod
    def _money(value: Any) -> str:
        """Format a numeric value as Brazilian reais."""
        if pd.isna(value):
            return "Not available"

        try:
            return f"BRL {float(value):,.2f}"
        except (TypeError, ValueError):
            return "Not available"

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """Normalize text for accent-insensitive and case-insensitive matching."""
        if value is None or pd.isna(value):
            return ""

        text = str(value).strip().casefold()
        normalized = unicodedata.normalize("NFKD", text)

        return "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )

    def _category_name(self, category_id: Any) -> str:
        """Return the English display name for a category ID."""
        category_number = self._safe_int(
            category_id,
            default=-1,
        )

        return self.CATEGORY_NAMES.get(
            category_number,
            "Uncategorized",
        )

    @staticmethod
    def _parse_specs(raw: Any) -> dict[str, Any]:
        """Parse product specifications stored as JSON in the CSV."""
        if pd.isna(raw):
            return {}

        try:
            value = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}

        return value if isinstance(value, dict) else {}

    def _category_ids(self, category: str) -> list[int]:
        """Resolve English or Portuguese category names to category IDs."""
        needle = self._normalize_text(category)

        if not needle:
            return []

        # Normalize all synonyms before matching.
        normalized_synonyms = [
            (self._normalize_text(label), category_ids)
            for label, category_ids in self.CATEGORY_SYNONYMS.items()
        ]

        matching_synonyms = [
            (label, category_ids)
            for label, category_ids in normalized_synonyms
            if label and label in needle
        ]

        # Prefer the most specific synonym.
        # "electric guitars" -> [1]
        # "acoustic guitars" -> [5]
        # "guitars" -> [1, 5]
        if matching_synonyms:
            matching_synonyms.sort(
                key=lambda item: len(item[0]),
                reverse=True,
            )

            return matching_synonyms[0][1]

        # Normalize category names from the original CSV.
        normalized_names = self.categories["name"].apply(
            self._normalize_text
        )

        # Check for exact category-name matches.
        exact_matches = self.categories[
            normalized_names == needle
        ]

        if not exact_matches.empty:
            return [
                self._safe_int(value, default=-1)
                for value in exact_matches["category_id"].tolist()
            ]

        # Normalize category descriptions.
        normalized_descriptions = self.categories["description"].apply(
            self._normalize_text
        )

        # Search category names and descriptions.
        matches = self.categories[
            normalized_names.str.contains(
                needle,
                na=False,
                regex=False,
            )
            | normalized_descriptions.str.contains(
                needle,
                na=False,
                regex=False,
            )
        ]

        return [
            self._safe_int(value, default=-1)
            for value in matches["category_id"].tolist()
        ]

    def search_products(
        self,
        query: str = "",
        category: str = "",
        max_price_brl: float | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Search products using text, category, price and availability filters."""
        frame = self.products.copy()

        query = str(query or "").strip()
        category = str(category or "").strip()

        # Apply text search.
        if query:
            needle = self._normalize_text(query)
            search_terms = {needle}

            for english, portuguese in self.SEARCH_TRANSLATIONS.items():
                normalized_english = self._normalize_text(english)

                if normalized_english in needle:
                    search_terms.add(
                        self._normalize_text(portuguese)
                    )

            searchable = (
                frame["name"].fillna("").astype(str).apply(
                    self._normalize_text
                )
                + " "
                + frame["description"].fillna("").astype(str).apply(
                    self._normalize_text
                )
                + " "
                + frame["specs"].fillna("").astype(str).apply(
                    self._normalize_text
                )
            )

            mask = pd.Series(False, index=frame.index)

            for term in search_terms:
                if term:
                    mask |= searchable.str.contains(
                        term,
                        na=False,
                        regex=False,
                    )

            frame = frame[mask]

        # Apply category filter.
        if category:
            category_ids = self._category_ids(category)

            if not category_ids:
                return []

            frame = frame[
                frame["category_id"].isin(category_ids)
            ]

        # Apply maximum price filter.
        if max_price_brl is not None:
            try:
                maximum_price = float(max_price_brl)
            except (TypeError, ValueError):
                raise ValueError(
                    "max_price_brl must be a valid number."
                )

            if maximum_price < 0:
                raise ValueError(
                    "max_price_brl cannot be negative."
                )

            frame = frame[
                frame["price_brl"] <= maximum_price
            ]

        # Apply stock and product status filters.
        if in_stock_only:
            active_status = (
                frame["status"]
                .fillna("")
                .astype(str)
                .str.casefold()
            )

            frame = frame[
                (frame["stock_quantity"] > 0)
                & (active_status == "active")
            ]

        # Validate and constrain result limit.
        try:
            result_limit = max(
                1,
                min(int(limit), 20),
            )
        except (TypeError, ValueError):
            result_limit = 8

        # Sort by price and product name.
        frame = frame.sort_values(
            ["price_brl", "name"],
            ascending=[True, True],
            na_position="last",
        ).head(result_limit)

        results = []

        for _, row in frame.iterrows():
            results.append({
                "product_id": self._safe_int(row["product_id"]),
                "name": str(row["name"]),
                "category": self._category_name(row["category_id"]),
                "price": self._money(row["price_brl"]),
                "stock_quantity": self._safe_int(
                    row["stock_quantity"]
                ),
                "status": str(row["status"]),
                "description": str(row["description"]),
                "specifications": self._parse_specs(row["specs"]),
            })

        return results

    def get_product(
        self,
        product_name: str,
    ) -> dict[str, Any] | None:
        """Retrieve a product by exact or partial name."""
        product_name = str(product_name or "").strip()

        if not product_name:
            return None

        normalized_names = self.products["name"].fillna("").astype(str).apply(
            self._normalize_text
        )

        needle = self._normalize_text(product_name)

        matches = self.products[
            normalized_names == needle
        ]

        if matches.empty:
            matches = self.products[
                normalized_names.str.contains(
                    needle,
                    na=False,
                    regex=False,
                )
            ]

        if matches.empty:
            return None

        row = matches.iloc[0]

        return {
            "product_id": self._safe_int(row["product_id"]),
            "name": str(row["name"]),
            "price": self._money(row["price_brl"]),
            "stock_quantity": self._safe_int(
                row["stock_quantity"]
            ),
            "status": str(row["status"]),
            "category": self._category_name(row["category_id"]),
            "description": str(row["description"]),
            "specifications": self._parse_specs(row["specs"]),
        }

    def get_promotions(
        self,
        product_name: str = "",
    ) -> list[dict[str, Any]]:
        """Return active promotions, including product availability information."""
        active = self.promotions[
            self.promotions["is_active"]
        ].merge(
            self.products[
                [
                    "product_id",
                    "name",
                    "price_brl",
                    "stock_quantity",
                    "status",
                ]
            ],
            on="product_id",
            how="inner",
        )

        product_name = str(product_name or "").strip()

        if product_name:
            names = active["name"].fillna("").astype(str).apply(
                self._normalize_text
            )

            active = active[
                names.str.contains(
                    self._normalize_text(product_name),
                    na=False,
                    regex=False,
                )
            ]

        results = []

        for row in active.itertuples(index=False):
            stock_quantity = self._safe_int(
                row.stock_quantity
            )

            product_status = str(row.status)

            results.append({
                "product": str(row.name),
                "discount_percent": float(row.discount_percent),
                "promotion_description": str(row.description),
                "regular_price": self._money(row.price_brl),
                "stock_quantity": stock_quantity,
                "product_status": product_status,
                "available_for_purchase": (
                    stock_quantity > 0
                    and product_status.casefold() == "active"
                ),
            })

        return results

    def get_order_status(
        self,
        order_id: int,
        customer_email: str,
    ) -> dict[str, Any]:
        """Return order details only after matching its ID and customer email."""
        try:
            requested_order_id = int(order_id)
        except (TypeError, ValueError):
            return {
                "found": False,
                "message": "Please provide a valid order ID.",
            }

        customer_email = (
            str(customer_email or "")
            .strip()
            .casefold()
        )

        if not customer_email:
            return {
                "found": False,
                "message": (
                    "A customer email address is required "
                    "to verify the order."
                ),
            }

        order = self.orders[
            self.orders["order_id"] == requested_order_id
        ]

        if order.empty:
            return {
                "found": False,
                "message": "No order was found with that order ID.",
            }

        row = order.iloc[0]

        customer = self.customers[
            self.customers["customer_id"] == row["customer_id"]
        ]

        if (
            customer.empty
            or customer.iloc[0]["email"].strip().casefold()
            != customer_email
        ):
            return {
                "found": False,
                "message": (
                    "The order could not be verified with that email "
                    "address. Please check the details or contact "
                    "store support."
                ),
            }

        return {
            "found": True,
            "order_id": self._safe_int(row["order_id"]),
            "status": str(row["status"]),
            "order_date": str(row["order_date"]),
            "total": self._money(row["total_brl"]),
            "payment_method": str(row["payment_method"]),
            "tracking_code": (
                None
                if pd.isna(row["tracking_code"])
                else str(row["tracking_code"])
            ),
            "estimated_delivery": (
                None
                if pd.isna(row["estimated_delivery"])
                else str(row["estimated_delivery"])
            ),
            "notes": (
                None
                if pd.isna(row["notes"])
                else str(row["notes"])
            ),
        }

    def get_customer_orders(
        self,
        customer_email: str,
        order_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve a verified order.

        An order ID is required in addition to the customer's email.
        Email-only order-history lookups are intentionally not supported.
        """
        if order_id is None:
            raise ValueError(
                "An order ID is required. Email-only order-history "
                "lookups are not supported."
            )

        result = self.get_order_status(
            order_id,
            customer_email,
        )

        if not result["found"]:
            return []

        return [
            {
                "order_id": result["order_id"],
                "status": result["status"],
                "order_date": result["order_date"],
            }
        ]
