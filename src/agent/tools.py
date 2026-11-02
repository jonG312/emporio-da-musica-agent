"""Agent tools for deterministic catalog operations and policy retrieval."""

from langchain_core.tools import tool

from src.database.csv_repository import CSVRepository
from src.rag.retriever import PolicyRetriever


def build_tools(
    repository: CSVRepository,
    policy_retriever: PolicyRetriever,
):
    """Build the tools available to the customer-support agent."""

    @tool
    def search_products(
        query: str = "",
        category: str = "",
        max_price_brl: float | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> list[dict]:
        """Search the product catalog.

        Use this tool when customers ask for product recommendations,
        products matching a description, products in a category, or products
        within a budget.

        Args:
            query: Product name or keywords to search for.
            category: Product category, such as guitars, keyboards, drums,
                or acoustic guitars.
            max_price_brl: Optional maximum regular price in Brazilian reais.
            in_stock_only: If true, return only active products with stock.
            limit: Maximum number of results, from 1 to 20.

        Do not invent product names, prices, or stock quantities.
        Use the returned catalog data when answering.
        """
        return repository.search_products(
            query=query,
            category=category,
            max_price_brl=max_price_brl,
            in_stock_only=in_stock_only,
            limit=limit,
        )

    @tool
    def get_product_price(product_name: str) -> dict:
        """Retrieve the catalog details for one product.

        Use this tool when a customer asks for the price or details of a
        specific product. The result includes its listed price, stock,
        category, status, description, and specifications.

        Args:
            product_name: Product name or a distinctive part of its name.

        Never estimate or invent a product price.
        """
        result = repository.get_product(product_name)

        if result is None:
            return {
                "found": False,
                "message": "No matching product was found in the catalog.",
            }

        return {
            "found": True,
            **result,
        }

    @tool
    def check_stock(product_name: str) -> dict:
        """Check the current catalog stock and sales status of a product.

        Use this tool when a customer asks whether a specific product is
        available for purchase.

        Args:
            product_name: Product name or a distinctive part of its name.

        A product is available to sell only when its stock quantity is
        greater than zero and its catalog status is active.
        """
        result = repository.get_product(product_name)

        if result is None:
            return {
                "found": False,
                "message": "No matching product was found in the catalog.",
            }

        status = str(result["status"]).strip().casefold()
        stock_quantity = result["stock_quantity"]

        return {
            "found": True,
            "product": result["name"],
            "stock_quantity": stock_quantity,
            "status": result["status"],
            "available_to_sell": (
                stock_quantity > 0 and status == "active"
            ),
        }

    @tool
    def get_active_promotions(product_name: str = "") -> list[dict]:
        """Retrieve active promotions from the promotions CSV.

        Use this tool when customers ask about current discounts, special
        offers, or promotions for a particular product.

        Args:
            product_name: Optional product name to filter promotions.
                Leave empty to retrieve all active promotions.

        Only report discounts returned by this tool. Check the
        available_for_purchase field before claiming a product can be bought.
        """
        return repository.get_promotions(product_name=product_name)

    @tool
    def get_order_status(order_id: int, customer_email: str) -> dict:
        """Retrieve an order after verifying its ID and customer email.

        Use this tool when a customer asks about the status, tracking,
        payment, or estimated delivery of a specific order.

        Args:
            order_id: The order ID provided by the customer.
            customer_email: The email address associated with the order.

        Always obtain both the order ID and email before calling this tool.
        Do not reveal order information if the verification fails.
        """
        return repository.get_order_status(
            order_id=order_id,
            customer_email=customer_email,
        )

    @tool
    def get_customer_orders(
        customer_email: str,
        order_id: int,
    ) -> list[dict]:
        """Retrieve basic information about one verified customer order.

        Both the order ID and the customer's email address are required.
        This tool does not support listing all orders using an email address
        alone.

        Args:
            customer_email: The email address associated with the order.
            order_id: The specific order ID to verify.

        Use this tool only when both pieces of information are available.
        """
        return repository.get_customer_orders(
            customer_email=customer_email,
            order_id=order_id,
        )

    @tool
    def search_store_policies(question: str) -> list[dict]:
        """Search the internal store policy PDF.

        Use this tool for questions about store policies and procedures,
        including payments, returns, exchanges, warranties, shipping,
        promotions, privacy, and customer service.

        Also use it for company information, contact details, address,
        and opening hours when relevant.

        Args:
            question: The customer's question or the specific policy topic.

        Base policy answers on the retrieved document content.
        If the retrieved results do not contain the answer, say that the
        information could not be confirmed from the policy manual.
        """
        documents = policy_retriever.search(question, k=5)

        return [
            {
                "page": document.metadata.get("page", "Unknown"),
                "source": document.metadata.get(
                    "source", "Store policy manual"
                ),
                "content": document.page_content,
            }
            for document in documents
        ]

    @tool
    def get_store_information() -> list[dict]:
        """Retrieve official company and contact information from the PDF.

        Use this tool when customers ask for the store's address, company
        name, legal name, CNPJ, telephone, WhatsApp, email, or opening hours.

        Do not answer these questions from memory or assumptions.
        Use only information supported by the retrieved PDF content.
        """
        queries = [
            "Endereço completo da loja Empório da Música",
            "Razão social nome fantasia CNPJ e informações da empresa",
            "Telefone email e canais de contato da loja",
            "Horário de funcionamento da loja",
            "Número de WhatsApp para atendimento ao cliente",
        ]

        results = []
        seen_content = set()

        for query in queries:
            documents = policy_retriever.search(query, k=3)

            for document in documents:
                content = document.page_content.strip()

                if not content or content in seen_content:
                    continue

                seen_content.add(content)

                results.append({
                    "page": document.metadata.get("page", "Unknown"),
                    "source": document.metadata.get(
                        "source", "Store policy manual"
                    ),
                    "content": content,
                })

        return results

    return [
        search_products,
        get_product_price,
        check_stock,
        get_active_promotions,
        get_order_status,
        get_customer_orders,
        search_store_policies,
        get_store_information,
    ]