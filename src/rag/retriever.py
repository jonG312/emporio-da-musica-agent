
"""Policy document retrieval using ChromaDB and Hugging Face embeddings."""

import os
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.document_loader import PolicyDocumentLoader


class PolicyRetriever:
    """Retrieve relevant sections from the store policy PDF."""

    def __init__(
        self,
        pdf_path=None,
        persist_directory=None,
        embedding_model=None,
    ):
        # Resolve project root independently of the execution directory.
        self.project_root = Path(__file__).resolve().parents[2]

        # Configure PDF path.
        configured_pdf = pdf_path or os.getenv("POLICY_PDF")

        if configured_pdf:
            self.pdf_path = Path(configured_pdf)

            if not self.pdf_path.is_absolute():
                self.pdf_path = self.project_root / self.pdf_path
        else:
            self.pdf_path = (
                self.project_root / "data" / "store_policies_source.pdf"
            )

        # Configure ChromaDB persistence path.
        configured_chroma = (
            persist_directory or os.getenv("CHROMA_DIR")
        )

        if configured_chroma:
            chroma_path = Path(configured_chroma)

            if not chroma_path.is_absolute():
                chroma_path = self.project_root / chroma_path
        else:
            chroma_path = self.project_root / "storage" / "chroma"

        self.persist_directory = str(chroma_path)

        # Configure multilingual embedding model.
        self.embedding_model_name = (
            embedding_model
            or os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
        )

        print(f"[RAG] PDF path: {self.pdf_path}")
        print(f"[RAG] PDF exists: {self.pdf_path.exists()}")
        print(f"[RAG] ChromaDB path: {self.persist_directory}")

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"Policy PDF not found: {self.pdf_path}"
            )

        # Generate a fingerprint to detect PDF changes.
        self.pdf_hash = self._calculate_pdf_hash()

        # Initialize embeddings on CPU.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = None

        # Initialize ChromaDB and synchronize documents.
        self._initialize_vector_store()

    def _calculate_pdf_hash(self):
        """Calculate PDF fingerprint to detect document changes."""

        sha256 = hashlib.sha256()

        with open(self.pdf_path, "rb") as pdf_file:
            for chunk in iter(lambda: pdf_file.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _initialize_vector_store(self):
        """Initialize ChromaDB and synchronize the policy index."""

        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        # Load and split PDF documents.
        loader = PolicyDocumentLoader(pdf_path=self.pdf_path)
        chunks = loader.load_and_split()

        if not chunks:
            raise ValueError(
                "No policy document chunks were generated."
            )

        print(f"[RAG] PDF loaded successfully: {len(chunks)} chunks")

        # Connect to ChromaDB.
        self.vector_store = Chroma(
            collection_name="store_policies",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        # Check whether the stored index belongs to the current PDF.
        collection = self.vector_store._collection
        existing_count = collection.count()

        # Collection metadata can be None when no metadata exists.
        collection_metadata = collection.metadata or {}
        stored_hash = collection_metadata.get("pdf_hash")

        if existing_count > 0 and stored_hash != self.pdf_hash:
            print("[RAG] PDF changed or index has no fingerprint.")
            print("[RAG] Rebuilding policy index...")

            existing_ids = collection.get()["ids"]

            if existing_ids:
                collection.delete(ids=existing_ids)

            existing_count = 0

        # Index documents when the collection is empty.
        if existing_count == 0:
            print("[RAG] Creating policy embeddings...")

            self.vector_store.add_documents(chunks)

            collection.modify(
                metadata={
                    "pdf_hash": self.pdf_hash,
                    "embedding_model": self.embedding_model_name,
                }
            )

            print("[RAG] Policy index created successfully.")

        else:
            print(
                f"[RAG] Existing policy index loaded: "
                f"{existing_count} chunks"
            )

    def search(self, query: str, k: int = 4):
        """Return the most relevant policy sections."""

        if not query or not query.strip():
            return []

        if self.vector_store is None:
            return []

        try:
            results = self.vector_store.similarity_search(
                query=query.strip(),
                k=k,
            )

            print(
                f"[RAG] Query: {query.strip()} | "
                f"Results: {len(results)}"
            )

            return results

        except Exception as error:
            raise RuntimeError(
                f"Policy retrieval failed: {error}"
            ) from error

    def get_relevant_context(self, query: str, k: int = 4) -> str:
        """Format retrieved policy sections for the LLM."""

        documents = self.search(query, k=k)

        if not documents:
            return "No relevant policy information was found."

        context_parts = []

        for index, document in enumerate(documents, start=1):
            page = document.metadata.get("page", "Unknown")
            source = document.metadata.get(
                "source",
                self.pdf_path.name,
            )

            context_parts.append(
                f"[Policy Reference {index}]\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{document.page_content}"
            )

        return "\n\n".join(context_parts)
