
"""PDF document loading and text extraction utilities.

This module loads the store's internal policy PDF, extracts its text,
and splits the content into smaller chunks suitable for RAG indexing.
"""

from pathlib import Path
from typing import List

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PolicyDocumentLoader:
    """Load and split the store policy PDF into searchable documents."""

    def __init__(
        self,
        pdf_path: str | Path = "data/store_policies_source.pdf",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialize the policy document loader.

        Args:
            pdf_path: Path to the source PDF file.
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.pdf_path = Path(pdf_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def load_pdf(self) -> List[Document]:
        """Extract text from every page of the policy PDF.

        Returns:
            A list of LangChain Document objects, one per non-empty page.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the PDF contains no extractable text.
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"Policy PDF not found: {self.pdf_path}"
            )

        documents: List[Document] = []

        try:
            with pymupdf.open(self.pdf_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    page_text = page.get_text("text").strip()

                    if not page_text:
                        continue

                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": str(self.pdf_path),
                                "page": page_number,
                                "document_type": "store_policy",
                            },
                        )
                    )

        except pymupdf.FileDataError as exc:
            raise ValueError(
                f"Unable to read PDF file: {self.pdf_path}"
            ) from exc

        if not documents:
            raise ValueError(
                "The policy PDF contains no extractable text. "
                "It may be empty or scanned as images."
            )

        return documents

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Split loaded documents into smaller overlapping chunks.

        Page metadata is preserved and each chunk receives a chunk index.

        Args:
            documents: Documents returned by load_pdf().

        Returns:
            A list of smaller LangChain Document objects.
        """
        if not documents:
            return []

        chunks = self.text_splitter.split_documents(documents)

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk"] = index

        return chunks

    def load_and_split(self) -> List[Document]:
        """Load the PDF and return its text as searchable chunks."""
        documents = self.load_pdf()
        return self.split_documents(documents)
