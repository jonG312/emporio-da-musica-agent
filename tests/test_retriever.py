import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Starting retriever test...")

from src.rag.retriever import PolicyRetriever

print("Import successful!")

retriever = PolicyRetriever()

print("Retriever initialized successfully!")

questions = [
    "Qual é o número de WhatsApp da loja?",
    "Qual é o horário de funcionamento?",
    "Quais são as formas de pagamento?",
    "Qual é a política de devolução?",
    "A loja oferece aulas gratuitas de guitarra?"
]

for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    results = retriever.search(question, k=5)

    if not results:
        print("NO RESULTS FOUND")
        continue

    for i, document in enumerate(results, start=1):
        print(f"\nRESULT {i}")
        print("PAGE:", document.metadata.get("page"))
        print("CONTENT:", document.page_content)