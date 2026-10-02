"""
RAG: Build a chroma DB vector store from the product catalog.

The Vector store is created once at import time and reused by the 
search_product_catalog tool.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import embeddings, get_logger
from src.data import PRODUCT_CATALOG

logger = get_logger("rag")

def build_documents() -> list[Document]:
    """Convert every catalog entry into a langchain Document"""
    docs: list[Document] = []
    for p in PRODUCT_CATALOG:
        content = (
            f"Product: {p['name']}\n"
            f"Brand: {p['brand']}\n"
            f"Category: {p['category']}\n"
            f"Price: ₹{p['price']}\n"
            f"Rating: {p['rating']}/5\n"
            f"Features: {', '.join(p['features'])}\n"
            f"Description: {p['description']}\n"
            f"Colors: {', '.join(p['colors'])}\n"
            f"In Stock: {'Yes' if p['in_stock'] else 'No'}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata = {
                    "id":p["id"],
                    "name":p["name"],
                    "brand":p["brand"],
                    "category":p["category"],
                    "price": p["price"],
                    "rating":p["rating"]
                }
            )
        )
        return docs

def build_vectorstore() -> Chroma:
    """create an in-memory chroma collection from the product catalog"""
    docs = build_documents()
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="axiom-cart-products"
    )
    logger.info("Vector store ready (%d products indexed)", len(docs))
    return store

product_vectorstore = build_vectorstore()
