from qdrant_client import QdrantClient
from backend.core.config import settings
import logging

def get_qdrant_client() -> QdrantClient:
    try:
        client = QdrantClient(url=settings.QDRANT_URL)
        return client
    except Exception as e:
        logging.error(f"Failed to connect to Qdrant at {settings.QDRANT_URL}: {e}")
        raise
