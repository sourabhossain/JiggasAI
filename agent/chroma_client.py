import chromadb
from django.conf import settings

_client = None
_collection = None


def get_client() -> chromadb.HttpClient:
    global _client
    if _client is None:
        _client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name="jiggasai_documents",
        )
    return _collection


def delete_document_chunks(doc_id: str) -> int:
    collection = get_collection()

    results = collection.get(
        where={"document_id": doc_id},
        include=["metadatas"],
    )

    chunk_ids = results.get("ids", [])

    if not chunk_ids:
        print(f"[CHROMA DELETE] No chunks found for document_id={doc_id}")
        return 0

    collection.delete(ids=chunk_ids)
    print(f"[CHROMA DELETE] Deleted {len(chunk_ids)} chunks for document_id={doc_id}")
    return len(chunk_ids)
