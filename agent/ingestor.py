import logging
import os
import re
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from django.conf import settings
from .chroma_client import get_collection

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    filename = os.path.basename(filename or "")
    filename = re.sub(r"[^\w\s\-.]", "", filename)
    filename = re.sub(r"\s+", " ", filename).strip()
    return filename[:100] if filename else "unknown.pdf"


def ingest_document(pdf_path: str, doc_id: str) -> int:
    logger.info(f"[ingestor] Loading PDF: {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
    except Exception as e:
        raise ValueError(
            "Could not read the PDF. The file may be corrupted or unsupported."
        ) from e

    if not pages:
        raise ValueError("No pages found in PDF.")

    logger.info(f"[ingestor] Loaded {len(pages)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)
    logger.info(f"[ingestor] Split into {len(chunks)} chunks")

    if not chunks or all(not c.page_content.strip() for c in chunks):
        raise ValueError(
            "No text could be extracted from this PDF. "
            "The file may be scanned/image-based or password protected."
        )

    embeddings = OllamaEmbeddings(
        model=settings.OLLAMA_EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )

    texts = [chunk.page_content for chunk in chunks]
    safe_source = sanitize_filename(pdf_path)
    metadatas = [
        {
            "document_id": str(doc_id),
            "source": safe_source,
            "page": str(chunk.metadata.get("page", 0)),
        }
        for chunk in chunks
    ]
    ids = [f"doc{doc_id}_chunk{i}" for i in range(len(chunks))]

    logger.info(f"[ingestor] Embedding {len(texts)} chunks via Ollama ({settings.OLLAMA_EMBED_MODEL})...")
    BATCH_SIZE = 50
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        logger.info(f"[ingestor] Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)")
        for attempt in range(3):
            try:
                all_embeddings.extend(embeddings.embed_documents(batch))
                break
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"[ingestor] Retry {attempt + 1} after error: {e}")
                    time.sleep(2)
                else:
                    raise
    logger.info(f"[ingestor] All {len(all_embeddings)} embeddings done.")

    collection = get_collection()
    collection.add(
        documents=texts,
        embeddings=all_embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    logger.info(f"[ingestor] Done. Stored {len(chunks)} chunks for document {doc_id}")
    return len(chunks)
