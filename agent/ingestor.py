import logging
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.GOOGLE_API_KEY,
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

    logger.info(f"[ingestor] Embedding {len(texts)} chunks via Google API...")
    embedded_vectors = embeddings.embed_documents(texts)

    collection = get_collection()
    collection.add(
        documents=texts,
        embeddings=embedded_vectors,
        metadatas=metadatas,
        ids=ids,
    )

    logger.info(f"[ingestor] Done. Stored {len(chunks)} chunks for document {doc_id}")
    return len(chunks)


ingest_pdf = ingest_document
