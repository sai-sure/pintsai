"""
rag_engine.py — Document ingestion and semantic retrieval.
Loads PDF/TXT policy documents, chunks them, embeds with Google
Generative AI, and stores in an in-memory FAISS vector store.

Guardrail built in: retrieve_with_relevance_check() returns an
is_in_scope flag so the agent can refuse out-of-scope claims.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("pints.rag")

# Relevance threshold for FAISS L2 distance.
# Scores below this = relevant; above = probably not in scope.
RELEVANCE_THRESHOLD = 1.2


class RAGEngine:
    """
    Manages the knowledge base (policy docs, procedure manuals, etc.).
    Call add_documents() whenever new files are uploaded via the sidebar.
    Call retrieve_with_relevance_check() from the claims agent.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
        self.vectorstore: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # Registry of loaded files for display in the UI
        self.document_registry: List[Dict] = []

    # ------------------------------------------------------------------ #
    #  Document loading                                                     #
    # ------------------------------------------------------------------ #

    def _load_uploaded_file(self, uploaded_file) -> List[Document]:
        """Read a Streamlit UploadedFile and return LangChain Documents."""
        suffix = Path(uploaded_file.name).suffix.lower()
        docs: List[Document] = []

        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            elif suffix in (".txt", ".md"):
                loader = TextLoader(tmp_path, encoding="utf-8")
                docs = loader.load()
            else:
                logger.warning(f"Unsupported file type ignored: {suffix}")
        except Exception as exc:
            logger.error(f"Error loading '{uploaded_file.name}': {exc}")
        finally:
            os.unlink(tmp_path)

        # Tag every page with the originating filename for source citation
        for doc in docs:
            doc.metadata["source_file"] = uploaded_file.name

        return docs

    def add_documents(self, uploaded_files) -> Dict:
        """
        Ingest a list of Streamlit UploadedFile objects into the vector store.
        Returns a status dict suitable for displaying in the UI.
        """
        all_docs: List[Document] = []

        for f in uploaded_files:
            pages = self._load_uploaded_file(f)
            if not pages:
                logger.warning(f"No content extracted from '{f.name}'")
                continue
            all_docs.extend(pages)
            self.document_registry.append(
                {"filename": f.name, "pages": len(pages), "size_kb": round(f.size / 1024, 1)}
            )

        if not all_docs:
            return {"status": "error", "message": "No content could be extracted from the uploaded files."}

        chunks = self.text_splitter.split_documents(all_docs)
        logger.info(f"Split {len(all_docs)} pages → {len(chunks)} chunks")

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        return {
            "status": "success",
            "chunks": len(chunks),
            "pages": len(all_docs),
            "files": len(uploaded_files),
        }

    # ------------------------------------------------------------------ #
    #  Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def retrieve_with_relevance_check(
        self, query: str, k: int = 6
    ) -> Tuple[List[Document], bool]:
        """
        Returns (docs, is_in_scope).
        is_in_scope=False means no relevant policy content was found —
        the agent should mark the claim as 'out_of_scope'.
        """
        if self.vectorstore is None:
            logger.warning("Vector store is empty — no RAG context available")
            return [], False

        try:
            results: List[Tuple[Document, float]] = (
                self.vectorstore.similarity_search_with_score(query, k=k)
            )
        except Exception as exc:
            logger.error(f"FAISS retrieval failed: {exc}")
            return [], False

        # Filter by relevance threshold (L2: lower = more similar)
        relevant = [(doc, score) for doc, score in results if score < RELEVANCE_THRESHOLD]

        if not relevant:
            logger.warning(
                f"No relevant docs found (best score={results[0][1]:.3f} > "
                f"threshold={RELEVANCE_THRESHOLD}) — marking out_of_scope"
            )
            return [], False

        docs = [doc for doc, _ in relevant]
        logger.info(f"Retrieved {len(docs)} relevant chunks (threshold={RELEVANCE_THRESHOLD})")
        return docs, True

    def retrieve(self, query: str, k: int = 6) -> List[Document]:
        docs, _ = self.retrieve_with_relevance_check(query, k)
        return docs

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def is_ready(self) -> bool:
        return self.vectorstore is not None

    def get_registry(self) -> List[Dict]:
        return self.document_registry
    