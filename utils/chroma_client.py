import io
import contextlib
import chromadb


def make_chroma_client(path: str) -> chromadb.PersistentClient:
    """Create a ChromaDB PersistentClient, suppressing telemetry noise."""
    settings = chromadb.Settings(anonymized_telemetry=False)
    with contextlib.redirect_stderr(io.StringIO()):
        return chromadb.PersistentClient(path=path, settings=settings)
