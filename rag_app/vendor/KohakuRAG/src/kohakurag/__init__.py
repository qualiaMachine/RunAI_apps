"""Public package surface for KohakuRAG."""

from .datastore import (
    HierarchicalNodeStore,
    InMemoryNodeStore,
    KVaultNodeStore,
    matches_to_snippets,
)
from .embeddings import (
    EmbeddingModel,
    average_embeddings,
)

# Torch-dependent classes — only available when torch/transformers are installed.
try:
    from .embeddings import (
        JinaEmbeddingModel,
        LocalHFEmbeddingModel,
    )
except Exception:  # ImportError or numpy/scipy crash
    JinaEmbeddingModel = None  # type: ignore[assignment,misc]
    LocalHFEmbeddingModel = None  # type: ignore[assignment,misc]

from .indexer import DocumentIndexer
from .llm import OpenAIChatModel

try:
    from .llm import HuggingFaceLocalChatModel
except Exception:
    HuggingFaceLocalChatModel = None  # type: ignore[assignment,misc]

from .parsers import (
    dict_to_payload,
    markdown_to_payload,
    payload_to_dict,
    text_to_payload,
)
from .pipeline import (
    LLMQueryPlanner,
    MockChatModel,
    PromptTemplate,
    QueryPlanner,
    RAGPipeline,
    SimpleQueryPlanner,
    StructuredAnswer,
    StructuredAnswerResult,
    format_snippets,
)

# Optional components — degrade gracefully when dependencies are missing.
try:
    from .reranker import CrossEncoderReranker
except Exception:
    CrossEncoderReranker = None  # type: ignore[assignment,misc]

from .semantic_scholar import SemanticScholarRetriever
from .types import (
    ContextSnippet,
    DocumentPayload,
    NodeKind,
    RetrievalMatch,
    StoredNode,
    TreeNode,
)

__all__ = [
    "average_embeddings",
    "ContextSnippet",
    "CrossEncoderReranker",
    "DocumentIndexer",
    "DocumentPayload",
    "EmbeddingModel",
    "HierarchicalNodeStore",
    "HuggingFaceLocalChatModel",
    "InMemoryNodeStore",
    "KVaultNodeStore",
    "JinaEmbeddingModel",
    "LLMQueryPlanner",
    "LocalHFEmbeddingModel",
    "MockChatModel",
    "NodeKind",
    "OpenAIChatModel",
    "PromptTemplate",
    "QueryPlanner",
    "RAGPipeline",
    "RetrievalMatch",
    "SemanticScholarRetriever",
    "SimpleQueryPlanner",
    "StoredNode",
    "StructuredAnswer",
    "StructuredAnswerResult",
    "TreeNode",
    "dict_to_payload",
    "format_snippets",
    "markdown_to_payload",
    "payload_to_dict",
    "text_to_payload",
    "matches_to_snippets",
]
