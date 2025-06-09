import os
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict
from datetime import datetime
from enum import Enum

class ChunkingMethod(str, Enum):
    TOKEN_SENTENCE = "token_sentence"
    SENTENCE_WINDOW = "sentence_window"
    LANGCHAIN_KONLPY = "langchain_konlpy"
    RECURSIVE_CHARACTER = "recursive_character"
    SEMANTIC_CHUNKER = "semantic_chunker"

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    VLLM = "vllm"
    OLLAMA = "ollama"

class ParserType(str, Enum):
    DOCLING = "docling"
    PDFMINER = "pdfminer"
    PDFPLUMBER = "pdfplumber"
    PYPDFIUM2 = "pypdfium2"
    PYPDF = "pypdf"
    PYMUPDF = "pymupdf"

class QueryType(str, Enum):
    DOCUMENT_SEARCH = "document_search"
    GENERAL_KNOWLEDGE = "general_knowledge"
    GREETING = "greeting"
    SYSTEM_INFO = "system_info"

class ChunkingSettings(BaseModel):
    method: ChunkingMethod = ChunkingMethod.TOKEN_SENTENCE
    chunk_size: int = 512
    overlap: int = 50
    unit: Literal["token", "sentence"] = "token"

class LLMSettings(BaseModel):
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    vllm_base_url: Optional[str] = None
    ollama_base_url: Optional[str] = None
    temperature: float = 0.1

class ParserSettings(BaseModel):
    parser_type: ParserType = ParserType.DOCLING

class EmbeddingSettings(BaseModel):
    model_name: str = os.getenv('EMBEDDING_MODEL', 'jinaai/jina-embeddings-v3')

class RetrievalSettings(BaseModel):
    search_limit: int = 5
    score_threshold: float = 0.7
    use_hybrid: bool = False  # 이전 버전과의 호환성을 위해 유지

class SystemSettings(BaseModel):
    chunking: ChunkingSettings = ChunkingSettings()
    llm: LLMSettings = LLMSettings()
    parser: ParserSettings = ParserSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()

class SettingsUpdateRequest(BaseModel):
    chunking: Optional[ChunkingSettings] = None
    llm: Optional[LLMSettings] = None
    parser: Optional[ParserSettings] = None
    embedding: Optional[EmbeddingSettings] = None
    retrieval: Optional[RetrievalSettings] = None

class VectorStoreDocument(BaseModel):
    id: str
    title: str
    metadata: Dict
    chunk_count: int
    created_at: datetime

class VectorStoreResponse(BaseModel):
    documents: List[VectorStoreDocument]
    total_count: int

class DocumentCreate(BaseModel):
    title: str
    content: str
    metadata: Optional[dict] = {}

class DocumentResponse(BaseModel):
    id: str
    title: str
    metadata: dict
    created_at: datetime

class ChunkResponse(BaseModel):
    content: str
    metadata: Dict
    score: float
    chunk_index: Optional[str] = None
    document_id: Optional[str] = None

class SearchRequest(BaseModel):
    """검색 요청 모델"""
    query: Optional[str] = None
    question: Optional[str] = None
    limit: int = 5
    threshold: float = 0.7
    use_hybrid: Optional[bool] = None
    strategy: Optional[str] = None
    document_id: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[ChunkResponse]
    total_count: int

class QARequest(BaseModel):
    question: str
    limit: int = 5
    threshold: float = 0.7
    use_hybrid: Optional[bool] = None  # 이전 버전과의 호환성을 위해 유지
    strategy: Optional[str] = "hybrid"  # 'vector', 'bm25', 'hybrid' 중 하나

class QueryAnalysisResult(BaseModel):
    needs_documents: bool
    query_type: QueryType
    keywords: List[str] = []
    general_response: Optional[str] = None
    confidence: float = 1.0
    reasoning: Optional[str] = None

class QAResponse(BaseModel):
    question: str
    answer: str
    context: List[ChunkResponse] = []
    model: str
    success: bool
    error: Optional[str] = None

class DocumentCountResponse(BaseModel):
    total_documents: int
