from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from typing import List, Dict, Optional
import re
from .models import ChunkingMethod, ChunkingSettings
from .config import settings

class TextChunker:
    def __init__(self, chunking_settings: Optional[ChunkingSettings] = None):
        if chunking_settings:
            self.settings = chunking_settings
        else:
            # 기본 설정
            self.settings = ChunkingSettings()
        
        self.setup_chunker()
    
    def setup_chunker(self):
        """청킹 방식에 따라 splitter를 설정합니다."""
        if self.settings.method == ChunkingMethod.TOKEN_SENTENCE:
            self.splitter = self._create_token_sentence_splitter()
        elif self.settings.method == ChunkingMethod.SENTENCE_WINDOW:
            self.splitter = self._create_sentence_window_splitter()
        elif self.settings.method == ChunkingMethod.LANGCHAIN_KONLPY:
            self.splitter = self._create_konlpy_splitter()
        elif self.settings.method == ChunkingMethod.RECURSIVE_CHARACTER:
            self.splitter = self._create_recursive_character_splitter()
        elif self.settings.method == ChunkingMethod.SEMANTIC_CHUNKER:
            self.splitter = self._create_semantic_chunker()
        else:
            # 기본값
            self.splitter = self._create_recursive_character_splitter()
    
    def _create_token_sentence_splitter(self):
        """토큰/문장 기반 splitter 생성"""

            # 문장 단위로 분할
        return SentenceBasedSplitter(
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.overlap
        )
    
    def _create_sentence_window_splitter(self):
        """문장 윈도우 + Kiwi 기반 splitter"""
        return SentenceWindowSplitter(
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.overlap
        )
    
    def _create_konlpy_splitter(self):
        """KoNLPy 기반 splitter (나중에 구현)"""
        # TODO: KoNLPy 설치 후 구현
        return RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
    
    def _create_recursive_character_splitter(self):
        """Recursive Character 기반 splitter"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _create_semantic_chunker(self):
        """Semantic Chunker 기반 splitter (임베딩 기반)"""

        from langchain_huggingface import HuggingFaceEmbeddings
        #embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        embeddings = HuggingFaceEmbeddings(
        model_name = settings.embedding_model,
            model_kwargs={
            "device":'cuda',
            "trust_remote_code": True
            },
        )
        return SemanticChunker(embeddings=embeddings)
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """텍스트를 청크로 나누고 메타데이터를 추가합니다."""
        if metadata is None:
            metadata = {}
        
        chunks = self.splitter.split_text(text)
        total_chunks = len(chunks)
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "chunk_size": len(chunk),
                "total_chunks": total_chunks,
                "chunking_method": self.settings.method.value
            }
            
            result.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
        
        return result
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """여러 문서를 청크로 나눕니다."""
        all_chunks = []
        
        for doc_index, doc in enumerate(documents):
            text = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # 문서 인덱스를 메타데이터에 추가
            doc_metadata = {**metadata, "document_index": doc_index}
            
            chunks = self.chunk_text(text, doc_metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def update_settings(self, new_settings: ChunkingSettings):
        """청킹 설정을 업데이트하고 splitter를 재설정합니다."""
        self.settings = new_settings
        self.setup_chunker()


class SentenceBasedSplitter:
    """문장 기반 텍스트 분할기"""
    
    def __init__(self, chunk_size: int = 5, overlap: int = 1):
        self.chunk_size = chunk_size  # 문장 개수
        self.overlap = overlap  # 겹치는 문장 개수
    
    def split_text(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할합니다."""
        # 문장 분할 패턴 (한국어 고려)
        sentence_pattern = r'[.!?]+\s*'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), self.chunk_size - self.overlap):
            chunk_sentences = sentences[i:i + self.chunk_size]
            chunk = ' '.join(chunk_sentences)
            if chunk:
                chunks.append(chunk)
        
        return chunks


class SentenceWindowSplitter:
    """문장 윈도우 방식의 텍스트 분할기 (Kiwi 사용)"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Kiwi 사용 가능한 경우 사용, 아니면 기본 문장 분할 사용
        try:
            from kiwipiepy import Kiwi
            self.kiwi = Kiwi()
            self.use_kiwi = True
        except ImportError:
            self.use_kiwi = False
    
    def split_text(self, text: str) -> List[str]:
        """텍스트를 Kiwi 또는 기본 방식으로 분할합니다."""
        if self.use_kiwi:
            return self._split_with_kiwi(text)
        else:
            return self._split_with_regex(text)
    
    def _split_with_kiwi(self, text: str) -> List[str]:
        """Kiwi를 사용한 문장 분할"""
        sentences = []
        for sent in self.kiwi.split_into_sents(text):
            sentences.append(sent.text)
        
        return self._create_chunks_from_sentences(sentences)
    
    def _split_with_regex(self, text: str) -> List[str]:
        """정규식을 사용한 문장 분할"""
        sentence_pattern = r'[.!?]+\s*'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return self._create_chunks_from_sentences(sentences)
    
    def _create_chunks_from_sentences(self, sentences: List[str]) -> List[str]:
        """문장들을 청크로 만듭니다."""
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
