import json
import os
from typing import Dict, Any
from .models import SystemSettings, ChunkingSettings, LLMSettings, ParserSettings, EmbeddingSettings, RetrievalSettings

class SettingsManager:
    def __init__(self, config_file: str = "settings.json"):
        self.config_file = config_file
        self.settings = SystemSettings()
        self.load_settings()
        self._load_env_variables()
    
    def _load_env_variables(self):
        """환경변수에서 API 키 등을 로드합니다."""
        # OpenAI API 키
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and not self.settings.llm.api_key:
            self.settings.llm.api_key = openai_key
        
        # Google API 키
        google_key = os.getenv('GOOGLE_API_KEY')
        if google_key and self.settings.llm.provider == 'gemini' and not self.settings.llm.api_key:
            self.settings.llm.api_key = google_key
    
    def load_settings(self):
        """설정 파일에서 설정을 로드합니다."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.settings = SystemSettings(**data)
            except Exception as e:
                print(f"설정 로드 오류: {e}")
                self.settings = SystemSettings()
        else:
            self.save_settings()
    
    def save_settings(self):
        """현재 설정을 파일에 저장합니다."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings.model_dump(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"설정 저장 오류: {e}")
    
    def update_chunking_settings(self, settings: ChunkingSettings):
        """청킹 설정을 업데이트합니다."""
        self.settings.chunking = settings
        self.save_settings()
    
    def update_llm_settings(self, settings: LLMSettings):
        """LLM 설정을 업데이트합니다."""
        self.settings.llm = settings
        self.save_settings()
    
    def update_parser_settings(self, settings: ParserSettings):
        """파서 설정을 업데이트합니다."""
        self.settings.parser = settings
        self.save_settings()
    
    def update_embedding_settings(self, settings: EmbeddingSettings):
        """임베딩 설정을 업데이트합니다."""
        self.settings.embedding = settings
        self.save_settings()
    
    def update_retrieval_settings(self, settings: RetrievalSettings):
        """검색 설정을 업데이트합니다."""
        self.settings.retrieval = settings
        self.save_settings()
    
    def get_settings(self) -> SystemSettings:
        """현재 설정을 반환합니다."""
        return self.settings
    
    def get_chunking_settings(self) -> ChunkingSettings:
        """청킹 설정을 반환합니다."""
        return self.settings.chunking
    
    def get_llm_settings(self) -> LLMSettings:
        """LLM 설정을 반환합니다."""
        return self.settings.llm
    
    def get_parser_settings(self) -> ParserSettings:
        """파서 설정을 반환합니다."""
        return self.settings.parser
    
    def get_embedding_settings(self) -> EmbeddingSettings:
        """임베딩 설정을 반환합니다."""
        return self.settings.embedding
    
    def get_retrieval_settings(self) -> RetrievalSettings:
        """검색 설정을 반환합니다."""
        return self.settings.retrieval
