from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "documents"
    embedding_model: str = "jinaai/jina-embeddings-v3"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    vllm_base_url: str = "http://localhost:8000/v1"
    ollama_base_url: str = "http://localhost:11434/v1"
    search_limit: int = 5
    score_threshold: float = 0.7
    
    class Config:
        env_file = ".env"

settings = Settings()
