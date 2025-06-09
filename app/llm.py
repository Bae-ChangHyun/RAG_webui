import json
import logging
import re
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from .models import LLMSettings, LLMProvider
from .models import QueryAnalysisResult, QueryType

# LlamaIndex 임포트
try:
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.gemini import Gemini
    from llama_index.core.output_parsers import PydanticOutputParser
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core import Response
except ImportError:
    pass

logger = logging.getLogger(__name__)

class LLMService:
    """LLM 서비스를 관리하는 클래스"""
    
    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.client = None
        self.available = False
        self._initialize_client()
    
    def _initialize_client(self):
        """LLM 클라이언트를 초기화합니다."""
        try:
            if self.settings.provider == LLMProvider.OPENAI:
                import openai
                self.client = openai.OpenAI(api_key=self.settings.api_key)
                
            elif self.settings.provider == LLMProvider.GEMINI:
                import google.generativeai as genai
                genai.configure(api_key=self.settings.api_key)
                self.client = genai
                
            elif self.settings.provider == LLMProvider.VLLM:
                import openai
                self.client = openai.OpenAI(
                    base_url=self.settings.vllm_base_url,
                    api_key="dummy"
                )
                
            elif self.settings.provider == LLMProvider.OLLAMA:
                import openai
                self.client = openai.OpenAI(
                    base_url=self.settings.ollama_base_url,
                    api_key="dummy"
                )
            
            self.available = True
            logger.info(f"LLM 클라이언트 초기화 완료: {self.settings.provider}")
            
        except Exception as e:
            logger.error(f"LLM 클라이언트 초기화 실패: {e}")
            self.available = False
            
    def _remove_think_tags(self, text: str) -> str:
        """<think> 태그와 그 내용을 제거합니다."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """컨텍스트를 기반으로 답변을 생성합니다."""
        try:
            if not self.is_available():
                raise Exception("LLM 서비스가 사용 불가능합니다.")
            
            # 컨텍스트 포맷팅
            context_text = "\n\n".join([
                f"문서 {i+1}:\n{chunk['content']}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            # 프롬프트 생성
            system_prompt = self._get_system_prompt()
            user_prompt = self._get_user_prompt(question, context_text)
            
            # LLM 호출
            if self.settings.provider == LLMProvider.OPENAI:
                response = self._generate_with_openai(system_prompt, user_prompt)
            elif self.settings.provider == LLMProvider.GEMINI:
                response = self._generate_with_gemini(system_prompt, user_prompt)
            elif self.settings.provider == LLMProvider.VLLM:
                response = self._generate_with_vllm(system_prompt, user_prompt)
            elif self.settings.provider == LLMProvider.OLLAMA:
                response = self._generate_with_ollama(system_prompt, user_prompt)
            else:
                raise ValueError(f"지원하지 않는 LLM 제공자: {self.settings.provider}")
            
            return response
            
        except Exception as e:
            logger.error(f"답변 생성 중 오류 발생: {e}")
            raise

    def generate_general_response(self, question: str) -> str:
        """일반적인 답변을 생성합니다."""
        try:
            if not self.is_available():
                raise Exception("LLM 서비스가 사용 불가능합니다.")
            
            # 프롬프트 생성
            system_prompt = self._get_general_system_prompt()
            user_prompt = f"질문: {question}\n\n답변:"
            
            # LLM 호출
            if self.settings.provider == LLMProvider.OPENAI:
                response = self._generate_with_openai(system_prompt, user_prompt)
            elif self.settings.provider == LLMProvider.GEMINI:
                response = self._generate_with_gemini(system_prompt, user_prompt)
            elif self.settings.provider == LLMProvider.VLLM:
                response = self._generate_with_vllm(system_prompt, user_prompt)
            elif self.settings.provider == LLMProvider.OLLAMA:
                response = self._generate_with_ollama(system_prompt, user_prompt)
            else:
                raise ValueError(f"지원하지 않는 LLM 제공자: {self.settings.provider}")
            
            return response
            
        except Exception as e:
            logger.error(f"일반 답변 생성 중 오류 발생: {e}")
            raise
    
    def _generate_with_openai(self, system_prompt: str, user_prompt: str) -> str:
        """OpenAI로 답변 생성"""
        response = self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return self._remove_think_tags(response.choices[0].message.content)
    
    def _generate_with_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Gemini로 답변 생성"""
        model = self.client.GenerativeModel(self.settings.model_name)
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1500,
            }
        )
        
        return self._remove_think_tags(response.text)
    
    def _generate_with_vllm(self, system_prompt: str, user_prompt: str) -> str:
        """vLLM으로 답변 생성"""
        response = self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return self._remove_think_tags(response.choices[0].message.content)
    
    def _generate_with_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Ollama로 답변 생성"""
        response = self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return self._remove_think_tags(response.choices[0].message.content)
    
    def _get_system_prompt(self) -> str:
        """시스템 프롬프트를 반환합니다."""
        return """당신은 주어진 문서를 바탕으로 질문에 답하는 도움이 되는 AI 어시스턴트입니다. 다음 규칙을 따르세요:

1. 주어진 문서의 내용만을 바탕으로 답변하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 답변은 정확하고 구체적이어야 합니다
4. 한국어로 명확하고 도움이 되는 답변을 제공하세요
5. 만약 문서에 답이 없다면, 일반적인 답변을 하세요
6. 질문과 관계없는 부가적인 말을 하지마세요"""

    def _get_user_prompt(self, question: str, context_text: str) -> str:
        """사용자 프롬프트를 반환합니다."""
        return f"""문서 내용:
{context_text}

질문: {question}

위 문서를 바탕으로 질문에 답해주세요. 문서에 답이 없다면 일반적인 답변을 하세요."""

    def _get_general_system_prompt(self) -> str:
        """일반 답변용 시스템 프롬프트를 반환합니다."""
        return """당신은 일반적인 지식과 개념에 대해 명확하고 이해하기 쉽게 설명하는 AI 어시스턴트입니다. 한국어로 답변하며, 정확하고 도움이 되는 정보를 제공하세요. 만약 확실하지 않은 정보라면 그렇다고 명시하세요."""
    
    def is_available(self) -> bool:
        """LLM 서비스 사용 가능 여부를 반환합니다."""
        return self.available
    
    def update_settings(self, new_settings: LLMSettings):
        """LLM 설정을 업데이트합니다."""
        self.settings = new_settings
        self._initialize_client()