import tempfile
import os
import re
from typing import Union, Optional
import aiofiles
from .models import ParserType, ParserSettings
import logging

logger = logging.getLogger(__name__)

class DocumentParser:
    def __init__(self, parser_settings: Optional[ParserSettings] = None):
        if parser_settings:
            self.settings = parser_settings
        else:
            self.settings = ParserSettings()
        
        self.setup_parser()
    
    async def parse_file(self, file_content: bytes, filename: str) -> str:
        """파일 내용을 파싱하여 텍스트를 추출합니다."""
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # 파서 타입에 따라 처리
            if self.settings.parser_type == ParserType.DOCLING:
                return await self._parse_with_docling(tmp_file_path)
            else:
                return await self._parse_with_langchain(tmp_file_path)
                
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    async def _parse_with_docling(self, file_path: str) -> str:
        """Docling으로 파싱"""
        if hasattr(self, 'use_langchain_docling') and self.use_langchain_docling:
            # LangChain Docling 사용
            from langchain_community.document_loaders import DoclingLoader
            loader = DoclingLoader(file_path)
            docs = loader.load()
            return "\n\n".join([doc.page_content for doc in docs])
        elif hasattr(self, 'converter') and self.converter:
            # 기존 Docling 사용
            result = self.converter.convert(file_path)
            if result.document:
                return result.document.export_to_markdown()
            else:
                raise ValueError("문서를 파싱할 수 없습니다.")
        else:
            # converter가 없는 경우 LangChain Docling 시도
            try:
                from langchain_community.document_loaders import DoclingLoader
                loader = DoclingLoader(file_path)
                docs = loader.load()
                return "\n\n".join([doc.page_content for doc in docs])
            except ImportError:
                raise ValueError("Docling 파서를 사용할 수 없습니다. 적절한 패키지가 설치되지 않았습니다.")
    
    async def _parse_with_langchain(self, file_path: str) -> str:
        """LangChain 파서로 파싱"""
        try:
            loader = self.loader_class(file_path)
            docs = loader.load()
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.warning(f"LangChain 파서 오류: {e}, Docling으로 fallback을 시도합니다.")
            # Fallback to docling
            return await self._parse_with_docling(file_path)
    
    def parse_text(self, text: str) -> str:
        """일반 텍스트를 파싱합니다."""
        # <think> 태그 제거
        return self.remove_think_tags(text)
    
    def remove_think_tags(self, text: str) -> str:
        """<think> 태그와 그 내용을 제거합니다."""
        if text is None:
            return ""
        
        # <think> 태그 제거
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned_text.strip()
    
    def get_supported_formats(self) -> list:
        """지원되는 파일 형식을 반환합니다."""
        return [
            "pdf", "docx", "pptx", "xlsx", 
            "html", "txt", "md"
        ]
    
    def setup_parser(self):
        """파서 타입에 따라 파서를 설정합니다."""
        if self.settings.parser_type == ParserType.DOCLING:
            self._setup_docling_parser()
        elif self.settings.parser_type == ParserType.PDFMINER:
            self._setup_pdfminer_parser()
        elif self.settings.parser_type == ParserType.PDFPLUMBER:
            self._setup_pdfplumber_parser()
        elif self.settings.parser_type == ParserType.PYPDFIUM2:
            self._setup_pypdfium2_parser()
        elif self.settings.parser_type == ParserType.PYPDF:
            self._setup_pypdf_parser()
        elif self.settings.parser_type == ParserType.PYMUPDF:
            self._setup_pymupdf_parser()
        else:
            # 기본값은 Docling
            self._setup_docling_parser()
    
    def _setup_docling_parser(self):
        """LangChain Docling 파서 설정"""
        try:
            from langchain_community.document_loaders import DoclingLoader
            self.use_langchain_docling = True
        except ImportError:
            # Fallback to original docling
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()
            self.use_langchain_docling = False
    
    def _setup_pdfminer_parser(self):
        """PDFMiner 파서 설정"""
        try:
            from langchain_community.document_loaders import PDFMinerLoader
            self.loader_class = PDFMinerLoader
            logger.info("PDFMinerLoader가 설정되었습니다.")
        except ImportError:
            # Fallback to docling
            self._setup_docling_parser()
    
    def _setup_pdfplumber_parser(self):
        """PDFPlumber 파서 설정"""
        try:
            from langchain_community.document_loaders import PDFPlumberLoader
            self.loader_class = PDFPlumberLoader
            logger.info("PDFPlumberLoader가 설정되었습니다.")
        except ImportError:
            # Fallback to docling
            self._setup_docling_parser()
    
    def _setup_pypdfium2_parser(self):
        """PyPDFium2 파서 설정"""
        try:
            from langchain_community.document_loaders import PyPDFium2Loader
            self.loader_class = PyPDFium2Loader
            logger.info("PyPDFium2Loader가 설정되었습니다.")
        except ImportError:
            # Fallback to docling
            self._setup_docling_parser()
    
    def _setup_pypdf_parser(self):
        """PyPDF 파서 설정"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            self.loader_class = PyPDFLoader
            logger.info("PyPDFLoader가 설정되었습니다.")
        except ImportError:
            # Fallback to docling
            self._setup_docling_parser()
    
    def _setup_pymupdf_parser(self):
        """PyMuPDF 파서 설정"""
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
            self.loader_class = PyMuPDFLoader
            logger.info("PyMuPDFLoader가 설정되었습니다.")
        except ImportError:
            # Fallback to docling
            self._setup_docling_parser()
    
    def update_settings(self, new_settings: ParserSettings):
        """파서 설정을 업데이트하고 파서를 재설정합니다."""
        self.settings = new_settings
        self.setup_parser()
