from typing import List, Dict, Optional
import logging

from langchain_qdrant import RetrievalMode

from .database import QdrantDatabase
from .llm import LLMService
from .settings import SettingsManager

settings_manager = SettingsManager()
logger = logging.getLogger(__name__)

class DocumentRetriever:
    """문서 검색을 위한 retriever 클래스 - langchain-qdrant의 RetrievalMode 활용"""
    
    def __init__(self, db : QdrantDatabase, llm_service: LLMService = None):
        """초기화"""
        self.db = db
        self.llm_service = llm_service
    
    def search_documents(self, 
                        query: str, 
                        limit: int = 5, 
                        threshold: float = 0.0, 
                        document_id: Optional[str] = None,
                        strategy: str = "dense") -> Dict:
        """문서에서 유사한 내용을 검색합니다."""
        try:
            if not query or query.strip() == "":
                return {
                    "query": query,
                    "results": [],
                    "total_count": 0
                }
            
            # 문자열을 RetrievalMode로 변환
            try:
                if strategy.lower() == "dense" or strategy.lower() == "vector":
                    retrieval_mode = RetrievalMode.DENSE
                elif strategy.lower() == "sparse" or strategy.lower() == "bm25":
                    retrieval_mode = RetrievalMode.SPARSE
                elif strategy.lower() == "hybrid":
                    retrieval_mode = RetrievalMode.HYBRID
                else:
                    logger.warning(f"알 수 없는 검색 전략: {strategy}, 기본값(dense) 사용")
                    retrieval_mode = RetrievalMode.DENSE
            except ValueError:
                logger.warning(f"검색 전략 변환 실패: {strategy}, 기본값(dense) 사용")
                retrieval_mode = RetrievalMode.DENSE
            
            # 메타데이터 필터 설정
            filters = None
            if document_id:
                filters = {
                    "document_id": document_id
                }
            
            # QdrantVectorStore의 search 메서드 사용
            results = self.db.search(
                query=query,
                k=limit,
                filter=filters,
                retrieval_mode=retrieval_mode
            )
            
            # 점수 필터링
            filtered_results = [
                result for result in results
                if result["score"] >= threshold
            ]
            
            # unique_chunk_index 추가
            for result in filtered_results:
                result["unique_chunk_index"] = f"{result.get('document_id', 'unknown')}#{result.get('metadata', {}).get('chunk_index', 0)}"
        
            
            # 점수 기준 정렬
            filtered_results.sort(key=lambda x: x["score"], reverse=True)
            
            # === 상위 2개 문서의 앞뒤 청크 추가 ===
            # 1. 상위 2개 추출
            top2 = filtered_results[:2]
            extra_chunks = []
            seen_keys = set(r["unique_chunk_index"] for r in filtered_results)
            for doc in top2:
                doc_id = doc.get("document_id")
                chunk_idx = doc.get("chunk_index")
                if doc_id is None or chunk_idx is None:
                    continue
                for offset in range(-2, 3):
                    idx = chunk_idx + offset
                    if idx < 0:
                        continue
                    qdrant_filter = {
                        "must": [
                            {"key": "metadata.document_id", "match": {"value": doc_id}},
                            {"key": "metadata.chunk_index", "match": {"value": idx}}
                        ]
                    }
                    chunk_results = self.db.search(
                        query="",  # 내용 무관, 전체에서 필터만 적용
                        k=1,
                        filter=qdrant_filter,
                        retrieval_mode=retrieval_mode
                    )
                    for chunk in chunk_results:
                        key = f"{doc_id}#{idx}"
                        if key in seen_keys:
                            # 기존에 있던 버전 삭제
                            filtered_results = [r for r in filtered_results if r["unique_chunk_index"] != key]
                        else:
                            chunk["unique_chunk_index"] = key
                            seen_keys.add(key)
                        # main chunk보다 0.01 낮은 score로 추가
                        main_score = doc.get("score", 0.0)
                        chunk["unique_chunk_index"] = key
                        chunk["score"] = max(main_score - 0.01, 0.0)
                        extra_chunks.append(chunk)
            # extra_chunks를 기존 results에 추가
            filtered_results.extend(extra_chunks)

            # content 길이 20 미만 제거
            filtered_results = [
                result for result in filtered_results
                if len(result.get("content", "")) >= 20
            ]

            # 점수 기준 재정렬
            filtered_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            # 중복 unique_chunk_index 제거 (최초 등장만 유지)
            deduped = {}
            for r in filtered_results:
                key = r["unique_chunk_index"]
                if key not in deduped:
                    deduped[key] = r
            filtered_results = list(deduped.values())

            return {
                    "query": query,
                    "results": filtered_results[:limit],
                    "total_count": len(filtered_results)
                }
            
        except Exception as e:
            logger.error(f"문서 검색 오류: {e}")
            return {
                "query": query,
                "results": [],
                "total_count": 0,
                "error": str(e)
            }
    
    def search_by_document_id(self, document_id: str, query: str = "", limit: int = 1000, threshold: float = 0.0, strategy: str = "dense") -> Dict:
        """특정 문서 ID 내에서 모든 청크를 검색합니다."""
        return self.search_documents(
            query=query,
            limit=limit,
            threshold=threshold,
            document_id=document_id,
            strategy=strategy
        )
    
    def get_document_statistics(self) -> Dict:
        """문서 통계 정보를 반환합니다."""
        try:
            document_count = self.db.get_document_count()
            collection_info = self.db.get_collection_info()
            
            return {
                "total_documents": document_count,
                "total_chunks": collection_info["total_points"],
                "average_chunks_per_document": collection_info["total_points"] / max(document_count, 1),
                "collection_info": collection_info
            }
        except Exception as e:
            logger.error(f"문서 통계 조회 오류: {e}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "error": str(e)
            }
