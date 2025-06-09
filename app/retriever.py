from typing import List, Dict, Optional
import logging
from enum import Enum

from pydantic import Field
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever

from .database import QdrantDatabase
from .llm import LLMService
from .settings import SettingsManager
settings_manager = SettingsManager()
logger = logging.getLogger(__name__)

class VectorRetriever(BaseRetriever):
    """Qdrant 벡터 검색을 위한 LangChain 호환 retriever"""
    
    database: QdrantDatabase = Field(..., description="Qdrant database instance")
    k: int = Field(default=5, description="Number of documents to retrieve")
    filters: Optional[Dict] = Field(default=None, description="Search filters")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """벡터 검색을 수행하고 Document 리스트를 반환합니다."""
        try:
            results = self.database.search(
                query=query,
                k=self.k,
                filter=self.filters
            )
            
            documents = []
            for result in results:
                metadata = result["metadata"]
                doc = Document(
                    page_content=result["content"],
                    metadata={
                        **metadata,
                        "score": result["score"],
                        "chunk_index": result.get("chunk_index"),
                        "document_id": result.get("document_id"),
                        "unique_chunk_index": f"{result.get('document_id', 'unknown')}#{metadata.get('chunk_index', 0)}"
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"벡터 검색 중 오류 발생: {e}")
            return []

class RetrievalStrategy(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25" 
    HYBRID = "hybrid"

class DocumentRetriever:
    def __init__(self, llm_service: LLMService):
        self.db = QdrantDatabase(embedding_model_name=settings_manager.get_embedding_settings().model_name)
        self.llm_service = llm_service
    
    def search_documents(self, 
                        query: str, 
                        limit: int = 5, 
                        threshold: float = 0.0, 
                        document_id: Optional[str] = None,
                        strategy: str = "hybrid") -> Dict:
        """문서에서 유사한 내용을 검색합니다."""
        try:
            if not query or query.strip() == "":
                return {
                    "query": query,
                    "results": [],
                    "total_count": 0
                }
            
            # 문자열을 RetrievalStrategy enum으로 변환
            try:
                strategy_enum = RetrievalStrategy(strategy.lower())
            except ValueError:
                logger.warning(f"알 수 없는 검색 전략: {strategy}, 기본값(hybrid) 사용")
                strategy_enum = RetrievalStrategy.HYBRID
            
            # 메타데이터 필터 설정
            filters = None
            if document_id:
                filters = {
                    "document_id": document_id
                }
            
            # 전략에 따른 검색기 생성 및 검색 수행
            results = self._retrieve_with_strategy(
                query=query,
                limit=limit,
                threshold=threshold,
                filters=filters,
                strategy=strategy_enum
            )
            
            return {
                "query": query,
                "results": results,
                "total_count": len(results)
            }
            
        except Exception as e:
            logger.error(f"문서 검색 오류: {e}")
            return {
                "query": query,
                "results": [],
                "total_count": 0,
                "error": str(e)
            }
    
    def _retrieve_with_strategy(self, 
                          query: str, 
                          limit: int,
                          threshold: float,
                          filters: Optional[Dict],
                          strategy: RetrievalStrategy) -> List[Dict]:
        """선택된 전략으로 검색을 수행합니다."""
        
        try:
            if strategy == RetrievalStrategy.VECTOR:
                results = self.db.search(
                    query=query,
                    k=limit,
                    filter=filters
                )
                # unique_chunk_index 추가
                for result in results:
                    result["unique_chunk_index"] = f"{result.get('document_id', 'unknown')}#{result.get('metadata', {}).get('chunk_index', 0)}"
            
            elif strategy == RetrievalStrategy.BM25:
                # 모든 문서 가져오기
                all_docs = []
                try:
                    points, _ = self.db.client.scroll(
                        collection_name=self.db.collection_name,
                        limit=10000,
                        with_payload=True
                    )
                    
                    for point in points:
                        if point.payload and "content" in point.payload:
                            all_docs.append(Document(
                                page_content=point.payload["content"],
                                metadata=point.payload.get("metadata", {})
                            ))
                    
                    if not all_docs:
                        logger.warning("BM25 검색을 위한 문서가 없습니다.")
                        return []
                    
                    # BM25 검색기 생성
                    bm25_retriever = BM25Retriever.from_documents(all_docs)
                    bm25_results = bm25_retriever.invoke(query)
                    
                    # 결과 포맷팅 - 실제 BM25 점수 계산
                    results = []
                    for i, doc in enumerate(bm25_results):
                        # BM25에서는 순위를 기반으로 점수 계산 (높은 순위일수록 높은 점수)
                        bm25_score = max(0.1, 1.0 - (i * 0.1))  # 순위 기반 점수 (0.1~1.0)
                        results.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": bm25_score,
                            "chunk_index": doc.metadata.get("chunk_index"),
                            "document_id": doc.metadata.get("document_id"),
                            "unique_chunk_index": f"{doc.metadata.get('document_id', 'unknown')}#{doc.metadata.get('chunk_index', 0)}"
                        })
                    results = results[:limit]  # 지정된 개수만큼 제한
                except Exception as e:
                    logger.error(f"BM25 검색 중 오류 발생: {e}")
                    return []
                
            elif strategy == RetrievalStrategy.HYBRID:
                # EnsembleRetriever를 사용한 RRF 기반 하이브리드 검색
                try:
                    # 벡터 검색기 생성
                    vector_retriever = VectorRetriever(
                        database=self.db,
                        k=limit,
                        filters=filters
                    )
                    
                    # BM25용 모든 문서 가져오기
                    all_docs = []
                    points, _ = self.db.client.scroll(
                        collection_name=self.db.collection_name,
                        limit=10000,
                        with_payload=True
                    )
                    
                    for point in points:
                        if point.payload and "content" in point.payload:
                            # 필터가 있는 경우 필터링 적용
                            if filters:
                                metadata = point.payload.get("metadata", {})
                                if not all(metadata.get(k) == v for k, v in filters.items()):
                                    continue
                            
                            all_docs.append(Document(
                                page_content=point.payload["content"],
                                metadata=point.payload.get("metadata", {})
                            ))
                    
                    if not all_docs:
                        logger.warning("하이브리드 검색을 위한 문서가 없습니다.")
                        # 벡터 검색만 수행
                        vector_docs = vector_retriever._get_relevant_documents(query)
                        results = [
                            {
                                "content": doc.page_content,
                                "metadata": {k: v for k, v in doc.metadata.items() if k not in ["score", "chunk_index", "document_id", "unique_chunk_index"]},
                                "score": doc.metadata.get("score", 0.0),
                                "chunk_index": doc.metadata.get("chunk_index"),
                                "document_id": doc.metadata.get("document_id"),
                                "unique_chunk_index": doc.metadata.get("unique_chunk_index")
                            }
                            for doc in vector_docs
                        ]
                        return results
                    
                    # BM25 검색기 생성
                    bm25_retriever = BM25Retriever.from_documents(all_docs)
                    bm25_retriever.k = limit
                    
                    # EnsembleRetriever 생성 (RRF 사용)
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[vector_retriever, bm25_retriever],
                        weights=[0.7, 0.3]  # 벡터 검색과 BM25 검색에 동일한 가중치
                    )
                    
                    # 검색 수행
                    ensemble_docs = ensemble_retriever.invoke(query)
                    
                    # 결과 포맷팅
                    results = []
                    for doc in ensemble_docs:
                        metadata = {k: v for k, v in doc.metadata.items() if k not in ["score", "chunk_index", "document_id", "unique_chunk_index"]}
                        result = {
                            "content": doc.page_content,
                            "metadata": metadata,
                            "score": doc.metadata.get("score", 0.0),
                            "chunk_index": doc.metadata.get("chunk_index"),
                            "document_id": doc.metadata.get("document_id"),
                            "unique_chunk_index": doc.metadata.get("unique_chunk_index", f"{doc.metadata.get('document_id', 'unknown')}#{metadata.get('chunk_index', 0)}")
                        }
                        results.append(result)
                    
                except Exception as e:
                    logger.error(f"하이브리드 검색 중 오류 발생: {e}")
                    # 폴백으로 벡터 검색만 수행
                    return self.db.search(
                        query=query,
                        k=limit,
                        filter=filters
                    )
            else:
                results = self.db.search(
                    query=query,
                    k=limit,
                    filter=filters
                )

            # === 상위 2개 문서의 앞뒤 청크 추가 ===
            # 1. 상위 2개 추출
            top2 = results[:2]
            extra_chunks = []
            seen_keys = set(r["unique_chunk_index"] for r in results)
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
                        filter=qdrant_filter
                    )
                    for chunk in chunk_results:
                        key = f"{doc_id}#{idx}"
                        if key in seen_keys:
                            # 기존에 있던 버전 삭제
                            results = [r for r in results if r["unique_chunk_index"] != key]
                        else:
                            chunk["unique_chunk_index"] = key
                            seen_keys.add(key)
                        # main chunk보다 0.01 낮은 score로 추가
                        main_score = doc.get("score", 0.0)
                        chunk["unique_chunk_index"] = key
                        chunk["score"] = max(main_score - 0.01, 0.0)
                        extra_chunks.append(chunk)
            # extra_chunks를 기존 results에 추가
            results.extend(extra_chunks)

            # 점수 필터링
            results = [
                result for result in results
                if result.get("score", 0.0) >= threshold
            ]

            # content 길이 20 미만 제거
            results = [
                result for result in results
                if len(result.get("content", "")) >= 20
            ]

            # 점수 기준 재정렬
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            # 중복 unique_chunk_index 제거 (최초 등장만 유지)
            deduped = {}
            for r in results:
                key = r["unique_chunk_index"]
                if key not in deduped:
                    deduped[key] = r
            results = list(deduped.values())

            return results[:limit]
            
        except Exception as e:
            logger.error(f"검색 중 오류 발생: {e}")
            return []
    
    def search_by_document_id(self, document_id: str, query: str = "", limit: int = 1000, threshold: float = 0.0, strategy: str = "vector") -> Dict:
        """특정 문서 ID 내에서 모든 청크를 검색합니다."""
        # reuse search_documents with document_id filter
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
