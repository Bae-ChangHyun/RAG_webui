# filepath: /home/bch/Project/main_project/RAG/rag_test/app/database.py
import uuid
import logging
from typing import List, Dict, Optional
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from qdrant_client import QdrantClient
from qdrant_client import models


from .config import settings

logger = logging.getLogger(__name__)

class QdrantDatabase:
    
    def __init__(self, embedding_model_name: str = None, retrieval_mode: RetrievalMode = RetrievalMode.HYBRID):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection_name = settings.qdrant_collection_name
        self.retrieval_mode = retrieval_mode

        model_name = embedding_model_name or settings.embedding_model
        self.embedding_model  = HuggingFaceEmbeddings(
                                    model_name = model_name,
                                    model_kwargs={
                                    "device":'cuda',
                                    "trust_remote_code": True
                                    },)
        
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        self._setup_vector_store()
    
    def _setup_vector_store(self):
        """벡터스토어 초기화"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "text-dense": models.VectorParams(
                            size=1024, 
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "text-sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams()
                        )
                    },
                )
                logger.info(f"컬렉션 '{self.collection_name}' 생성됨")
            
            # Qdrant 벡터스토어 설정
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding_model,
                content_payload_key="content",
                metadata_payload_key="metadata",
                retrieval_mode=self.retrieval_mode,
                sparse_embedding= self.sparse_embeddings,
                vector_name="text-dense",
                sparse_vector_name="text-sparse",
            )
            
        except Exception as e:
            logger.error(f"벡터스토어 초기화 오류: {e}")
            raise

    def add_documents(self, documents: List[Dict], metadata: Dict = None):
        """문서를 벡터스토어에 추가합니다."""
        try:
            # 딕셔너리 형태의 문서를 Document 객체로 변환
            docs = []
            for doc in documents:
                if isinstance(doc, dict):
                    doc_metadata = doc.get('metadata', {}).copy()
                    if metadata:
                        doc_metadata.update(metadata)
                    
                    # Document 객체 생성
                    docs.append(Document(
                        page_content=doc.get('content', ''),
                        metadata=doc_metadata
                    ))
                else:
                    # 이미 Document 객체인 경우
                    if metadata:
                        doc.metadata.update(metadata)
                    docs.append(doc)

            self.vector_store.add_documents(docs)
            logger.info(f"{len(docs)}개의 문서 청크가 추가됨")
            
        except Exception as e:
            logger.error(f"문서 추가 오류: {e}")
            raise

    def search(self, query: str, k: int = 5, filter: Dict = None, retrieval_mode: RetrievalMode = None) -> List[Dict]:
        """문서를 검색합니다."""
        try:
            self.vector_store.retrieval_mode = retrieval_mode
          
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "document_id": doc.metadata.get("document_id")
                }
                for doc, score in results
            ]
            
        except Exception as e:
            logger.error(f"검색 오류: {e}")
            return []



    def get_document_count(self) -> int:
        """저장된 고유 문서 수를 반환합니다."""
        try:
            document_ids = set()
            
            # Qdrant scroll 사용하여 모든 문서 ID 수집
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True
            )
            
            # 데이터 구조 디버깅을 위해 첫 번째 포인트 로그
            if points:
                logger.info(f"첫 번째 포인트 구조: {points[0].payload}")
            
            # 문서 ID 수집
            for point in points:
                if point.payload:
                    # 직접 document_id 확인
                    document_id = point.payload.get("document_id")
                    if document_id:
                        document_ids.add(document_id)
                    else:
                        # metadata 안에 document_id가 있는지 확인
                        metadata = point.payload.get("metadata", {})
                        if isinstance(metadata, dict):
                            doc_id_in_metadata = metadata.get("document_id")
                            if doc_id_in_metadata:
                                document_ids.add(doc_id_in_metadata)
            
            logger.info(f"총 {len(document_ids)}개의 고유 문서 발견")
            logger.info(f"발견된 문서 ID들: {list(document_ids)[:5]}...")  # 처음 5개만 로그
            return len(document_ids)
            
        except Exception as e:
            logger.error(f"문서 수 조회 오류: {e}")
            return 0

    def get_collection_info(self) -> Dict:
        """컬렉션 정보를 반환합니다."""
        try:
            info = self.client.get_collection(self.collection_name)
            total_points = info.points_count
            
            # 고유 문서 수 계산
            document_ids = set()
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True
            )
            
            for point in points:
                if point.payload:
                    # 직접 document_id 확인
                    document_id = point.payload.get("document_id")
                    if document_id:
                        document_ids.add(document_id)
                    else:
                        # metadata 안에 document_id가 있는지 확인
                        metadata = point.payload.get("metadata", {})
                        if isinstance(metadata, dict):
                            doc_id_in_metadata = metadata.get("document_id")
                            if doc_id_in_metadata:
                                document_ids.add(doc_id_in_metadata)
            
            return {
                "total_points": total_points,
                "total_documents": len(document_ids),
                "vector_size": info.config.params.vectors.get("text-dense", {}).get("size", "Unknown"),
                "distance_metric": info.config.params.vectors.get("text-dense", {}).get("distance", "Unknown")
            }
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 오류: {e}")
            return {"error": str(e)}

    def get_all_documents(self) -> Dict:
        """벡터스토어에 저장된 모든 문서 정보를 반환합니다."""
        try:
            # 모든 포인트의 document_id와 메타데이터를 가져옵니다
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True
            )
            
            documents = {}
            for point in points:
                if point.payload:
                    # 직접 document_id 확인
                    document_id = point.payload.get("document_id")
                    if not document_id:
                        # metadata 안에 document_id가 있는지 확인
                        metadata = point.payload.get("metadata", {})
                        if isinstance(metadata, dict):
                            document_id = metadata.get("document_id")
                    
                    if document_id:
                        if document_id not in documents:
                            # title 찾기 (직접 또는 metadata에서)
                            title = point.payload.get("title")
                            if not title:
                                metadata = point.payload.get("metadata", {})
                                if isinstance(metadata, dict):
                                    title = metadata.get("title", "Unknown")
                            
                            documents[document_id] = {
                                "title": title or "Unknown",
                                "metadata": {k: v for k, v in point.payload.items() 
                                           if k not in ["content", "document_id"]},
                                "chunk_count": 0,
                                "created_at": point.payload.get("created_at") or 
                                             (point.payload.get("metadata", {}) or {}).get("created_at")
                            }
                        documents[document_id]["chunk_count"] += 1
            
            return documents
            
        except Exception as e:
            logger.error(f"문서 목록 조회 오류: {e}")
            return {}
    
    def delete_document_chunks(self, document_id: str):
        """특정 문서의 모든 청크를 삭제합니다."""
        try:
            # 해당 문서의 모든 포인트 삭제
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True
            )
            
            point_ids_to_delete = []
            for point in points:
                if point.payload:
                    # 직접 document_id 확인
                    doc_id = point.payload.get("document_id")
                    if not doc_id:
                        # metadata 안에 document_id가 있는지 확인
                        metadata = point.payload.get("metadata", {})
                        if isinstance(metadata, dict):
                            doc_id = metadata.get("document_id")
                    
                    if doc_id == document_id:
                        point_ids_to_delete.append(point.id)
            
            if point_ids_to_delete:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids_to_delete
                )
                logger.info(f"문서 {document_id}의 {len(point_ids_to_delete)}개 청크가 삭제됨")
            else:
                logger.warning(f"문서 {document_id}를 찾을 수 없습니다")
                
        except Exception as e:
            logger.error(f"문서 삭제 오류: {e}")
            raise
    
    def clear_all_documents(self):
        """모든 문서를 삭제합니다."""
        try:
            # 컬렉션 삭제 후 재생성
            self.client.delete_collection(self.collection_name)
            self._setup_vector_store()
            logger.info("모든 문서가 삭제되었습니다")
        except Exception as e:
            logger.error(f"모든 문서 삭제 오류: {e}")
            raise
