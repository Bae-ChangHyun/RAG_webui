from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uuid
from datetime import datetime
from typing import List, Optional
import logging

from .models import (
    DocumentCreate, DocumentResponse, SearchRequest, 
    SearchResponse, DocumentCountResponse, ChunkResponse,
    QARequest, QAResponse, SystemSettings, SettingsUpdateRequest,
    VectorStoreResponse, VectorStoreDocument
)
from .parser import DocumentParser
from .chunker import TextChunker
from .database import QdrantDatabase
from .retriever import DocumentRetriever
from .llm import LLMService
from .settings import SettingsManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document System",
    description="FastAPI와 Qdrant를 이용한 RAG 시스템",
    version="1.0.0"
)

# 정적 파일과 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 전역 인스턴스
settings_manager = SettingsManager()
parser = DocumentParser(settings_manager.get_parser_settings())
chunker = TextChunker(settings_manager.get_chunking_settings())
db = QdrantDatabase(embedding_model_name=settings_manager.get_embedding_settings().model_name)
llm_service = LLMService(settings_manager.get_llm_settings())
retriever = DocumentRetriever(db = db,llm_service=llm_service)
retrieval_settings = settings_manager.get_retrieval_settings()

@app.get("/api/status")
async def api_status():
    """API 상태 확인"""
    return {"message": "RAG Document System API", "status": "running"}

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    folder_path: Optional[str] = None,
    folder_name: Optional[str] = None
):
    """파일을 업로드하고 벡터 DB에 저장합니다."""
    try:
        content = await file.read()
        
        doc_title = title or file.filename or "Untitled Document"
        
        try:
            parsed_content = await parser.parse_file(content, file.filename)
        except Exception as e:
            logger.error(f"파일 파싱 오류: {e}")
            raise HTTPException(status_code=400, detail=f"파일 파싱 실패: {str(e)}")
        
        document_id = str(uuid.uuid4())

        metadata = {
            "document_id": document_id,
            "title": doc_title,
            "filename": file.filename,
            "file_size": len(content),
            "content_type": file.content_type,
            "created_at": datetime.now().isoformat(),
            "folder_path": folder_path,
            "folder_name": folder_name
        }
        
        # 텍스트 청킹
        chunks = chunker.chunk_text(parsed_content, metadata)
        
        # Qdrant에 저장
        db.add_documents(chunks, metadata)
        
        logger.info(f"문서 '{doc_title}' 저장 완료: {len(chunks)}개 청크")
        
        return DocumentResponse(
            id=document_id,
            title=doc_title,
            metadata=metadata,
            created_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"문서 업로드 오류: {e}")
        raise HTTPException(status_code=500, detail=f"문서 업로드 실패: {str(e)}")

@app.post("/documents/text", response_model=DocumentResponse)
async def create_text_document(document: DocumentCreate):
    """텍스트 문서를 생성하고 저장합니다."""
    try:
        # 문서 ID 생성
        document_id = str(uuid.uuid4())
        
        # 메타데이터 설정
        metadata = {
            "document_id": document_id,
            **document.metadata,
            "title": document.title,
            "created_at": datetime.now().isoformat(),
            "content_type": "text/plain"
        }
        
        # 텍스트 청킹
        chunks = chunker.chunk_text(document.content, metadata)
        
        # Qdrant에 저장
        db.add_documents(chunks, metadata)
        
        logger.info(f"텍스트 문서 '{document.title}' 저장 완료: {len(chunks)}개 청크")
        
        return DocumentResponse(
            id=document_id,
            title=document.title,
            metadata=metadata,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"텍스트 문서 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"문서 생성 실패: {str(e)}")

@app.post("/search")
async def search_documents(request: SearchRequest):
    """문서를 검색합니다."""
    try:
        # 검색 전략 결정 - strategy 필드를 우선적으로 사용
        strategy = "hybrid"  # 기본값
        if request.strategy:
            strategy = request.strategy

        
        # 문서 검색
        results = retriever.search_documents(
            query=request.query,
            limit=request.limit,
            threshold=request.threshold,
            strategy=strategy
        )
        
        logger.info(f"검색 결과: {len(results['results'])}개 문서 발견")
        
        return {
            "success": True,
            "results": results["results"],
            "total_count": len(results["results"])
        }
            
    except Exception as e:
        logger.error(f"문서 검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa")
async def question_answer(request: SearchRequest):
    """질문에 대한 답변을 생성합니다."""
    try:
        # 질문 필드 확인
        if not request.question and not request.query:
            raise HTTPException(status_code=400, detail="질문이 필요합니다.")
        
        question = request.question or request.query
        
        # 검색 전략 결정 - strategy 필드를 우선적으로 사용
        strategy = "hybrid"  # 기본값
        if request.strategy: strategy = request.strategy
        
        # 문서 검색
        search_results = retriever.search_documents(
            query=question,
            limit=request.limit,
            threshold=request.threshold,
            strategy=strategy
        )
        
        logger.info(f"검색 결과: {len(search_results['results'])}개 문서 발견")
        
        # 컨텍스트 추출
        context_chunks = [
            {
                "content": result["content"],
                "metadata": result["metadata"],
                "score": result["score"],
                "document_id": result["document_id"],
                "chunk_index": result["chunk_index"],
                "unique_chunk_index": result.get("unique_chunk_index", f"{result['document_id']}#{result['metadata'].get('chunk_index', 0)}")
            }
            for result in search_results["results"]
        ]
        
        # LLM을 사용하여 답변 생성
        if context_chunks:
            answer = llm_service.generate_answer(
                question=question,
                context_chunks=context_chunks
            )
        else:
            answer = llm_service.generate_general_response(question)
        
        return {
            "success": True,
            "answer": answer,
            "context_chunks": context_chunks,
            "results": search_results["results"],
            "total_count": len(search_results["results"])
        }
            
    except Exception as e:
        logger.error(f"질문 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/count", response_model=DocumentCountResponse)
async def get_document_count():
    """저장된 문서의 총 개수를 조회합니다."""
    try:
        count = db.get_document_count()
        return DocumentCountResponse(total_documents=count)
        
    except Exception as e:
        logger.error(f"문서 개수 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"문서 개수 조회 실패: {str(e)}")

@app.get("/documents/statistics")
async def get_document_statistics():
    """문서 통계 정보를 조회합니다."""
    try:
        stats = retriever.get_document_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """특정 문서를 삭제합니다."""
    try:
        db.delete_document_chunks(document_id)
        logger.info(f"문서 {document_id} 삭제 완료")
        
        return {"message": f"문서 {document_id}가 성공적으로 삭제되었습니다."}
        
    except Exception as e:
        logger.error(f"문서 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=f"문서 삭제 실패: {str(e)}")

@app.get("/documents/{document_id}/content")
async def get_document_full_content(document_id: str):
    """특정 문서의 전체 내용을 조회합니다."""
    try:
        # Qdrant에서 해당 문서의 모든 청크를 직접 가져옵니다
        points, _ = db.client.scroll(
            collection_name=db.collection_name,
            limit=10000,
            with_payload=True
        )
        # 필터링 및 청크 수집
        chunks = []
        for point in points:
            payload = point.payload or {}
            md = payload.get("metadata", {})
            if md.get("document_id") == document_id:
                chunks.append({
                    "content": payload.get("content", ""),
                    "metadata": md
                })
        if not chunks:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
        # 정렬 및 합치기
        chunks = sorted(chunks, key=lambda x: x["metadata"].get("chunk_index", 0))
        full_content = "\n\n".join([c["content"] for c in chunks])
        metadata = chunks[0]["metadata"]
        
        # 청크별 정보도 함께 반환
        chunks_info = []
        for chunk in chunks:
            chunks_info.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "unique_chunk_index": f"{document_id}#{chunk['metadata'].get('chunk_index', 0)}"
            })
        
        return {
            "document_id": document_id,
            "title": metadata.get("title", "Unknown"),
            "content": full_content,
            "metadata": metadata,
            "chunk_count": len(chunks),
            "chunks": chunks_info  # 청크별 상세 정보 추가
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"문서 내용 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"문서 내용 조회 실패: {str(e)}")

@app.get("/documents/{document_id}/search")
async def search_in_document(
    document_id: str,
    query: str,
    limit: int = 5
):
    """특정 문서 내에서만 검색합니다."""
    try:
        results = retriever.search_by_document_id(
            document_id=document_id,
            query=query,
            limit=limit
        )
        
        return SearchResponse(
            query=results["query"],
            results=[
                ChunkResponse(
                    content=result["content"],
                    metadata=result["metadata"],
                    score=result["score"],
                    chunk_index=result.get("chunk_index"),
                    document_id=result.get("document_id")
                )
                for result in results["results"]
            ],
            total_count=results["total_count"]
        )
        
    except Exception as e:
        logger.error(f"문서별 검색 오류: {e}")
        raise HTTPException(status_code=500, detail=f"문서별 검색 실패: {str(e)}")

@app.get("/health")
async def health_check():
    """시스템 상태 체크"""
    try:
        # Qdrant 연결 확인
        collection_info = db.get_collection_info()
        
        return {
            "status": "healthy",
            "qdrant_connected": True,
            "collection_info": collection_info,
            "supported_formats": parser.get_supported_formats()
        }
        
    except Exception as e:
        logger.error(f"헬스체크 오류: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/settings", response_model=SystemSettings)
async def get_settings():
    """현재 시스템 설정을 조회합니다."""
    try:
        return settings_manager.get_settings()
    except Exception as e:
        logger.error(f"설정 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"설정 조회 실패: {str(e)}")

@app.post("/settings")
async def update_settings(settings_update: SettingsUpdateRequest):
    """시스템 설정을 업데이트합니다."""
    try:
        global parser, chunker, llm_service, retrieval_settings, db, retriever
        
        # 임베딩 모델 변경 확인
        embedding_changed = False
        if settings_update.embedding:
            current_embedding = settings_manager.get_embedding_settings()
            if current_embedding.model_name != settings_update.embedding.model_name:
                embedding_changed = True
                # 임베딩 설정 업데이트
                settings_manager.update_embedding_settings(settings_update.embedding)
        
        # 청킹 설정 업데이트
        if settings_update.chunking:
            settings_manager.update_chunking_settings(settings_update.chunking)
            chunker.update_settings(settings_update.chunking)
        
        # LLM 설정 업데이트
        if settings_update.llm:
            settings_manager.update_llm_settings(settings_update.llm)
            llm_service.update_settings(settings_update.llm)
        
        # 파서 설정 업데이트
        if settings_update.parser:
            settings_manager.update_parser_settings(settings_update.parser)
            parser.update_settings(settings_update.parser)
        
        # 검색 설정 업데이트
        if settings_update.retrieval:
            settings_manager.update_retrieval_settings(settings_update.retrieval)
            retrieval_settings = settings_update.retrieval
        
        response_data = {"message": "설정이 성공적으로 업데이트되었습니다."}
        
        # 임베딩 모델이 변경된 경우 응답에 표시
        if embedding_changed:
            response_data["embedding_changed"] = True
            response_data["message"] = "설정이 업데이트되었습니다. 임베딩 모델 변경으로 인해 벡터스토어 초기화가 필요합니다."
        
        return response_data
        
    except Exception as e:
        logger.error(f"설정 업데이트 오류: {e}")
        raise HTTPException(status_code=500, detail=f"설정 업데이트 실패: {str(e)}")

@app.get("/vectorstore/documents", response_model=VectorStoreResponse)
async def get_vectorstore_documents():
    """벡터스토어에 저장된 모든 문서를 조회합니다."""
    try:
        documents = db.get_all_documents()
        
        document_list = []
        for doc_id, doc_info in documents.items():
            document_list.append(VectorStoreDocument(
                id=doc_id,
                title=doc_info.get("title", "Unknown"),
                metadata=doc_info.get("metadata", {}),
                chunk_count=doc_info.get("chunk_count", 0),
                created_at=doc_info.get("created_at", datetime.now())
            ))
        
        return VectorStoreResponse(
            documents=document_list,
            total_count=len(document_list)
        )
        
    except Exception as e:
        logger.error(f"문서 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"문서 목록 조회 실패: {str(e)}")

@app.delete("/vectorstore/clear")
async def clear_vectorstore():
    """벡터스토어의 모든 문서를 삭제합니다."""
    try:
        db.clear_all_documents()
        logger.info("벡터스토어의 모든 문서가 삭제되었습니다.")
        
        return {"message": "벡터스토어의 모든 문서가 성공적으로 삭제되었습니다."}
        
    except Exception as e:
        logger.error(f"벡터스토어 초기화 오류: {e}")
        raise HTTPException(status_code=500, detail=f"벡터스토어 초기화 실패: {str(e)}")

@app.post("/vectorstore/reinitialize")
async def reinitialize_vectorstore():
    """임베딩 모델 변경 후 벡터스토어를 재초기화합니다."""
    try:
        global db, retriever
        
        # 현재 문서들을 삭제
        db.clear_all_documents()
        
        # 새로운 임베딩 모델로 데이터베이스 재초기화
        from .database import QdrantDatabase
        db = QdrantDatabase(embedding_model_name=settings_manager.get_embedding_settings().model_name)
        
        # retriever도 재초기화
        retriever = DocumentRetriever(llm_service)
        
        logger.info("벡터스토어가 새로운 임베딩 모델로 재초기화되었습니다.")
        
        return {"message": "벡터스토어가 새로운 임베딩 모델로 성공적으로 재초기화되었습니다."}
        
    except Exception as e:
        logger.error(f"벡터스토어 재초기화 오류: {e}")
        raise HTTPException(status_code=500, detail=f"벡터스토어 재초기화 실패: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

