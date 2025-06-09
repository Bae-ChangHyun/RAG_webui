// RAG Document System JavaScript

// 전역 상수
const API_BASE_URL = '';
const TOAST_DURATION = 3000;

// 유틸리티 함수들
class Utils {
    static formatTimestamp(date = new Date()) {
        return date.toLocaleTimeString('ko-KR', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }

    static formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    static showToast(message, type = 'info') {
        const toast = document.createElement('div');
        const bgColor = {
            'success': 'bg-green-500',
            'error': 'bg-red-500',
            'warning': 'bg-yellow-500',
            'info': 'bg-blue-500'
        }[type] || 'bg-blue-500';

        toast.className = `toast ${bgColor} text-white px-6 py-3 rounded-lg shadow-lg`;
        toast.textContent = message;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'toastSlideOut 0.3s ease-in forwards';
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 300);
        }, TOAST_DURATION);
    }

    static async request(url, options = {}) {
        try {
            const response = await fetch(API_BASE_URL + url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
                throw new Error(error.detail || `Request failed with status ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    static scrollToBottom(element) {
        element.scrollTop = element.scrollHeight;
    }

    static copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            Utils.showToast('클립보드에 복사되었습니다', 'success');
        }).catch(() => {
            Utils.showToast('복사에 실패했습니다', 'error');
        });
    }
}

// 설정 관리자
class SettingsManager {
    constructor() {
        this.defaultSettings = {
            chunking: {
                method: 'token_sentence',
                chunk_size: 512,
                overlap: 50,
                unit: 'token'
            },
            llm: {
                provider: 'openai',
                model_name: 'gpt-3.5-turbo',
                api_key: '',
                api_url: '',
                temperature: 0.1
            },
            parser: {
                parser_type: 'docling'
            }
        };
        this.currentSettings = { ...this.defaultSettings };
    }

    async loadSettings() {
        try {
            const settings = await Utils.request('/settings');
            this.currentSettings = settings;
            return settings;
        } catch (error) {
            console.error('설정 로드 실패:', error);
            Utils.showToast('설정을 불러오는데 실패했습니다', 'error');
            return this.defaultSettings;
        }
    }

    async saveSettings(settings) {
        try {
            await Utils.request('/settings', {
                method: 'POST',
                body: JSON.stringify(settings)
            });
            this.currentSettings = { ...this.currentSettings, ...settings };
            Utils.showToast('설정이 저장되었습니다', 'success');
            return true;
        } catch (error) {
            console.error('설정 저장 실패:', error);
            Utils.showToast('설정 저장에 실패했습니다: ' + error.message, 'error');
            return false;
        }
    }

    getSettings() {
        return { ...this.currentSettings };
    }
}

// 문서 관리자
class DocumentManager {
    constructor() {
        this.documents = [];
    }

    async uploadFile(file, title = '') {
        const formData = new FormData();
        formData.append('file', file);
        if (title) {
            formData.append('title', title);
        }

        try {
            const response = await fetch('/documents/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
                throw new Error(error.detail);
            }

            const result = await response.json();
            Utils.showToast(`파일 "${result.title}"이 업로드되었습니다`, 'success');
            return result;
        } catch (error) {
            console.error('파일 업로드 실패:', error);
            Utils.showToast('파일 업로드에 실패했습니다: ' + error.message, 'error');
            throw error;
        }
    }

    async createTextDocument(title, content) {
        try {
            const result = await Utils.request('/documents/text', {
                method: 'POST',
                body: JSON.stringify({
                    title,
                    content,
                    metadata: {}
                })
            });

            Utils.showToast(`문서 "${result.title}"이 생성되었습니다`, 'success');
            return result;
        } catch (error) {
            console.error('텍스트 문서 생성 실패:', error);
            Utils.showToast('문서 생성에 실패했습니다: ' + error.message, 'error');
            throw error;
        }
    }

    async getDocumentCount() {
        try {
            const result = await Utils.request('/documents/count');
            return result.total_documents;
        } catch (error) {
            console.error('문서 개수 조회 실패:', error);
            return 0;
        }
    }

    async getVectorStoreDocuments() {
        try {
            const result = await Utils.request('/vectorstore/documents');
            this.documents = result.documents;
            return result;
        } catch (error) {
            console.error('문서 목록 조회 실패:', error);
            Utils.showToast('문서 목록을 불러오는데 실패했습니다', 'error');
            return { documents: [], total_count: 0 };
        }
    }

    async deleteDocument(documentId) {
        try {
            await Utils.request(`/documents/${documentId}`, {
                method: 'DELETE'
            });
            Utils.showToast('문서가 삭제되었습니다', 'success');
            return true;
        } catch (error) {
            console.error('문서 삭제 실패:', error);
            Utils.showToast('문서 삭제에 실패했습니다: ' + error.message, 'error');
            return false;
        }
    }

    async clearAllDocuments() {
        try {
            await Utils.request('/vectorstore/clear', {
                method: 'DELETE'
            });
            this.documents = [];
            Utils.showToast('모든 문서가 삭제되었습니다', 'success');
            return true;
        } catch (error) {
            console.error('벡터스토어 초기화 실패:', error);
            Utils.showToast('문서 삭제에 실패했습니다: ' + error.message, 'error');
            return false;
        }
    }
}

// 검색 및 QA 관리자
class SearchManager {
    constructor() {
        this.searchHistory = [];
    }

    async askQuestion(question, options = {}) {
        const defaultOptions = {
            limit: 5,
            threshold: 0.3,
            use_hybrid: true
        };

        const requestData = {
            question,
            ...defaultOptions,
            ...options
        };

        try {
            const result = await Utils.request('/qa', {
                method: 'POST',
                body: JSON.stringify(requestData)
            });

            // 검색 기록에 추가
            this.searchHistory.unshift({
                question,
                timestamp: new Date(),
                success: result.success
            });

            // 최대 50개까지만 보관
            if (this.searchHistory.length > 50) {
                this.searchHistory = this.searchHistory.slice(0, 50);
            }

            return result;
        } catch (error) {
            console.error('질문 처리 실패:', error);
            throw error;
        }
    }

    async searchDocuments(query, options = {}) {
        const defaultOptions = {
            limit: 5,
            threshold: 0.7,
            use_hybrid: true
        };

        const requestData = {
            query,
            ...defaultOptions,
            ...options
        };

        try {
            const result = await Utils.request('/search', {
                method: 'POST',
                body: JSON.stringify(requestData)
            });

            return result;
        } catch (error) {
            console.error('문서 검색 실패:', error);
            throw error;
        }
    }

    getSearchHistory() {
        return [...this.searchHistory];
    }

    clearSearchHistory() {
        this.searchHistory = [];
        Utils.showToast('검색 기록이 삭제되었습니다', 'info');
    }
}

// UI 관리자
class UIManager {
    constructor() {
        this.searchManager = new SearchManager();
        this.currentResults = [];
        this.isLoading = false;
    }

    async handleQuestion(question, options = {}) {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.showLoading();
        
        try {
            const result = await this.searchManager.askQuestion(question, options);
            this.displayResults(result);
        } catch (error) {
            this.showError(error);
        } finally {
            this.isLoading = false;
            this.hideLoading();
        }
    }

    async handleSearch(query, options = {}) {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.showLoading();
        
        try {
            const result = await this.searchManager.searchDocuments(query, options);
            this.displayResults(result);
        } catch (error) {
            this.showError(error);
        } finally {
            this.isLoading = false;
            this.hideLoading();
        }
    }

    displayResults(data) {
        const resultsContainer = document.getElementById('results');
        resultsContainer.innerHTML = '';

        console.log('Response data:', data);

        if (data.error) {
            this.showError(data.error);
            return;
        }

        // 답변 표시
        if (data.answer) {
            const answerSection = document.createElement('div');
            answerSection.className = 'answer-section';
            answerSection.innerHTML = `
                <h3>답변</h3>
                <div class="answer-content">${data.answer}</div>
            `;
            resultsContainer.appendChild(answerSection);
        }

        // 참고 문서(컨텍스트) 표시
        let contextItems = [];
        if (Array.isArray(data)) {
            contextItems = data;
        } else if (data.context) {
            contextItems = Array.isArray(data.context) ? data.context : [data.context];
        }

        if (contextItems.length > 0) {
            const contextSection = document.createElement('div');
            contextSection.className = 'context-section';
            contextSection.innerHTML = '<h4>참고 문서</h4>';
            
            contextItems.forEach((chunk, idx) => {
                const item = document.createElement('div');
                item.className = 'context-item';
                const metadata = chunk.metadata || {};
                item.innerHTML = `
                    <div style="font-size:12px;color:#888;">
                        <b>문서:</b> ${metadata.title || '-'} 
                        <b>Score:</b> ${(chunk.score * 100).toFixed(2)}%
                    </div>
                    <div style="margin:4px 0 12px 0;">${chunk.content}</div>
                `;
                contextSection.appendChild(item);
            });
            resultsContainer.appendChild(contextSection);
        }

        // 검색 결과 표시 (원하면 아래도 유지)
        if (data.results && data.results.length > 0) {
            const searchSection = document.createElement('div');
            searchSection.className = 'search-results-section';
            searchSection.innerHTML = `
                <h3>검색 결과</h3>
                <div class="search-results-list">
                    ${data.results.map(result => `
                        <div class="search-result-item">
                            <div class="result-header">
                                <span class="document-id">문서 ID: ${result.document_id}</span>
                                <span class="similarity-score">유사도: ${(result.score * 100).toFixed(2)}%</span>
                            </div>
                            <div class="result-content">${result.content}</div>
                        </div>
                    `).join('')}
                </div>
            `;
            resultsContainer.appendChild(searchSection);
        } else if (!data.answer && (!data.context || data.context.length === 0)) {
            // 아무것도 없을 때
            const noResultsSection = document.createElement('div');
            noResultsSection.className = 'no-results-section';
            noResultsSection.innerHTML = `
                <div class="no-results-message">
                    검색 결과가 없습니다.
                </div>
            `;
            resultsContainer.appendChild(noResultsSection);
        }
    }

    showLoading() {
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'loading';
        loadingDiv.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        loadingDiv.innerHTML = `
            <div class="bg-white p-4 rounded-lg shadow-lg">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
                <p class="mt-2 text-gray-700">처리 중...</p>
            </div>
        `;
        document.body.appendChild(loadingDiv);
    }

    hideLoading() {
        const loadingDiv = document.getElementById('loading');
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }

    showError(error) {
        Utils.showToast(error.message || '오류가 발생했습니다.', 'error');
    }
}

// 파일 드래그 앤 드롭 핸들러
class FileDropHandler {
    constructor(dropZone, onFileDrop) {
        this.dropZone = dropZone;
        this.onFileDrop = onFileDrop;
        this.init();
    }

    init() {
        this.dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        this.dropZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.dropZone.addEventListener('drop', this.handleDrop.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        this.dropZone.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        if (!this.dropZone.contains(e.relatedTarget)) {
            this.dropZone.classList.remove('dragover');
        }
    }

    handleDrop(e) {
        e.preventDefault();
        this.dropZone.classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            this.onFileDrop(files[0]); // 첫 번째 파일만 처리
        }
    }
}

// Alpine.js용 전역 함수들
window.Utils = Utils;
window.SettingsManager = SettingsManager;
window.DocumentManager = DocumentManager;
window.SearchManager = SearchManager;
window.FileDropHandler = FileDropHandler;
window.UIManager = UIManager;
