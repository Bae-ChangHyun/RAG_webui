import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage
    MCP_AVAILABLE = True
except ImportError as e:
    MCP_AVAILABLE = False
    logging.warning(f"MCP 도구 라이브러리가 설치되지 않았습니다: {e}")

logger = logging.getLogger(__name__)

class MCPService:
    """MCP(Model Context Protocol) 서비스를 관리하는 클래스"""
    
    def __init__(self, mcp_config_path: str = "mcp.json"):
        self.mcp_config_path = Path(mcp_config_path)
        self.client = None
        self.tools = []
        self.available = False
        self._load_mcp_client()
    
    def _load_mcp_client(self):
        """MCP 설정 파일을 읽고 클라이언트를 로드합니다."""
        try:
            if not MCP_AVAILABLE:
                logger.warning("MCP 도구 라이브러리가 사용 불가능합니다.")
                return
                
            if not self.mcp_config_path.exists():
                logger.warning(f"MCP 설정 파일을 찾을 수 없습니다: {self.mcp_config_path}")
                return
            
            with open(self.mcp_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # MCP 서버 설정 읽기
            servers = config.get('mcpServers', {}) if isinstance(config, dict) else config
            
            if not servers:
                logger.warning("MCP 서버 설정이 없습니다.")
                return
            
            # MultiServerMCPClient에서 사용할 수 있는 형태로 변환
            client_config = {}
            for server_name, server_config in servers.items():
                command = server_config.get('command', '')
                args = server_config.get('args', [])
                env = server_config.get('env', {})
                
                # stdio transport 형태로 변환
                client_config[server_name] = {
                    "command": command,
                    "args": args,
                    "transport": "stdio",
                    "env": env
                }
            
            # MultiServerMCPClient 생성
            self.client = MultiServerMCPClient(client_config)
            self.available = True
            logger.info(f"MCP 클라이언트 초기화 완료: {len(client_config)}개 서버")
                
        except Exception as e:
            logger.error(f"MCP 클라이언트 로드 실패: {e}")
            self.available = False
    
    async def get_all_tools(self) -> List[Any]:
        """모든 MCP 도구들을 반환합니다."""
        try:
            if not self.available or not self.client:
                return []
            
            if not self.tools:
                self.tools = await self.client.get_tools()
                logger.info(f"MCP 도구 로드 완료: {len(self.tools)}개")
                for tool in self.tools:
                    logger.info(f"- {tool.name}: {tool.description}")
            
            return self.tools
        except Exception as e:
            logger.error(f"MCP 도구 가져오기 실패: {e}")
            return []
    
    def is_available(self) -> bool:
        """MCP 서비스 사용 가능 여부를 반환합니다."""
        return self.available and MCP_AVAILABLE
    
    async def get_tool_names(self) -> List[str]:
        """사용 가능한 도구 이름들을 반환합니다."""
        tools = await self.get_all_tools()
        return [tool.name for tool in tools]
    
    def reload_tools(self):
        """MCP 도구들을 다시 로드합니다."""
        self.tools = []
        self._load_mcp_client()


class MCPLLMService:
    """MCP 도구를 사용하는 LLM 서비스"""
    
    def __init__(self, llm_service, mcp_service: MCPService):
        self.llm_service = llm_service
        self.mcp_service = mcp_service
        self.available = llm_service.is_available() and mcp_service.is_available()
    
    async def generate_tool_answer(self, question: str) -> Dict[str, Any]:
        """MCP 도구를 사용하여 답변을 생성합니다."""
        try:
            if not self.is_available():
                raise Exception("MCP LLM 서비스가 사용 불가능합니다.")
            
            # MCP 도구들 가져오기
            tools = await self.mcp_service.get_all_tools()
            
            if not tools:
                raise Exception("사용 가능한 MCP 도구가 없습니다.")
            
            # LLM 클라이언트 준비
            llm = self._get_langchain_llm()
            
            # langgraph의 react_agent 생성
            agent = create_react_agent(llm, tools)
            
            # 에이전트 실행
            result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
            
            # 결과에서 최종 답변 추출
            final_message = result["messages"][-1]
            answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            # 사용된 도구들 추출
            used_tools = []
            for message in result["messages"]:
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        used_tools.append(tool_call.get('name', ''))
                elif hasattr(message, 'name') and message.name:
                    used_tools.append(message.name)
            
            return {
                "answer": answer,
                "used_tools": list(set(used_tools)),  # 중복 제거
                "search_type": "mcp_tools",
                "tool_names": await self.mcp_service.get_tool_names()
            }
                
        except Exception as e:
            logger.error(f"MCP 도구 사용 답변 생성 중 오류 발생: {e}")
            raise
    
    def _get_langchain_llm(self):
        """LangChain 호환 LLM 객체를 반환합니다."""
        try:
            if self.llm_service.settings.provider.value == "openai":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=self.llm_service.settings.model_name,
                    api_key=self.llm_service.settings.api_key,
                    temperature=0.3
                )
            elif self.llm_service.settings.provider.value == "vllm":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=self.llm_service.settings.model_name,
                    base_url=self.llm_service.settings.vllm_base_url,
                    api_key="dummy",
                    temperature=0.3
                )
            elif self.llm_service.settings.provider.value == "ollama":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=self.llm_service.settings.model_name,
                    base_url=self.llm_service.settings.ollama_base_url,
                    api_key="dummy",
                    temperature=0.3
                )
            elif self.llm_service.settings.provider.value == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=self.llm_service.settings.model_name,
                    google_api_key=self.llm_service.settings.api_key,
                    temperature=0.3
                )
            else:
                raise ValueError(f"지원하지 않는 LLM 제공자: {self.llm_service.settings.provider}")
                
        except Exception as e:
            logger.error(f"LangChain LLM 생성 실패: {e}")
            raise
    
    def is_available(self) -> bool:
        """MCP LLM 서비스 사용 가능 여부를 반환합니다."""
        return self.available
