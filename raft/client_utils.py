"""
이 코드는 OpenAI 클라이언트에 사용자 정의 HTTP 클라이언트를 전달하여
proxies 매개변수 문제를 해결합니다.
"""

# 이 파일을 client_utils.py로 저장하세요

from abc import ABC
from typing import Any, Dict, Optional, Union
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from openai import OpenAI, AzureOpenAI
import httpx
import logging
from os import environ, getenv
import time
from threading import Lock
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.identity import get_bearer_token_provider

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("client_utils")

def build_openai_client(**kwargs: Any) -> Union[OpenAI, AzureOpenAI]:
    """
    OpenAI 클라이언트 생성
    사용자 정의 HTTP 클라이언트를 사용하여 proxies 매개변수 문제 해결
    """
    logger.info("build_openai_client 함수 호출됨")
    
    # API 키 확인 (다양한 이름 지원)
    api_key = kwargs.get('api_key') or kwargs.get('openai_key') or getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("API 키를 찾을 수 없습니다.")
        raise ValueError("API 키를 찾을 수 없습니다. --openai_key 매개변수 또는 OPENAI_API_KEY 환경 변수를 통해 제공해주세요.")
    
    try:
        # 사용자 정의 HTTP 클라이언트 생성 (proxies 매개변수 없이)
        # limits 매개변수는 기본값 사용
        http_client = httpx.Client(timeout=60.0)  # 60초 타임아웃 설정
        logger.info("사용자 정의 HTTP 클라이언트 생성 성공")
        
        # Azure 사용 여부 확인
        if is_azure():
            # Azure OpenAI 클라이언트 생성
            endpoint = getenv("AZURE_OPENAI_ENDPOINT")
            deployment = getenv("AZURE_OPENAI_DEPLOYMENT")
            api_version = getenv("AZURE_OPENAI_API_VERSION") or "2024-02-15-preview"
            
            # Azure API 키 또는 기본 API 키 사용
            azure_key = getenv("AZURE_OPENAI_KEY") or api_key
            
            # 클라이언트 생성 - http_client 명시적 전달
            try:
                logger.info("Azure OpenAI 클라이언트 생성")
                return AzureOpenAI(
                    api_key=azure_key,
                    azure_endpoint=endpoint,
                    azure_deployment=deployment,
                    api_version=api_version,
                    http_client=http_client  # 사용자 정의 HTTP 클라이언트 전달
                )
            except Exception as e:
                logger.error(f"Azure OpenAI 클라이언트 생성 실패: {e}")
                raise
        else:
            # 표준 OpenAI 클라이언트 생성
            try:
                # 기본 매개변수
                client_args = {
                    "api_key": api_key,
                    "http_client": http_client  # 사용자 정의 HTTP 클라이언트 전달
                }
                
                # 추가 매개변수
                for key in ['organization', 'base_url']:
                    if key in kwargs and kwargs[key]:
                        client_args[key] = kwargs[key]
                    elif env_val := getenv(f"OPENAI_{key.upper()}"):
                        client_args[key] = env_val
                
                logger.info(f"OpenAI 클라이언트 생성: {list(client_args.keys())}")
                return OpenAI(**client_args)
            except Exception as e:
                logger.error(f"OpenAI 클라이언트 생성 실패: {e}")
                raise
    except Exception as e:
        logger.error(f"HTTP 클라이언트 생성 중 오류: {e}")
        # 최후의 방법: 표준 방식으로 생성 시도
        logger.info("기본 방식으로 OpenAI 클라이언트 생성 시도")
        try:
            return OpenAI(api_key=api_key)
        except Exception as e2:
            logger.error(f"기본 방식으로도 OpenAI 클라이언트 생성 실패: {e2}")
            raise

def build_langchain_embeddings(**kwargs: Any) -> OpenAIEmbeddings:
    """
    LangChain OpenAI 임베딩 클라이언트 생성
    """
    logger.info("build_langchain_embeddings 함수 호출됨")
    
    # API 키 확인 (다양한 이름 지원)
    api_key = (kwargs.get('api_key') or kwargs.get('openai_key') or 
              kwargs.get('openai_api_key') or getenv("OPENAI_API_KEY"))
    
    if not api_key:
        logger.error("임베딩용 API 키를 찾을 수 없습니다.")
        raise ValueError("임베딩용 API 키를 찾을 수 없습니다.")
    
    # 모델 확인
    model = kwargs.get('model') or getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-ada-002"
    
    try:
        # Azure 사용 여부 확인
        if is_azure():
            # Azure OpenAI 임베딩 클라이언트 생성
            endpoint = getenv("AZURE_OPENAI_ENDPOINT")
            api_version = getenv("AZURE_OPENAI_API_VERSION") or "2024-02-15-preview"
            
            # 배포 이름 결정 (임베딩 전용 또는 일반)
            embedding_deployment = getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            general_deployment = getenv("AZURE_OPENAI_DEPLOYMENT")
            deployment = embedding_deployment or general_deployment
            
            # Azure API 키 또는 기본 API 키 사용
            azure_key = getenv("AZURE_OPENAI_KEY") or api_key
            
            try:
                # 클라이언트 생성
                logger.info("Azure OpenAI 임베딩 클라이언트 생성")
                return AzureOpenAIEmbeddings(
                    azure_deployment=deployment,
                    openai_api_version=api_version,
                    azure_endpoint=endpoint,
                    openai_api_key=azure_key
                )
            except Exception as e:
                logger.warning(f"Azure OpenAI 임베딩 클라이언트 생성 실패, 대체 방법 시도: {e}")
                # LangChain에서는 매개변수 이름이 다를 수 있음
                return AzureOpenAIEmbeddings(
                    deployment=deployment,
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=azure_key
                )
        else:
            # 표준 OpenAI 임베딩 클라이언트 생성
            logger.info("OpenAI 임베딩 클라이언트 생성")
            # LangChain에서는 openai_api_key 이름이 필요할 수 있음
            return OpenAIEmbeddings(
                model=model,
                openai_api_key=api_key
            )
    except Exception as e:
        logger.error(f"임베딩 클라이언트 생성 실패: {e}")
        # 최후의 방법: 가능한 모든 키 이름 시도
        try:
            return OpenAIEmbeddings(
                model=model,
                api_key=api_key,
                openai_api_key=api_key
            )
        except Exception as e2:
            logger.error(f"대체 방법으로도 임베딩 클라이언트 생성 실패: {e2}")
            raise

def is_azure() -> bool:
    """
    Azure OpenAI 서비스 사용 여부 확인
    """
    return ("AZURE_OPENAI_ENDPOINT" in environ or 
            "AZURE_OPENAI_KEY" in environ or 
            "AZURE_OPENAI_AD_TOKEN" in environ)

class UsageStats:
    """토큰 사용량 통계 추적 클래스"""
    
    def __init__(self) -> None:
        self.start = time.time()
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.end = None
        self.duration = 0
        self.calls = 0

    def __add__(self, other: 'UsageStats') -> 'UsageStats':
        stats = UsageStats()
        stats.start = min(self.start, other.start) if self.start is not None and other.start is not None else (self.start or other.start)
        stats.end = max(self.end, other.end) if self.end is not None and other.end is not None else (self.end or other.end)
        stats.completion_tokens = self.completion_tokens + other.completion_tokens
        stats.prompt_tokens = self.prompt_tokens + other.prompt_tokens
        stats.total_tokens = self.total_tokens + other.total_tokens
        stats.duration = self.duration + other.duration
        stats.calls = self.calls + other.calls
        return stats

class StatsCompleter(ABC):
    """OpenAI API 호출 통계를 추적하는 래퍼 클래스"""
    
    def __init__(self, create_func):
        self.create_func = create_func
        self.stats = None
        self.lock = Lock()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # 'proxies' 매개변수 제거
        filtered_kwds = {k: v for k, v in kwds.items() if k != 'proxies'}
        
        # 제거된 매개변수가 있는 경우 로깅
        if len(filtered_kwds) != len(kwds):
            logger.warning("StatsCompleter에서 'proxies' 매개변수 제거")
        
        # API 호출
        response = self.create_func(*args, **filtered_kwds)
        
        # 통계 업데이트
        self.lock.acquire()
        try:
            if not self.stats:
                self.stats = UsageStats()
            self.stats.completion_tokens += response.usage.completion_tokens
            self.stats.prompt_tokens += response.usage.prompt_tokens
            self.stats.total_tokens += response.usage.total_tokens
            self.stats.calls += 1
            return response
        finally:
            self.lock.release()
    
    def get_stats_and_reset(self) -> UsageStats:
        """현재 통계를 반환하고 재설정"""
        self.lock.acquire()
        try:
            end = time.time()
            stats = self.stats
            if stats:
                stats.end = end
                stats.duration = end - self.stats.start
                self.stats = None
            return stats
        finally:
            self.lock.release()

class ChatCompleter(StatsCompleter):
    """채팅 완성 API 호출 래퍼"""
    
    def __init__(self, client):
        super().__init__(client.chat.completions.create)

class CompletionsCompleter(StatsCompleter):
    """텍스트 완성 API 호출 래퍼"""
    
    def __init__(self, client):
        super().__init__(client.completions.create)