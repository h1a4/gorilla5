# openai_patch.py - OpenAI 라이브러리 proxies 오류를 해결하기 위한 패치
# 범위를 OpenAI 네임스페이스로 엄격하게 제한
import logging
import os
import inspect
from functools import wraps

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openai_patch")

logger.info("OpenAI 라이브러리에 제한된 패치 적용 중...")

# 환경 변수에서 프록시 설정 제거
for env_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    if env_var in os.environ:
        logger.info(f"환경 변수 {env_var} 제거")
        del os.environ[env_var]

# 특정 클래스의 __init__ 메서드만 패치하는 함수
def patch_init(cls, name):
    if not cls.__module__.startswith('openai'):
        return False  # openai 패키지에 속하지 않으면 패치하지 않음
    
    try:
        original_init = cls.__init__
        
        @wraps(original_init)
        def patched_init(self, *args, **kwargs):
            # proxies 매개변수 제거
            if 'proxies' in kwargs:
                logger.info(f"{name}에서 'proxies' 매개변수 제거")
                del kwargs['proxies']
            
            # 원래 초기화 메서드 호출
            original_init(self, *args, **kwargs)
        
        cls.__init__ = patched_init
        return True
    except (AttributeError, TypeError) as e:
        logger.warning(f"{name} 패치 실패: {e}")
        return False

# 특정 모듈만 직접 패치
def patch_specific_modules():
    """
    특정 OpenAI 모듈만 직접 패치
    """
    try:
        # 1. OpenAI 클라이언트 패치
        import openai
        
        # OpenAI 클래스 패치
        if hasattr(openai, 'OpenAI'):
            if patch_init(openai.OpenAI, "openai.OpenAI"):
                logger.info("openai.OpenAI 패치 완료")
        
        # AzureOpenAI 클래스 패치
        if hasattr(openai, 'AzureOpenAI'):
            if patch_init(openai.AzureOpenAI, "openai.AzureOpenAI"):
                logger.info("openai.AzureOpenAI 패치 완료")
        
        # 2. Base Client 패치
        try:
            from openai._base_client import SyncHttpxClientWrapper, BaseClient
            
            if patch_init(SyncHttpxClientWrapper, "SyncHttpxClientWrapper"):
                logger.info("SyncHttpxClientWrapper 패치 완료")
                
            if patch_init(BaseClient, "BaseClient"):
                logger.info("BaseClient 패치 완료")
        except ImportError:
            logger.warning("Base Client 클래스를 가져올 수 없음")
        
        # 3. httpx Client 패치 (단, openai 내부에서 사용되는 경우에만)
        try:
            import httpx
            
            original_httpx_init = httpx.Client.__init__
            
            @wraps(original_httpx_init)
            def patched_httpx_init(self, *args, **kwargs):
                # proxies 매개변수가 있고 호출 스택에 openai가 있는 경우에만 제거
                if 'proxies' in kwargs and any('openai' in frame.filename for frame in inspect.stack()):
                    logger.info("openai에서 호출된 httpx.Client에서 'proxies' 매개변수 제거")
                    del kwargs['proxies']
                
                # 원래 초기화 메서드 호출
                original_httpx_init(self, *args, **kwargs)
            
            # httpx.Client 클래스의 초기화 메서드 교체
            httpx.Client.__init__ = patched_httpx_init
            logger.info("httpx.Client 패치 완료 (openai 호출일 경우에만 적용)")
            
            # AsyncClient도 비슷하게 패치
            if hasattr(httpx, 'AsyncClient'):
                original_async_init = httpx.AsyncClient.__init__
                
                @wraps(original_async_init)
                def patched_async_init(self, *args, **kwargs):
                    if 'proxies' in kwargs and any('openai' in frame.filename for frame in inspect.stack()):
                        logger.info("openai에서 호출된 httpx.AsyncClient에서 'proxies' 매개변수 제거")
                        del kwargs['proxies']
                    
                    original_async_init(self, *args, **kwargs)
                
                httpx.AsyncClient.__init__ = patched_async_init
                logger.info("httpx.AsyncClient 패치 완료 (openai 호출일 경우에만 적용)")
        except ImportError:
            logger.warning("httpx 모듈을 가져올 수 없음")
        
        # 4. 주요 API 리소스 클래스만 패치
        try:
            # Chat API 패치
            if hasattr(openai.resources, 'chat'):
                try:
                    from openai.resources.chat import Completions
                    if patch_init(Completions, "openai.resources.chat.Completions"):
                        logger.info("openai.resources.chat.Completions 패치 완료")
                except ImportError:
                    pass
            
            # Embeddings API 패치
            if hasattr(openai.resources, 'embeddings'):
                try:
                    from openai.resources.embeddings import Embeddings
                    if patch_init(Embeddings, "openai.resources.embeddings.Embeddings"):
                        logger.info("openai.resources.embeddings.Embeddings 패치 완료")
                except ImportError:
                    pass
        except AttributeError:
            logger.warning("리소스 클래스를 가져올 수 없음")
        
    except ImportError as e:
        logger.error(f"OpenAI 모듈을 가져올 수 없음: {e}")
    except Exception as e:
        logger.error(f"패치 중 오류 발생: {e}")

# 특정 모듈만 패치 실행
patch_specific_modules()

logger.info("OpenAI 라이브러리 패치 완료. 범위를 제한하여 다른 라이브러리에 영향이 없도록 했습니다.")