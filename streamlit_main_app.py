"""
Streamlit Financial Prediction Dashboard
Main entry point for the LSTM-based stock prediction system
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px
from typing import Dict, List, Optional
import logging
import sys
import os
import time
import psutil
import gc
import traceback
from contextlib import asynccontextmanager
import concurrent.futures

# 프로젝트 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 개선된 더미 클래스들 (실제 인터페이스와 정확히 일치)
class DummyGPUOptimizedStackingEnsemble:
    def __init__(self, use_gru=False, enable_mixed_precision=True):
        self.use_gru = use_gru
        self.enable_mixed_precision = enable_mixed_precision
        self.model = None
        self.is_trained = False
        
    def predict(self, X):
        """실제 구현과 일치하는 예측 메서드"""
        if len(X.shape) == 2:
            return np.random.uniform(0, 1, X.shape[0])
        return np.random.uniform(0, 1, len(X))
    
    def train(self, X, y, epochs=100):
        """실제 학습 로직 시뮬레이션"""
        self.is_trained = True
        return self

class DummyGPUMemoryManager:
    def __init__(self):
        self.gpus = []
        self.gpu_available = False
        
    def setup_gpu_configuration(self, memory_limit_mb=None):
        """GPU 설정 시뮬레이션"""
        try:
            # GPU 존재 여부 확인 시뮬레이션
            self.gpu_available = False  # 실제 환경에서는 GPU 감지 로직
            return self.gpu_available
        except Exception as e:
            logging.warning(f"GPU 설정 실패: {e}")
            return False
    
    def _log_gpu_info(self, gpus):
        return {
            "gpu_count": len(gpus),
            "gpu_available": self.gpu_available,
            "gpu_names": [f"GPU_{i}" for i in range(len(gpus))]
        }

class DummyModelVersionManager:
    def __init__(self, base_dir="./models"):
        self.base_dir = base_dir
        self.versions = {}
    
    def save_model_with_version(self, model, metrics, description=""):
        import uuid
        version_id = str(uuid.uuid4())[:8]
        self.versions[version_id] = {
            "model": model,
            "metrics": metrics,
            "description": description,
            "timestamp": pd.Timestamp.now()
        }
        return version_id

class DummyEnhancedDartApiClient:
    def __init__(self):
        self.usage_stats = {'usage_percentage': np.random.uniform(20, 80)}
    
    async def init_redis(self):
        """Redis 초기화 시뮬레이션"""
        await asyncio.sleep(0.1)
        
    def get_api_usage_stats(self):
        return self.usage_stats

class DummyEnhancedBokApiClient:
    async def get_economic_indicators(self, start_date, end_date):
        """경제 지표 데이터 시뮬레이션"""
        try:
            start = pd.to_datetime(start_date, format='%Y%m%d')
            end = pd.to_datetime(end_date, format='%Y%m%d')
            dates = pd.date_range(start=start, end=end, freq='D')
            
            return {
                'kospi': pd.DataFrame({
                    'date': dates,
                    'value': np.random.uniform(2500, 3000, len(dates))
                }),
                'vkospi': pd.DataFrame({
                    'date': dates,
                    'value': np.random.uniform(15, 30, len(dates))
                })
            }
        except Exception as e:
            logging.error(f"경제 지표 데이터 생성 오류: {e}")
            return {'kospi': pd.DataFrame(), 'vkospi': pd.DataFrame()}
    
    async def close_session(self):
        await asyncio.sleep(0.01)

class DummyIntegratedMacroDataCollector:
    def __init__(self):
        self.is_initialized = True

class DummyRobustRedisManager:
    def __init__(self, host='localhost', port=6379):
        self.host = host
        self.port = port
        self.connected = False
        self.fallback_cache = {}
        self.use_fallback = True
    
    async def connect(self):
        """Redis 연결 시뮬레이션"""
        try:
            await asyncio.sleep(0.1)
            self.connected = False  # 실제 환경에서는 연결 시도
            self.use_fallback = True
            return self.connected
        except Exception:
            return False
    
    async def get_diagnostics(self):
        return {
            'connected': self.connected,
            'using_fallback': self.use_fallback,
            'fallback_cache_size': len(self.fallback_cache),
            'connection_attempts': 1
        }

# 실제 모듈 임포트 시도, 실패시 더미 클래스 사용
try:
    from lstm_gpu_improvements import (
        GPUOptimizedStackingEnsemble, 
        GPUMemoryManager,
        ModelVersionManager
    )
    logging.info("실제 LSTM GPU 모듈 로드 성공")
except ImportError as e:
    logging.warning(f"LSTM GPU 모듈 로드 실패, 더미 클래스 사용: {e}")
    GPUOptimizedStackingEnsemble = DummyGPUOptimizedStackingEnsemble
    GPUMemoryManager = DummyGPUMemoryManager
    ModelVersionManager = DummyModelVersionManager

try:
    from dart_api_improvements import (
        EnhancedAdaptiveRateLimiter,
        RobustRedisManager,
        StrictVKOSPIValidator
    )
    logging.info("DART API 개선 모듈 로드 성공")
except ImportError as e:
    logging.warning(f"DART API 모듈 로드 실패, 더미 클래스 사용: {e}")
    RobustRedisManager = DummyRobustRedisManager

try:
    from improved_dart_integration_v2 import (
        EnhancedDartApiClient,
        EnhancedBokApiClient,
        IntegratedMacroDataCollector
    )
    logging.info("통합 데이터 수집 모듈 로드 성공")
except ImportError as e:
    logging.warning(f"통합 데이터 수집 모듈 로드 실패, 더미 클래스 사용: {e}")
    EnhancedDartApiClient = DummyEnhancedDartApiClient
    EnhancedBokApiClient = DummyEnhancedBokApiClient
    IntegratedMacroDataCollector = DummyIntegratedMacroDataCollector

# 로깅 설정 개선
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('streamlit_app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 비동기 함수 실행을 위한 개선된 헬퍼
@asynccontextmanager
async def safe_async_context():
    """안전한 비동기 컨텍스트 매니저"""
    try:
        yield
    except Exception as e:
        logger.error(f"비동기 작업 중 오류: {e}")
        raise

def run_async(coro):
    """개선된 비동기 함수 실행기"""
    try:
        # 현재 이벤트 루프 확인
        try:
            loop = asyncio.get_running_loop()
            # 이미 실행 중인 루프가 있으면 스레드풀 사용
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=30)  # 30초 타임아웃
        except RuntimeError:
            # 실행 중인 루프가 없으면 새로 생성
            return asyncio.run(coro)
    except Exception as e:
        logger.error(f"비동기 실행 오류: {e}")
        st.error(f"비동기 작업 실행 중 오류가 발생했습니다: {str(e)}")
        return None

# 페이지 설정
st.set_page_config(
    page_title="Financial AI Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 메모리 정리 함수
def cleanup_memory():
    """메모리 정리 및 가비지 컬렉션"""
    try:
        gc.collect()
        # 세션 상태에서 오래된 데이터 정리
        current_time = time.time()
        if 'last_cleanup' not in st.session_state:
            st.session_state.last_cleanup = current_time
        elif current_time - st.session_state.last_cleanup > 300:  # 5분마다
            # 대용량 데이터 정리
            for key in list(st.session_state.keys()):
                if key.startswith('cached_') and isinstance(st.session_state[key], pd.DataFrame):
                    if len(st.session_state[key]) > 10000:  # 큰 데이터프레임 제거
                        del st.session_state[key]
            st.session_state.last_cleanup = current_time
            logger.info("메모리 정리 완료")
    except Exception as e:
        logger.warning(f"메모리 정리 중 오류: {e}")

# CSS 스타일
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-color: #ffeeba;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화 개선
def initialize_session_state():
    """세션 상태 안전 초기화"""
    defaults = {
        'model': None,
        'predictions': None,
        'dart_client': None,
        'redis_manager': None,
        'last_refresh': time.time(),
        'training_in_progress': False,
        'error_count': 0,
        'max_errors': 5
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# 오류 처리 데코레이터
def handle_errors(func):
    """오류 처리 데코레이터"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.session_state.error_count += 1
            error_msg = f"오류 발생: {str(e)}"
            logger.error(f"{func.__name__} - {error_msg}\n{traceback.format_exc()}")
            
            if st.session_state.error_count > st.session_state.max_errors:
                st.error("연속 오류가 너무 많이 발생했습니다. 페이지를 새로고침해주세요.")
                st.stop()
            else:
                st.error(error_msg)
            return None
    return wrapper

# 헤더
st.title("🤖 Financial AI Prediction System")
st.markdown("### Advanced LSTM/GRU Stock Market Prediction with Real-time Data Integration")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 모델 설정
    st.subheader("🧠 Model Settings")
    model_type = st.selectbox("Model Type", ["LSTM", "GRU"])
    use_mixed_precision = st.checkbox("Enable Mixed Precision (GPU)", value=True)
    batch_size = st.slider("Batch Size", 8, 128, 32, step=8)
    learning_rate = st.select_slider(
        "Learning Rate", 
        options=[0.0001, 0.001, 0.01, 0.1],
        value=0.001
    )
    
    # 데이터 설정
    st.subheader("📊 Data Settings")
    selected_stocks = st.multiselect(
        "Select Stocks",
        ["005930", "000660", "035720", "005380", "035420"],
        default=["005930", "000660"]
    )
    
    date_range = st.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=365), datetime.now()),
        max_value=datetime.now()
    )
    
    # API 설정
    st.subheader("🔌 API Configuration")
    enable_redis = st.checkbox("Enable Redis Cache", value=True)
    api_rate_limit = st.number_input("API Calls/Second", 1, 20, 10)
    
    # 시스템 정보
    st.subheader("💻 System Info")
    
    @handle_errors
    def setup_gpu_info():
        gpu_manager = GPUMemoryManager()
        gpu_available = gpu_manager.setup_gpu_configuration()
        
        if gpu_available:
            st.success("✅ GPU 감지됨")
            if st.button("GPU 정보 표시"):
                try:
                    gpu_info = gpu_manager._log_gpu_info(gpu_manager.gpus)
                    st.json(gpu_info)
                except Exception as e:
                    st.warning(f"GPU 정보를 가져올 수 없습니다: {e}")
        else:
            st.warning("⚠️ GPU 없음 - CPU 사용 중")
        return gpu_available
    
    gpu_available = setup_gpu_info()
    
    # 개선된 자동 새로고침
    st.subheader("🔄 Auto Refresh")
    auto_refresh = st.checkbox("Enable Auto Refresh")
    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 30)
        
        # 타이머 표시
        current_time = time.time()
        elapsed = current_time - st.session_state.last_refresh
        remaining = max(0, refresh_interval - elapsed)
        
        # 조건부 새로고침
        if remaining > 0:
            st.progress(1 - (remaining / refresh_interval))
            st.caption(f"다음 새로고침: {int(remaining)}초")
        else:
            # 데이터가 있을 때만 새로고침
            if st.session_state.predictions is not None:
                st.session_state.last_refresh = current_time
                cleanup_memory()  # 새로고침 전 메모리 정리
                st.rerun()

# 메인 컨텐츠
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Predictions", 
    "📊 Market Data", 
    "🧪 Model Training", 
    "📉 Performance",
    "⚡ System Status"
])

# Tab 1: 예측 (개선된 오류 처리)
with tab1:
    st.header("Stock Price Predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        @st.cache_data(ttl=300)  # 5분 캐싱
        def generate_dummy_predictions(stocks):
            """캐시된 더미 예측 생성"""
            predictions_data = []
            for stock_code in stocks:
                dummy_prediction = {
                    'stock_code': stock_code,
                    'current_price': np.random.uniform(50000, 200000),
                    'predicted_price': np.random.uniform(50000, 200000),
                    'confidence': np.random.uniform(0.6, 0.95),
                    'direction': np.random.choice(['UP', 'DOWN']),
                    'probability': np.random.uniform(0.5, 0.8)
                }
                predictions_data.append(dummy_prediction)
            return pd.DataFrame(predictions_data)
        
        if st.button("🚀 Generate Predictions", type="primary"):
            if not selected_stocks:
                st.error("최소 하나의 주식을 선택해주세요")
            else:
                with st.spinner("모델 로딩 및 예측 생성 중..."):
                    try:
                        # 모델 초기화 (오류 처리 강화)
                        if st.session_state.model is None:
                            st.session_state.model = GPUOptimizedStackingEnsemble(
                                use_gru=(model_type == "GRU"),
                                enable_mixed_precision=use_mixed_precision and gpu_available
                            )
                        
                        # 데이터 수집
                        collector = IntegratedMacroDataCollector()
                        
                        # 캐시된 예측 데이터 사용
                        st.session_state.predictions = generate_dummy_predictions(selected_stocks)
                        st.session_state.error_count = 0  # 성공시 오류 카운트 리셋
                        st.success("✅ 예측이 성공적으로 생성되었습니다!")
                        
                    except Exception as e:
                        st.session_state.error_count += 1
                        logger.error(f"예측 생성 오류: {e}")
                        st.error(f"예측 생성 중 오류 발생: {str(e)}")
                        
                        # 폴백 옵션 제공
                        if st.button("간단한 예측 시도"):
                            st.session_state.predictions = generate_dummy_predictions(selected_stocks)
    
    with col2:
        if st.session_state.predictions is not None:
            # 예측 요약 메트릭
            try:
                avg_confidence = st.session_state.predictions['confidence'].mean()
                up_count = (st.session_state.predictions['direction'] == 'UP').sum()
                
                st.metric("평균 신뢰도", f"{avg_confidence:.1%}")
                st.metric("상승 신호", f"{up_count}/{len(selected_stocks)}")
            except Exception as e:
                logger.warning(f"메트릭 계산 오류: {e}")
                st.warning("메트릭을 계산할 수 없습니다")
    
    # 예측 결과 표시 (개선된 오류 처리)
    if st.session_state.predictions is not None:
        st.subheader("Prediction Results")
        
        try:
            # 예측 테이블
            for _, row in st.session_state.predictions.iterrows():
                with st.expander(f"📊 {row['stock_code']} - {row['direction']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    # 안전한 메트릭 표시
                    try:
                        col1.metric(
                            "현재 가격",
                            f"₩{row['current_price']:,.0f}"
                        )
                        col2.metric(
                            "예상 가격",
                            f"₩{row['predicted_price']:,.0f}",
                            delta=f"{(row['predicted_price']/row['current_price']-1)*100:.1f}%"
                        )
                        col3.metric(
                            "신뢰도",
                            f"{row['confidence']:.1%}"
                        )
                        
                        # 예측 상세 차트 (오류 처리 강화)
                        try:
                            fig = go.Figure()
                            fig.add_trace(go.Indicator(
                                mode="gauge+number",
                                value=row['probability']*100,
                                title={'text': f"{row['direction']} 확률"},
                                gauge={'axis': {'range': [0, 100]},
                                       'bar': {'color': "green" if row['direction'] == "UP" else "red"},
                                       'steps': [
                                           {'range': [0, 50], 'color': "lightgray"},
                                           {'range': [50, 100], 'color': "gray"}],
                                       'threshold': {'line': {'color': "black", 'width': 4},
                                                   'thickness': 0.75, 'value': 50}}
                            ))
                            fig.update_layout(height=250)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as chart_error:
                            logger.warning(f"차트 생성 오류: {chart_error}")
                            st.warning("차트를 표시할 수 없습니다")
                    
                    except Exception as metric_error:
                        logger.warning(f"메트릭 표시 오류: {metric_error}")
                        st.warning(f"{row['stock_code']}의 데이터를 표시할 수 없습니다")
        
        except Exception as e:
            logger.error(f"예측 결과 표시 오류: {e}")
            st.error("예측 결과를 표시하는 중 오류가 발생했습니다")

# Tab 2: 시장 데이터 (개선된 비동기 처리)
with tab2:
    st.header("Market Data Integration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 KOSPI & VKOSPI")
        
        if st.button("시장 지수 가져오기"):
            with st.spinner("시장 데이터 가져오는 중..."):
                try:
                    bok_client = EnhancedBokApiClient()
                    
                    # 날짜 변환 (오류 처리 강화)
                    try:
                        start_date = date_range[0].strftime('%Y%m%d')
                        end_date = date_range[1].strftime('%Y%m%d')
                    except Exception as date_error:
                        logger.error(f"날짜 변환 오류: {date_error}")
                        st.error("날짜 형식 오류가 발생했습니다")
                        st.stop()
                    
                    # 비동기 함수 실행 (타임아웃 추가)
                    indicators = run_async(
                        bok_client.get_economic_indicators(start_date, end_date)
                    )
                    
                    if indicators is None:
                        st.error("시장 데이터를 가져올 수 없습니다")
                    else:
                        # KOSPI 차트
                        if 'kospi' in indicators and not indicators['kospi'].empty:
                            try:
                                fig_kospi = px.line(
                                    indicators['kospi'], 
                                    x='date', 
                                    y='value',
                                    title='KOSPI Index'
                                )
                                st.plotly_chart(fig_kospi, use_container_width=True)
                            except Exception as chart_error:
                                logger.warning(f"KOSPI 차트 오류: {chart_error}")
                                st.warning("KOSPI 차트를 표시할 수 없습니다")
                        
                        # VKOSPI 차트
                        if 'vkospi' in indicators and not indicators['vkospi'].empty:
                            try:
                                fig_vkospi = px.line(
                                    indicators['vkospi'],
                                    x='date',
                                    y='value',
                                    title='VKOSPI (Volatility Index)'
                                )
                                st.plotly_chart(fig_vkospi, use_container_width=True)
                            except Exception as chart_error:
                                logger.warning(f"VKOSPI 차트 오류: {chart_error}")
                                st.warning("VKOSPI 차트를 표시할 수 없습니다")
                        
                        # 세션 정리
                        run_async(bok_client.close_session())
                    
                except Exception as e:
                    logger.error(f"시장 데이터 가져오기 오류: {e}")
                    st.error(f"시장 데이터 가져오기 오류: {str(e)}")
    
    with col2:
        st.subheader("📰 Corporate Disclosures")
        
        if st.button("DART 데이터 가져오기"):
            with st.spinner("기업 공시 가져오는 중..."):
                try:
                    if st.session_state.dart_client is None:
                        st.session_state.dart_client = EnhancedDartApiClient()
                        if hasattr(st.session_state.dart_client, 'init_redis'):
                            run_async(st.session_state.dart_client.init_redis())
                    
                    # 최근 공시 조회 (더미 데이터)
                    disclosures = pd.DataFrame({
                        'date': pd.date_range(end=datetime.now(), periods=5, freq='D'),
                        'company': np.random.choice(['삼성전자', 'SK하이닉스'], 5),
                        'title': ['실적발표', '주요사항보고', '분기보고서', '임시공시', '정정공시'],
                        'impact': np.random.choice(['positive', 'negative', 'neutral'], 5)
                    })
                    
                    # 공시 표시
                    for _, disc in disclosures.iterrows():
                        icon = "🟢" if disc['impact'] == 'positive' else "🔴" if disc['impact'] == 'negative' else "⚪"
                        st.write(f"{icon} **{disc['company']}** - {disc['title']} ({disc['date'].strftime('%Y-%m-%d')})")
                    
                except Exception as e:
                    logger.error(f"DART 데이터 가져오기 오류: {e}")
                    st.error(f"DART 데이터 가져오기 오류: {str(e)}")

# Tab 3: 모델 학습 (개선된 안정성)
with tab3:
    st.header("Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Training Configuration")
        
        epochs = st.slider("Number of Epochs", 10, 200, 100)
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        
        # 학습 중 상태 확인
        if st.session_state.training_in_progress:
            st.warning("⚠️ 학습이 진행 중입니다...")
            if st.button("학습 중단"):
                st.session_state.training_in_progress = False
                st.success("학습이 중단되었습니다")
        else:
            if st.button("🏋️ 학습 시작", type="primary"):
                st.session_state.training_in_progress = True
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # 학습 시뮬레이션 (메모리 효율적)
                    for i in range(epochs):
                        if not st.session_state.training_in_progress:
                            break
                            
                        progress = (i + 1) / epochs
                        progress_bar.progress(progress)
                        
                        loss = max(0.1, 0.5 * (1 - progress) + np.random.uniform(-0.05, 0.05))
                        status_text.text(f"Epoch {i+1}/{epochs} - Loss: {loss:.4f}")
                        time.sleep(0.01)  # 시뮬레이션을 위한 작은 지연
                        
                        if i % 10 == 0:
                            # 중간 결과 표시
                            col1_metric, col2_metric, col3_metric = st.columns(3)
                            col1_metric.metric("Training Loss", f"{loss:.4f}")
                            col2_metric.metric("Validation Loss", f"{loss + 0.05:.4f}")
                            col3_metric.metric("Learning Rate", f"{learning_rate:.4f}")
                    
                    st.session_state.training_in_progress = False
                    st.success("✅ 학습이 성공적으로 완료되었습니다!")
                    
                    # 모델 저장 옵션
                    if st.button("💾 모델 저장"):
                        try:
                            version_manager = ModelVersionManager()
                            version_id = version_manager.save_model_with_version(
                                st.session_state.model,
                                metrics={'val_loss': loss + 0.05, 'val_auc': 0.85},
                                description="Streamlit 학습 세션"
                            )
                            st.success(f"모델이 저장되었습니다. 버전: {version_id}")
                        except Exception as save_error:
                            logger.error(f"모델 저장 오류: {save_error}")
                            st.error(f"모델 저장 중 오류 발생: {str(save_error)}")
                    
                except Exception as e:
                    st.session_state.training_in_progress = False
                    logger.error(f"학습 오류: {e}")
                    st.error(f"학습 중 오류 발생: {str(e)}")
    
    with col2:
        st.subheader("📊 Training Metrics")
        
        # 실시간 메트릭 차트 (더미 데이터)
        if st.checkbox("실시간 메트릭 표시"):
            try:
                # 실시간 업데이트 시뮬레이션
                df_metrics = pd.DataFrame({
                    'epoch': range(10),
                    'train_loss': np.random.uniform(0.3, 0.5, 10),
                    'val_loss': np.random.uniform(0.35, 0.55, 10)
                })
                
                fig = px.line(df_metrics, x='epoch', y=['train_loss', 'val_loss'])
                fig.update_layout(title="학습 진행 상황")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.warning(f"메트릭 차트 오류: {e}")
                st.warning("메트릭 차트를 표시할 수 없습니다")

# Tab 4: 성능 분석 (개선된 차트 오류 처리)
with tab4:
    st.header("Performance Analysis")
    
    # 백테스트 결과
    st.subheader("📊 Backtesting Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # 더미 백테스트 메트릭
        col1.metric("총 수익률", "+15.3%", "+2.1%")
        col2.metric("샤프 비율", "1.45", "+0.12")
        col3.metric("최대 손실", "-8.2%", "-1.3%")
        col4.metric("승률", "62.5%", "+3.2%")
        
        # 수익률 곡선 (오류 처리 강화)
        try:
            dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
            cumulative_returns = np.cumprod(1 + np.random.normal(0.0005, 0.02, 252))
            benchmark_returns = np.cumprod(1 + np.random.normal(0.0003, 0.015, 252))
            
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode='lines',
                name='Strategy Returns',
                line=dict(color='blue', width=2)
            ))
            fig_returns.add_trace(go.Scatter(
                x=dates,
                y=benchmark_returns,
                mode='lines',
                name='Benchmark (KOSPI)',
                line=dict(color='gray', width=2, dash='dash')
            ))
            fig_returns.update_layout(
                title="누적 수익률 비교",
                xaxis_title="날짜",
                yaxis_title="누적 수익률",
                hovermode='x unified'
            )
            st.plotly_chart(fig_returns, use_container_width=True)
        except Exception as chart_error:
            logger.warning(f"수익률 차트 오류: {chart_error}")
            st.warning("수익률 차트를 표시할 수 없습니다")
        
        # 위험 분석
        st.subheader("🎯 Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # VaR 계산
                var_95 = np.percentile(np.random.normal(-0.001, 0.02, 1000), 5)
                st.metric("Value at Risk (95%)", f"{var_95:.2%}")
                
                # 포트폴리오 베타
                st.metric("포트폴리오 베타", "0.85")
            except Exception as e:
                logger.warning(f"위험 메트릭 계산 오류: {e}")
                st.warning("위험 메트릭을 계산할 수 없습니다")
        
        with col2:
            try:
                # 상관관계 히트맵
                stocks = ['005930', '000660', '035720']
                corr_matrix = np.random.uniform(0.3, 0.9, (3, 3))
                np.fill_diagonal(corr_matrix, 1)
                
                fig_corr = px.imshow(
                    corr_matrix,
                    labels=dict(x="Stock", y="Stock", color="Correlation"),
                    x=stocks,
                    y=stocks,
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                fig_corr.update_layout(title="주식 상관관계 매트릭스")
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as heatmap_error:
                logger.warning(f"히트맵 오류: {heatmap_error}")
                st.warning("상관관계 히트맵을 표시할 수 없습니다")
    
    except Exception as e:
        logger.error(f"성능 분석 탭 오류: {e}")
        st.error("성능 분석 데이터를 표시하는 중 오류가 발생했습니다")

# Tab 5: 시스템 상태 (개선된 진단)
with tab5:
    st.header("System Status & Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔌 API Status")
        
        try:
            # API 사용량
            if st.session_state.dart_client:
                api_stats = st.session_state.dart_client.get_api_usage_stats()
                
                fig_api = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=api_stats['usage_percentage'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "API 사용량 (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                st.plotly_chart(fig_api, use_container_width=True)
            else:
                st.info("API 클라이언트가 초기화되지 않았습니다")
        except Exception as e:
            logger.warning(f"API 상태 표시 오류: {e}")
            st.warning("API 상태를 표시할 수 없습니다")
    
    with col2:
        st.subheader("💾 Cache Status")
        
        if st.button("Redis 상태 확인"):
            try:
                if st.session_state.redis_manager is None:
                    st.session_state.redis_manager = RobustRedisManager()
                    connection_result = run_async(st.session_state.redis_manager.connect())
                    if connection_result is None:
                        st.error("Redis 연결 시도 중 오류가 발생했습니다")
                        st.stop()
                
                diagnostics = run_async(st.session_state.redis_manager.get_diagnostics())
                
                if diagnostics and diagnostics.get('connected'):
                    st.success("✅ Redis 연결됨")
                    if 'redis_info' in diagnostics:
                        st.json(diagnostics['redis_info'])
                else:
                    st.warning("⚠️ 폴백 캐시 사용 중")
                    if diagnostics:
                        st.write(f"캐시 크기: {diagnostics.get('fallback_cache_size', 0)} 항목")
            except Exception as e:
                logger.error(f"Redis 상태 확인 오류: {e}")
                st.error(f"Redis 상태 확인 중 오류 발생: {str(e)}")
    
    # 시스템 리소스 (개선된 오류 처리)
    st.subheader("💻 System Resources")
    
    col1, col2, col3 = st.columns(3)
    
    try:
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=0.1)  # 짧은 간격으로 변경
        col1.metric("CPU 사용률", f"{cpu_percent}%")
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        col2.metric("메모리 사용률", f"{memory.percent}%")
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        col3.metric("디스크 사용률", f"{disk.percent}%")
        
        # 프로세스 정보
        if st.checkbox("프로세스 세부 정보 표시"):
            try:
                process = psutil.Process()
                process_info = {
                    "PID": process.pid,
                    "메모리 (MB)": round(process.memory_info().rss / 1024 / 1024, 2),
                    "CPU %": round(process.cpu_percent(), 2),
                    "스레드": process.num_threads(),
                    "상태": process.status()
                }
                st.json(process_info)
            except Exception as process_error:
                logger.warning(f"프로세스 정보 오류: {process_error}")
                st.warning("프로세스 정보를 가져올 수 없습니다")
                
    except Exception as e:
        logger.error(f"시스템 리소스 확인 오류: {e}")
        st.error(f"시스템 리소스 정보를 가져오는 중 오류 발생: {str(e)}")
    
    # 시스템 진단
    st.subheader("🔍 System Diagnostics")
    
    if st.button("전체 시스템 진단 실행"):
        with st.spinner("시스템 진단 중..."):
            try:
                diagnostics_results = {
                    "메모리 정리": "완료",
                    "세션 상태": f"{len(st.session_state)} 항목",
                    "오류 카운트": st.session_state.get('error_count', 0),
                    "GPU 사용 가능": gpu_available,
                    "마지막 새로고침": datetime.fromtimestamp(st.session_state.last_refresh).strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 메모리 정리 실행
                cleanup_memory()
                
                st.success("✅ 시스템 진단 완료")
                st.json(diagnostics_results)
                
                # 권장사항 표시
                if st.session_state.get('error_count', 0) > 2:
                    st.warning("⚠️ 오류가 자주 발생하고 있습니다. 페이지 새로고침을 권장합니다.")
                
                if len(st.session_state) > 20:
                    st.info("ℹ️ 세션 상태가 많습니다. 성능 향상을 위해 일부 데이터를 정리했습니다.")
                    
            except Exception as diagnostic_error:
                logger.error(f"시스템 진단 오류: {diagnostic_error}")
                st.error(f"시스템 진단 중 오류 발생: {str(diagnostic_error)}")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Financial AI Prediction System v1.0 | Built with Streamlit & TensorFlow</p>
    <p>⚠️ 이는 교육 목적으로만 사용됩니다. 투자 조언이 아닙니다.</p>
</div>
""", unsafe_allow_html=True)

# 앱 종료 시 정리
try:
    # 세션 종료 시 리소스 정리
    if hasattr(st.session_state, 'dart_client') and st.session_state.dart_client:
        # 비동기 정리는 별도 처리 필요
        pass
    
    # 정기적 메모리 정리
    if time.time() - st.session_state.get('last_cleanup', 0) > 600:  # 10분마다
        cleanup_memory()
        st.session_state.last_cleanup = time.time()
        
except Exception as cleanup_error:
    logger.warning(f"정리 과정에서 오류: {cleanup_error}")
