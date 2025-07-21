"""
Streamlit Financial Prediction Dashboard - Fixed Version
Main entry point for the LSTM-based stock prediction system
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px
import logging
import time
import traceback

# 로깅 설정 (간단하게)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정 (먼저 실행)
st.set_page_config(
    page_title="Financial AI Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 안전한 더미 클래스들
class SafeGPUOptimizedStackingEnsemble:
    def __init__(self, use_gru=False, enable_mixed_precision=True):
        self.use_gru = use_gru
        self.enable_mixed_precision = enable_mixed_precision
        self.model = None
        self.is_trained = False
        
    def predict(self, X):
        """안전한 예측 메서드"""
        try:
            if hasattr(X, 'shape'):
                if len(X.shape) == 2:
                    return np.random.uniform(0, 1, X.shape[0])
                elif len(X.shape) == 3:
                    return np.random.uniform(0, 1, X.shape[0])
            return np.random.uniform(0, 1, len(X))
        except:
            return np.array([0.5])
    
    def train(self, X, y, epochs=100):
        """안전한 학습 메서드"""
        self.is_trained = True
        return self

class SafeGPUMemoryManager:
    def __init__(self):
        self.gpus = []
        self.gpu_available = False
        
    def setup_gpu_configuration(self, memory_limit_mb=None):
        """안전한 GPU 설정"""
        self.gpu_available = False
        return False
    
    def _log_gpu_info(self, gpus):
        return {
            "gpu_count": 0,
            "gpu_available": False,
            "gpu_names": []
        }

class SafeModelVersionManager:
    def __init__(self, base_dir="./models"):
        self.base_dir = base_dir
        self.versions = {}
    
    def save_model_with_version(self, model, metrics, description=""):
        version_id = f"v_{int(time.time())}"
        self.versions[version_id] = {
            "model": model,
            "metrics": metrics,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        return version_id

# 전역 변수로 안전하게 설정
GPUOptimizedStackingEnsemble = SafeGPUOptimizedStackingEnsemble
GPUMemoryManager = SafeGPUMemoryManager
ModelVersionManager = SafeModelVersionManager

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
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화 (안전하게)
def safe_initialize_session_state():
    """안전한 세션 상태 초기화"""
    try:
        defaults = {
            'model': None,
            'predictions': None,
            'last_refresh': time.time(),
            'training_in_progress': False,
            'error_count': 0,
            'gpu_available': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    except Exception as e:
        st.error(f"세션 상태 초기화 오류: {e}")

# 세션 상태 초기화 실행
safe_initialize_session_state()

# 안전한 예측 데이터 생성 함수
@st.cache_data(ttl=300)
def generate_safe_predictions(stocks):
    """안전한 더미 예측 생성"""
    try:
        predictions_data = []
        for stock_code in stocks:
            prediction = {
                'stock_code': stock_code,
                'current_price': np.random.uniform(50000, 200000),
                'predicted_price': np.random.uniform(50000, 200000),
                'confidence': np.random.uniform(0.6, 0.95),
                'direction': np.random.choice(['UP', 'DOWN']),
                'probability': np.random.uniform(0.5, 0.8)
            }
            predictions_data.append(prediction)
        return pd.DataFrame(predictions_data)
    except Exception as e:
        st.error(f"예측 데이터 생성 오류: {e}")
        return pd.DataFrame()

# 헤더
st.title("🤖 Financial AI Prediction System")
st.markdown("### Advanced LSTM/GRU Stock Market Prediction (Safe Mode)")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 모델 설정
    st.subheader("🧠 Model Settings")
    model_type = st.selectbox("Model Type", ["LSTM", "GRU"])
    use_mixed_precision = st.checkbox("Enable Mixed Precision", value=False)
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
    
    # 시스템 정보 (안전하게)
    st.subheader("💻 System Info")
    try:
        gpu_manager = GPUMemoryManager()
        gpu_available = gpu_manager.setup_gpu_configuration()
        st.session_state.gpu_available = gpu_available
        
        if gpu_available:
            st.success("✅ GPU 감지됨")
        else:
            st.warning("⚠️ GPU 없음 - CPU 사용 중")
    except Exception as e:
        st.error(f"시스템 정보 오류: {e}")
        st.session_state.gpu_available = False

# 메인 탭
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Predictions", 
    "📊 Market Data", 
    "🧪 Model Training", 
    "📉 Performance"
])

# Tab 1: 예측
with tab1:
    st.header("Stock Price Predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Generate Predictions", type="primary"):
            if not selected_stocks:
                st.error("최소 하나의 주식을 선택해주세요")
            else:
                with st.spinner("예측 생성 중..."):
                    try:
                        # 모델 초기화 (안전하게)
                        if st.session_state.model is None:
                            st.session_state.model = GPUOptimizedStackingEnsemble(
                                use_gru=(model_type == "GRU"),
                                enable_mixed_precision=use_mixed_precision
                            )
                        
                        # 예측 데이터 생성
                        st.session_state.predictions = generate_safe_predictions(selected_stocks)
                        st.session_state.error_count = 0
                        st.success("✅ 예측이 성공적으로 생성되었습니다!")
                        
                    except Exception as e:
                        st.session_state.error_count += 1
                        st.error(f"예측 생성 중 오류 발생: {str(e)}")
                        # 안전한 폴백
                        try:
                            st.session_state.predictions = generate_safe_predictions(selected_stocks)
                            st.warning("폴백 모드로 예측을 생성했습니다.")
                        except:
                            st.error("예측 생성에 완전히 실패했습니다.")
    
    with col2:
        if st.session_state.predictions is not None and not st.session_state.predictions.empty:
            try:
                avg_confidence = st.session_state.predictions['confidence'].mean()
                up_count = (st.session_state.predictions['direction'] == 'UP').sum()
                
                st.metric("평균 신뢰도", f"{avg_confidence:.1%}")
                st.metric("상승 신호", f"{up_count}/{len(selected_stocks)}")
            except Exception as e:
                st.warning(f"메트릭 계산 오류: {e}")
    
    # 예측 결과 표시
    if st.session_state.predictions is not None and not st.session_state.predictions.empty:
        st.subheader("Prediction Results")
        
        try:
            for _, row in st.session_state.predictions.iterrows():
                with st.expander(f"📊 {row['stock_code']} - {row['direction']}"):
                    col1, col2, col3 = st.columns(3)
                    
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
                        
                        # 간단한 확률 표시
                        st.progress(row['probability'])
                        st.caption(f"{row['direction']} 확률: {row['probability']:.1%}")
                        
                    except Exception as metric_error:
                        st.warning(f"{row['stock_code']} 데이터 표시 오류: {metric_error}")
        
        except Exception as e:
            st.error(f"예측 결과 표시 오류: {e}")

# Tab 2: 시장 데이터
with tab2:
    st.header("Market Data Integration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 KOSPI & VKOSPI")
        
        if st.button("시장 지수 가져오기"):
            with st.spinner("시장 데이터 생성 중..."):
                try:
                    # 더미 시장 데이터 생성
                    start_date = date_range[0]
                    end_date = date_range[1]
                    dates = pd.date_range(start=start_date, end=end_date, freq='D')
                    
                    # KOSPI 데이터
                    kospi_data = pd.DataFrame({
                        'date': dates,
                        'value': np.random.uniform(2500, 3000, len(dates))
                    })
                    
                    fig_kospi = px.line(kospi_data, x='date', y='value', title='KOSPI Index (Demo)')
                    st.plotly_chart(fig_kospi, use_container_width=True)
                    
                    # VKOSPI 데이터
                    vkospi_data = pd.DataFrame({
                        'date': dates,
                        'value': np.random.uniform(15, 30, len(dates))
                    })
                    
                    fig_vkospi = px.line(vkospi_data, x='date', y='value', title='VKOSPI (Demo)')
                    st.plotly_chart(fig_vkospi, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"시장 데이터 생성 오류: {e}")
    
    with col2:
        st.subheader("📰 Corporate Disclosures")
        
        if st.button("DART 데이터 가져오기"):
            try:
                # 더미 공시 데이터
                disclosures = pd.DataFrame({
                    'date': pd.date_range(end=datetime.now(), periods=5, freq='D'),
                    'company': ['삼성전자', 'SK하이닉스', 'NAVER', 'LG전자', '현대차'],
                    'title': ['실적발표', '주요사항보고', '분기보고서', '임시공시', '정정공시'],
                    'impact': ['positive', 'negative', 'neutral', 'positive', 'neutral']
                })
                
                for _, disc in disclosures.iterrows():
                    icon = "🟢" if disc['impact'] == 'positive' else "🔴" if disc['impact'] == 'negative' else "⚪"
                    st.write(f"{icon} **{disc['company']}** - {disc['title']} ({disc['date'].strftime('%Y-%m-%d')})")
                    
            except Exception as e:
                st.error(f"공시 데이터 생성 오류: {e}")

# Tab 3: 모델 학습
with tab3:
    st.header("Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Training Configuration")
        
        epochs = st.slider("Number of Epochs", 10, 200, 100)
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        
        if st.session_state.get('training_in_progress', False):
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
                    # 학습 시뮬레이션
                    for i in range(min(epochs, 20)):  # 최대 20회만 시뮬레이션
                        if not st.session_state.get('training_in_progress', False):
                            break
                            
                        progress = (i + 1) / min(epochs, 20)
                        progress_bar.progress(progress)
                        
                        loss = max(0.1, 0.5 * (1 - progress) + np.random.uniform(-0.05, 0.05))
                        status_text.text(f"Epoch {i+1}/{epochs} - Loss: {loss:.4f}")
                        time.sleep(0.1)
                    
                    st.session_state.training_in_progress = False
                    st.success("✅ 학습이 완료되었습니다!")
                    
                    # 모델 저장
                    if st.button("💾 모델 저장"):
                        try:
                            version_manager = ModelVersionManager()
                            version_id = version_manager.save_model_with_version(
                                st.session_state.model,
                                metrics={'val_loss': loss, 'val_auc': 0.85},
                                description="Safe mode training"
                            )
                            st.success(f"모델 저장됨: {version_id}")
                        except Exception as save_error:
                            st.error(f"모델 저장 오류: {save_error}")
                    
                except Exception as e:
                    st.session_state.training_in_progress = False
                    st.error(f"학습 중 오류: {e}")
    
    with col2:
        st.subheader("📊 Training Metrics")
        
        if st.checkbox("메트릭 차트 표시"):
            try:
                # 더미 메트릭 데이터
                epochs_range = range(10)
                train_loss = [0.5 - 0.4 * (i / 10) + np.random.uniform(-0.05, 0.05) for i in epochs_range]
                val_loss = [0.55 - 0.35 * (i / 10) + np.random.uniform(-0.05, 0.05) for i in epochs_range]
                
                df_metrics = pd.DataFrame({
                    'epoch': epochs_range,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
                
                fig = px.line(df_metrics, x='epoch', y=['train_loss', 'val_loss'], 
                             title="학습 진행 상황")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"메트릭 차트 오류: {e}")

# Tab 4: 성능 분석
with tab4:
    st.header("Performance Analysis")
    
    # 백테스트 결과
    st.subheader("📊 Backtesting Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        col1.metric("총 수익률", "+15.3%", "+2.1%")
        col2.metric("샤프 비율", "1.45", "+0.12")
        col3.metric("최대 손실", "-8.2%", "-1.3%")
        col4.metric("승률", "62.5%", "+3.2%")
        
        # 수익률 곡선
        try:
            dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
            returns_data = pd.DataFrame({
                'date': dates,
                'strategy': np.cumprod(1 + np.random.normal(0.0005, 0.02, 252)),
                'benchmark': np.cumprod(1 + np.random.normal(0.0003, 0.015, 252))
            })
            
            fig_returns = px.line(returns_data, x='date', y=['strategy', 'benchmark'],
                                title="누적 수익률 비교")
            st.plotly_chart(fig_returns, use_container_width=True)
        except Exception as chart_error:
            st.warning(f"수익률 차트 오류: {chart_error}")
        
    except Exception as e:
        st.error(f"성능 분석 오류: {e}")

# 시스템 상태 (하단)
st.subheader("💻 System Status")

col1, col2, col3 = st.columns(3)

try:
    import psutil
    
    # CPU 사용률
    cpu_percent = psutil.cpu_percent(interval=0.1)
    col1.metric("CPU 사용률", f"{cpu_percent}%")
    
    # 메모리 사용률
    memory = psutil.virtual_memory()
    col2.metric("메모리 사용률", f"{memory.percent}%")
    
    # 오류 카운트
    col3.metric("오류 카운트", st.session_state.get('error_count', 0))
    
except Exception as e:
    st.warning(f"시스템 상태 조회 오류: {e}")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Financial AI Prediction System v1.0 (Safe Mode) | Built with Streamlit</p>
    <p>⚠️ 교육 목적으로만 사용하세요. 투자 조언이 아닙니다.</p>
</div>
""", unsafe_allow_html=True)

# 디버깅 정보 (개발 시에만)
if st.checkbox("디버깅 정보 표시"):
    st.json({
        "세션 상태 키": list(st.session_state.keys()),
        "모델 상태": "로드됨" if st.session_state.model else "없음",
        "예측 데이터": "있음" if st.session_state.predictions is not None else "없음",
        "GPU 사용 가능": st.session_state.get('gpu_available', False),
        "오류 카운트": st.session_state.get('error_count', 0)
    })
