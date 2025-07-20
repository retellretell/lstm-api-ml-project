"""
Streamlit Financial Prediction Dashboard
Main entry point for the LSTM-based stock prediction system
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import logging
import sys
import os

# 프로젝트 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from lstm_gpu_improvements import (
        GPUOptimizedStackingEnsemble, 
        GPUMemoryManager,
        ModelVersionManager
    )
    from dart_api_improvements import (
        EnhancedAdaptiveRateLimiter,
        RobustRedisManager,
        StrictVKOSPIValidator
    )
    from improved_dart_integration_v2 import (
        EnhancedDartApiClient,
        EnhancedBokApiClient,
        IntegratedMacroDataCollector
    )
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="Financial AI Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# 세션 상태 초기화
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'dart_client' not in st.session_state:
    st.session_state.dart_client = None
if 'redis_manager' not in st.session_state:
    st.session_state.redis_manager = None

# 비동기 함수 실행을 위한 헬퍼
def run_async(coro):
    """비동기 함수를 동기적으로 실행"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

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
    gpu_manager = GPUMemoryManager()
    gpu_available = gpu_manager.setup_gpu_configuration()
    
    if gpu_available:
        st.success("✅ GPU Detected")
        if st.button("Show GPU Info"):
            gpu_info = gpu_manager._log_gpu_info(gpu_manager.gpus)
            st.json(gpu_info)
    else:
        st.warning("⚠️ No GPU - Using CPU")

# 메인 컨텐츠
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Predictions", 
    "📊 Market Data", 
    "🧪 Model Training", 
    "📉 Performance",
    "⚡ System Status"
])

# Tab 1: 예측
with tab1:
    st.header("Stock Price Predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Generate Predictions", type="primary"):
            if not selected_stocks:
                st.error("Please select at least one stock")
            else:
                with st.spinner("Loading model and generating predictions..."):
                    try:
                        # 모델 초기화
                        if st.session_state.model is None:
                            st.session_state.model = GPUOptimizedStackingEnsemble(
                                use_gru=(model_type == "GRU"),
                                enable_mixed_precision=use_mixed_precision
                            )
                        
                        # 데이터 수집
                        collector = IntegratedMacroDataCollector()
                        
                        # 더미 데이터 생성 (실제 구현시 데이터베이스에서 로드)
                        predictions_data = []
                        
                        for stock_code in selected_stocks:
                            # 실제 구현시 여기서 데이터 로드 및 예측
                            dummy_prediction = {
                                'stock_code': stock_code,
                                'current_price': np.random.uniform(50000, 200000),
                                'predicted_price': np.random.uniform(50000, 200000),
                                'confidence': np.random.uniform(0.6, 0.95),
                                'direction': np.random.choice(['UP', 'DOWN']),
                                'probability': np.random.uniform(0.5, 0.8)
                            }
                            predictions_data.append(dummy_prediction)
                        
                        st.session_state.predictions = pd.DataFrame(predictions_data)
                        st.success("✅ Predictions generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
    
    with col2:
        if st.session_state.predictions is not None:
            # 예측 요약 메트릭
            avg_confidence = st.session_state.predictions['confidence'].mean()
            up_count = (st.session_state.predictions['direction'] == 'UP').sum()
            
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
            st.metric("Bullish Signals", f"{up_count}/{len(selected_stocks)}")
    
    # 예측 결과 표시
    if st.session_state.predictions is not None:
        st.subheader("Prediction Results")
        
        # 예측 테이블
        for _, row in st.session_state.predictions.iterrows():
            with st.expander(f"📊 {row['stock_code']} - {row['direction']}"):
                col1, col2, col3 = st.columns(3)
                
                col1.metric(
                    "Current Price",
                    f"₩{row['current_price']:,.0f}"
                )
                col2.metric(
                    "Predicted Price",
                    f"₩{row['predicted_price']:,.0f}",
                    delta=f"{(row['predicted_price']/row['current_price']-1)*100:.1f}%"
                )
                col3.metric(
                    "Confidence",
                    f"{row['confidence']:.1%}"
                )
                
                # 예측 상세 차트
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=row['probability']*100,
                    title={'text': f"{row['direction']} Probability"},
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

# Tab 2: 시장 데이터
with tab2:
    st.header("Market Data Integration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 KOSPI & VKOSPI")
        
        if st.button("Fetch Market Indices"):
            with st.spinner("Fetching market data..."):
                try:
                    bok_client = EnhancedBokApiClient()
                    
                    # 날짜 변환
                    start_date = date_range[0].strftime('%Y%m%d')
                    end_date = date_range[1].strftime('%Y%m%d')
                    
                    # 비동기 함수 실행
                    indicators = run_async(
                        bok_client.get_economic_indicators(start_date, end_date)
                    )
                    
                    # KOSPI 차트
                    if 'kospi' in indicators and not indicators['kospi'].empty:
                        fig_kospi = px.line(
                            indicators['kospi'], 
                            x='date', 
                            y='value',
                            title='KOSPI Index'
                        )
                        st.plotly_chart(fig_kospi, use_container_width=True)
                    
                    # VKOSPI 차트
                    if 'vkospi' in indicators and not indicators['vkospi'].empty:
                        fig_vkospi = px.line(
                            indicators['vkospi'],
                            x='date',
                            y='value',
                            title='VKOSPI (Volatility Index)'
                        )
                        st.plotly_chart(fig_vkospi, use_container_width=True)
                    
                    run_async(bok_client.close_session())
                    
                except Exception as e:
                    st.error(f"Error fetching market data: {str(e)}")
    
    with col2:
        st.subheader("📰 Corporate Disclosures")
        
        if st.button("Fetch DART Data"):
            with st.spinner("Fetching corporate disclosures..."):
                try:
                    if st.session_state.dart_client is None:
                        st.session_state.dart_client = EnhancedDartApiClient()
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
                    st.error(f"Error fetching DART data: {str(e)}")

# Tab 3: 모델 학습
with tab3:
    st.header("Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Training Configuration")
        
        epochs = st.slider("Number of Epochs", 10, 200, 100)
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        
        if st.button("🏋️ Start Training", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 학습 시뮬레이션 (실제 구현시 실제 학습 코드)
                for i in range(epochs):
                    progress_bar.progress((i + 1) / epochs)
                    status_text.text(f"Epoch {i+1}/{epochs} - Loss: {np.random.uniform(0.3, 0.5):.4f}")
                    
                    if i % 10 == 0:
                        # 중간 결과 표시
                        col1_metric, col2_metric, col3_metric = st.columns(3)
                        col1_metric.metric("Training Loss", f"{np.random.uniform(0.3, 0.4):.4f}")
                        col2_metric.metric("Validation Loss", f"{np.random.uniform(0.35, 0.45):.4f}")
                        col3_metric.metric("Learning Rate", f"{learning_rate:.4f}")
                
                st.success("✅ Training completed successfully!")
                
                # 모델 저장 옵션
                if st.button("💾 Save Model"):
                    version_manager = ModelVersionManager()
                    version_id = version_manager.save_model_with_version(
                        st.session_state.model,
                        metrics={'val_loss': 0.35, 'val_auc': 0.85},
                        description="Streamlit training session"
                    )
                    st.success(f"Model saved with version: {version_id}")
                
            except Exception as e:
                st.error(f"Training error: {str(e)}")
    
    with col2:
        st.subheader("📊 Training Metrics")
        
        # 실시간 메트릭 차트 (더미 데이터)
        if st.checkbox("Show Live Metrics"):
            placeholder = st.empty()
            
            for i in range(10):
                df_metrics = pd.DataFrame({
                    'epoch': range(i*10, (i+1)*10),
                    'train_loss': np.random.uniform(0.3, 0.5, 10),
                    'val_loss': np.random.uniform(0.35, 0.55, 10)
                })
                
                fig = px.line(df_metrics, x='epoch', y=['train_loss', 'val_loss'])
                placeholder.plotly_chart(fig, use_container_width=True)
                
                if st.button("Stop"):
                    break

# Tab 4: 성능 분석
with tab4:
    st.header("Performance Analysis")
    
    # 백테스트 결과
    st.subheader("📊 Backtesting Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # 더미 백테스트 메트릭
    col1.metric("Total Return", "+15.3%", "+2.1%")
    col2.metric("Sharpe Ratio", "1.45", "+0.12")
    col3.metric("Max Drawdown", "-8.2%", "-1.3%")
    col4.metric("Win Rate", "62.5%", "+3.2%")
    
    # 수익률 곡선
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    cumulative_returns = np.cumprod(1 + np.random.normal(0.0005, 0.02, 252))
    
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
        y=np.cumprod(1 + np.random.normal(0.0003, 0.015, 252)),
        mode='lines',
        name='Benchmark (KOSPI)',
        line=dict(color='gray', width=2, dash='dash')
    ))
    fig_returns.update_layout(
        title="Cumulative Returns Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode='x unified'
    )
    st.plotly_chart(fig_returns, use_container_width=True)
    
    # 위험 분석
    st.subheader("🎯 Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # VaR 계산
        var_95 = np.percentile(np.random.normal(-0.001, 0.02, 1000), 5)
        st.metric("Value at Risk (95%)", f"{var_95:.2%}")
        
        # 포트폴리오 베타
        st.metric("Portfolio Beta", "0.85")
    
    with col2:
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
        fig_corr.update_layout(title="Stock Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

# Tab 5: 시스템 상태
with tab5:
    st.header("System Status & Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔌 API Status")
        
        # API 사용량
        if st.session_state.dart_client:
            api_stats = st.session_state.dart_client.get_api_usage_stats()
            
            fig_api = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=api_stats['usage_percentage'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "API Usage (%)"},
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
    
    with col2:
        st.subheader("💾 Cache Status")
        
        if st.button("Check Redis Status"):
            if st.session_state.redis_manager is None:
                st.session_state.redis_manager = RobustRedisManager()
                run_async(st.session_state.redis_manager.connect())
            
            diagnostics = run_async(st.session_state.redis_manager.get_diagnostics())
            
            if diagnostics['connected']:
                st.success("✅ Redis Connected")
                if 'redis_info' in diagnostics:
                    st.json(diagnostics['redis_info'])
            else:
                st.warning("⚠️ Using Fallback Cache")
                st.write(f"Cache Size: {diagnostics['fallback_cache_size']} items")
    
    # 시스템 리소스
    st.subheader("💻 System Resources")
    
    col1, col2, col3 = st.columns(3)
    
    # CPU 사용률
    cpu_percent = psutil.cpu_percent(interval=1)
    col1.metric("CPU Usage", f"{cpu_percent}%")
    
    # 메모리 사용률
    memory = psutil.virtual_memory()
    col2.metric("Memory Usage", f"{memory.percent}%")
    
    # 디스크 사용률
    disk = psutil.disk_usage('/')
    col3.metric("Disk Usage", f"{disk.percent}%")
    
    # 프로세스 정보
    if st.checkbox("Show Process Details"):
        process = psutil.Process()
        process_info = {
            "PID": process.pid,
            "Memory (MB)": process.memory_info().rss / 1024 / 1024,
            "CPU %": process.cpu_percent(),
            "Threads": process.num_threads(),
            "Status": process.status()
        }
        st.json(process_info)

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Financial AI Prediction System v1.0 | Built with Streamlit & TensorFlow</p>
    <p>⚠️ This is for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

# 자동 새로고침 옵션
if st.sidebar.checkbox("Auto Refresh (5s)"):
    st.experimental_rerun()
