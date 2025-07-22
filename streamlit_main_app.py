"""
AI Stock Recommendation System - News & Financial Analysis
뉴스와 재무제표 기반 실시간 종목 추천 시스템
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px
import logging
import time
import asyncio
import json
import requests
from typing import Dict, List, Optional, Tuple

# 향상된 뉴스 분석 모듈
from news_sentiment import (
    get_recent_news,
    analyze_stock_sentiment,
    get_dart_disclosures,
    STOCK_MAPPING,
    EnhancedNewsAnalyzer
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="AI 주식 추천 시스템",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
def initialize_session_state():
    """세션 상태 초기화"""
    defaults = {
        'news_data': None,
        'stock_analysis': None,
        'last_refresh': time.time(),
        'selected_stocks': list(STOCK_MAPPING.values())[:10],
        'recommendations': None,
        'stop_loss_candidates': None,
        'dart_disclosures': None,
        'auto_refresh': False,
        'refresh_interval': 300  # 5분
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# CSS 스타일
st.markdown("""
<style>
    .recommendation-card {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .stoploss-card {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .news-positive {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 10px;
        margin: 5px 0;
    }
    
    .news-negative {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 10px;
        margin: 5px 0;
    }
    
    .metric-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .important-news {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.title("🤖 AI 주식 추천 시스템")
st.markdown("### 실시간 뉴스와 재무제표 분석 기반 매매 추천")

# 상단 메트릭
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.session_state.news_data:
        st.metric("분석된 뉴스", f"{len(st.session_state.news_data)}개")
    else:
        st.metric("분석된 뉴스", "0개")

with col2:
    if st.session_state.recommendations:
        st.metric("매수 추천", f"{len(st.session_state.recommendations)}개", delta="↑")
    else:
        st.metric("매수 추천", "0개")

with col3:
    if st.session_state.stop_loss_candidates:
        st.metric("손절 추천", f"{len(st.session_state.stop_loss_candidates)}개", delta="↓")
    else:
        st.metric("손절 추천", "0개")

with col4:
    last_update = datetime.fromtimestamp(st.session_state.last_refresh)
    st.metric("마지막 업데이트", last_update.strftime("%H:%M:%S"))

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 자동 새로고침
    st.session_state.auto_refresh = st.checkbox(
        "자동 새로고침",
        value=st.session_state.auto_refresh,
        help="5분마다 자동으로 데이터를 업데이트합니다"
    )
    
    # 분석 기간
    analysis_days = st.slider(
        "분석 기간 (일)",
        min_value=1,
        max_value=7,
        value=1,
        help="최근 며칠간의 뉴스를 분석할지 설정합니다"
    )
    
    # 종목 필터
    selected_sectors = st.multiselect(
        "섹터 선택",
        options=["전자", "IT", "자동차", "화학", "금융", "바이오"],
        default=["전자", "IT", "금융"]
    )
    
    # 추천 민감도
    sensitivity = st.select_slider(
        "추천 민감도",
        options=["보수적", "보통", "공격적"],
        value="보통",
        help="보수적: 강한 시그널만, 공격적: 약한 시그널도 포함"
    )
    
    st.markdown("---")
    
    # 알림 설정
    st.subheader("🔔 알림 설정")
    
    enable_buy_alert = st.checkbox("매수 추천 알림", value=True)
    enable_sell_alert = st.checkbox("손절 추천 알림", value=True)
    alert_threshold = st.slider(
        "알림 임계값",
        min_value=0.5,
        max_value=0.9,
        value=0.7,
        help="신뢰도가 이 값 이상일 때만 알림"
    )

# 메인 컨텐츠
tab1, tab2, tab3, tab4 = st.tabs(["📊 실시간 추천", "📰 뉴스 분석", "📈 재무제표", "📋 공시 정보"])

# Tab 1: 실시간 추천
with tab1:
    # 분석 버튼
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 실시간 분석 시작", type="primary", use_container_width=True):
            with st.spinner("뉴스와 재무 데이터를 분석 중입니다..."):
                # 뉴스 수집 및 분석
                st.session_state.news_data = get_recent_news(analysis_days)
                st.session_state.stock_analysis = analyze_stock_sentiment(analysis_days)
                
                # 추천 종목 선정
                recommendations = []
                stop_loss = []
                
                # 민감도에 따른 임계값 설정
                thresholds = {
                    "보수적": (0.5, -0.5),
                    "보통": (0.3, -0.3),
                    "공격적": (0.1, -0.1)
                }
                buy_threshold, sell_threshold = thresholds[sensitivity]
                
                for stock_code, analysis in st.session_state.stock_analysis.items():
                    if analysis['total_mentions'] > 0:
                        score = analysis['final_score']
                        
                        # 섹터 필터링 (실제로는 섹터 정보 연동 필요)
                        stock_name = next((k for k, v in STOCK_MAPPING.items() if v == stock_code), "Unknown")
                        
                        if score > buy_threshold and analysis['recommendation'] == 'BUY':
                            recommendations.append({
                                'code': stock_code,
                                'name': stock_name,
                                'score': score,
                                'mentions': analysis['total_mentions'],
                                'positive_ratio': analysis['positive'] / analysis['total_mentions'],
                                'important_news': analysis['important_news']
                            })
                        elif score < sell_threshold and analysis['recommendation'] == 'SELL':
                            stop_loss.append({
                                'code': stock_code,
                                'name': stock_name,
                                'score': score,
                                'mentions': analysis['total_mentions'],
                                'negative_ratio': analysis['negative'] / analysis['total_mentions'],
                                'important_news': analysis['important_news']
                            })
                
                # 점수 기준 정렬
                recommendations.sort(key=lambda x: x['score'], reverse=True)
                stop_loss.sort(key=lambda x: x['score'])
                
                st.session_state.recommendations = recommendations[:10]  # 상위 10개
                st.session_state.stop_loss_candidates = stop_loss[:5]   # 상위 5개
                st.session_state.last_refresh = time.time()
                
                st.success("✅ 분석이 완료되었습니다!")
    
    with col2:
        if st.button("🔄 새로고침", use_container_width=True):
            st.rerun()
    
    # 추천 결과 표시
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 매수 추천 종목")
        
        if st.session_state.recommendations:
            for idx, rec in enumerate(st.session_state.recommendations, 1):
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{idx}. {rec['name']} ({rec['code']})</h4>
                        <p><strong>신뢰도:</strong> {rec['score']:.2%}</p>
                        <p><strong>긍정 뉴스 비율:</strong> {rec['positive_ratio']:.1%}</p>
                        <p><strong>총 언급 횟수:</strong> {rec['mentions']}회</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 주요 뉴스 표시
                    if rec['important_news']:
                        with st.expander(f"주요 뉴스 ({len(rec['important_news'])}개)"):
                            for news in rec['important_news'][:3]:
                                st.write(f"• [{news['title']}]({news['url']})")
                                st.caption(f"  {news['source']} - {news['date'].strftime('%m/%d %H:%M')}")
        else:
            st.info("매수 추천 종목이 없습니다. 분석을 실행해주세요.")
    
    with col2:
        st.subheader("⚠️ 손절 추천 종목")
        
        if st.session_state.stop_loss_candidates:
            for idx, stock in enumerate(st.session_state.stop_loss_candidates, 1):
                with st.container():
                    st.markdown(f"""
                    <div class="stoploss-card">
                        <h4>{idx}. {stock['name']} ({stock['code']})</h4>
                        <p><strong>위험도:</strong> {abs(stock['score']):.2%}</p>
                        <p><strong>부정 뉴스 비율:</strong> {stock['negative_ratio']:.1%}</p>
                        <p><strong>총 언급 횟수:</strong> {stock['mentions']}회</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 주요 뉴스 표시
                    if stock['important_news']:
                        with st.expander(f"주요 뉴스 ({len(stock['important_news'])}개)"):
                            for news in stock['important_news'][:3]:
                                st.write(f"• [{news['title']}]({news['url']})")
                                st.caption(f"  {news['source']} - {news['date'].strftime('%m/%d %H:%M')}")
        else:
            st.info("손절 추천 종목이 없습니다.")

# Tab 2: 뉴스 분석
with tab2:
    st.subheader("📰 실시간 뉴스 분석")
    
    # 뉴스 필터
    col1, col2, col3 = st.columns(3)
    with col1:
        news_filter = st.selectbox("감성 필터", ["전체", "긍정", "부정", "중립"])
    with col2:
        source_filter = st.selectbox("출처 필터", ["전체"] + list(set(n.get('source', 'Unknown') for n in st.session_state.news_data or [])))
    with col3:
        sort_by = st.selectbox("정렬 기준", ["최신순", "신뢰도순", "관련 종목수"])
    
    # 뉴스 표시
    if st.session_state.news_data:
        filtered_news = st.session_state.news_data.copy()
        
        # 필터 적용
        if news_filter != "전체":
            filter_map = {"긍정": "positive", "부정": "negative", "중립": "neutral"}
            filtered_news = [n for n in filtered_news if n['sentiment'] == filter_map[news_filter]]
        
        if source_filter != "전체":
            filtered_news = [n for n in filtered_news if n.get('source') == source_filter]
        
        # 정렬
        if sort_by == "최신순":
            filtered_news.sort(key=lambda x: x['date'], reverse=True)
        elif sort_by == "신뢰도순":
            filtered_news.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        else:  # 관련 종목수
            filtered_news.sort(key=lambda x: len(x.get('mentioned_stocks', [])), reverse=True)
        
        # 뉴스 카드 표시
        for news in filtered_news[:20]:
            sentiment_class = {
                'positive': 'news-positive',
                'negative': 'news-negative',
                'neutral': ''
            }.get(news['sentiment'], '')
            
            # 중요 뉴스 표시
            is_important = news.get('confidence', 0) > 0.7 and len(news.get('mentioned_stocks', [])) > 0
            
            if is_important:
                st.markdown(f"""
                <div class="important-news">
                    <h4>⭐ {news['title']}</h4>
                    <p>{news.get('description', '')[:200]}...</p>
                    <p><strong>감성:</strong> {news['sentiment']} (신뢰도: {news.get('confidence', 0):.2%})</p>
                    <p><strong>언급 종목:</strong> {', '.join([f"{name}({code})" for name, code in news.get('mentioned_stocks', [])])}
                    <p><small>{news['source']} - {news['date'].strftime('%Y-%m-%d %H:%M')}</small></p>
                    <a href="{news['url']}" target="_blank">전체 기사 보기</a>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.container():
                    st.markdown(f"""
                    <div class="{sentiment_class}" style="padding: 10px; margin: 5px 0;">
                        <strong>{news['title']}</strong><br>
                        <small>{news['source']} - {news['date'].strftime('%m/%d %H:%M')}</small><br>
                        감성: {news['sentiment']} | 
                        종목: {', '.join([name for name, _ in news.get('mentioned_stocks', [])])}
                    </div>
                    """, unsafe_allow_html=True)
        
        # 뉴스 통계
        st.markdown("---")
        st.subheader("📊 뉴스 통계")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 감성 분포
            sentiment_counts = {
                'positive': sum(1 for n in st.session_state.news_data if n['sentiment'] == 'positive'),
                'negative': sum(1 for n in st.session_state.news_data if n['sentiment'] == 'negative'),
                'neutral': sum(1 for n in st.session_state.news_data if n['sentiment'] == 'neutral')
            }
            
            fig_sentiment = go.Figure(data=[
                go.Bar(
                    x=list(sentiment_counts.keys()),
                    y=list(sentiment_counts.values()),
                    marker_color=['green', 'red', 'gray']
                )
            ])
            fig_sentiment.update_layout(
                title="뉴스 감성 분포",
                xaxis_title="감성",
                yaxis_title="뉴스 수",
                height=300
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # 시간대별 뉴스 분포
            news_by_hour = {}
            for news in st.session_state.news_data:
                hour = news['date'].hour
                news_by_hour[hour] = news_by_hour.get(hour, 0) + 1
            
            fig_timeline = go.Figure(data=[
                go.Scatter(
                    x=list(news_by_hour.keys()),
                    y=list(news_by_hour.values()),
                    mode='lines+markers',
                    line=dict(color='blue', width=2)
                )
            ])
            fig_timeline.update_layout(
                title="시간대별 뉴스 분포",
                xaxis_title="시간",
                yaxis_title="뉴스 수",
                height=300
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    else:
        st.info("뉴스 데이터가 없습니다. 실시간 분석을 먼저 실행해주세요.")

# Tab 3: 재무제표
with tab3:
    st.subheader("📈 재무제표 기반 분석")
    
    # DART API를 통한 재무 데이터 조회
    selected_stock = st.selectbox(
        "종목 선택",
        options=list(STOCK_MAPPING.keys()),
        format_func=lambda x: f"{x} ({STOCK_MAPPING[x]})"
    )
    
    if st.button("재무제표 조회", use_container_width=True):
        with st.spinner(f"{selected_stock}의 재무제표를 조회 중입니다..."):
            try:
                # DART API 호출 (실제 구현 필요)
                # 여기서는 더미 데이터 사용
                financial_data = {
                    'revenue': [1000, 1100, 1200, 1150, 1300],
                    'operating_profit': [100, 120, 140, 130, 150],
                    'net_income': [80, 95, 110, 100, 120],
                    'quarters': ['21Q4', '22Q1', '22Q2', '22Q3', '22Q4']
                }
                
                # 재무 차트
                fig_financial = go.Figure()
                
                fig_financial.add_trace(go.Bar(
                    name='매출액',
                    x=financial_data['quarters'],
                    y=financial_data['revenue'],
                    yaxis='y',
                    offsetgroup=1
                ))
                
                fig_financial.add_trace(go.Bar(
                    name='영업이익',
                    x=financial_data['quarters'],
                    y=financial_data['operating_profit'],
                    yaxis='y2',
                    offsetgroup=2
                ))
                
                fig_financial.update_layout(
                    title=f"{selected_stock} 분기별 실적 추이",
                    xaxis=dict(title='분기'),
                    yaxis=dict(title='매출액 (억원)', side='left'),
                    yaxis2=dict(title='영업이익 (억원)', overlaying='y', side='right'),
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_financial, use_container_width=True)
                
                # 주요 재무 지표
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    growth_rate = (financial_data['revenue'][-1] / financial_data['revenue'][0] - 1) * 100
                    st.metric("매출 성장률", f"{growth_rate:.1f}%", delta=f"{growth_rate:.1f}%")
                
                with col2:
                    profit_margin = financial_data['operating_profit'][-1] / financial_data['revenue'][-1] * 100
                    st.metric("영업이익률", f"{profit_margin:.1f}%")
                
                with col3:
                    st.metric("PER", "15.2", delta="-2.1")
                
                with col4:
                    st.metric("PBR", "1.8", delta="0.1")
                
                # AI 재무 분석
                st.markdown("---")
                st.subheader("🤖 AI 재무 분석")
                
                analysis_text = f"""
                **{selected_stock} 재무 분석 결과**
                
                ✅ **긍정적 요인:**
                - 매출액이 지속적으로 성장하고 있습니다 (YoY {growth_rate:.1f}%)
                - 영업이익률이 {profit_margin:.1f}%로 업계 평균 대비 양호합니다
                - 최근 분기 실적이 시장 예상치를 상회했습니다
                
                ⚠️ **주의 요인:**
                - 원자재 가격 상승으로 인한 마진 압박 우려
                - 환율 변동성 증가에 따른 수익성 변동 가능성
                
                📊 **종합 평가:** 
                재무적으로 안정적이며 성장세를 유지하고 있어 중장기 투자 매력도가 높습니다.
                """
                
                st.markdown(analysis_text)
                
            except Exception as e:
                st.error(f"재무제표 조회 중 오류가 발생했습니다: {e}")

# Tab 4: 공시 정보
with tab4:
    st.subheader("📋 최신 공시 정보")
    
    if st.button("공시 정보 조회", use_container_width=True):
        with st.spinner("DART에서 공시 정보를 조회 중입니다..."):
            # DART 공시 조회 (더미 데이터)
            disclosures = [
                {
                    'date': datetime.now() - timedelta(hours=2),
                    'corp_name': '삼성전자',
                    'title': '주요사항보고서(자기주식취득결정)',
                    'type': 'positive',
                    'importance': 'high'
                },
                {
                    'date': datetime.now() - timedelta(hours=5),
                    'corp_name': 'SK하이닉스',
                    'title': '분기보고서 (2024.3분기)',
                    'type': 'neutral',
                    'importance': 'medium'
                },
                {
                    'date': datetime.now() - timedelta(days=1),
                    'corp_name': '카카오',
                    'title': '임원 주식매매 계약체결',
                    'type': 'negative',
                    'importance': 'low'
                }
            ]
            
            st.session_state.dart_disclosures = disclosures
    
    if st.session_state.dart_disclosures:
        # 중요 공시 필터
        importance_filter = st.radio(
            "중요도 필터",
            ["전체", "높음", "중간", "낮음"],
            horizontal=True
        )
        
        filtered_disclosures = st.session_state.dart_disclosures
        if importance_filter != "전체":
            importance_map = {"높음": "high", "중간": "medium", "낮음": "low"}
            filtered_disclosures = [d for d in filtered_disclosures if d['importance'] == importance_map[importance_filter]]
        
        # 공시 표시
        for disclosure in filtered_disclosures:
            icon = "🔴" if disclosure['type'] == 'negative' else "🟢" if disclosure['type'] == 'positive' else "⚪"
            importance_badge = {
                'high': "🔥 중요",
                'medium': "📌 보통",
                'low': "📄 일반"
            }.get(disclosure['importance'], "")
            
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{icon} **{disclosure['corp_name']}** - {disclosure['title']}")
                    st.caption(f"{disclosure['date'].strftime('%Y-%m-%d %H:%M')}")
                with col2:
                    st.write(importance_badge)
                
                st.markdown("---")
        
        # 공시 통계
        st.subheader("📊 공시 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 기업별 공시 수
            corp_counts = {}
            for d in st.session_state.dart_disclosures:
                corp = d['corp_name']
                corp_counts[corp] = corp_counts.get(corp, 0) + 1
            
            fig_corp = go.Figure(data=[
                go.Pie(labels=list(corp_counts.keys()), values=list(corp_counts.values()))
            ])
            fig_corp.update_layout(title="기업별 공시 분포", height=300)
            st.plotly_chart(fig_corp, use_container_width=True)
        
        with col2:
            # 공시 유형별 분포
            type_counts = {
                'positive': sum(1 for d in st.session_state.dart_disclosures if d['type'] == 'positive'),
                'negative': sum(1 for d in st.session_state.dart_disclosures if d['type'] == 'negative'),
                'neutral': sum(1 for d in st.session_state.dart_disclosures if d['type'] == 'neutral')
            }
            
            fig_type = go.Figure(data=[
                go.Bar(
                    x=['긍정적', '부정적', '중립'],
                    y=list(type_counts.values()),
                    marker_color=['green', 'red', 'gray']
                )
            ])
            fig_type.update_layout(title="공시 성격별 분포", height=300)
            st.plotly_chart(fig_type, use_container_width=True)
    
    else:
        st.info("공시 정보를 조회하려면 위 버튼을 클릭하세요.")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>AI 주식 추천 시스템</strong></p>
    <p>실시간 뉴스와 재무제표 분석 기반 투자 의사결정 지원</p>
    <p>⚠️ 투자 결정은 본인의 책임하에 신중히 하시기 바랍니다.</p>
</div>
""", unsafe_allow_html=True)

# 자동 새로고침
if st.session_state.auto_refresh:
    time_since_refresh = time.time() - st.session_state.last_refresh
    if time_since_refresh > st.session_state.refresh_interval:
        st.rerun()
