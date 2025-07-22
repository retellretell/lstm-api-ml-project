##integrated_recommendation.py

"""
통합 추천 시스템
뉴스, 재무제표, 기술적 분석을 통합한 원클릭 투자 추천
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

from news_sentiment import analyze_stock_sentiment, get_recent_news, EnhancedNewsAnalyzer
from dart_financial_analyzer import DartFinancialAnalyzer, InvestmentRecommender

logger = logging.getLogger(__name__)


class IntegratedRecommendationSystem:
    """통합 추천 시스템"""
    
    def __init__(self):
        self.news_analyzer = EnhancedNewsAnalyzer()
        self.financial_analyzer = DartFinancialAnalyzer()
        self.recommender = InvestmentRecommender()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def one_click_analysis(self, 
                                sensitivity: str = "보통",
                                sectors: List[str] = None) -> Dict:
        """원클릭 종합 분석"""
        
        try:
            # 1. 뉴스 분석 (비동기)
            news_task = asyncio.create_task(self._analyze_news_async())
            
            # 2. 기본 종목 리스트 (섹터 필터 적용)
            target_stocks = self._get_target_stocks(sectors)
            
            # 3. 재무 분석 (병렬 처리)
            financial_tasks = []
            for stock_code in target_stocks:
                task = asyncio.create_task(self._analyze_financial_async(stock_code))
                financial_tasks.append(task)
            
            # 4. 모든 분석 완료 대기
            news_analysis = await news_task
            financial_results = await asyncio.gather(*financial_tasks)
            
            # 5. 종합 점수 계산
            combined_analysis = self._combine_analysis(
                news_analysis, 
                dict(zip(target_stocks, financial_results)),
                sensitivity
            )
            
            # 6. 최종 추천 생성
            recommendations = self._generate_final_recommendations(
                combined_analysis,
                sensitivity
            )
            
            return {
                'status': 'success',
                'timestamp': datetime.now(),
                'recommendations': recommendations['buy'][:5],
                'stop_loss': recommendations['sell'][:3],
                'market_summary': self._generate_market_summary(news_analysis),
                'analysis_details': combined_analysis
            }
            
        except Exception as e:
            logger.error(f"One-click analysis failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'recommendations': [],
                'stop_loss': []
            }
    
    async def _analyze_news_async(self) -> Dict:
        """비동기 뉴스 분석"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            analyze_stock_sentiment,
            1  # 최근 1일
        )
    
    async def _analyze_financial_async(self, stock_code: str) -> Dict:
        """비동기 재무 분석"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.financial_analyzer.analyze_financial_health,
            stock_code
        )
    
    def _get_target_stocks(self, sectors: List[str] = None) -> List[str]:
        """분석 대상 종목 선정"""
        from news_sentiment import STOCK_MAPPING
        
        # 섹터별 종목 매핑 (실제로는 더 정교한 분류 필요)
        sector_mapping = {
            "전자": ["005930", "000660", "006400"],
            "IT": ["035720", "035420"],
            "자동차": ["005380"],
            "화학": ["051910"],
            "금융": ["105560", "055550"],
            "바이오": ["068270", "207940"]
        }
        
        if sectors:
            stocks = []
            for sector in sectors:
                stocks.extend(sector_mapping.get(sector, []))
            return list(set(stocks))
        else:
            return list(STOCK_MAPPING.values())[:20]  # 상위 20개
    
    def _combine_analysis(self, 
                         news_analysis: Dict,
                         financial_analysis: Dict,
                         sensitivity: str) -> Dict:
        """뉴스와 재무 분석 통합"""
        
        combined = {}
        
        # 가중치 설정
        weights = {
            "보수적": {'news': 0.3, 'financial': 0.5, 'technical': 0.2},
            "보통": {'news': 0.4, 'financial': 0.4, 'technical': 0.2},
            "공격적": {'news': 0.5, 'financial': 0.3, 'technical': 0.2}
        }
        
        weight = weights[sensitivity]
        
        for stock_code in financial_analysis:
            # 뉴스 점수
            news_score = 0
            if stock_code in news_analysis:
                news_data = news_analysis[stock_code]
                if news_data['total_mentions'] > 0:
                    news_score = news_data['final_score']
            
            # 재무 점수
            financial_score = financial_analysis[stock_code]['score'] / 100
            
            # 기술적 점수 (더미)
            technical_score = np.random.uniform(-0.5, 0.5)
            
            # 종합 점수
            total_score = (
                news_score * weight['news'] +
                financial_score * weight['financial'] +
                technical_score * weight['technical']
            )
            
            combined[stock_code] = {
                'news_score': news_score,
                'financial_score': financial_score,
                'technical_score': technical_score,
                'total_score': total_score,
                'financial_grade': financial_analysis[stock_code]['grade'],
                'news_sentiment': 'positive' if news_score > 0 else 'negative' if news_score < 0 else 'neutral'
            }
        
        return combined
    
    def _generate_final_recommendations(self, 
                                      combined_analysis: Dict,
                                      sensitivity: str) -> Dict:
        """최종 추천 생성"""
        
        buy_recommendations = []
        sell_recommendations = []
        
        # 임계값 설정
        thresholds = {
            "보수적": {'buy': 0.5, 'sell': -0.5},
            "보통": {'buy': 0.3, 'sell': -0.3},
            "공격적": {'buy': 0.1, 'sell': -0.1}
        }
        
        threshold = thresholds[sensitivity]
        
        for stock_code, analysis in combined_analysis.items():
            score = analysis['total_score']
            
            # 종목명 조회
            from news_sentiment import STOCK_MAPPING
            stock_name = next((name for name, code in STOCK_MAPPING.items() if code == stock_code), "Unknown")
            
            recommendation = {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'score': score,
                'confidence': abs(score),
                'financial_grade': analysis['financial_grade'],
                'news_sentiment': analysis['news_sentiment'],
                'reasons': self._generate_reasons(analysis)
            }
            
            if score > threshold['buy']:
                buy_recommendations.append(recommendation)
            elif score < threshold['sell']:
                sell_recommendations.append(recommendation)
        
        # 정렬
        buy_recommendations.sort(key=lambda x: x['score'], reverse=True)
        sell_recommendations.sort(key=lambda x: x['score'])
        
        return {
            'buy': buy_recommendations,
            'sell': sell_recommendations
        }
    
    def _generate_reasons(self, analysis: Dict) -> List[str]:
        """추천 이유 생성"""
        reasons = []
        
        # 뉴스 기반
        if analysis['news_score'] > 0.3:
            reasons.append("긍정적인 뉴스 다수")
        elif analysis['news_score'] < -0.3:
            reasons.append("부정적인 뉴스 우세")
        
        # 재무 기반
        if analysis['financial_grade'] in ['A', 'B']:
            reasons.append("우수한 재무 건전성")
        elif analysis['financial_grade'] in ['D', 'F']:
            reasons.append("재무 건전성 취약")
        
        # 기술적 기반
        if analysis['technical_score'] > 0.2:
            reasons.append("기술적 상승 신호")
        elif analysis['technical_score'] < -0.2:
            reasons.append("기술적 하락 신호")
        
        return reasons
    
    def _generate_market_summary(self, news_analysis: Dict) -> Dict:
        """시장 요약 생성"""
        
        total_mentions = sum(data['total_mentions'] for data in news_analysis.values())
        positive_mentions = sum(data['positive'] for data in news_analysis.values())
        negative_mentions = sum(data['negative'] for data in news_analysis.values())
        
        market_sentiment = "neutral"
        if total_mentions > 0:
            positive_ratio = positive_mentions / total_mentions
            if positive_ratio > 0.6:
                market_sentiment = "bullish"
            elif positive_ratio < 0.4:
                market_sentiment = "bearish"
        
        return {
            'overall_sentiment': market_sentiment,
            'total_news': total_mentions,
            'positive_ratio': positive_mentions / total_mentions if total_mentions > 0 else 0,
            'negative_ratio': negative_mentions / total_mentions if total_mentions > 0 else 0,
            'most_mentioned': self._get_most_mentioned_stocks(news_analysis)
        }
    
    def _get_most_mentioned_stocks(self, news_analysis: Dict) -> List[Dict]:
        """가장 많이 언급된 종목"""
        mentioned = []
        
        for stock_code, data in news_analysis.items():
            if data['total_mentions'] > 0:
                from news_sentiment import STOCK_MAPPING
                stock_name = next((name for name, code in STOCK_MAPPING.items() if code == stock_code), "Unknown")
                
                mentioned.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'mentions': data['total_mentions'],
                    'sentiment': 'positive' if data['final_score'] > 0 else 'negative' if data['final_score'] < 0 else 'neutral'
                })
        
        mentioned.sort(key=lambda x: x['mentions'], reverse=True)
        return mentioned[:5]


# 간편 사용을 위한 헬퍼 함수
async def get_instant_recommendations(sensitivity: str = "보통") -> Dict:
    """즉시 사용 가능한 추천 함수"""
    system = IntegratedRecommendationSystem()
    return await system.one_click_analysis(sensitivity=sensitivity)


def get_recommendations_sync(sensitivity: str = "보통") -> Dict:
    """동기식 추천 함수 (Streamlit용)"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_instant_recommendations(sensitivity))
    finally:
        loop.close()
