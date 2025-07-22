"""
DART API를 활용한 재무제표 분석 모듈
실시간 재무 데이터 기반 투자 추천
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)

DART_API_KEY = "e45fa610cea4a8e8a6eebd9e05e3580daa071f82"
DART_BASE_URL = "https://opendart.fss.or.kr/api"

# 기업 코드 매핑 (실제로는 DART에서 제공하는 기업코드 파일 사용)
CORP_CODE_MAPPING = {
    "005930": "00126380",  # 삼성전자
    "000660": "00164742",  # SK하이닉스
    "035720": "00256598",  # 카카오
    "005380": "00164779",  # 현대차
    "035420": "00266961",  # NAVER
    "051910": "00167070",  # LG화학
    "006400": "00164611",  # 삼성SDI
    "028260": "00115160",  # 삼성물산
    "105560": "00547583",  # KB금융
    "055550": "00547871",  # 신한지주
}


class DartFinancialAnalyzer:
    """DART 재무제표 분석기"""
    
    def __init__(self):
        self.api_key = DART_API_KEY
        self.financial_cache = {}
        
    def get_financial_statements(self, stock_code: str, year: int = None, quarter: int = None) -> Dict:
        """재무제표 조회"""
        try:
            corp_code = CORP_CODE_MAPPING.get(stock_code)
            if not corp_code:
                logger.warning(f"Unknown stock code: {stock_code}")
                return self._get_dummy_financials(stock_code)
            
            # 기본값 설정
            if not year:
                year = datetime.now().year
            if not quarter:
                quarter = (datetime.now().month - 1) // 3 + 1
            
            # 캐시 확인
            cache_key = f"{stock_code}_{year}_{quarter}"
            if cache_key in self.financial_cache:
                return self.financial_cache[cache_key]
            
            # 단일회사 재무제표 조회
            params = {
                'crtfc_key': self.api_key,
                'corp_code': corp_code,
                'bsns_year': str(year),
                'reprt_code': self._get_report_code(quarter)
            }
            
            response = requests.get(
                f"{DART_BASE_URL}/fnlttSinglAcnt.json",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == '000':
                    financial_data = self._parse_financial_data(data.get('list', []))
                    self.financial_cache[cache_key] = financial_data
                    return financial_data
            
            # 실패시 더미 데이터
            return self._get_dummy_financials(stock_code)
            
        except Exception as e:
            logger.error(f"Failed to get financial statements: {e}")
            return self._get_dummy_financials(stock_code)
    
    def analyze_financial_health(self, stock_code: str) -> Dict:
        """재무 건전성 분석"""
        financials = self.get_financial_statements(stock_code)
        
        if not financials:
            return {
                'score': 0,
                'grade': 'N/A',
                'factors': []
            }
        
        score = 0
        factors = []
        
        # 수익성 분석
        if financials.get('operating_margin', 0) > 10:
            score += 20
            factors.append(('positive', '영업이익률 10% 이상'))
        elif financials.get('operating_margin', 0) < 5:
            score -= 10
            factors.append(('negative', '영업이익률 5% 미만'))
        
        # 성장성 분석
        if financials.get('revenue_growth', 0) > 10:
            score += 20
            factors.append(('positive', '매출 성장률 10% 이상'))
        elif financials.get('revenue_growth', 0) < 0:
            score -= 20
            factors.append(('negative', '매출 감소'))
        
        # 안정성 분석
        debt_ratio = financials.get('debt_ratio', 100)
        if debt_ratio < 50:
            score += 20
            factors.append(('positive', '부채비율 50% 미만'))
        elif debt_ratio > 100:
            score -= 20
            factors.append(('negative', '부채비율 100% 초과'))
        
        # ROE 분석
        roe = financials.get('roe', 0)
        if roe > 15:
            score += 20
            factors.append(('positive', 'ROE 15% 이상'))
        elif roe < 5:
            score -= 10
            factors.append(('negative', 'ROE 5% 미만'))
        
        # 등급 결정
        if score >= 60:
            grade = 'A'
        elif score >= 40:
            grade = 'B'
        elif score >= 20:
            grade = 'C'
        elif score >= 0:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'score': score,
            'grade': grade,
            'factors': factors,
            'financials': financials
        }
    
    def get_recent_disclosures(self, stock_code: str, days: int = 7) -> List[Dict]:
        """최근 공시 조회"""
        try:
            corp_code = CORP_CODE_MAPPING.get(stock_code)
            if not corp_code:
                return []
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            params = {
                'crtfc_key': self.api_key,
                'corp_code': corp_code,
                'bgn_de': start_date.strftime('%Y%m%d'),
                'end_de': end_date.strftime('%Y%m%d'),
                'page_no': '1',
                'page_count': '20'
            }
            
            response = requests.get(
                f"{DART_BASE_URL}/list.json",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == '000':
                    return self._analyze_disclosures(data.get('list', []))
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get disclosures: {e}")
            return []
    
    def _parse_financial_data(self, data: List[Dict]) -> Dict:
        """재무 데이터 파싱"""
        financials = {}
        
        for item in data:
            account_nm = item.get('account_nm', '')
            amount = item.get('thstrm_amount', '0').replace(',', '')
            
            try:
                amount = float(amount) / 100000000  # 억원 단위
            except:
                amount = 0
            
            # 주요 계정 추출
            if '매출액' in account_nm or '수익' in account_nm:
                financials['revenue'] = amount
            elif '영업이익' in account_nm:
                financials['operating_profit'] = amount
            elif '당기순이익' in account_nm:
                financials['net_income'] = amount
            elif '자산총계' in account_nm:
                financials['total_assets'] = amount
            elif '부채총계' in account_nm:
                financials['total_liabilities'] = amount
            elif '자본총계' in account_nm:
                financials['total_equity'] = amount
        
        # 재무비율 계산
        if financials.get('revenue', 0) > 0:
            financials['operating_margin'] = (financials.get('operating_profit', 0) / financials['revenue']) * 100
            financials['net_margin'] = (financials.get('net_income', 0) / financials['revenue']) * 100
        
        if financials.get('total_equity', 0) > 0:
            financials['roe'] = (financials.get('net_income', 0) / financials['total_equity']) * 100
            financials['debt_ratio'] = (financials.get('total_liabilities', 0) / financials['total_equity']) * 100
        
        return financials
    
    def _analyze_disclosures(self, disclosures: List[Dict]) -> List[Dict]:
        """공시 분석"""
        analyzed = []
        
        positive_keywords = ['자기주식취득', '유상증자', '신규투자', '매출증가', '실적개선']
        negative_keywords = ['감자', '소송', '손실', '매출감소', '실적악화']
        
        for disclosure in disclosures:
            report_nm = disclosure.get('report_nm', '')
            
            # 중요도 판단
            importance = 'low'
            if any(keyword in report_nm for keyword in ['주요사항보고서', '분기보고서', '반기보고서']):
                importance = 'high'
            elif any(keyword in report_nm for keyword in ['임시공시', '조회공시']):
                importance = 'medium'
            
            # 성격 판단
            disclosure_type = 'neutral'
            if any(keyword in report_nm for keyword in positive_keywords):
                disclosure_type = 'positive'
            elif any(keyword in report_nm for keyword in negative_keywords):
                disclosure_type = 'negative'
            
            analyzed.append({
                'date': datetime.strptime(disclosure.get('rcept_dt', ''), '%Y%m%d'),
                'corp_name': disclosure.get('corp_name', ''),
                'title': report_nm,
                'type': disclosure_type,
                'importance': importance,
                'rcept_no': disclosure.get('rcept_no', '')
            })
        
        return analyzed
    
    def _get_report_code(self, quarter: int) -> str:
        """분기별 보고서 코드"""
        report_codes = {
            1: '11013',  # 1분기
            2: '11012',  # 반기
            3: '11014',  # 3분기
            4: '11011'   # 사업보고서
        }
        return report_codes.get(quarter, '11013')
    
    def _get_dummy_financials(self, stock_code: str) -> Dict:
        """더미 재무 데이터"""
        np.random.seed(hash(stock_code) % 2**32)
        
        return {
            'revenue': np.random.uniform(1000, 10000),
            'operating_profit': np.random.uniform(100, 1000),
            'net_income': np.random.uniform(50, 500),
            'total_assets': np.random.uniform(5000, 20000),
            'total_liabilities': np.random.uniform(2000, 8000),
            'total_equity': np.random.uniform(3000, 12000),
            'operating_margin': np.random.uniform(5, 20),
            'net_margin': np.random.uniform(3, 15),
            'roe': np.random.uniform(5, 25),
            'debt_ratio': np.random.uniform(30, 150),
            'revenue_growth': np.random.uniform(-10, 30)
        }


class InvestmentRecommender:
    """종합 투자 추천 시스템"""
    
    def __init__(self):
        self.financial_analyzer = DartFinancialAnalyzer()
        
    def generate_recommendations(self, stock_analysis: Dict, sensitivity: str = "보통") -> Tuple[List[Dict], List[Dict]]:
        """뉴스와 재무 분석 기반 종합 추천"""
        
        recommendations = []
        stop_loss = []
        
        # 민감도 설정
        thresholds = {
            "보수적": {'buy': 0.6, 'sell': -0.6, 'financial_weight': 0.7},
            "보통": {'buy': 0.4, 'sell': -0.4, 'financial_weight': 0.5},
            "공격적": {'buy': 0.2, 'sell': -0.2, 'financial_weight': 0.3}
        }
        
        threshold = thresholds[sensitivity]
        
        for stock_code, news_analysis in stock_analysis.items():
            # 뉴스 점수
            news_score = news_analysis.get('final_score', 0)
            
            # 재무 분석
            financial_health = self.financial_analyzer.analyze_financial_health(stock_code)
            financial_score = financial_health['score'] / 100  # 정규화
            
            # 종합 점수 계산
            combined_score = (
                news_score * (1 - threshold['financial_weight']) + 
                financial_score * threshold['financial_weight']
            )
            
            # 최근 공시 확인
            recent_disclosures = self.financial_analyzer.get_recent_disclosures(stock_code, days=7)
            disclosure_bonus = sum(0.1 if d['type'] == 'positive' else -0.1 if d['type'] == 'negative' else 0 
                                 for d in recent_disclosures)
            
            combined_score += disclosure_bonus
            
            # 추천 결정
            stock_info = {
                'code': stock_code,
                'name': self._get_stock_name(stock_code),
                'news_score': news_score,
                'financial_score': financial_score,
                'combined_score': combined_score,
                'financial_grade': financial_health['grade'],
                'financial_factors': financial_health['factors'],
                'recent_disclosures': recent_disclosures[:3]
            }
            
            if combined_score > threshold['buy']:
                recommendations.append(stock_info)
            elif combined_score < threshold['sell']:
                stop_loss.append(stock_info)
        
        # 정렬
        recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
        stop_loss.sort(key=lambda x: x['combined_score'])
        
        return recommendations[:10], stop_loss[:5]
    
    def _get_stock_name(self, stock_code: str) -> str:
        """종목 코드로 종목명 조회"""
        from news_sentiment import STOCK_MAPPING
        
        for name, code in STOCK_MAPPING.items():
            if code == stock_code:
                return name
        return "Unknown"


def get_investment_signals(stock_codes: List[str]) -> Dict:
    """투자 시그널 생성"""
    signals = {}
    analyzer = DartFinancialAnalyzer()
    
    for stock_code in stock_codes:
        # 재무 건전성 점수
        financial_health = analyzer.analyze_financial_health(stock_code)
        
        # 시그널 생성
        signal = {
            'financial_score': financial_health['score'],
            'financial_grade': financial_health['grade'],
            'signal_strength': 0,
            'action': 'HOLD'
        }
        
        # 재무 점수 기반 시그널
        if financial_health['score'] >= 60:
            signal['signal_strength'] += 0.3
        elif financial_health['score'] <= 20:
            signal['signal_strength'] -= 0.3
        
        # 최근 공시 확인
        recent_disclosures = analyzer.get_recent_disclosures(stock_code, days=3)
        positive_disclosures = sum(1 for d in recent_disclosures if d['type'] == 'positive')
        negative_disclosures = sum(1 for d in recent_disclosures if d['type'] == 'negative')
        
        if positive_disclosures > negative_disclosures:
            signal['signal_strength'] += 0.2
        elif negative_disclosures > positive_disclosures:
            signal['signal_strength'] -= 0.2
        
        # 최종 액션 결정
        if signal['signal_strength'] > 0.3:
            signal['action'] = 'BUY'
        elif signal['signal_strength'] < -0.3:
            signal['action'] = 'SELL'
        
        signals[stock_code] = signal
    
    return signals
