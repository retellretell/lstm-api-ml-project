# DART API Integration Module for Real-time Corporate Disclosures

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import redis.asyncio as redis
import hashlib

logger = logging.getLogger(__name__)

class EnhancedDartApiClient:
    """개선된 DART API 클라이언트 with 실시간 데이터 통합"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('DART_API_KEY')
        self.base_url = "https://opendart.fss.or.kr/api"
        self.session = None
        self.cache_ttl = 3600  # 1시간 캐시
        
        # Redis 캐시 설정
        self.redis_client = None
        self.use_cache = True
        
        # 속도 제한 설정
        self.rate_limiter = EnhancedAdaptiveRateLimiter(
            max_calls_per_second=10,
            max_calls_per_day=10000
        )
        
        if not self.api_key:
            logger.warning("DART API 키가 설정되지 않았습니다. 더미 데이터를 사용합니다.")
    
    async def initialize(self):
        """비동기 초기화"""
        try:
            # aiohttp 세션 생성
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Redis 연결 시도
            if self.use_cache:
                try:
                    self.redis_client = redis.Redis(
                        host='localhost',
                        port=6379,
                        decode_responses=True,
                        socket_connect_timeout=5
                    )
                    await self.redis_client.ping()
                    logger.info("✅ Redis 캐시 연결 성공")
                except Exception as e:
                    logger.warning(f"Redis 연결 실패, 로컬 캐시 사용: {e}")
                    self.redis_client = None
                    
        except Exception as e:
            logger.error(f"DART API 클라이언트 초기화 실패: {e}")
    
    async def close(self):
        """리소스 정리"""
        if self.session:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def fetch_company_disclosures(self, corp_code: str, 
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None) -> List[Dict]:
        """기업 공시 정보 조회"""
        try:
            # API 키 확인
            if not self.api_key:
                return self._generate_dummy_disclosures(corp_code)
            
            # 날짜 설정
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # 캐시 키 생성
            cache_key = f"dart:disclosures:{corp_code}:{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}"
            
            # 캐시 확인
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    logger.debug(f"캐시 히트: {cache_key}")
                    return json.loads(cached_data)
            
            # 속도 제한 확인
            await self.rate_limiter.check_and_wait()
            
            # API 호출
            params = {
                'crtfc_key': self.api_key,
                'corp_code': corp_code,
                'bgn_de': start_date.strftime('%Y%m%d'),
                'end_de': end_date.strftime('%Y%m%d'),
                'page_no': '1',
                'page_count': '100'
            }
            
            url = f"{self.base_url}/list.json"
            
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data.get('status') == '000':  # 성공
                    disclosures = data.get('list', [])
                    
                    # 캐시 저장
                    if self.redis_client and disclosures:
                        await self.redis_client.setex(
                            cache_key,
                            self.cache_ttl,
                            json.dumps(disclosures)
                        )
                    
                    return disclosures
                else:
                    logger.error(f"DART API 오류: {data.get('message')}")
                    return []
                    
        except Exception as e:
            logger.error(f"공시 정보 조회 실패 ({corp_code}): {e}")
            return self._generate_dummy_disclosures(corp_code)
    
    async def fetch_financial_statements(self, corp_code: str, year: int, quarter: int) -> Dict:
        """재무제표 조회"""
        try:
            if not self.api_key:
                return self._generate_dummy_financial_data(corp_code, year, quarter)
            
            # 캐시 키
            cache_key = f"dart:financial:{corp_code}:{year}:Q{quarter}"
            
            # 캐시 확인
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            
            # 속도 제한
            await self.rate_limiter.check_and_wait()
            
            # API 호출
            params = {
                'crtfc_key': self.api_key,
                'corp_code': corp_code,
                'bsns_year': str(year),
                'reprt_code': f'1101{quarter}'  # 분기보고서 코드
            }
            
            url = f"{self.base_url}/fnlttSinglAcnt.json"
            
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data.get('status') == '000':
                    financial_data = self._parse_financial_statements(data.get('list', []))
                    
                    # 캐시 저장
                    if self.redis_client:
                        await self.redis_client.setex(
                            cache_key,
                            self.cache_ttl * 24,  # 재무제표는 24시간 캐시
                            json.dumps(financial_data)
                        )
                    
                    return financial_data
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"재무제표 조회 실패 ({corp_code}): {e}")
            return self._generate_dummy_financial_data(corp_code, year, quarter)
    
    async def get_company_overview(self, corp_code: str) -> Dict:
        """기업 개요 조회"""
        try:
            if not self.api_key:
                return self._generate_dummy_company_overview(corp_code)
            
            # 캐시 확인
            cache_key = f"dart:company:{corp_code}"
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            
            # API 호출
            params = {
                'crtfc_key': self.api_key,
                'corp_code': corp_code
            }
            
            url = f"{self.base_url}/company.json"
            
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data.get('status') == '000':
                    company_info = {
                        'corp_name': data.get('corp_name'),
                        'corp_name_eng': data.get('corp_name_eng'),
                        'stock_name': data.get('stock_name'),
                        'stock_code': data.get('stock_code'),
                        'ceo_nm': data.get('ceo_nm'),
                        'corp_cls': data.get('corp_cls'),
                        'jurir_no': data.get('jurir_no'),
                        'bizr_no': data.get('bizr_no'),
                        'adres': data.get('adres'),
                        'hm_url': data.get('hm_url'),
                        'ir_url': data.get('ir_url'),
                        'phn_no': data.get('phn_no'),
                        'fax_no': data.get('fax_no'),
                        'induty_code': data.get('induty_code'),
                        'est_dt': data.get('est_dt'),
                        'acc_mt': data.get('acc_mt')
                    }
                    
                    # 캐시 저장 (7일)
                    if self.redis_client:
                        await self.redis_client.setex(
                            cache_key,
                            self.cache_ttl * 24 * 7,
                            json.dumps(company_info)
                        )
                    
                    return company_info
                    
        except Exception as e:
            logger.error(f"기업 개요 조회 실패 ({corp_code}): {e}")
            return self._generate_dummy_company_overview(corp_code)
    
    def analyze_disclosure_impact(self, disclosure: Dict) -> Dict:
        """공시 영향도 분석"""
        try:
            report_nm = disclosure.get('report_nm', '').lower()
            
            # 긍정적 키워드
            positive_keywords = [
                '실적', '증가', '상승', '개선', '신규', '계약', '수주', '특허',
                '배당', '자사주', '매입', '흑자전환', '영업이익', '매출액'
            ]
            
            # 부정적 키워드
            negative_keywords = [
                '감소', '하락', '손실', '적자', '감자', '소송', '제재', '처분',
                '정정', '취소', '철회', '부도', '파산', '구조조정'
            ]
            
            # 중요도 가중치
            importance_weights = {
                '주요사항보고서': 0.8,
                '분기보고서': 0.7,
                '사업보고서': 0.7,
                '임시공시': 0.6,
                '정정공시': 0.5,
                '기타공시': 0.3
            }
            
            # 영향도 점수 계산
            positive_score = sum(1 for keyword in positive_keywords if keyword in report_nm)
            negative_score = sum(1 for keyword in negative_keywords if keyword in report_nm)
            
            # 공시 유형별 가중치
            disclosure_type = disclosure.get('report_nm', '').split()[0]
            weight = importance_weights.get(disclosure_type, 0.5)
            
            # 최종 영향도 계산
            impact_score = (positive_score - negative_score) * weight
            
            # 영향도 분류
            if impact_score > 0.5:
                impact = 'positive'
                confidence = min(0.9, 0.6 + impact_score * 0.1)
            elif impact_score < -0.5:
                impact = 'negative'
                confidence = min(0.9, 0.6 + abs(impact_score) * 0.1)
            else:
                impact = 'neutral'
                confidence = 0.5
            
            return {
                'impact': impact,
                'score': impact_score,
                'confidence': confidence,
                'positive_signals': positive_score,
                'negative_signals': negative_score,
                'importance_weight': weight
            }
            
        except Exception as e:
            logger.error(f"공시 영향도 분석 실패: {e}")
            return {
                'impact': 'neutral',
                'score': 0,
                'confidence': 0.5
            }
    
    def _parse_financial_statements(self, data: List[Dict]) -> Dict:
        """재무제표 데이터 파싱"""
        try:
            financial_metrics = {}
            
            for item in data:
                account_nm = item.get('account_nm', '')
                thstrm_amount = item.get('thstrm_amount', '0')
                
                # 주요 재무 지표 추출
                if '매출액' in account_nm:
                    financial_metrics['revenue'] = int(thstrm_amount.replace(',', ''))
                elif '영업이익' in account_nm:
                    financial_metrics['operating_profit'] = int(thstrm_amount.replace(',', ''))
                elif '당기순이익' in account_nm:
                    financial_metrics['net_income'] = int(thstrm_amount.replace(',', ''))
                elif '자산총계' in account_nm:
                    financial_metrics['total_assets'] = int(thstrm_amount.replace(',', ''))
                elif '부채총계' in account_nm:
                    financial_metrics['total_liabilities'] = int(thstrm_amount.replace(',', ''))
                elif '자본총계' in account_nm:
                    financial_metrics['total_equity'] = int(thstrm_amount.replace(',', ''))
            
            # 재무 비율 계산
            if financial_metrics.get('revenue'):
                financial_metrics['profit_margin'] = (
                    financial_metrics.get('net_income', 0) / financial_metrics['revenue']
                )
            
            if financial_metrics.get('total_equity') and financial_metrics.get('total_assets'):
                financial_metrics['debt_ratio'] = (
                    financial_metrics.get('total_liabilities', 0) / financial_metrics['total_assets']
                )
                financial_metrics['roe'] = (
                    financial_metrics.get('net_income', 0) / financial_metrics['total_equity']
                )
            
            return financial_metrics
            
        except Exception as e:
            logger.error(f"재무제표 파싱 실패: {e}")
            return {}
    
    def _generate_dummy_disclosures(self, corp_code: str) -> List[Dict]:
        """더미 공시 데이터 생성"""
        disclosures = []
        base_date = datetime.now()
        
        disclosure_types = [
            ('주요사항보고서', ['신규시설투자', '타법인주식취득', '유상증자결정']),
            ('분기보고서', ['2024년 1분기', '2024년 2분기', '2024년 3분기']),
            ('임시공시', ['단일판매계약체결', '최대주주변경', '주식매수선택권부여']),
            ('기타공시', ['주주총회소집공고', '사업보고서', '감사보고서'])
        ]
        
        for i in range(10):
            disc_type, subtypes = np.random.choice(disclosure_types, p=[0.3, 0.2, 0.4, 0.1])
            subtype = np.random.choice(subtypes)
            
            disclosures.append({
                'rcept_no': f'20240{i:05d}',
                'corp_cls': 'Y',
                'corp_code': corp_code,
                'corp_name': f'테스트기업_{corp_code[-4:]}',
                'report_nm': f'{disc_type}({subtype})',
                'rcept_dt': (base_date - timedelta(days=i*3)).strftime('%Y%m%d'),
                'rm': ''
            })
        
        return disclosures
    
    def _generate_dummy_financial_data(self, corp_code: str, year: int, quarter: int) -> Dict:
        """더미 재무 데이터 생성"""
        np.random.seed(hash(f"{corp_code}{year}{quarter}") % 2**32)
        
        base_revenue = np.random.uniform(1000, 10000) * 1e8  # 1000억 ~ 1조
        
        return {
            'revenue': int(base_revenue),
            'operating_profit': int(base_revenue * np.random.uniform(0.05, 0.15)),
            'net_income': int(base_revenue * np.random.uniform(0.03, 0.10)),
            'total_assets': int(base_revenue * np.random.uniform(1.5, 3.0)),
            'total_liabilities': int(base_revenue * np.random.uniform(0.5, 1.5)),
            'total_equity': int(base_revenue * np.random.uniform(1.0, 2.0)),
            'profit_margin': np.random.uniform(0.03, 0.10),
            'debt_ratio': np.random.uniform(0.3, 0.7),
            'roe': np.random.uniform(0.05, 0.20)
        }
    
    def _generate_dummy_company_overview(self, corp_code: str) -> Dict:
        """더미 기업 개요 생성"""
        companies = {
            '00126380': {
                'corp_name': '삼성전자',
                'stock_code': '005930',
                'ceo_nm': '한종희',
                'induty_code': '전자부품 제조업'
            },
            '00164779': {
                'corp_name': 'SK하이닉스',
                'stock_code': '000660',
                'ceo_nm': '박정호',
                'induty_code': '반도체 제조업'
            }
        }
        
        default_company = {
            'corp_name': f'테스트기업_{corp_code[-4:]}',
            'stock_code': corp_code[-6:],
            'ceo_nm': '홍길동',
            'induty_code': '제조업'
        }
        
        company = companies.get(corp_code, default_company)
        
        return {
            'corp_name': company['corp_name'],
            'corp_name_eng': company['corp_name'] + ' Co., Ltd.',
            'stock_name': company['corp_name'],
            'stock_code': company['stock_code'],
            'ceo_nm': company['ceo_nm'],
            'corp_cls': 'Y',
            'jurir_no': f'{corp_code[-10:]}',
            'bizr_no': f'{corp_code[-10:]}',
            'adres': '서울특별시 강남구 테헤란로 123',
            'hm_url': f'http://www.{company["corp_name"]}.com',
            'ir_url': f'http://ir.{company["corp_name"]}.com',
            'phn_no': '02-1234-5678',
            'fax_no': '02-1234-5679',
            'induty_code': company['induty_code'],
            'est_dt': '19900101',
            'acc_mt': '12'
        }


class DartDataProcessor:
    """DART 데이터 처리 및 분석"""
    
    def __init__(self):
        self.dart_client = None
    
    async def initialize(self):
        """초기화"""
        self.dart_client = EnhancedDartApiClient()
        await self.dart_client.initialize()
    
    async def process_company_disclosures(self, stock_codes: List[str], 
                                        days_back: int = 30) -> pd.DataFrame:
        """여러 기업의 공시 정보 처리"""
        try:
            all_disclosures = []
            
            for stock_code in stock_codes:
                # 종목 코드를 기업 코드로 변환 (실제로는 매핑 필요)
                corp_code = self._get_corp_code(stock_code)
                
                # 공시 정보 조회
                disclosures = await self.dart_client.fetch_company_disclosures(
                    corp_code,
                    datetime.now() - timedelta(days=days_back),
                    datetime.now()
                )
                
                # 영향도 분석 추가
                for disclosure in disclosures:
                    impact_analysis = self.dart_client.analyze_disclosure_impact(disclosure)
                    disclosure.update(impact_analysis)
                    disclosure['stock_code'] = stock_code
                    all_disclosures.append(disclosure)
            
            # DataFrame 변환
            df = pd.DataFrame(all_disclosures)
            
            if not df.empty:
                # 날짜 변환
                df['rcept_dt'] = pd.to_datetime(df['rcept_dt'], format='%Y%m%d')
                
                # 정렬
                df = df.sort_values('rcept_dt', ascending=False)
            
            return df
            
        except Exception as e:
            logger.error(f"공시 데이터 처리 실패: {e}")
            return pd.DataFrame()
    
    async def get_financial_comparison(self, stock_codes: List[str], 
                                     year: int, quarter: int) -> pd.DataFrame:
        """재무제표 비교 분석"""
        try:
            financial_data = []
            
            for stock_code in stock_codes:
                corp_code = self._get_corp_code(stock_code)
                
                # 재무제표 조회
                financials = await self.dart_client.fetch_financial_statements(
                    corp_code, year, quarter
                )
                
                if financials:
                    financials['stock_code'] = stock_code
                    financials['year'] = year
                    financials['quarter'] = quarter
                    financial_data.append(financials)
            
            # DataFrame 생성
            df = pd.DataFrame(financial_data)
            
            if not df.empty:
                # 재무 비율 추가 계산
                df['operating_margin'] = df['operating_profit'] / df['revenue']
                df['asset_turnover'] = df['revenue'] / df['total_assets']
                
                # 순위 추가
                for col in ['revenue', 'operating_profit', 'net_income', 'roe']:
                    if col in df.columns:
                        df[f'{col}_rank'] = df[col].rank(ascending=False, method='min')
            
            return df
            
        except Exception as e:
            logger.error(f"재무 데이터 비교 실패: {e}")
            return pd.DataFrame()
    
    def _get_corp_code(self, stock_code: str) -> str:
        """종목 코드를 기업 코드로 변환 (더미 매핑)"""
        # 실제로는 DART에서 제공하는 기업코드 매핑 파일 사용
        corp_code_mapping = {
            '005930': '00126380',  # 삼성전자
            '000660': '00164779',  # SK하이닉스
            '035720': '00256598',  # 카카오
            '005380': '00164742',  # 현대차
            '035420': '00266961',  # NAVER
        }
        
        return corp_code_mapping.get(stock_code, f'00{stock_code}')
    
    async def close(self):
        """리소스 정리"""
        if self.dart_client:
            await self.dart_client.close()


# Rate Limiter 클래스 (dart_api_improvements.py에서 가져온 것)
class EnhancedAdaptiveRateLimiter:
    """향상된 적응형 API 속도 제한 관리자"""
    
    def __init__(self, max_calls_per_second: int, max_calls_per_day: int):
        self.max_calls_per_second = max_calls_per_second
        self.max_calls_per_day = max_calls_per_day
        self.call_times = []
        self.daily_calls = 0
        self.last_reset = datetime.now()
        self.latency_window = []
    
    async def check_and_wait(self):
        """호출 제한 확인 및 대기"""
        now = datetime.now()
        
        # 일일 제한 리셋
        if now.date() > self.last_reset.date():
            self.daily_calls = 0
            self.last_reset = now
            self.call_times.clear()
        
        # 일일 제한 확인
        if self.daily_calls >= self.max_calls_per_day:
            wait_time = (self.last_reset + timedelta(days=1) - now).total_seconds()
            logger.warning(f"일일 API 제한 도달. {wait_time:.0f}초 대기")
            await asyncio.sleep(wait_time)
            self.daily_calls = 0
            self.last_reset = datetime.now()
        
        # 초당 제한 확인
        self.call_times = [t for t in self.call_times if (now - t).total_seconds() < 1]
        
        if len(self.call_times) >= self.max_calls_per_second:
            sleep_time = 1 - (now - min(self.call_times)).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.call_times.append(now)
        self.daily_calls += 1
