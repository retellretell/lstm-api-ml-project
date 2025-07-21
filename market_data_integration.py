# Market Data Integration Module for Real-time Stock and Index Data

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import os
from collections import deque
import pickle

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    yf = None

try:
    from pykrx import stock
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False
    stock = None

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)

class MarketDataCollector:
    """통합 시장 데이터 수집기"""
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.cache_ttl = 300  # 5분 캐시
        self.redis_client = None
        self.local_cache = {}
        self.session = None
        
        # 데이터 소스 우선순위
        self.data_sources = {
            'pykrx': PYKRX_AVAILABLE,
            'yfinance': YF_AVAILABLE,
            'manual': True  # 항상 사용 가능한 수동 입력
        }
        
        logger.info(f"MarketDataCollector 초기화 - PyKRX: {PYKRX_AVAILABLE}, yfinance: {YF_AVAILABLE}")
    
    async def initialize(self):
        """비동기 초기화"""
        try:
            # aiohttp 세션 생성
            self.session = aiohttp.ClientSession()
            
            # Redis 연결 시도
            if self.use_cache and REDIS_AVAILABLE:
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
            logger.error(f"MarketDataCollector 초기화 실패: {e}")
    
    async def close(self):
        """리소스 정리"""
        if self.session:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()
    
    async def get_stock_data(self, stock_code: str, 
                           start_date: datetime, 
                           end_date: datetime) -> pd.DataFrame:
        """주식 데이터 조회 (다중 소스 지원)"""
        try:
            # 캐시 확인
            cache_key = f"stock:{stock_code}:{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}"
            cached_data = await self._get_cache(cache_key)
            
            if cached_data is not None:
                return cached_data
            
            # PyKRX 시도
            if PYKRX_AVAILABLE:
                try:
                    df = await self._get_pykrx_data(stock_code, start_date, end_date)
                    if not df.empty:
                        await self._set_cache(cache_key, df)
                        return df
                except Exception as e:
                    logger.warning(f"PyKRX 데이터 조회 실패: {e}")
            
            # yfinance 시도
            if YF_AVAILABLE:
                try:
                    df = await self._get_yfinance_data(stock_code, start_date, end_date)
                    if not df.empty:
                        await self._set_cache(cache_key, df)
                        return df
                except Exception as e:
                    logger.warning(f"yfinance 데이터 조회 실패: {e}")
            
            # 더미 데이터 생성
            logger.info(f"실제 데이터를 가져올 수 없어 더미 데이터 생성: {stock_code}")
            df = self._generate_dummy_stock_data(stock_code, start_date, end_date)
            await self._set_cache(cache_key, df)
            return df
            
        except Exception as e:
            logger.error(f"주식 데이터 조회 실패 ({stock_code}): {e}")
            return pd.DataFrame()
    
    async def get_index_data(self, index_code: str,
                           start_date: datetime,
                           end_date: datetime) -> pd.DataFrame:
        """지수 데이터 조회"""
        try:
            # 캐시 확인
            cache_key = f"index:{index_code}:{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}"
            cached_data = await self._get_cache(cache_key)
            
            if cached_data is not None:
                return cached_data
            
            # PyKRX로 지수 데이터 조회
            if PYKRX_AVAILABLE:
                try:
                    df = stock.get_index_ohlcv_by_date(
                        start_date.strftime("%Y%m%d"),
                        end_date.strftime("%Y%m%d"),
                        index_code
                    )
                    if not df.empty:
                        df = df.reset_index()
                        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                        await self._set_cache(cache_key, df)
                        return df
                except Exception as e:
                    logger.warning(f"PyKRX 지수 데이터 조회 실패: {e}")
            
            # 더미 데이터 생성
            df = self._generate_dummy_index_data(index_code, start_date, end_date)
            await self._set_cache(cache_key, df)
            return df
            
        except Exception as e:
            logger.error(f"지수 데이터 조회 실패 ({index_code}): {e}")
            return pd.DataFrame()
    
    async def get_vkospi_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """VKOSPI (변동성 지수) 데이터 조회"""
        try:
            # 캐시 확인
            cache_key = f"vkospi:{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}"
            cached_data = await self._get_cache(cache_key)
            
            if cached_data is not None:
                return cached_data
            
            # 실제 VKOSPI 데이터 조회 (API 구현 필요)
            # 현재는 더미 데이터 생성
            df = self._generate_dummy_vkospi_data(start_date, end_date)
            await self._set_cache(cache_key, df)
            return df
            
        except Exception as e:
            logger.error(f"VKOSPI 데이터 조회 실패: {e}")
            return pd.DataFrame()
    
    async def get_market_sentiment(self, stock_codes: List[str]) -> Dict:
        """시장 심리 지표 계산"""
        try:
            sentiment_data = {
                'fear_greed_index': 0,
                'market_momentum': 0,
                'sector_rotation': {},
                'volume_trend': 0,
                'volatility_regime': 'normal'
            }
            
            # 최근 30일 데이터로 분석
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # 주식별 데이터 수집
            price_changes = []
            volume_changes = []
            
            for stock_code in stock_codes:
                df = await self.get_stock_data(stock_code, start_date, end_date)
                if not df.empty:
                    # 가격 변화율
                    price_change = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                    price_changes.append(price_change)
                    
                    # 거래량 변화
                    recent_volume = df['volume'].iloc[-5:].mean()
                    past_volume = df['volume'].iloc[:-5].mean()
                    volume_change = (recent_volume / past_volume - 1) * 100
                    volume_changes.append(volume_change)
            
            # Fear & Greed Index 계산 (0-100)
            if price_changes:
                avg_price_change = np.mean(price_changes)
                volatility = np.std(price_changes)
                
                # 가격 상승 + 낮은 변동성 = Greed
                # 가격 하락 + 높은 변동성 = Fear
                fear_greed = 50 + avg_price_change * 2 - volatility
                sentiment_data['fear_greed_index'] = np.clip(fear_greed, 0, 100)
            
            # 시장 모멘텀
            if price_changes:
                positive_stocks = sum(1 for p in price_changes if p > 0)
                sentiment_data['market_momentum'] = (positive_stocks / len(price_changes)) * 100
            
            # 거래량 트렌드
            if volume_changes:
                sentiment_data['volume_trend'] = np.mean(volume_changes)
            
            # 변동성 체제 판단
            vkospi_df = await self.get_vkospi_data(start_date, end_date)
            if not vkospi_df.empty:
                current_vkospi = vkospi_df['value'].iloc[-1]
                if current_vkospi < 20:
                    sentiment_data['volatility_regime'] = 'low'
                elif current_vkospi > 30:
                    sentiment_data['volatility_regime'] = 'high'
                else:
                    sentiment_data['volatility_regime'] = 'normal'
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"시장 심리 분석 실패: {e}")
            return {
                'fear_greed_index': 50,
                'market_momentum': 50,
                'sector_rotation': {},
                'volume_trend': 0,
                'volatility_regime': 'normal'
            }
    
    async def _get_pykrx_data(self, stock_code: str, 
                            start_date: datetime, 
                            end_date: datetime) -> pd.DataFrame:
        """PyKRX를 통한 데이터 조회"""
        df = stock.get_market_ohlcv_by_date(
            start_date.strftime("%Y%m%d"),
            end_date.strftime("%Y%m%d"),
            stock_code
        )
        
        if not df.empty:
            df = df.reset_index()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            # 기술적 지표 추가
            df = self._add_technical_indicators(df)
        
        return df
    
    async def _get_yfinance_data(self, stock_code: str,
                               start_date: datetime,
                               end_date: datetime) -> pd.DataFrame:
        """yfinance를 통한 데이터 조회"""
        # 한국 주식은 .KS 또는 .KQ 접미사 필요
        ticker = f"{stock_code}.KS"
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                # KOSDAQ 시도
                ticker = f"{stock_code}.KQ"
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
            
            if not df.empty:
                df = df.reset_index()
                df.columns = df.columns.str.lower()
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                
                # 기술적 지표 추가
                df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"yfinance 데이터 조회 실패: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 추가"""
        try:
            # 이동평균
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA60'] = df['close'].rolling(window=60).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['Signal']
            
            # 볼린저 밴드
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + 2 * bb_std
            df['BB_lower'] = df['BB_middle'] - 2 * bb_std
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()
            
            # 거래량 지표
            df['Volume_MA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
            
            return df
            
        except Exception as e:
            logger.error(f"기술적 지표 계산 실패: {e}")
            return df
    
    def _generate_dummy_stock_data(self, stock_code: str,
                                 start_date: datetime,
                                 end_date: datetime) -> pd.DataFrame:
        """더미 주식 데이터 생성"""
        # 시드 설정으로 일관된 데이터 생성
        np.random.seed(hash(stock_code) % 2**32)
        
        # 거래일 생성
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 초기 가격 설정
        base_price = np.random.uniform(10000, 200000)
        
        data = []
        for date in dates:
            # 일일 변동률 (평균 0%, 표준편차 2%)
            daily_return = np.random.normal(0, 0.02)
            
            # OHLC 계산
            open_price = base_price
            close_price = base_price * (1 + daily_return)
            high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.02)
            low_price = min(open_price, close_price) * np.random.uniform(0.98, 1.0)
            
            # 거래량 (기본 거래량에 랜덤 변동)
            base_volume = np.random.uniform(1000000, 10000000)
            volume = int(base_volume * np.random.uniform(0.5, 2.0))
            
            data.append({
                'date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
            # 다음 날 기준 가격 업데이트
            base_price = close_price
        
        df = pd.DataFrame(data)
        
        # 기술적 지표 추가
        df = self._add_technical_indicators(df)
        
        return df
    
    def _generate_dummy_index_data(self, index_code: str,
                                 start_date: datetime,
                                 end_date: datetime) -> pd.DataFrame:
        """더미 지수 데이터 생성"""
        np.random.seed(hash(index_code) % 2**32)
        
        # 지수별 기본값 설정
        index_params = {
            '1001': {'name': 'KOSPI', 'base': 2500, 'volatility': 0.01},
            '2001': {'name': 'KOSDAQ', 'base': 850, 'volatility': 0.015},
            '1028': {'name': 'KOSPI200', 'base': 350, 'volatility': 0.012}
        }
        
        params = index_params.get(index_code, {'name': 'INDEX', 'base': 1000, 'volatility': 0.01})
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        base_value = params['base']
        
        data = []
        for date in dates:
            # 지수는 개별 주식보다 변동성이 낮음
            daily_return = np.random.normal(0, params['volatility'])
            
            open_value = base_value
            close_value = base_value * (1 + daily_return)
            high_value = max(open_value, close_value) * np.random.uniform(1.0, 1.005)
            low_value = min(open_value, close_value) * np.random.uniform(0.995, 1.0)
            
            # 지수 거래량
            volume = int(np.random.uniform(1e9, 5e9))
            
            data.append({
                'date': date,
                'open': open_value,
                'high': high_value,
                'low': low_value,
                'close': close_value,
                'volume': volume
            })
            
            base_value = close_value
        
        return pd.DataFrame(data)
    
    def _generate_dummy_vkospi_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """더미 VKOSPI 데이터 생성"""
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # VKOSPI는 평균 회귀 특성을 가짐
        mean_vkospi = 20
        current_vkospi = mean_vkospi
        
        data = []
        for date in dates:
            # 평균 회귀 + 랜덤 충격
            mean_reversion = 0.1 * (mean_vkospi - current_vkospi)
            random_shock = np.random.normal(0, 1.5)
            
            current_vkospi = current_vkospi + mean_reversion + random_shock
            current_vkospi = max(10, min(50, current_vkospi))  # 10-50 범위 제한
            
            data.append({
                'date': date,
                'value': current_vkospi
            })
        
        return pd.DataFrame(data)
    
    async def _get_cache(self, key: str) -> Optional[pd.DataFrame]:
        """캐시에서 데이터 조회"""
        try:
            if self.redis_client:
                data = await self.redis_client.get(key)
                if data:
                    return pd.read_json(data)
            elif key in self.local_cache:
                return self.local_cache[key]
        except Exception as e:
            logger.debug(f"캐시 조회 실패: {e}")
        
        return None
    
    async def _set_cache(self, key: str, df: pd.DataFrame):
        """캐시에 데이터 저장"""
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    key,
                    self.cache_ttl,
                    df.to_json()
                )
            else:
                self.local_cache[key] = df
                # 로컬 캐시 크기 제한
                if len(self.local_cache) > 100:
                    # 가장 오래된 항목 제거
                    oldest_key = next(iter(self.local_cache))
                    del self.local_cache[oldest_key]
        except Exception as e:
            logger.debug(f"캐시 저장 실패: {e}")


class TradingSignalGenerator:
    """거래 신호 생성기"""
    
    def __init__(self):
        self.market_collector = MarketDataCollector()
        self.signal_history = deque(maxlen=1000)
    
    async def initialize(self):
        """초기화"""
        await self.market_collector.initialize()
    
    async def generate_signals(self, stock_codes: List[str]) -> List[Dict]:
        """여러 주식에 대한 거래 신호 생성"""
        signals = []
        
        for stock_code in stock_codes:
            try:
                signal = await self._analyze_single_stock(stock_code)
                if signal:
                    signals.append(signal)
                    self.signal_history.append(signal)
            except Exception as e:
                logger.error(f"신호 생성 실패 ({stock_code}): {e}")
        
        return signals
    
    async def _analyze_single_stock(self, stock_code: str) -> Optional[Dict]:
        """개별 주식 분석 및 신호 생성"""
        try:
            # 최근 60일 데이터 조회
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            df = await self.market_collector.get_stock_data(stock_code, start_date, end_date)
            
            if df.empty or len(df) < 60:
                return None
            
            # 현재 가격 및 지표
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # 신호 초기화
            signal = {
                'stock_code': stock_code,
                'timestamp': datetime.now(),
                'price': current_price,
                'signals': [],
                'strength': 0,
                'action': 'HOLD'
            }
            
            # 1. 이동평균 크로스오버
            ma_signal = self._check_ma_crossover(df)
            if ma_signal:
                signal['signals'].append(ma_signal)
            
            # 2. RSI 신호
            rsi_signal = self._check_rsi_signal(df)
            if rsi_signal:
                signal['signals'].append(rsi_signal)
            
            # 3. MACD 신호
            macd_signal = self._check_macd_signal(df)
            if macd_signal:
                signal['signals'].append(macd_signal)
            
            # 4. 볼린저 밴드 신호
            bb_signal = self._check_bollinger_bands(df)
            if bb_signal:
                signal['signals'].append(bb_signal)
            
            # 5. 거래량 신호
            volume_signal = self._check_volume_signal(df)
            if volume_signal:
                signal['signals'].append(volume_signal)
            
            # 종합 신호 강도 계산
            if signal['signals']:
                buy_signals = sum(1 for s in signal['signals'] if s['type'] == 'BUY')
                sell_signals = sum(1 for s in signal['signals'] if s['type'] == 'SELL')
                
                signal['strength'] = (buy_signals - sell_signals) / len(signal['signals'])
                
                if signal['strength'] > 0.3:
                    signal['action'] = 'BUY'
                elif signal['strength'] < -0.3:
                    signal['action'] = 'SELL'
                else:
                    signal['action'] = 'HOLD'
            
            return signal
            
        except Exception as e:
            logger.error(f"주식 분석 실패 ({stock_code}): {e}")
            return None
    
    def _check_ma_crossover(self, df: pd.DataFrame) -> Optional[Dict]:
        """이동평균 크로스오버 확인"""
        try:
            if 'MA5' not in df.columns or 'MA20' not in df.columns:
                return None
            
            # 최근 2일간의 MA 데이터
            ma5_current = df['MA5'].iloc[-1]
            ma5_prev = df['MA5'].iloc[-2]
            ma20_current = df['MA20'].iloc[-1]
            ma20_prev = df['MA20'].iloc[-2]
            
            # 골든크로스 (단기 이평선이 장기 이평선을 상향 돌파)
            if ma5_prev <= ma20_prev and ma5_current > ma20_current:
                return {
                    'type': 'BUY',
                    'indicator': 'MA_CROSSOVER',
                    'description': 'Golden Cross (MA5 > MA20)',
                    'strength': 0.8
                }
            
            # 데드크로스 (단기 이평선이 장기 이평선을 하향 돌파)
            elif ma5_prev >= ma20_prev and ma5_current < ma20_current:
                return {
                    'type': 'SELL',
                    'indicator': 'MA_CROSSOVER',
                    'description': 'Death Cross (MA5 < MA20)',
                    'strength': 0.8
                }
            
            return None
            
        except Exception:
            return None
    
    def _check_rsi_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """RSI 신호 확인"""
        try:
            if 'RSI' not in df.columns:
                return None
            
            rsi = df['RSI'].iloc[-1]
            
            # 과매도 구간
            if rsi < 30:
                return {
                    'type': 'BUY',
                    'indicator': 'RSI',
                    'description': f'Oversold (RSI: {rsi:.1f})',
                    'strength': 0.7
                }
            
            # 과매수 구간
            elif rsi > 70:
                return {
                    'type': 'SELL',
                    'indicator': 'RSI',
                    'description': f'Overbought (RSI: {rsi:.1f})',
                    'strength': 0.7
                }
            
            return None
            
        except Exception:
            return None
    
    def _check_macd_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """MACD 신호 확인"""
        try:
            if 'MACD' not in df.columns or 'Signal' not in df.columns:
                return None
            
            macd_current = df['MACD'].iloc[-1]
            macd_prev = df['MACD'].iloc[-2]
            signal_current = df['Signal'].iloc[-1]
            signal_prev = df['Signal'].iloc[-2]
            
            # MACD가 시그널선을 상향 돌파
            if macd_prev <= signal_prev and macd_current > signal_current:
                return {
                    'type': 'BUY',
                    'indicator': 'MACD',
                    'description': 'MACD crosses above signal',
                    'strength': 0.6
                }
            
            # MACD가 시그널선을 하향 돌파
            elif macd_prev >= signal_prev and macd_current < signal_current:
                return {
                    'type': 'SELL',
                    'indicator': 'MACD',
                    'description': 'MACD crosses below signal',
                    'strength': 0.6
                }
            
            return None
            
        except Exception:
            return None
    
    def _check_bollinger_bands(self, df: pd.DataFrame) -> Optional[Dict]:
        """볼린저 밴드 신호 확인"""
        try:
            if 'BB_upper' not in df.columns or 'BB_lower' not in df.columns:
                return None
            
            close = df['close'].iloc[-1]
            bb_upper = df['BB_upper'].iloc[-1]
            bb_lower = df['BB_lower'].iloc[-1]
            bb_middle = df['BB_middle'].iloc[-1]
            
            # 가격이 하단 밴드 근처
            if close <= bb_lower * 1.02:
                return {
                    'type': 'BUY',
                    'indicator': 'BOLLINGER',
                    'description': 'Price near lower band',
                    'strength': 0.6
                }
            
            # 가격이 상단 밴드 근처
            elif close >= bb_upper * 0.98:
                return {
                    'type': 'SELL',
                    'indicator': 'BOLLINGER',
                    'description': 'Price near upper band',
                    'strength': 0.6
                }
            
            return None
            
        except Exception:
            return None
    
    def _check_volume_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """거래량 신호 확인"""
        try:
            if 'Volume_Ratio' not in df.columns:
                return None
            
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            price_change = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
            
            # 거래량 급증 + 가격 상승
            if volume_ratio > 2.0 and price_change > 1.0:
                return {
                    'type': 'BUY',
                    'indicator': 'VOLUME',
                    'description': f'Volume surge ({volume_ratio:.1f}x) with price up',
                    'strength': 0.7
                }
            
            # 거래량 급증 + 가격 하락 (매도 압력)
            elif volume_ratio > 2.0 and price_change < -1.0:
                return {
                    'type': 'SELL',
                    'indicator': 'VOLUME',
                    'description': f'Volume surge ({volume_ratio:.1f}x) with price down',
                    'strength': 0.7
                }
            
            return None
            
        except Exception:
            return None
    
    async def close(self):
        """리소스 정리"""
        await self.market_collector.close()


class MarketAnalyzer:
    """시장 분석 도구"""
    
    def __init__(self):
        self.market_collector = MarketDataCollector()
        self.sector_mapping = {
            '005930': '전자',
            '000660': '전자',
            '035720': 'IT',
            '005380': '자동차',
            '035420': 'IT',
            '051910': '화학',
            '006400': '전자부품',
            '028260': '무역',
            '105560': '금융',
            '055550': '금융'
        }
    
    async def initialize(self):
        """초기화"""
        await self.market_collector.initialize()
    
    async def analyze_market_breadth(self, stock_codes: List[str]) -> Dict:
        """시장 폭 분석"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            advances = 0
            declines = 0
            unchanged = 0
            
            advance_volume = 0
            decline_volume = 0
            
            for stock_code in stock_codes:
                df = await self.market_collector.get_stock_data(stock_code, start_date, end_date)
                
                if not df.empty and len(df) >= 2:
                    prev_close = df['close'].iloc[-2]
                    current_close = df['close'].iloc[-1]
                    current_volume = df['volume'].iloc[-1]
                    
                    change = (current_close / prev_close - 1) * 100
                    
                    if change > 0.1:
                        advances += 1
                        advance_volume += current_volume
                    elif change < -0.1:
                        declines += 1
                        decline_volume += current_volume
                    else:
                        unchanged += 1
            
            total_stocks = advances + declines + unchanged
            
            breadth_data = {
                'advances': advances,
                'declines': declines,
                'unchanged': unchanged,
                'advance_decline_ratio': advances / declines if declines > 0 else float('inf'),
                'advance_percentage': (advances / total_stocks * 100) if total_stocks > 0 else 0,
                'advance_volume': advance_volume,
                'decline_volume': decline_volume,
                'volume_ratio': advance_volume / decline_volume if decline_volume > 0 else float('inf'),
                'market_strength': 'bullish' if advances > declines else 'bearish' if declines > advances else 'neutral'
            }
            
            return breadth_data
            
        except Exception as e:
            logger.error(f"시장 폭 분석 실패: {e}")
            return {}
    
    async def analyze_sector_rotation(self, stock_codes: List[str], days: int = 30) -> Dict:
        """섹터 로테이션 분석"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            sector_performance = {}
            
            for stock_code in stock_codes:
                sector = self.sector_mapping.get(stock_code, 'Unknown')
                
                df = await self.market_collector.get_stock_data(stock_code, start_date, end_date)
                
                if not df.empty:
                    performance = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                    
                    if sector not in sector_performance:
                        sector_performance[sector] = []
                    
                    sector_performance[sector].append(performance)
            
            # 섹터별 평균 성과 계산
            sector_averages = {}
            for sector, performances in sector_performance.items():
                sector_averages[sector] = {
                    'average_return': np.mean(performances),
                    'median_return': np.median(performances),
                    'best_return': np.max(performances),
                    'worst_return': np.min(performances),
                    'volatility': np.std(performances),
                    'stock_count': len(performances)
                }
            
            # 섹터 순위
            sorted_sectors = sorted(
                sector_averages.items(),
                key=lambda x: x[1]['average_return'],
                reverse=True
            )
            
            return {
                'sector_performance': sector_averages,
                'sector_ranking': [s[0] for s in sorted_sectors],
                'best_sector': sorted_sectors[0][0] if sorted_sectors else None,
                'worst_sector': sorted_sectors[-1][0] if sorted_sectors else None,
                'rotation_signal': self._determine_rotation_signal(sorted_sectors)
            }
            
        except Exception as e:
            logger.error(f"섹터 로테이션 분석 실패: {e}")
            return {}
    
    def _determine_rotation_signal(self, sorted_sectors: List[Tuple[str, Dict]]) -> str:
        """섹터 로테이션 신호 판단"""
        if not sorted_sectors:
            return 'unknown'
        
        # 상위 섹터 확인
        top_sectors = [s[0] for s in sorted_sectors[:2]]
        
        # 경기 순환에 따른 섹터 로테이션
        if '금융' in top_sectors and '산업재' in top_sectors:
            return 'early_cycle'
        elif 'IT' in top_sectors and '전자' in top_sectors:
            return 'mid_cycle'
        elif '필수소비재' in top_sectors and '유틸리티' in top_sectors:
            return 'late_cycle'
        elif '화학' in top_sectors and '자동차' in top_sectors:
            return 'recovery'
        else:
            return 'mixed'
    
    async def calculate_market_indicators(self, index_code: str = '1001') -> Dict:
        """주요 시장 지표 계산"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            # KOSPI 데이터
            kospi_df = await self.market_collector.get_index_data(index_code, start_date, end_date)
            
            if kospi_df.empty:
                return {}
            
            # 변동성 지표
            returns = kospi_df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 연율화
            
            # 추세 지표
            sma_20 = kospi_df['close'].rolling(window=20).mean().iloc[-1]
            sma_50 = kospi_df['close'].rolling(window=50).mean().iloc[-1]
            current_price = kospi_df['close'].iloc[-1]
            
            trend = 'uptrend' if current_price > sma_20 > sma_50 else 'downtrend' if current_price < sma_20 < sma_50 else 'sideways'
            
            # 모멘텀 지표
            momentum_1m = (kospi_df['close'].iloc[-1] / kospi_df['close'].iloc[-20] - 1) * 100
            momentum_3m = (kospi_df['close'].iloc[-1] / kospi_df['close'].iloc[-60] - 1) * 100 if len(kospi_df) >= 60 else 0
            
            # 고점/저점
            high_52w = kospi_df['close'].rolling(window=252).max().iloc[-1] if len(kospi_df) >= 252 else kospi_df['close'].max()
            low_52w = kospi_df['close'].rolling(window=252).min().iloc[-1] if len(kospi_df) >= 252 else kospi_df['close'].min()
            
            return {
                'current_level': current_price,
                'volatility_annual': volatility * 100,
                'trend': trend,
                'momentum_1m': momentum_1m,
                'momentum_3m': momentum_3m,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'distance_from_high': (current_price / high_52w - 1) * 100,
                'distance_from_low': (current_price / low_52w - 1) * 100,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'above_sma20': current_price > sma_20,
                'above_sma50': current_price > sma_50
            }
            
        except Exception as e:
            logger.error(f"시장 지표 계산 실패: {e}")
            return {}
    
    async def close(self):
        """리소스 정리"""
        await self.market_collector.close()


class CorrelationAnalyzer:
    """상관관계 분석기"""
    
    def __init__(self):
        self.market_collector = MarketDataCollector()
    
    async def initialize(self):
        """초기화"""
        await self.market_collector.initialize()
    
    async def calculate_correlation_matrix(self, stock_codes: List[str], days: int = 60) -> pd.DataFrame:
        """주식 간 상관관계 매트릭스 계산"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 모든 주식의 수익률 데이터 수집
            returns_data = {}
            
            for stock_code in stock_codes:
                df = await self.market_collector.get_stock_data(stock_code, start_date, end_date)
                
                if not df.empty:
                    returns = df['close'].pct_change().dropna()
                    returns_data[stock_code] = returns
            
            # DataFrame으로 변환
            returns_df = pd.DataFrame(returns_data)
            
            # 상관관계 매트릭스 계산
            correlation_matrix = returns_df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"상관관계 계산 실패: {e}")
            return pd.DataFrame()
    
    async def find_pairs_trading_opportunities(self, stock_codes: List[str], 
                                             min_correlation: float = 0.7) -> List[Dict]:
        """페어 트레이딩 기회 탐색"""
        try:
            # 상관관계 매트릭스 계산
            corr_matrix = await self.calculate_correlation_matrix(stock_codes)
            
            if corr_matrix.empty:
                return []
            
            pairs = []
            
            # 높은 상관관계를 가진 페어 찾기
            for i in range(len(stock_codes)):
                for j in range(i + 1, len(stock_codes)):
                    stock1 = stock_codes[i]
                    stock2 = stock_codes[j]
                    
                    if stock1 in corr_matrix.columns and stock2 in corr_matrix.columns:
                        correlation = corr_matrix.loc[stock1, stock2]
                        
                        if correlation >= min_correlation:
                            # 스프레드 분석
                            spread_analysis = await self._analyze_spread(stock1, stock2)
                            
                            if spread_analysis:
                                pairs.append({
                                    'stock1': stock1,
                                    'stock2': stock2,
                                    'correlation': correlation,
                                    'spread_mean': spread_analysis['mean'],
                                    'spread_std': spread_analysis['std'],
                                    'current_spread': spread_analysis['current'],
                                    'z_score': spread_analysis['z_score'],
                                    'signal': spread_analysis['signal']
                                })
            
            # Z-score 기준으로 정렬
            pairs.sort(key=lambda x: abs(x['z_score']), reverse=True)
            
            return pairs
            
        except Exception as e:
            logger.error(f"페어 트레이딩 분석 실패: {e}")
            return []
    
    async def _analyze_spread(self, stock1: str, stock2: str) -> Optional[Dict]:
        """두 주식 간 스프레드 분석"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            df1 = await self.market_collector.get_stock_data(stock1, start_date, end_date)
            df2 = await self.market_collector.get_stock_data(stock2, start_date, end_date)
            
            if df1.empty or df2.empty:
                return None
            
            # 날짜 정렬 및 매칭
            df1 = df1.set_index('date')
            df2 = df2.set_index('date')
            
            # 공통 날짜만 선택
            common_dates = df1.index.intersection(df2.index)
            
            if len(common_dates) < 20:
                return None
            
            prices1 = df1.loc[common_dates, 'close']
            prices2 = df2.loc[common_dates, 'close']
            
            # 가격 비율 (스프레드)
            spread = prices1 / prices2
            
            # 통계 계산
            spread_mean = spread.mean()
            spread_std = spread.std()
            current_spread = spread.iloc[-1]
            
            # Z-score 계산
            z_score = (current_spread - spread_mean) / spread_std
            
            # 신호 생성
            if z_score > 2:
                signal = 'SELL_SPREAD'  # Stock1 매도, Stock2 매수
            elif z_score < -2:
                signal = 'BUY_SPREAD'   # Stock1 매수, Stock2 매도
            else:
                signal = 'NEUTRAL'
            
            return {
                'mean': spread_mean,
                'std': spread_std,
                'current': current_spread,
                'z_score': z_score,
                'signal': signal
            }
            
        except Exception as e:
            logger.error(f"스프레드 분석 실패: {e}")
            return None
    
    async def close(self):
        """리소스 정리"""
        await self.market_collector.close()


# 유틸리티 함수들
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """샤프 비율 계산"""
    try:
        excess_returns = returns - risk_free_rate / 252  # 일일 무위험 수익률
        
        if len(excess_returns) == 0 or excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe
        
    except Exception:
        return 0.0


def calculate_max_drawdown(prices: pd.Series) -> float:
    """최대 낙폭 계산"""
    try:
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
        
    except Exception:
        return 0.0


def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """베타 계산"""
    try:
        covariance = stock_returns.cov(market_returns)
        market_variance = market_returns.var()
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return beta
        
    except Exception:
        return 1.0