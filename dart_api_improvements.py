# Improved API Rate Limiting and Redis Management

import asyncio
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Deque
from collections import deque
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class EnhancedAdaptiveRateLimiter:
    """향상된 적응형 API 속도 제한 관리자"""
    
    def __init__(self, max_calls_per_second: int, max_calls_per_day: int):
        self.max_calls_per_second = max_calls_per_second
        self.max_calls_per_day = max_calls_per_day
        self.call_times = deque(maxlen=10000)
        self.daily_calls = 0
        self.last_reset = datetime.now()
        
        # 향상된 네트워크 지연 추적
        self.latency_window = deque(maxlen=100)  # 최근 100개 요청
        self.latency_percentiles = {}  # P50, P90, P99 저장
        self.spike_detection_threshold = 2.0  # 표준편차 기준
        self.adaptive_coefficient = 0.1  # 기본 적응 계수
        
    async def check_and_wait(self):
        """향상된 호출 제한 확인 with 급격한 변동 대응"""
        now = datetime.now()
        
        # 일일 제한 리셋
        if now.date() > self.last_reset.date():
            self._reset_daily_limits()
        
        # 일일 제한 확인
        if self.daily_calls >= self.max_calls_per_day:
            await self._wait_for_daily_reset(now)
        
        # 초당 제한 확인 with 동적 조정
        await self._check_rate_limit_with_adaptation(now)
        
        # 호출 기록
        self.call_times.append(now)
        self.daily_calls += 1
    
    def _reset_daily_limits(self):
        """일일 제한 초기화"""
        self.daily_calls = 0
        self.last_reset = datetime.now()
        self.call_times.clear()
        logger.info("Daily API limits reset")
    
    async def _wait_for_daily_reset(self, now: datetime):
        """일일 제한 대기"""
        wait_time = (self.last_reset + timedelta(days=1) - now).total_seconds()
        logger.warning(f"Daily API limit reached. Waiting {wait_time:.0f} seconds.")
        await asyncio.sleep(wait_time)
        self._reset_daily_limits()
    
    async def _check_rate_limit_with_adaptation(self, now: datetime):
        """적응형 속도 제한 확인"""
        recent_calls = [t for t in self.call_times if (now - t).total_seconds() < 1]
        
        if len(recent_calls) >= self.max_calls_per_second:
            # 기본 대기 시간
            sleep_time = 1 - (now - min(self.call_times, default=now)).total_seconds()
            
            # 급격한 네트워크 변동 감지 및 대응
            adaptive_delay = self._calculate_adaptive_delay()
            
            # 백오프 전략 with 지터
            jitter = np.random.uniform(0, 0.2)  # 0-200ms 랜덤 지터
            total_sleep = max(0, sleep_time + adaptive_delay + jitter)
            
            if total_sleep > 0:
                logger.debug(f"Rate limit wait: {total_sleep:.3f}s "
                           f"(base: {sleep_time:.3f}, adaptive: {adaptive_delay:.3f}, "
                           f"jitter: {jitter:.3f})")
                await asyncio.sleep(total_sleep)
    
    def _calculate_adaptive_delay(self) -> float:
        """급격한 변동을 고려한 적응형 지연 계산"""
        if len(self.latency_window) < 10:
            return 0.0
        
        # 통계 계산
        latencies = list(self.latency_window)
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        # 백분위수 계산
        self.latency_percentiles = {
            'p50': np.percentile(latencies, 50),
            'p90': np.percentile(latencies, 90),
            'p99': np.percentile(latencies, 99)
        }
        
        # 최근 지연과 비교
        recent_latencies = list(self.latency_window)[-10:]
        recent_mean = np.mean(recent_latencies)
        
        # 급격한 변동 감지
        if recent_mean > mean_latency + self.spike_detection_threshold * std_latency:
            # 급격한 증가 감지 - 적극적 대응
            self.adaptive_coefficient = min(0.3, self.adaptive_coefficient * 1.5)
            adaptive_delay = min(recent_mean * self.adaptive_coefficient, 2.0)
            logger.warning(f"Latency spike detected: recent={recent_mean:.2f}s, "
                         f"mean={mean_latency:.2f}s, adaptive_delay={adaptive_delay:.2f}s")
        elif recent_mean < mean_latency - std_latency:
            # 개선 감지 - 계수 감소
            self.adaptive_coefficient = max(0.05, self.adaptive_coefficient * 0.8)
            adaptive_delay = recent_mean * self.adaptive_coefficient
        else:
            # 정상 상태
            if mean_latency > 1.0:
                # P90 기반 조정
                adaptive_delay = min(self.latency_percentiles['p90'] * 0.2, 1.0)
            else:
                adaptive_delay = mean_latency * self.adaptive_coefficient
        
        return adaptive_delay
    
    def record_latency(self, latency: float):
        """네트워크 지연 기록"""
        self.latency_window.append(latency)
        
        # 이상치 필터링 (IQR 방법)
        if len(self.latency_window) > 20:
            q1 = np.percentile(list(self.latency_window), 25)
            q3 = np.percentile(list(self.latency_window), 75)
            iqr = q3 - q1
            
            # 극단적 이상치 제거
            if latency > q3 + 3 * iqr or latency < q1 - 3 * iqr:
                self.latency_window.pop()
                logger.debug(f"Filtered outlier latency: {latency:.2f}s")
    
    def get_stats(self) -> Dict:
        """속도 제한 통계"""
        return {
            'daily_calls': self.daily_calls,
            'max_daily_calls': self.max_calls_per_day,
            'usage_percentage': (self.daily_calls / self.max_calls_per_day) * 100,
            'adaptive_coefficient': self.adaptive_coefficient,
            'latency_percentiles': self.latency_percentiles,
            'last_reset': self.last_reset.isoformat()
        }


class RobustRedisManager:
    """강건한 Redis 관리자 with 명시적 폴백"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379):
        self.host = host
        self.port = port
        self.redis_client = None
        self.connection_attempts = 0
        self.last_error = None
        self.fallback_cache = {}  # 로컬 캐시 폴백
        self.use_fallback = False
        
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type((redis.ConnectionError, redis.TimeoutError))
    )
    async def connect(self) -> bool:
        """Redis 연결 with 명시적 폴백 전략"""
        self.connection_attempts += 1
        
        try:
            # Redis 클라이언트 생성
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
                max_connections=50
            )
            
            # 연결 테스트
            await self.redis_client.ping()
            
            # 연결 성공 - 상세 정보 로깅
            info = await self.redis_client.info()
            logger.info(f"✓ Redis connected to {self.host}:{self.port}")
            logger.info(f"  Version: {info.get('redis_version', 'unknown')}")
            logger.info(f"  Memory: {info.get('used_memory_human', 'unknown')}")
            logger.info(f"  Clients: {info.get('connected_clients', 'unknown')}")
            
            self.use_fallback = False
            return True
            
        except redis.ConnectionError as e:
            self.last_error = {
                'type': 'ConnectionError',
                'message': str(e),
                'details': {
                    'cause': str(e.__cause__) if e.__cause__ else 'Unknown',
                    'host': self.host,
                    'port': self.port
                },
                'attempt': self.connection_attempts,
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"Redis connection failed (attempt {self.connection_attempts}): "
                        f"{self.last_error}")
            raise
            
        except redis.TimeoutError as e:
            self.last_error = {
                'type': 'TimeoutError',
                'message': str(e),
                'details': {
                    'timeout_settings': {
                        'connect_timeout': 5,
                        'socket_timeout': 5
                    }
                },
                'attempt': self.connection_attempts,
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"Redis timeout (attempt {self.connection_attempts}): "
                        f"{self.last_error}")
            raise
            
        except Exception as e:
            # 예상치 못한 오류 - 폴백으로 전환
            self.last_error = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': str(e.__traceback__) if hasattr(e, '__traceback__') else None,
                'attempt': self.connection_attempts,
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"Unexpected Redis error: {self.last_error}")
            
            # 명시적 폴백 전환
            self.redis_client = None
            self.use_fallback = True
            logger.warning("⚠ Switching to in-memory cache fallback")
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """데이터 조회 with 폴백"""
        if self.use_fallback:
            return self.fallback_cache.get(key)
        
        try:
            return await self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            # 자동 폴백 전환
            self.use_fallback = True
            return self.fallback_cache.get(key)
    
    async def set(self, key: str, value: str, ttl: int = 86400) -> bool:
        """데이터 저장 with 폴백"""
        if self.use_fallback:
            self.fallback_cache[key] = value
            # 메모리 관리 - 캐시 크기 제한
            if len(self.fallback_cache) > 10000:
                # LRU 방식으로 오래된 항목 제거
                oldest_keys = list(self.fallback_cache.keys())[:1000]
                for k in oldest_keys:
                    del self.fallback_cache[k]
            return True
        
        try:
            await self.redis_client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            # 자동 폴백 전환
            self.use_fallback = True
            self.fallback_cache[key] = value
            return True
    
    async def get_diagnostics(self) -> Dict:
        """상세 진단 정보"""
        diagnostics = {
            'connected': self.redis_client is not None and not self.use_fallback,
            'using_fallback': self.use_fallback,
            'connection_attempts': self.connection_attempts,
            'last_error': self.last_error,
            'fallback_cache_size': len(self.fallback_cache) if self.use_fallback else 0
        }
        
        if self.redis_client and not self.use_fallback:
            try:
                info = await self.redis_client.info()
                stats = await self.redis_client.info('stats')
                
                diagnostics.update({
                    'redis_info': {
                        'version': info.get('redis_version'),
                        'uptime_days': info.get('uptime_in_days'),
                        'connected_clients': info.get('connected_clients'),
                        'used_memory_human': info.get('used_memory_human'),
                        'used_memory_peak_human': info.get('used_memory_peak_human'),
                        'total_commands_processed': stats.get('total_commands_processed'),
                        'instantaneous_ops_per_sec': stats.get('instantaneous_ops_per_sec')
                    }
                })
            except Exception as e:
                diagnostics['diagnostics_error'] = str(e)
        
        return diagnostics
    
    async def health_check(self) -> bool:
        """헬스 체크 with 자동 복구 시도"""
        if self.use_fallback and self.redis_client is None:
            # 폴백 모드에서 주기적으로 재연결 시도
            logger.info("Attempting to reconnect to Redis...")
            try:
                connected = await self.connect()
                if connected:
                    logger.info("✓ Redis connection restored")
                    # 폴백 캐시 데이터 마이그레이션
                    await self._migrate_fallback_cache()
                return connected
            except Exception:
                return False
        
        if self.redis_client:
            try:
                await self.redis_client.ping()
                return True
            except Exception:
                self.use_fallback = True
                return False
        
        return False
    
    async def _migrate_fallback_cache(self):
        """폴백 캐시를 Redis로 마이그레이션"""
        if not self.fallback_cache:
            return
        
        logger.info(f"Migrating {len(self.fallback_cache)} items from fallback cache to Redis")
        
        migrated = 0
        for key, value in self.fallback_cache.items():
            try:
                await self.redis_client.set(key, value)
                migrated += 1
            except Exception as e:
                logger.error(f"Failed to migrate key {key}: {e}")
        
        logger.info(f"Migrated {migrated}/{len(self.fallback_cache)} items")
        self.fallback_cache.clear()


class StrictVKOSPIValidator:
    """엄격한 VKOSPI 데이터 검증기"""
    
    def __init__(self):
        # 엄격한 품질 기준
        self.max_missing_ratio = 0.05  # 5% 결측치 허용
        self.valid_range = (5, 100)  # VKOSPI 유효 범위
        self.max_daily_change = 0.5  # 일일 최대 변동률 50%
        self.min_data_points = 20  # 최소 데이터 포인트
        
    def validate_vkospi_data(self, df: pd.DataFrame) -> tuple[bool, Dict]:
        """VKOSPI 데이터 품질 검증 with 상세 리포트"""
        validation_report = {
            'is_valid': False,
            'total_rows': len(df) if not df.empty else 0,
            'issues': []
        }
        
        # 기본 검증
        if df.empty:
            validation_report['issues'].append("Empty dataframe")
            return False, validation_report
        
        if 'value' not in df.columns:
            validation_report['issues'].append("Missing 'value' column")
            return False, validation_report
        
        # 데이터 포인트 검증
        if len(df) < self.min_data_points:
            validation_report['issues'].append(
                f"Insufficient data points: {len(df)} < {self.min_data_points}"
            )
            return False, validation_report
        
        # 결측치 검증 (엄격한 기준)
        missing_count = df['value'].isna().sum()
        missing_ratio = missing_count / len(df)
        validation_report['missing_ratio'] = missing_ratio
        
        if missing_ratio > self.max_missing_ratio:
            validation_report['issues'].append(
                f"High missing ratio: {missing_ratio:.1%} > {self.max_missing_ratio:.1%}"
            )
            return False, validation_report
        
        # 값 범위 검증
        valid_values = df['value'].dropna()
        min_val, max_val = valid_values.min(), valid_values.max()
        validation_report['value_range'] = (min_val, max_val)
        
        if min_val < self.valid_range[0] or max_val > self.valid_range[1]:
            validation_report['issues'].append(
                f"Values out of range: [{min_val:.1f}, {max_val:.1f}] "
                f"not in {self.valid_range}"
            )
            return False, validation_report
        
        # 일일 변동률 검증
        daily_changes = valid_values.pct_change().abs()
        max_change = daily_changes.max()
        validation_report['max_daily_change'] = max_change
        
        if max_change > self.max_daily_change:
            validation_report['issues'].append(
                f"Excessive daily change: {max_change:.1%} > "
                f"{self.max_daily_change:.1%}"
            )
            # 경고만 하고 통과 (실제 시장에서 발생 가능)
            logger.warning(f"VKOSPI daily change warning: {max_change:.1%}")
        
        # 통계적 이상치 검증
        q1 = valid_values.quantile(0.25)
        q3 = valid_values.quantile(0.75)
        iqr = q3 - q1
        outliers = ((valid_values < q1 - 3 * iqr) | (valid_values > q3 + 3 * iqr)).sum()
        outlier_ratio = outliers / len(valid_values)
        validation_report['outlier_ratio'] = outlier_ratio
        
        if outlier_ratio > 0.05:  # 5% 이상 이상치
            validation_report['issues'].append(
                f"High outlier ratio: {outlier_ratio:.1%}"
            )
        
        # 최종 판정
        validation_report['is_valid'] = len(validation_report['issues']) == 0
        
        if validation_report['is_valid']:
            logger.info("✓ VKOSPI data validation passed")
        else:
            logger.warning(f"✗ VKOSPI validation failed: {validation_report['issues']}")
        
        return validation_report['is_valid'], validation_report
