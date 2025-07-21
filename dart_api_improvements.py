recent_window = min(10, len(latencies))
            recent_latencies = latencies[-recent_window:]
            recent_mean = np.mean(recent_latencies)
            
            # 급격한 변동 감지 (개선된 알고리즘)
            if len(latencies) > 20 and std_latency > 0:
                z_score = (recent_mean - mean_latency) / std_latency
                
                if z_score > self.spike_detection_threshold:
                    # 급격한 증가 감지 - 적극적 대응
                    self.adaptive_coefficient = min(0.3, self.adaptive_coefficient * 1.2)
                    adaptive_delay = min(recent_mean * self.adaptive_coefficient, 2.0)
                    logger.warning(f"지연 급증 감지: recent={recent_mean:.2f}s, "
                                 f"z_score={z_score:.1f}, adaptive_delay={adaptive_delay:.2f}s")
                    
                    # 심각한 급증시 일시적 스로틀링
                    if z_score > 3.0:
                        self.is_throttled = True
                        self.throttle_until = datetime.now() + timedelta(seconds=30)
                        logger.error("심각한 지연 급증으로 30초 스로틀링 적용")
                        
                elif z_score < -0.5:
                    # 개선 감지 - 계수 감소
                    self.adaptive_coefficient = max(0.05, self.adaptive_coefficient * 0.9)
                    adaptive_delay = recent_mean * self.adaptive_coefficient
                else:
                    # 정상 상태 - 점진적 조정
                    if mean_latency > 1.0:
                        adaptive_delay = min(self.latency_percentiles['p90'] * 0.15, 1.0)
                    else:
                        adaptive_delay = mean_latency * self.adaptive_coefficient
            else:
                # 데이터 부족시 보수적 접근
                adaptive_delay = min(recent_mean * 0.1, 0.5)
            
            return max(0, adaptive_delay)
            
        except Exception as e:
            logger.debug(f"적응형 지연 계산 오류: {e}")
            return 0.1  # 안전한 기본값
    
    def record_latency(self, latency: float):
        """네트워크 지연 기록 with 개선된 이상치 필터링"""
        try:
            # 기본 유효성 검사
            if not isinstance(latency, (int, float)) or latency < 0:
                logger.debug(f"유효하지 않은 지연값 무시: {latency}")
                return
            
            # 극단적 값 사전 필터링 (30초 초과는 타임아웃으로 간주)
            if latency > 30.0:
                logger.warning(f"극단적 지연값 감지: {latency:.2f}s")
                self.timeout_requests += 1
                latency = min(latency, 30.0)  # 상한선 적용
            
            self.latency_window.append(latency)
            
            # 이상치 필터링 (IQR 방법, 개선됨)
            if len(self.latency_window) > 30:
                latencies = np.array(list(self.latency_window))
                q1, q3 = np.percentile(latencies, [25, 75])
                iqr = q3 - q1
                
                # 더 관대한 이상치 기준 (3 IQR -> 4 IQR)
                lower_bound = q1 - 4 * iqr
                upper_bound = q3 + 4 * iqr
                
                if latency < lower_bound or latency > upper_bound:
                    # 이상치를 완전히 제거하지 않고 조정
                    adjusted_latency = np.clip(latency, q1 - 2*iqr, q3 + 2*iqr)
                    self.latency_window[-1] = adjusted_latency
                    logger.debug(f"지연 이상치 조정: {latency:.2f}s -> {adjusted_latency:.2f}s")
        
        except Exception as e:
            logger.debug(f"지연 기록 오류: {e}")
    
    def record_failure(self, error_type: str = "unknown"):
        """API 호출 실패 기록"""
        self.failed_requests += 1
        
        # 실패율이 높으면 임시 스로틀링
        if self.total_requests > 20:
            failure_rate = self.failed_requests / self.total_requests
            if failure_rate > 0.3:  # 30% 이상 실패
                self.is_throttled = True
                self.throttle_until = datetime.now() + timedelta(seconds=60)
                logger.warning(f"높은 실패율({failure_rate:.1%})로 인한 스로틀링: {error_type}")
    
    def get_stats(self) -> Dict:
        """속도 제한 통계 (확장됨)"""
        try:
            current_time = datetime.now()
            uptime_hours = (current_time - self.last_reset).total_seconds() / 3600
            
            # 기본 통계
            stats = {
                'daily_calls': self.daily_calls,
                'max_daily_calls': self.max_calls_per_day,
                'usage_percentage': (self.daily_calls / self.max_calls_per_day) * 100,
                'adaptive_coefficient': self.adaptive_coefficient,
                'latency_percentiles': self.latency_percentiles,
                'last_reset': self.last_reset.isoformat(),
                'is_throttled': self.is_throttled,
                'throttle_until': self.throttle_until.isoformat() if self.throttle_until else None
            }
            
            # 성능 통계
            if self.total_requests > 0:
                stats.update({
                    'total_requests': self.total_requests,
                    'failed_requests': self.failed_requests,
                    'timeout_requests': self.timeout_requests,
                    'success_rate': ((self.total_requests - self.failed_requests) / self.total_requests) * 100,
                    'requests_per_hour': self.total_requests / max(uptime_hours, 0.1)
                })
            
            # 지연 통계
            if self.latency_window:
                latencies = list(self.latency_window)
                stats.update({
                    'avg_latency': np.mean(latencies),
                    'min_latency': np.min(latencies),
                    'max_latency': np.max(latencies),
                    'latency_samples': len(latencies)
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"통계 생성 오류: {e}")
            return {'error': str(e)}


class RobustRedisManager:
    """강건한 Redis 관리자 with 개선된 폴백 및 복구"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, password: Optional[str] = None):
        self.host = host
        self.port = port
        self.password = password
        self.redis_client = None
        self.connection_attempts = 0
        self.last_error = None
        self.fallback_cache = {}  # 로컬 캐시 폴백
        self.use_fallback = True  # 기본적으로 폴백 사용
        
        # 개선된 캐시 관리
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'errors': 0
        }
        self.max_fallback_size = 10000
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5분
        
        # 재연결 관리
        self.last_connection_attempt = 0
        self.connection_backoff = 60  # 1분
        
    async def connect(self) -> bool:
        """Redis 연결 with 개선된 오류 처리"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis 라이브러리가 설치되지 않음, 폴백 캐시 사용")
            return False
        
        current_time = time.time()
        if current_time - self.last_connection_attempt < self.connection_backoff:
            logger.debug("연결 백오프 중...")
            return False
        
        self.last_connection_attempt = current_time
        self.connection_attempts += 1
        
        try:
            # Redis 클라이언트 생성 (개선된 설정)
            connection_kwargs = {
                'host': self.host,
                'port': self.port,
                'decode_responses': True,
                'socket_connect_timeout': 5,
                'socket_timeout': 5,
                'retry_on_timeout': True,
                'health_check_interval': 30,
                'max_connections': 20  # 연결 풀 크기 조정
            }
            
            if self.password:
                connection_kwargs['password'] = self.password
            
            if hasattr(redis, 'Redis'):
                self.redis_client = redis.Redis(**connection_kwargs)
            else:
                logger.error("Redis 클라이언트를 생성할 수 없습니다")
                return False
            
            # 연결 테스트 (타임아웃 적용)
            if hasattr(self.redis_client, 'ping'):
                await asyncio.wait_for(self.redis_client.ping(), timeout=10)
            else:
                # 동기 Redis의 경우
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.redis_client.ping)
            
            # 연결 성공
            self.use_fallback = False
            self.connection_backoff = 60  # 백오프 리셋
            
            # 상세 정보 로깅
            await self._log_connection_info()
            
            # 폴백 캐시 마이그레이션
            await self._migrate_fallback_cache()
            
            return True
            
        except asyncio.TimeoutError:
            self._handle_connection_error("연결 타임아웃", "TimeoutError")
            return False
        except Exception as e:
            error_type = type(e).__name__
            self._handle_connection_error(str(e), error_type)
            return False
    
    def _handle_connection_error(self, message: str, error_type: str):
        """연결 오류 처리 통합"""
        self.last_error = {
            'type': error_type,
            'message': message,
            'attempt': self.connection_attempts,
            'timestamp': datetime.now().isoformat(),
            'host': self.host,
            'port': self.port
        }
        
        # 백오프 시간 증가 (최대 10분)
        self.connection_backoff = min(600, self.connection_backoff * 1.5)
        
        logger.error(f"Redis 연결 실패 (시도 {self.connection_attempts}): {message}")
        logger.info(f"다음 시도까지 {self.connection_backoff:.0f}초 대기")
        
        # 폴백으로 전환
        self.redis_client = None
        self.use_fallback = True
    
    async def _log_connection_info(self):
        """연결 정보 로깅"""
        try:
            if hasattr(self.redis_client, 'info'):
                info = await self.redis_client.info()
            else:
                # 동기 버전 처리
                loop = asyncio.get_event_loop()
                info = await loop.run_in_executor(None, self.redis_client.info)
            
            logger.info(f"✅ Redis 연결 성공: {self.host}:{self.port}")
            logger.info(f"  버전: {info.get('redis_version', 'unknown')}")
            logger.info(f"  메모리: {info.get('used_memory_human', 'unknown')}")
            logger.info(f"  클라이언트: {info.get('connected_clients', 'unknown')}")
            
        except Exception as e:
            logger.debug(f"Redis 정보 조회 실패: {e}")
    
    async def get(self, key: str) -> Optional[str]:
        """데이터 조회 with 강화된 폴백"""
        try:
            if self.use_fallback:
                self._periodic_cleanup()
                result = self.fallback_cache.get(key)
                if result is not None:
                    self.cache_stats['hits'] += 1
                    return result['value'] if isinstance(result, dict) else result
                else:
                    self.cache_stats['misses'] += 1
                    return None
            
            # Redis에서 조회
            if hasattr(self.redis_client, 'get'):
                result = await self.redis_client.get(key)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.redis_client.get, key)
            
            if result is not None:
                self.cache_stats['hits'] += 1
            else:
                self.cache_stats['misses'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Redis get 오류: {e}")
            self.cache_stats['errors'] += 1
            # 자동 폴백 전환
            if not self.use_fallback:
                logger.warning("Redis 오류로 인한 폴백 전환")
                self.use_fallback = True
            return self.fallback_cache.get(key, {}).get('value') if isinstance(self.fallback_cache.get(key), dict) else self.fallback_cache.get(key)
    
    async def set(self, key: str, value: str, ttl: int = 86400) -> bool:
        """데이터 저장 with 강화된 폴백"""
        try:
            if self.use_fallback:
                self._set_fallback_cache(key, value, ttl)
                self.cache_stats['sets'] += 1
                return True
            
            # Redis에 저장
            if hasattr(self.redis_client, 'setex'):
                await self.redis_client.setex(key, ttl, value)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.redis_client.setex, key, ttl, value)
            
            self.cache_stats['sets'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Redis set 오류: {e}")
            self.cache_stats['errors'] += 1
            # 자동 폴백 전환
            if not self.use_fallback:
                logger.warning("Redis 오류로 인한 폴백 전환")
                self.use_fallback = True
            self._set_fallback_cache(key, value, ttl)
            return True
    
    def _set_fallback_cache(self, key: str, value: str, ttl: int):
        """폴백 캐시에 저장 with TTL 관리"""
        current_time = time.time()
        self.fallback_cache[key] = {
            'value': value,
            'timestamp': current_time,
            'ttl': ttl,
            'expires_at': current_time + ttl
        }
        
        # 캐시 크기 관리
        if len(self.fallback_cache) > self.max_fallback_size:
            self._cleanup_fallback_cache(force=True)
    
    def _periodic_cleanup(self):
        """주기적 캐시 정리"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_fallback_cache()
            self.last_cleanup = current_time
    
    def _cleanup_fallback_cache(self, force: bool = False):
        """폴백 캐시 정리 (TTL 기반)"""
        current_time = time.time()
        expired_keys = []
        
        # 만료된 키 찾기
        for key, data in self.fallback_cache.items():
            if isinstance(data, dict) and data.get('expires_at', 0) < current_time:
                expired_keys.append(key)
        
        # 만료된 키 제거
        for key in expired_keys:
            del self.fallback_cache[key]
        
        # 강제 정리시 LRU 방식으로 추가 제거
        if force and len(self.fallback_cache) > self.max_fallback_size:
            # 타임스탬프 기준 정렬하여 오래된 항목 제거
            sorted_items = sorted(
                self.fallback_cache.items(),
                key=lambda x: x[1].get('timestamp', 0) if isinstance(x[1], dict) else 0
            )
            
            keep_count = self.max_fallback_size // 2
            keys_to_remove = [item[0] for item in sorted_items[:-keep_count]]
            
            for key in keys_to_remove:
                del self.fallback_cache[key]
        
        if expired_keys or force:
            logger.debug(f"캐시 정리 완료: {len(expired_keys)}개 만료, "
                        f"현재 크기: {len(self.fallback_cache)}")
    
    async def _migrate_fallback_cache(self):
        """폴백 캐시를 Redis로 마이그레이션 (개선됨)"""
        if not self.fallback_cache:
            return
        
        logger.info(f"폴백 캐시 마이그레이션 시작: {len(self.fallback_cache)}개 항목")
        
        migrated = 0
        failed = 0
        
        for key, data in list(self.fallback_cache.items()):
            try:
                if isinstance(data, dict):
                    value = data['value']
                    # 남은 TTL 계산
                    remaining_ttl = max(1, int(data['expires_at'] - time.time()))
                else:
                    value = data
                    remaining_ttl = 86400  # 기본 TTL
                
                if remaining_ttl > 0:
                    if hasattr(self.redis_client, 'setex'):
                        await self.redis_client.setex(key, remaining_ttl, value)
                    else:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, self.redis_client.setex, key, remaining_ttl, value)
                    migrated += 1
                
            except Exception as e:
                logger.debug(f"마이그레이션 실패 {key}: {e}")
                failed += 1
        
        logger.info(f"마이그레이션 완료: {migrated}개 성공, {failed}개 실패")
        
        # 성공적으로 마이그레이션된 경우 폴백 캐시 정리
        if migrated > 0:
            self.fallback_cache.clear()
    
    async def get_diagnostics(self) -> Dict:
        """상세 진단 정보 (확장됨)"""
        diagnostics = {
            'connected': self.redis_client is not None and not self.use_fallback,
            'using_fallback': self.use_fallback,
            'connection_attempts': self.connection_attempts,
            'last_error': self.last_error,
            'fallback_cache_size': len(self.fallback_cache),
            'cache_stats': self.cache_stats.copy(),
            'connection_backoff': self.connection_backoff,
            'last_connection_attempt': datetime.fromtimestamp(self.last_connection_attempt).isoformat() if self.last_connection_attempt else None
        }
        
        # 성능 메트릭 계산
        total_operations = self.cache_stats['hits'] + self.cache_stats['misses']
        if total_operations > 0:
            diagnostics['cache_hit_rate'] = (self.cache_stats['hits'] / total_operations) * 100
        
        # Redis 연결되어 있을 때 추가 정보
        if self.redis_client and not self.use_fallback:
            try:
                if hasattr(self.redis_client, 'info'):
                    info = await self.redis_client.info()
                    stats = await self.redis_client.info('stats')
                else:
                    loop = asyncio.get_event_loop()
                    info = await loop.run_in_executor(None, self.redis_client.info)
                    stats = await loop.run_in_executor(None, self.redis_client.info, 'stats')
                
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
                diagnostics['redis_diagnostics_error'] = str(e)
        
        return diagnostics
    
    async def health_check(self) -> bool:
        """헬스 체크 with 자동 복구 시도"""
        # 폴백 모드에서 재연결 시도
        if self.use_fallback and self.redis_client is None:
            logger.info("Redis 재연결 시도 중...")
            connected = await self.connect()
            return connected
        
        # 연결되어 있을 때 ping 테스트
        if self.redis_client and not self.use_fallback:
            try:
                if hasattr(self.redis_client, 'ping'):
                    await asyncio.wait_for(self.redis_client.ping(), timeout=5)
                else:
                    loop = asyncio.get_event_loop()
                    await asyncio.wait_for(
                        loop.run_in_executor(None, self.redis_client.ping), 
                        timeout=5
                    )
                return True
            except Exception as e:
                logger.warning(f"Redis 헬스 체크 실패: {e}")
                self.use_fallback = True
                return False
        
        # 폴백 모드는 항상 정상
        return self.use_fallback


class StrictVKOSPIValidator:
    """엄격한 VKOSPI 데이터 검증기 (개선됨)"""
    
    def __init__(self):
        # 엄격한 품질 기준
        self.max_missing_ratio = 0.05  # 5% 결측치 허용
        self.valid_range = (5, 100)  # VKOSPI 유효 범위
        self.max_daily_change = 0.5  # 일일 최대 변동률 50%
        self.min_data_points = 20  # 최소 데이터 포인트
        
        # 추가 검증 기준
        self.max_consecutive_missing = 5  # 연속 결측치 최대 개수
        self.outlier_threshold = 0.05  # 5% 이상 이상치시 경고
        
    def validate_vkospi_data(self, df: pd.DataFrame) -> tuple[bool, Dict]:
        """VKOSPI 데이터 품질 검증 with 상세 리포트"""
        validation_report = {
            'is_valid': False,
            'total_rows': len(df) if not df.empty else 0,
            'issues': [],
            'warnings': [],
            'quality_score': 0.0
        }
        
        try:
            # 기본 검증
            if df.empty:
                validation_report['issues'].append("빈 데이터프레임")
                return False, validation_report
            
            if 'value' not in df.columns:
                validation_report['issues'].append("'value' 컬럼 누락")
                return False, validation_report
            
            # 데이터 포인트 검증
            if len(df) < self.min_data_points:
                validation_report['issues'].append(
                    f"데이터 포인트 부족: {len(df)} < {self.min_data_points}"
                )
                return False, validation_report
            
            # 결측치 검증 (개선됨)
            missing_mask = df['value'].isna()
            missing_count = missing_mask.sum()
            missing_ratio = missing_count / len(df)
            validation_report['missing_ratio'] = missing_ratio
            
            if missing_ratio > self.max_missing_ratio:
                validation_report['issues'].append(
                    f"높은 결측치 비율: {missing_ratio:.1%} > {self.max_missing_ratio:.1%}"
                )
                return False, validation_report
            
            # 연속 결측치 검증
            consecutive_missing = self._check_consecutive_missing(missing_mask)
            validation_report['max_consecutive_missing'] = consecutive_missing
            
            if consecutive_missing > self.max_consecutive_missing:
                validation_report['issues'].append(
                    f"연속 결측치 과다: {consecutive_missing} > {self.max_consecutive_missing}"
                )
            
            # 값 범위 검증
            valid_values = df['value'].dropna()
            if len(valid_values) == 0:
                validation_report['issues'].append("유효한 값이 없음")
                return False, validation_report
            
            min_val, max_val = valid_values.min(), valid_values.max()
            validation_report['value_range'] = (float(min_val), float(max_val))
            
            if min_val < self.valid_range[0] or max_val > self.valid_range[1]:
                validation_report['issues'].append(
                    f"값 범위 초과: [{min_val:.1f}, {max_val:.1f}] "
                    f"not in {self.valid_range}"
                )
                return False, validation_report
            
            # 일일 변동률 검증 (개선됨)
            daily_changes = valid_values.pct_change().abs()
            max_change = daily_changes.max()
            extreme_changes = (daily_changes > self.max_daily_change).sum()
            
            validation_report['max_daily_change'] = float(max_change) if not np.isnan(max_change) else 0.0
            validation_report['extreme_change_count'] = int(extreme_changes)
            
            if max_change > self.max_daily_change:
                if extreme_changes > len(valid_values) * 0.02:  # 2% 이상이면 문제
                    validation_report['issues'].append(
                        f"과도한 일일 변동: {extreme_changes}개 관측치가 {self.max_daily_change:.1%} 초과"
                    )
                else:
                    validation_report['warnings'].append(
                        f"일부 극단적 변동 감지: 최대 {max_change:.1%}"
                    )
            
            # 통계적 이상치 검증 (개선됨)
            outlier_info = self._detect_outliers(valid_values)
            validation_report.update(outlier_info)
            
            if outlier_info['outlier_ratio'] > self.outlier_threshold:
                validation_report['warnings'].append(
                    f"높은 이상치 비율: {outlier_info['outlier_ratio']:.1%}"
                )
            
            # 품질 점수 계산
            quality_score = self._calculate_quality_score(validation_report, len(valid_values))
            validation_report['quality_score'] = quality_score
            
            # 최종 판정
            critical_issues = len(validation_report['issues'])
            validation_report['is_valid'] = critical_issues == 0 and quality_score >= 0.7
            
            # 로그 출력
            if validation_report['is_valid']:
                logger.info(f"✅ VKOSPI 데이터 검증 통과 (품질 점수: {quality_score:.2f})")
            else:
                logger.warning(f"✗ VKOSPI 검증 실패: {validation_report['issues']}")
            
            if validation_report['warnings']:
                logger.info(f"경고사항: {validation_report['warnings']}")
            
            return validation_report['is_valid'], validation_report
            
        except Exception as e:
            logger.error(f"VKOSPI 검증 중 오류: {e}")
            validation_report['issues'].append(f"검증 오류: {str(e)}")
            return False, validation_report
    
    def _check_consecutive_missing(self, missing_mask: pd.Series) -> int:
        """연속 결측치 개수 확인"""
        max_consecutive = 0
        current_consecutive = 0
        
        for is_missing in missing_mask:
            if is_missing:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _detect_outliers(self, values: pd.Series) -> Dict:
        """통계적 이상치 탐지 (개선된 다중 방법)"""
        try:
            outlier_info = {
                'outlier_ratio': 0.0,
                'outlier_count': 0,
                'outlier_methods': {}
            }
            
            if len(values) < 10:
                return outlier_info
            
            values_array = values.values
            total_count = len(values_array)
            
            # 방법 1: IQR 방법
            q1, q3 = np.percentile(values_array, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                iqr_outliers = ((values_array < lower_bound) | (values_array > upper_bound)).sum()
                outlier_info['outlier_methods']['iqr'] = {
                    'count': int(iqr_outliers),
                    'ratio': iqr_outliers / total_count,
                    'bounds': (float(lower_bound), float(upper_bound))
                }
            
            # 방법 2: Z-Score 방법 (3σ 기준)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            if std_val > 0:
                z_scores = np.abs((values_array - mean_val) / std_val)
                zscore_outliers = (z_scores > 3).sum()
                outlier_info['outlier_methods']['zscore'] = {
                    'count': int(zscore_outliers),
                    'ratio': zscore_outliers / total_count,
                    'threshold': 3.0
                }
            
            # 방법 3: Modified Z-Score (중앙값 기반)
            median_val = np.median(values_array)
            mad = np.median(np.abs(values_array - median_val))
            if mad > 0:
                modified_z_scores = 0.6745 * (values_array - median_val) / mad
                mad_outliers = (np.abs(modified_z_scores) > 3.5).sum()
                outlier_info['outlier_methods']['modified_zscore'] = {
                    'count': int(mad_outliers),
                    'ratio': mad_outliers / total_count,
                    'threshold': 3.5
                }
            
            # 가장 보수적인 방법의 결과 사용 (가장 적은 이상치 개수)
            method_counts = [
                method_info.get('count', 0) 
                for method_info in outlier_info['outlier_methods'].values()
            ]
            
            if method_counts:
                min_outliers = min(method_counts)
                outlier_info['outlier_count'] = min_outliers
                outlier_info['outlier_ratio'] = min_outliers / total_count
            
            return outlier_info
            
        except Exception as e:
            logger.debug(f"이상치 탐지 오류: {e}")
            return {
                'outlier_ratio': 0.0,
                'outlier_count': 0,
                'outlier_methods': {},
                'error': str(e)
            }
    
    def _calculate_quality_score(self, validation_report: Dict, valid_count: int) -> float:
        """데이터 품질 점수 계산 (0-1 스케일)"""
        try:
            score = 1.0
            
            # 결측치 페널티
            missing_ratio = validation_report.get('missing_ratio', 0)
            score -= missing_ratio * 2  # 결측치는 2배 페널티
            
            # 연속 결측치 페널티
            max_consecutive = validation_report.get('max_consecutive_missing', 0)
            if max_consecutive > 0:
                score -= min(0.2, max_consecutive / self.max_consecutive_missing * 0.2)
            
            # 이상치 페널티
            outlier_ratio = validation_report.get('outlier_ratio', 0)
            score -= outlier_ratio * 0.5
            
            # 극단적 변동 페널티
            extreme_changes = validation_report.get('extreme_change_count', 0)
            if extreme_changes > 0:
                extreme_ratio = extreme_changes / valid_count
                score -= extreme_ratio * 0.3
            
            # 데이터 충분성 보너스
            if valid_count >= self.min_data_points * 2:
                score += 0.1
            
            # 범위 내 값 보너스
            value_range = validation_report.get('value_range', (0, 0))
            if value_range[0] >= self.valid_range[0] and value_range[1] <= self.valid_range[1]:
                score += 0.05
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.debug(f"품질 점수 계산 오류: {e}")
            return 0.5  # 기본값


class APICallTracker:
    """API 호출 추적 및 분석"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.call_history = deque(maxlen=max_history)
        self.daily_stats = {}
        
    def record_call(self, endpoint: str, success: bool, latency: float, 
                   response_size: int = 0, error_type: str = None):
        """API 호출 기록"""
        call_record = {
            'timestamp': datetime.now(),
            'endpoint': endpoint,
            'success': success,
            'latency': latency,
            'response_size': response_size,
            'error_type': error_type
        }
        
        self.call_history.append(call_record)
        self._update_daily_stats(call_record)
    
    def _update_daily_stats(self, call_record: Dict):
        """일일 통계 업데이트"""
        date_key = call_record['timestamp'].date().isoformat()
        
        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_latency': 0.0,
                'total_response_size': 0,
                'error_types': {}
            }
        
        stats = self.daily_stats[date_key]
        stats['total_calls'] += 1
        stats['total_latency'] += call_record['latency']
        stats['total_response_size'] += call_record['response_size']
        
        if call_record['success']:
            stats['successful_calls'] += 1
        else:
            stats['failed_calls'] += 1
            if call_record['error_type']:
                error_type = call_record['error_type']
                stats['error_types'][error_type] = stats['error_types'].get(error_type, 0) + 1
    
    def get_performance_summary(self, days: int = 7) -> Dict:
        """성능 요약 통계"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).date()
            recent_calls = [
                call for call in self.call_history 
                if call['timestamp'].date() >= cutoff_date
            ]
            
            if not recent_calls:
                return {'error': '데이터가 없습니다'}
            
            successful_calls = [call for call in recent_calls if call['success']]
            failed_calls = [call for call in recent_calls if not call['success']]
            
            # 기본 통계
            summary = {
                'period_days': days,
                'total_calls': len(recent_calls),
                'successful_calls': len(successful_calls),
                'failed_calls': len(failed_calls),
                'success_rate': len(successful_calls) / len(recent_calls) * 100,
                'average_latency': np.mean([call['latency'] for call in successful_calls]) if successful_calls else 0,
                'median_latency': np.median([call['latency'] for call in successful_calls]) if successful_calls else 0,
                'p95_latency': np.percentile([call['latency'] for call in successful_calls], 95) if successful_calls else 0
            }
            
            # 오류 분석
            if failed_calls:
                error_types = {}
                for call in failed_calls:
                    error_type = call.get('error_type', 'unknown')
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                summary['error_breakdown'] = error_types
            
            # 엔드포인트별 통계
            endpoint_stats = {}
            for call in recent_calls:
                endpoint = call['endpoint']
                if endpoint not in endpoint_stats:
                    endpoint_stats[endpoint] = {'calls': 0, 'failures': 0, 'latencies': []}
                
                endpoint_stats[endpoint]['calls'] += 1
                if not call['success']:
                    endpoint_stats[endpoint]['failures'] += 1
                else:
                    endpoint_stats[endpoint]['latencies'].append(call['latency'])
            
            # 엔드포인트 요약
            endpoint_summary = {}
            for endpoint, stats in endpoint_stats.items():
                endpoint_summary[endpoint] = {
                    'calls': stats['calls'],
                    'failure_rate': (stats['failures'] / stats['calls']) * 100,
                    'avg_latency': np.mean(stats['latencies']) if stats['latencies'] else 0
                }
            
            summary['endpoint_stats'] = endpoint_summary
            
            return summary
            
        except Exception as e:
            logger.error(f"성능 요약 생성 오류: {e}")
            return {'error': str(e)}


# 통합된 API 클라이언트 기본 클래스
class BaseAPIClient:
    """기본 API 클라이언트 with 공통 기능"""
    
    def __init__(self, base_url: str, rate_limiter: EnhancedAdaptiveRateLimiter,
                 redis_manager: Optional[RobustRedisManager] = None):
        self.base_url = base_url
        self.rate_limiter = rate_limiter
        self.redis_manager = redis_manager
        self.session = None
        self.call_tracker = APICallTracker()
        
        # 재시도 설정
        self.max_retries = 3
        self.retry_backoff = [1, 2, 4]  # 지수 백오프
        
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self._close_session()
    
    async def _ensure_session(self):
        """HTTP 세션 보장"""
        if self.session is None:
            try:
                import aiohttp
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                self.session = aiohttp.ClientSession(timeout=timeout)
            except ImportError:
                logger.error("aiohttp가 설치되지 않음")
                raise RuntimeError("aiohttp 라이브러리가 필요합니다")
    
    async def _close_session(self):
        """HTTP 세션 종료"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _make_request(self, endpoint: str, params: Dict = None, 
                          cache_key: str = None, cache_ttl: int = 3600) -> Optional[Dict]:
        """공통 HTTP 요청 처리"""
        # 캐시 확인
        if cache_key and self.redis_manager:
            cached_result = await self.redis_manager.get(cache_key)
            if cached_result:
                try:
                    return json.loads(cached_result)
                except json.JSONDecodeError:
                    logger.debug(f"캐시된 데이터 파싱 실패: {cache_key}")
        
        # Rate limiting 적용
        rate_limit_ok = await self.rate_limiter.check_and_wait()
        if not rate_limit_ok:
            self.call_tracker.record_call(endpoint, False, 0, error_type="rate_limited")
            return None
        
        # 세션 보장
        await self._ensure_session()
        
        # 재시도 로직을 포함한 요청
        for attempt in range(self.max_retries):
            start_time = time.time()
            
            try:
                url = f"{self.base_url}/{endpoint.lstrip('/')}"
                
                async with self.session.get(url, params=params) as response:
                    latency = time.time() - start_time
                    
                    if response.status == 200:
                        content = await response.text()
                        response_size = len(content.encode('utf-8'))
                        
                        # 성공 기록
                        self.rate_limiter.record_latency(latency)
                        self.call_tracker.record_call(endpoint, True, latency, response_size)
                        
                        # 캐싱
                        if cache_key and self.redis_manager:
                            await self.redis_manager.set(cache_key, content, cache_ttl)
                        
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            logger.warning(f"응답 JSON 파싱 실패: {endpoint}")
                            return {'raw_content': content}
                    
                    elif response.status == 429:  # Too Many Requests
                        latency = time.time() - start_time
                        self.rate_limiter.record_failure("rate_limited")
                        self.call_tracker.record_call(endpoint, False, latency, error_type="rate_limited")
                        
                        # 백오프 후 재시도
                        if attempt < self.max_retries - 1:
                            wait_time = self.retry_backoff[attempt]
                            logger.warning(f"Rate limit 도달, {wait_time}초 대기 후 재시도")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Rate limit 재시도 실패: {endpoint}")
                            return None
                    
                    else:
                        latency = time.time() - start_time
                        error_type = f"http_{response.status}"
                        self.rate_limiter.record_failure(error_type)
                        self.call_tracker.record_call(endpoint, False, latency, error_type=error_type)
                        
                        logger.error(f"HTTP 오류 {response.status}: {endpoint}")
                        if attempt == self.max_retries - 1:
                            return None
                        
                        # 백오프 후 재시도
                        await asyncio.sleep(self.retry_backoff[attempt])
            
            except asyncio.TimeoutError:
                latency = time.time() - start_time
                self.rate_limiter.record_failure("timeout")
                self.call_tracker.record_call(endpoint, False, latency, error_type="timeout")
                
                logger.warning(f"요청 타임아웃: {endpoint} (시도 {attempt + 1})")
                if attempt == self.max_retries - 1:
                    return None
                
                await asyncio.sleep(self.retry_backoff[attempt])
            
            except Exception as e:
                latency = time.time() - start_time
                error_type = type(e).__name__
                self.rate_limiter.record_failure(error_type)
                self.call_tracker.record_call(endpoint, False, latency, error_type=error_type)
                
                logger.error(f"요청 오류: {endpoint} - {str(e)} (시도 {attempt + 1})")
                if attempt == self.max_retries - 1:
                    return None
                
                await asyncio.sleep(self.retry_backoff[attempt])
        
        return None
    
    def get_api_usage_stats(self) -> Dict:
        """API 사용 통계 반환"""
        return {
            'rate_limiter_stats': self.rate_limiter.get_stats(),
            'performance_summary': self.call_tracker.get_performance_summary(),
            'redis_diagnostics': asyncio.create_task(self.redis_manager.get_diagnostics()) if self.redis_manager else None
        }# Improved API Rate Limiting and Redis Management

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Deque, Any, List
from collections import deque
import time
import json
import hashlib
from contextlib import asynccontextmanager

# Redis 임포트 (더 안전한 방식)
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis
        REDIS_AVAILABLE = True
        # 동기 버전을 비동기처럼 사용하기 위한 래퍼 필요
    except ImportError:
        REDIS_AVAILABLE = False
        redis = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    # 간단한 재시도 데코레이터 구현
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    stop_after_attempt = wait_exponential = retry_if_exception_type = lambda x: x

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
        
        # 통계 추적
        self.total_requests = 0
        self.failed_requests = 0
        self.timeout_requests = 0
        
        # 상태 관리
        self.is_throttled = False
        self.throttle_until = None
        
    async def check_and_wait(self) -> bool:
        """향상된 호출 제한 확인 with 급격한 변동 대응"""
        try:
            now = datetime.now()
            
            # 일일 제한 리셋 확인 (시간 동기화 개선)
            if now > self.last_reset + timedelta(days=1):
                self._reset_daily_limits()
            
            # 스로틀링 상태 확인
            if self.is_throttled and self.throttle_until:
                if now < self.throttle_until:
                    wait_time = (self.throttle_until - now).total_seconds()
                    logger.warning(f"스로틀링 중... {wait_time:.1f}초 대기")
                    await asyncio.sleep(min(wait_time, 60))  # 최대 60초
                    return False
                else:
                    self.is_throttled = False
                    self.throttle_until = None
                    logger.info("스로틀링 해제됨")
            
            # 일일 제한 확인
            if self.daily_calls >= self.max_calls_per_day:
                await self._wait_for_daily_reset(now)
                return False
            
            # 초당 제한 확인 with 동적 조정
            await self._check_rate_limit_with_adaptation(now)
            
            # 호출 기록
            self.call_times.append(now)
            self.daily_calls += 1
            self.total_requests += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limiter 오류: {e}")
            # 안전한 폴백: 기본 대기
            await asyncio.sleep(1)
            return False
    
    def _reset_daily_limits(self):
        """일일 제한 초기화 with 통계 로깅"""
        prev_daily_calls = self.daily_calls
        self.daily_calls = 0
        self.last_reset = datetime.now()
        self.call_times.clear()
        
        logger.info(f"일일 API 제한 리셋 - 이전 사용량: {prev_daily_calls}/{self.max_calls_per_day}")
        
        # 일일 통계 요약
        if prev_daily_calls > 0:
            failure_rate = (self.failed_requests / max(self.total_requests, 1)) * 100
            logger.info(f"일일 통계 - 실패율: {failure_rate:.1f}%, 총 요청: {self.total_requests}")
    
    async def _wait_for_daily_reset(self, now: datetime):
        """일일 제한 대기 with 점진적 백오프"""
        reset_time = self.last_reset + timedelta(days=1)
        wait_time = (reset_time - now).total_seconds()
        
        logger.warning(f"일일 API 제한 도달. {wait_time/3600:.1f}시간 대기 필요")
        
        # 긴 대기시간을 청크로 나누어 처리
        chunk_size = min(3600, wait_time)  # 최대 1시간씩
        while wait_time > 0:
            sleep_time = min(chunk_size, wait_time)
            logger.info(f"대기 중... {wait_time/3600:.1f}시간 남음")
            await asyncio.sleep(sleep_time)
            wait_time -= sleep_time
        
        self._reset_daily_limits()
    
    async def _check_rate_limit_with_adaptation(self, now: datetime):
        """적응형 속도 제한 확인 with 개선된 알고리즘"""
        # 최근 1초간의 호출 확인
        recent_calls = [t for t in self.call_times if (now - t).total_seconds() < 1]
        
        if len(recent_calls) >= self.max_calls_per_second:
            # 기본 대기 시간 계산
            oldest_recent = min(recent_calls) if recent_calls else now
            base_sleep = 1 - (now - oldest_recent).total_seconds()
            
            # 적응형 지연 계산
            adaptive_delay = self._calculate_adaptive_delay()
            
            # 백오프 전략 with 지터 (개선된 공식)
            jitter = np.random.uniform(0, 0.1)  # 줄어든 지터
            congestion_factor = min(len(recent_calls) / self.max_calls_per_second, 2.0)
            
            total_sleep = max(0, base_sleep + adaptive_delay * congestion_factor + jitter)
            
            if total_sleep > 0:
                logger.debug(f"Rate limit 대기: {total_sleep:.3f}s "
                           f"(base: {base_sleep:.3f}, adaptive: {adaptive_delay:.3f}, "
                           f"congestion: {congestion_factor:.1f}, jitter: {jitter:.3f})")
                await asyncio.sleep(total_sleep)
    
    def _calculate_adaptive_delay(self) -> float:
        """급격한 변동을 고려한 적응형 지연 계산 (개선됨)"""
        if len(self.latency_window) < 5:
            return 0.0
        
        try:
            # 통계 계산 (안전한 방식)
            latencies = np.array(list(self.latency_window))
            latencies = latencies[~np.isnan(latencies)]  # NaN 제거
            
            if len(latencies) == 0:
                return 0.0
            
            mean_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            # 백분위수 계산
            self.latency_percentiles = {
                'p50': np.percentile(latencies, 50),
                'p90': np.percentile(latencies, 90),
                'p99': np.percentile(latencies, 99)
            }
            
            # 최근 지연과 비교 (더 안정적인 윈도우)
            recent_window = min(10, len(latencies))
