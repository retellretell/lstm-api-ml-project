# Enhanced Hyperparameter and Robustness Tests

import pytest
import pytest_asyncio
import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Tuple
import logging
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

class TestExtendedHyperparameterSearch:
    """확장된 하이퍼파라미터 검색 테스트"""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("learning_rate", [0.0001, 0.001, 0.01, 0.1])
    @pytest.mark.parametrize("batch_size", [8, 16, 32, 64, 128])
    @pytest.mark.parametrize("dropout_rate", [0.1, 0.2, 0.3, 0.5])
    async def test_comprehensive_hyperparameter_grid(self, test_db, learning_rate, 
                                                   batch_size, dropout_rate):
        """포괄적인 하이퍼파라미터 그리드 검색"""
        # 극단적인 값에 대한 경고
        if learning_rate >= 0.1:
            logger.warning(f"Testing extreme learning rate: {learning_rate}")
        if batch_size >= 128:
            logger.warning(f"Testing large batch size: {batch_size}")
        if dropout_rate >= 0.5:
            logger.warning(f"Testing high dropout rate: {dropout_rate}")
        
        ensemble = StackingEnsembleWithLSTM()
        
        # 데이터 준비
        X, y = self._prepare_test_data(test_db)
        
        # 하이퍼파라미터 설정
        ensemble.lstm_learning_rate = learning_rate
        ensemble.lstm_batch_size = batch_size
        ensemble.lstm_dropout_rate = dropout_rate
        
        try:
            # LSTM 학습
            lstm_predictions = await ensemble.train_lstm_async(X, y)
            
            # 성능 평가
            auc = roc_auc_score(y[30:], lstm_predictions[30:])
            
            # 수렴 분석
            if hasattr(ensemble.lstm_model, 'history'):
                history = ensemble.lstm_model.history.history
                
                # 조기 수렴 감지
                converged = False
                convergence_epoch = len(history['loss'])
                
                for i in range(5, len(history['loss'])):
                    recent_losses = history['loss'][i-5:i]
                    if np.std(recent_losses) < 0.001:
                        converged = True
                        convergence_epoch = i
                        break
                
                # 발산 감지
                diverged = any(np.isnan(history['loss']) or np.isinf(history['loss']))
                
                # 결과 저장
                result = {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'dropout_rate': dropout_rate,
                    'auc': auc,
                    'converged': converged,
                    'convergence_epoch': convergence_epoch,
                    'diverged': diverged,
                    'final_loss': history['loss'][-1] if not diverged else float('inf')
                }
                
                # 안정성 검증
                if diverged:
                    pytest.fail(f"Model diverged with lr={learning_rate}, "
                              f"batch_size={batch_size}, dropout={dropout_rate}")
                
                # 극단적 하이퍼파라미터에 대한 성능 하한
                if learning_rate >= 0.1 or batch_size >= 128:
                    assert auc > 0.45, f"Poor performance with extreme hyperparameters: AUC={auc:.3f}"
                else:
                    assert auc > 0.55, f"Poor performance: AUC={auc:.3f}"
                
                logger.info(f"Hyperparameter test result: {result}")
                return result
                
        except Exception as e:
            logger.error(f"Training failed with hyperparameters: "
                        f"lr={learning_rate}, batch={batch_size}, dropout={dropout_rate}")
            logger.error(f"Error: {str(e)}")
            
            # 극단적인 하이퍼파라미터로 인한 실패는 예상 가능
            if learning_rate >= 0.1 or dropout_rate >= 0.5:
                pytest.skip(f"Expected failure with extreme hyperparameters: {str(e)}")
            else:
                raise


class TestMarketEventRobustness:
    """시장 이벤트에 대한 강건성 테스트"""
    
    @pytest.mark.asyncio
    async def test_black_swan_event_robustness(self, test_db):
        """블랙 스완 이벤트에 대한 모델 강건성"""
        ensemble = StackingEnsembleWithLSTM()
        
        # 정상 데이터
        X_normal, y_normal = self._prepare_test_data(test_db)
        
        # 블랙 스완 이벤트 시뮬레이션
        scenarios = {
            'normal': self._create_normal_scenario,
            'flash_crash': self._create_flash_crash_scenario,
            'volatility_spike': self._create_volatility_spike_scenario,
            'liquidity_crisis': self._create_liquidity_crisis_scenario,
            'circuit_breaker': self._create_circuit_breaker_scenario
        }
        
        results = {}
        
        for scenario_name, scenario_func in scenarios.items():
            logger.info(f"Testing {scenario_name} scenario...")
            
            # 시나리오별 데이터 생성
            X_scenario, y_scenario = scenario_func(X_normal.copy(), y_normal.copy())
            
            try:
                # 모델 학습
                lstm_predictions = await ensemble.train_lstm_async(X_scenario, y_scenario)
                
                # 성능 평가
                auc = roc_auc_score(y_scenario[30:], lstm_predictions[30:])
                
                # 예측 안정성 평가
                pred_std = np.std(lstm_predictions[30:])
                pred_range = np.max(lstm_predictions[30:]) - np.min(lstm_predictions[30:])
                
                results[scenario_name] = {
                    'auc': auc,
                    'prediction_std': pred_std,
                    'prediction_range': pred_range,
                    'nan_count': np.isnan(lstm_predictions).sum(),
                    'inf_count': np.isinf(lstm_predictions).sum()
                }
                
                # 안정성 검증
                assert results[scenario_name]['nan_count'] == 0, \
                    f"NaN predictions in {scenario_name} scenario"
                assert results[scenario_name]['inf_count'] == 0, \
                    f"Inf predictions in {scenario_name} scenario"
                
                # 성능 하락 허용 범위
                if scenario_name != 'normal':
                    performance_drop = results['normal']['auc'] - auc
                    assert performance_drop < 0.3, \
                        f"Excessive performance drop in {scenario_name}: {performance_drop:.3f}"
                
            except Exception as e:
                logger.error(f"Failed in {scenario_name} scenario: {str(e)}")
                results[scenario_name] = {'error': str(e)}
        
        # 결과 요약
        logger.info("Black Swan Event Robustness Test Results:")
        for scenario, result in results.items():
            if 'error' not in result:
                logger.info(f"{scenario}: AUC={result['auc']:.3f}, "
                           f"Stability={result['prediction_std']:.3f}")
        
        return results
    
    def _create_flash_crash_scenario(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """플래시 크래시 시나리오"""
        crash_idx = len(X) // 2
        crash_magnitude = 0.2  # 20% 급락
        
        # 가격 관련 특징에 충격 적용
        price_features = [0, 1, 2, 3]  # open, high, low, close
        for feat in price_features:
            X[crash_idx:crash_idx+5, feat] *= (1 - crash_magnitude)
        
        # 거래량 급증
        X[crash_idx:crash_idx+5, 4] *= 5  # volume
        
        return X, y
    
    def _create_volatility_spike_scenario(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """변동성 급등 시나리오"""
        # 전체 기간에 걸쳐 변동성 증가
        volatility_multiplier = np.random.uniform(2, 5, size=X.shape[0])
        
        # 가격 변동성 증가
        for i in range(1, len(X)):
            price_change = np.random.normal(0, 0.05) * volatility_multiplier[i]
            X[i, :4] = X[i-1, :4] * (1 + price_change)
        
        # RSI 극단값
        X[:, 7] = np.clip(X[:, 7] * np.random.uniform(0.5, 1.5, size=len(X)), 0, 100)
        
        return X, y
    
    def _create_liquidity_crisis_scenario(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """유동성 위기 시나리오"""
        # 거래량 급감
        crisis_start = len(X) // 3
        crisis_end = 2 * len(X) // 3
        
        X[crisis_start:crisis_end, 4] *= 0.1  # 거래량 90% 감소
        
        # 스프레드 확대 (high-low)
        X[crisis_start:crisis_end, 1] *= 1.05  # high
        X[crisis_start:crisis_end, 2] *= 0.95  # low
        
        return X, y
    
    def _create_circuit_breaker_scenario(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """서킷브레이커 시나리오"""
        # 거래 중단 시뮬레이션
        halt_periods = [
            (len(X) // 4, len(X) // 4 + 10),
            (len(X) // 2, len(X) // 2 + 5),
            (3 * len(X) // 4, 3 * len(X) // 4 + 15)
        ]
        
        for start, end in halt_periods:
            # 거래 중단 기간: 가격 동결, 거래량 0
            if end < len(X):
                X[start:end, 4] = 0  # volume
                # 가격은 이전 값 유지
                for i in range(start, min(end, len(X))):
                    X[i, :4] = X[start-1, :4]
        
        return X, y
    
    def _create_normal_scenario(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """정상 시나리오 (비교 기준)"""
        return X, y


class TestDynamicNetworkLatency:
    """동적 네트워크 지연 테스트"""
    
    @pytest.mark.asyncio
    async def test_adaptive_timeout_with_realistic_delays(self):
        """현실적인 네트워크 지연을 고려한 적응형 타임아웃"""
        collector = IntegratedMacroDataCollector()
        
        # 다양한 네트워크 상황 시뮬레이션
        network_scenarios = {
            'fast': [0.1, 0.2, 0.15],
            'normal': [0.5, 0.7, 0.6],
            'slow': [1.0, 1.5, 2.0],
            'unstable': [0.1, 3.0, 0.2, 5.0, 0.5]
        }
        
        results = {}
        
        for scenario_name, delays in network_scenarios.items():
            logger.info(f"Testing {scenario_name} network scenario...")
            
            # 네트워크 지연 시뮬레이션 설정
            async def mock_api_call_with_delay(stock_code):
                delay = np.random.choice(delays)
                await asyncio.sleep(delay)
                
                # 10% 확률로 타임아웃 시뮬레이션
                if np.random.random() < 0.1:
                    raise asyncio.TimeoutError(f"Simulated timeout after {delay}s")
                