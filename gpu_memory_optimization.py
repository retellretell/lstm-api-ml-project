# Enhanced GPU Memory Management and Mixed Precision Stability

import tensorflow as tf
import gc
import psutil
import logging
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """GPU 메모리 관리 및 OOM 방지 시스템"""
    
    def __init__(self):
        self.gpu_memory_limit = None
        self.oom_count = 0
        self.max_oom_retries = 3
        
    def setup_gpu_configuration(self, memory_limit_mb: Optional[int] = None):
        """향상된 GPU 설정 with OOM 핸들링"""
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            logger.warning("No GPU found. Using CPU.")
            return False
        
        for gpu_idx, gpu in enumerate(gpus):
            try:
                # 먼저 메모리 성장 설정 시도
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU {gpu_idx}: Memory growth enabled")
                
            except RuntimeError as e:
                logger.warning(f"Failed to set memory growth for GPU {gpu_idx}: {e}")
                
                # 메모리 제한 설정으로 폴백
                try:
                    if memory_limit_mb:
                        memory_limit = memory_limit_mb
                    else:
                        # GPU 메모리의 80%만 사용
                        total_memory = tf.config.experimental.get_memory_info(gpu)['total']
                        memory_limit = int(total_memory * 0.8 / (1024 * 1024))  # MB 단위
                    
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
                    logger.info(f"GPU {gpu_idx}: Memory limited to {memory_limit}MB")
                    self.gpu_memory_limit = memory_limit
                    
                except Exception as fallback_error:
                    logger.error(f"Failed to configure GPU {gpu_idx}: {fallback_error}")
                    return False
        
        # GPU 정보 로깅
        self._log_gpu_info(gpus)
        return True
    
    def _log_gpu_info(self, gpus: List):
        """GPU 정보 상세 로깅"""
        for i, gpu in enumerate(gpus):
            try:
                # GPU 메모리 정보
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                total_memory = memory_info.get('total', 0) / (1024**3)  # GB
                current_memory = memory_info.get('current', 0) / (1024**3)  # GB
                
                logger.info(f"GPU {i}: {gpu.name}")
                logger.info(f"  Total Memory: {total_memory:.2f} GB")
                logger.info(f"  Current Usage: {current_memory:.2f} GB")
                logger.info(f"  Available: {total_memory - current_memory:.2f} GB")
                
            except Exception as e:
                logger.warning(f"Could not get memory info for GPU {i}: {e}")
    
    def handle_oom(self, batch_size: int) -> int:
        """OOM 발생 시 배치 크기 자동 조정"""
        self.oom_count += 1
        
        if self.oom_count > self.max_oom_retries:
            raise RuntimeError(f"OOM occurred {self.oom_count} times. Stopping.")
        
        # 배치 크기 50% 감소
        new_batch_size = max(1, batch_size // 2)
        
        # 메모리 정리
        self._clear_memory()
        
        logger.warning(f"OOM detected (attempt {self.oom_count}). "
                      f"Reducing batch size: {batch_size} -> {new_batch_size}")
        
        return new_batch_size
    
    def _clear_memory(self):
        """GPU 및 시스템 메모리 정리"""
        # Python 가비지 컬렉션
        gc.collect()
        
        # TensorFlow 메모리 정리
        tf.keras.backend.clear_session()
        
        # GPU 메모리 정리 (가능한 경우)
        try:
            if tf.config.list_physical_devices('GPU'):
                # GPU 메모리 캐시 정리
                tf.config.experimental.reset_memory_stats('GPU:0')
        except Exception as e:
            logger.debug(f"Could not reset GPU memory stats: {e}")


class MixedPrecisionStabilizer:
    """Mixed Precision 학습 안정화"""
    
    def __init__(self, enable_mixed_precision: bool = True):
        self.enable_mixed_precision = enable_mixed_precision
        self.loss_scale_manager = None
        
    def setup_mixed_precision(self):
        """안정적인 Mixed Precision 설정"""
        if not self.enable_mixed_precision:
            return
        
        try:
            # Mixed precision 정책 설정
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            # 동적 손실 스케일링 설정
            self.loss_scale_manager = tf.keras.mixed_precision.LossScaleOptimizer(
                tf.keras.optimizers.Adam(learning_rate=0.001),
                dynamic=True,
                initial_scale=2**15,
                dynamic_growth_steps=2000
            )
            
            logger.info("Mixed precision training enabled with dynamic loss scaling")
            logger.info(f"Compute dtype: {policy.compute_dtype}")
            logger.info(f"Variable dtype: {policy.variable_dtype}")
            
        except Exception as e:
            logger.error(f"Failed to enable mixed precision: {e}")
            self.enable_mixed_precision = False
    
    def create_stable_lstm_model(self, input_shape: tuple, use_gru: bool = False):
        """Mixed Precision에 안정적인 LSTM/GRU 모델 생성"""
        model = tf.keras.Sequential()
        
        # 입력 레이어 - float32로 캐스팅
        if self.enable_mixed_precision:
            model.add(tf.keras.layers.Lambda(
                lambda x: tf.cast(x, tf.float32),
                input_shape=input_shape
            ))
        
        # RNN 레이어 with 안정성 개선
        rnn_layer = tf.keras.layers.GRU if use_gru else tf.keras.layers.LSTM
        
        # 첫 번째 RNN 레이어
        model.add(rnn_layer(
            128,
            return_sequences=True,
            input_shape=input_shape if not self.enable_mixed_precision else None,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
            recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
            dropout=0.2,
            recurrent_dropout=0.2,
            # Mixed precision 안정성을 위한 설정
            recurrent_activation='sigmoid' if not self.enable_mixed_precision else 'hard_sigmoid',
            use_bias=True,
            bias_initializer='zeros'
        ))
        
        # Gradient clipping을 위한 레이어
        if self.enable_mixed_precision:
            model.add(tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(x, -10, 10)
            ))
        
        model.add(tf.keras.layers.BatchNormalization())
        
        # 두 번째 RNN 레이어
        model.add(rnn_layer(
            64,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2
        ))
        
        model.add(tf.keras.layers.BatchNormalization())
        
        # 세 번째 RNN 레이어
        model.add(rnn_layer(
            32,
            dropout=0.2,
            recurrent_dropout=0.2
        ))
        
        # Dense 레이어 전 float32 캐스팅 (mixed precision 사용 시)
        if self.enable_mixed_precision:
            model.add(tf.keras.layers.Lambda(
                lambda x: tf.cast(x, tf.float32)
            ))
        
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(
            16,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
        ))
        model.add(tf.keras.layers.Dropout(0.3))
        
        # 출력 레이어 - 항상 float32
        model.add(tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            dtype='float32'  # Mixed precision에서도 float32 출력
        ))
        
        return model
    
    def compile_with_mixed_precision(self, model, learning_rate: float = 0.001):
        """Mixed Precision 최적화된 컴파일"""
        if self.enable_mixed_precision and self.loss_scale_manager:
            optimizer = self.loss_scale_manager
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # 손실 함수에 안정성 추가
        def stable_binary_crossentropy(y_true, y_pred):
            # 예측값 클리핑으로 log(0) 방지
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            return tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        model.compile(
            optimizer=optimizer,
            loss=stable_binary_crossentropy,
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model


class OptimizedBackupStrategy:
    """최적화된 백업 전략"""
    
    def __init__(self, backup_dir: str = "./backup", save_frequency: int = 5):
        self.backup_dir = backup_dir
        self.save_frequency = save_frequency
        self.backup_count = 0
        
    def create_backup_callbacks(self):
        """I/O 최적화된 백업 콜백 생성"""
        callbacks = []
        
        # 주기적 백업 (I/O 부하 감소)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.backup_dir}/checkpoint_{{epoch:02d}}.h5",
                save_freq=self.save_frequency * 100,  # steps 기준 (epochs * steps_per_epoch)
                save_weights_only=True,  # 가중치만 저장으로 I/O 감소
                verbose=0
            )
        )
        
        # 최고 성능 모델만 전체 저장
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.backup_dir}/best_model.h5",
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                save_weights_only=False,  # 최고 모델은 전체 저장
                verbose=1
            )
        )
        
        # 백업 정리 콜백
        callbacks.append(
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._cleanup_old_backups(epoch)
            )
        )
        
        return callbacks
    
    def _cleanup_old_backups(self, current_epoch: int):
        """오래된 백업 파일 정리"""
        import os
        import glob
        
        if current_epoch % 10 == 0:  # 10 에폭마다 정리
            # 최근 3개 체크포인트만 유지
            checkpoint_files = sorted(glob.glob(f"{self.backup_dir}/checkpoint_*.h5"))
            
            if len(checkpoint_files) > 3:
                for old_file in checkpoint_files[:-3]:
                    try:
                        os.remove(old_file)
                        logger.debug(f"Removed old checkpoint: {old_file}")
                    except Exception as e:
                        logger.warning(f"Could not remove {old_file}: {e}")


# 통합된 GPU 최적화 학습 함수
async def train_with_gpu_optimization(model_builder, X_train, y_train, X_val, y_val,
                                    batch_size: int = 32, epochs: int = 100):
    """GPU 최적화가 적용된 학습 함수"""
    
    # GPU 설정
    gpu_manager = GPUMemoryManager()
    gpu_available = gpu_manager.setup_gpu_configuration()
    
    # Mixed Precision 설정
    mp_stabilizer = MixedPrecisionStabilizer(enable_mixed_precision=gpu_available)
    mp_stabilizer.setup_mixed_precision()
    
    # 백업 전략
    backup_strategy = OptimizedBackupStrategy()
    
    # 모델 생성
    model = mp_stabilizer.create_stable_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
    model = mp_stabilizer.compile_with_mixed_precision(model)
    
    # 콜백 설정
    callbacks = backup_strategy.create_backup_callbacks()
    callbacks.extend([
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=15,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ])
    
    # OOM 방지를 위한 학습 루프
    current_batch_size = batch_size
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # TensorFlow 데이터셋 생성 (메모리 효율적)
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.batch(current_batch_size).prefetch(tf.data.AUTOTUNE)
            
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_dataset = val_dataset.batch(current_batch_size).prefetch(tf.data.AUTOTUNE)
            
            # 학습 실행
            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=callbacks,
                verbose=1
            )
            
            return model, history
            
        except tf.errors.ResourceExhaustedError as e:
            logger.error(f"OOM Error: {e}")
            current_batch_size = gpu_manager.handle_oom(current_batch_size)
            
            # 모델 재생성
            tf.keras.backend.clear_session()
            model = mp_stabilizer.create_stable_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2])
            )
            model = mp_stabilizer.compile_with_mixed_precision(model)
            
    raise RuntimeError("Failed to train model after multiple OOM attempts")