#!/usr/bin/env python3
import os
import torch
from ultralytics import YOLO
import pandas as pd
from config import Config

class ModelTrainer:
    def __init__(self):
        self.config = Config
        self.model = None
        self.results = None
        
    def setup_environment(self):
        print("Настройка окружения")
        self.config.setup_directories()
    
    def load_model(self):        
        try:
            print(f"Загружаем модель: {self.config.MODEL_NAME}")
            self.model = YOLO(self.config.MODEL_NAME)
            
            print(f"Модель успешно загружена: {self.config.MODEL_NAME}")
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            return False
    
    def setup_training_parameters(self):
        print("Настройка параметров обучения")
        
        training_params = {
            'data': self.config.DATASET_YAML,
            'epochs': self.config.EPOCHS,
            'imgsz': self.config.IMG_SIZE,
            'batch': self.config.BATCH_SIZE,
            'lr0': self.config.LEARNING_RATE,
            'patience': self.config.PATIENCE,
            'device': self.config.DEVICE,
            'workers': self.config.WORKERS, 
            'save': True,
            'exist_ok': True,
            'project': self.config.MODELS_PATH,
            'name': 'occupancy_detector',
            'verbose': True,
            
            'augment': True,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0,
            'perspective': 0.001,
            'flipud': 0.01,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.0,
            
            # РЕГУЛЯРИЗАЦИЯ
            'dropout': 0.1,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'close_mosaic': 10,
            
            # ДОПОЛНИТЕЛЬНЫЕ НАСТРОЙКИ
            'optimizer': 'AdamW',
            'momentum': 0.937,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'nbs': 64,  # Nominal batch size
            
            'overlap_mask': True,
            'mask_ratio': 4,
            'single_cls': True,
        }
        return training_params
    
    def train_model(self):
        print("Запуск обучения")
        
        try:
            train_params = self.setup_training_parameters()
            
            print("Параметры обучения:")
            for key, value in train_params.items():
                print(f"  {key}: {value}")
            
            print("Начинаем обучение...")
            self.results = self.model.train(**train_params)
            
            print("Обучение завершено")
            return True
            
        except Exception as e:
            print(f"Ошибка во время обучения: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate_model(self):
        print("Оцениваем модель на валидационных данных")

        best_model_path = self.config.BEST_MODEL
        
        if os.path.exists(best_model_path):
            print(f"Загружаем лучшую модель: {best_model_path}")
            best_model = YOLO(best_model_path)
            
            print("Проводим валидацию...")
            val_results = best_model.val()

            # Детальная валидация
            for conf_thresh in [0.1, 0.25, 0.3, 0.5]:
                detailed_results = best_model.val(
                    data=self.config.DATASET_YAML,
                    conf=conf_thresh,
                    iou=0.5
                )
                print(f"Confidence {conf_thresh}: mAP50={detailed_results.box.map50:.4f}, mAP50-95={detailed_results.box.map:.4f}")
            
            return val_results
        else:
            print("Лучшая модель не найдена")
            return None
    
    def analyze_training_results(self):
        if hasattr(self, 'results') and self.results:
            print("\nАнализ результатов обучения")
            try:
                # Получаем результаты из модели
                results_df = pd.DataFrame(self.results.results_dict)
                print(f"mAP50: {results_df['metrics/mAP50(B)'].max():.4f}")
                print(f"mAP50-95: {results_df['metrics/mAP50-95(B)'].max():.4f}")
                print(f"precision: {results_df['metrics/precision(B)'].max():.4f}")
                print(f"recall: {results_df['metrics/recall(B)'].max():.4f}")
            except Exception as e:
                print(f"Не удалось проанализировать результаты: {e}")
    
    def main(self):
        print("Обучение модели")
        
        try:
            # Настройка окружения
            self.setup_environment()
            
            # Загрузка модели
            if not self.load_model():
                return False
            
            # Обучение
            success = self.train_model()
            
            if success:
                # Оценка модели
                self.evaluate_model()
                
                # Анализ результатов
                self.analyze_training_results()
                
                print("\nМодель успешно обучена")
                return True
            else:
                print("Ошибка обучения модели")
                return False
                
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.main()