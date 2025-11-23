#!/usr/bin/env python3
import os
import torch

class Config:
    
    # ПУТИ К ДАННЫМ
    DATA_PATH = 'data'
    DATASET_PATH = 'dataset'
    MODELS_PATH = 'models'
    RESULTS_PATH = 'results'
    
    # Исходные данные
    TRAIN_IMAGES_DIR = os.path.join(DATA_PATH, 'train')
    TRAIN_LABELS_DIR = os.path.join(DATA_PATH, 'label')
    TRAIN_DF = os.path.join(DATA_PATH, 'train_df.csv')
    SAMPLE_SUBMISSION = os.path.join(DATA_PATH, 'sample_submission.csv')
    
    # YOLO датасет
    DATASET_YAML = os.path.join(DATASET_PATH, 'dataset.yaml')
    TRAIN_IMAGES_YOLO = os.path.join(DATASET_PATH, 'images/train')
    VAL_IMAGES_YOLO = os.path.join(DATASET_PATH, 'images/val')
    TRAIN_LABELS_YOLO = os.path.join(DATASET_PATH, 'labels/train')
    VAL_LABELS_YOLO = os.path.join(DATASET_PATH, 'labels/val')
    
    # Модели и результаты
    BEST_MODEL = os.path.join(MODELS_PATH, 'occupancy_detector/weights/best.pt')
    LAST_MODEL = os.path.join(MODELS_PATH, 'occupancy_detector/weights/last.pt')
    TRAINING_RESULTS_DIR = os.path.join(RESULTS_PATH, 'training')
    PREDICTIONS_DIR = os.path.join(RESULTS_PATH, 'predictions')
    SUBMISSIONS_DIR = os.path.join(RESULTS_PATH, 'submissions')
    
    # НАСТРОЙКИ МОДЕЛИ
    MODEL_NAME = 'yolov8m.pt'  
    IMG_SIZE = 960
    BATCH_SIZE = 4  # Уменьшить если не хватает памяти
    EPOCHS = 200    
    LEARNING_RATE = 0.0001 
    PATIENCE = 30   
    
    # НАСТРОЙКИ ОБУЧЕНИЯ
    DEVICE = '0' if torch.cuda.is_available() else 'cpu'
    WORKERS = 4
    AUGMENT = True
    
    # УЛУЧШЕННЫЕ НАСТРОЙКИ ИНФЕРЕНСА
    CONFIDENCE_THRESHOLD = 0.33 
    IOU_THRESHOLD = 0.43
    
    # НАСТРОЙКИ ВИЗУАЛИЗАЦИИ
    SHOW_LABELS = True 
    SHOW_CONFIDENCE = True 
    BOX_COLOR = (0, 255, 0)
    BOX_THICKNESS = 2
    FONT_SIZE = 0.8
    FONT_THICKNESS = 1
    
    @classmethod
    def get_training_params(cls):
        return {
            'data': cls.DATASET_YAML,
            'epochs': cls.EPOCHS,
            'imgsz': cls.IMG_SIZE,
            'batch': cls.BATCH_SIZE,
            'lr0': cls.LEARNING_RATE,
            'patience': cls.PATIENCE,
            'device': cls.DEVICE,
            'workers': cls.WORKERS,
            'augment': cls.AUGMENT,
            'save': True,
            'exist_ok': True,
            'project': cls.MODELS_PATH,
            'name': 'occupancy_detector',
            'verbose': True,
            # ДОБАВЛЯЕМ РАННЮЮ ОСТАНОВКУ
            'early_stopping': True,
            'early_stopping_patience': 50,
        }
    
    @classmethod
    def get_inference_params(cls):
        return {
            'conf': cls.CONFIDENCE_THRESHOLD,
            'iou': cls.IOU_THRESHOLD,
            'device': cls.DEVICE,
            'verbose': False,
        }

    @classmethod
    def get_visualization_params(cls):
        return {
            'boxes': True,
            'labels': cls.SHOW_LABELS,
            'conf': cls.SHOW_CONFIDENCE,
            'line_width': cls.BOX_THICKNESS,
        }

    @classmethod
    def setup_directories(cls):
        directories = [
            cls.MODELS_PATH,
            cls.TRAINING_RESULTS_DIR,
            cls.PREDICTIONS_DIR,
            cls.SUBMISSIONS_DIR,
            cls.DATASET_PATH,
            os.path.dirname(cls.TRAIN_IMAGES_YOLO),
            os.path.dirname(cls.VAL_IMAGES_YOLO),
            os.path.dirname(cls.TRAIN_LABELS_YOLO),
            os.path.dirname(cls.VAL_LABELS_YOLO),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def print_config(cls):
        print("Конфигурация")
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key}: {value}")

if __name__ == "__main__":
    Config.print_config()