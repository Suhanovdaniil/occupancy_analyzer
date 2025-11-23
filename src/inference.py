#!/usr/bin/env python3
import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from config import Config

class OccupancyPredictor:
    def __init__(self):
        self.config = Config
        
        model_path = self.config.BEST_MODEL
        
        if os.path.exists(model_path):
            print(f"Загрузка модели: {model_path}")
            self.model = YOLO(model_path)
            print("✅ Модель успешно загружена")
        else:
            print(f"❌ Модель не найдена: {model_path}")
            # Пробуем загрузить последнюю модель
            if os.path.exists(self.config.LAST_MODEL):
                print(f"Загружаем последнюю модель: {self.config.LAST_MODEL}")
                self.model = YOLO(self.config.LAST_MODEL)
            else:
                raise FileNotFoundError("Не найдены модели для инференса")
    
    def predict_single_image(self, image_path, output_path=None):
        if not os.path.exists(image_path):
            print(f"❌ Изображение не найдено: {image_path}")
            return 0
        
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Не удалось загрузить изображение: {image_path}")
            return 0
        
        # Делаем предсказание с улучшенными параметрами
        results = self.model.predict(
            image_path,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            device=self.config.DEVICE,
            verbose=False,
            augment=True  # Тестовое время аугментация
        )
        
        people_count = 0
        
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # только люди
                    people_count += 1
                    
                    # Получаем координаты
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    
                    # Рисуем прямоугольник
                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        self.config.BOX_COLOR,
                        self.config.BOX_THICKNESS
                    )
                    
                    # Добавляем текст с уверенностью
                    if self.config.SHOW_CONFIDENCE:
                        
                        cv2.rectangle(
                            image,
                            (x1, y1),
                            (x1, y1),
                            self.config.BOX_COLOR,
                            -1
                        )
                        
                       
        
        # Сохраняем результат
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            print(f"✅ Результат сохранен: {output_path}")
        
        return people_count
    
    def process_test_dataset(self, test_dir, save_visualizations=True):
        print(f"🔍 Обработка тестовых данных из: {test_dir}")
        
        # Получаем список всех тестовых изображений из sample_submission.csv
        submission_template = pd.read_csv(self.config.SAMPLE_SUBMISSION)
        test_image_ids = submission_template['IMG_ID'].tolist() 
        
        predictions = []
        processed_count = 0
        
        for img_id in test_image_ids:
            img_path = os.path.join(test_dir, f"{img_id}.jpg")
            
            if os.path.exists(img_path):
                if save_visualizations:
                    output_path = os.path.join(
                        self.config.PREDICTIONS_DIR,
                        f"pred_{img_id}.jpg"
                    )
                else:
                    output_path = None
                
                # Предсказываем количество людей
                people_count = self.predict_single_image(img_path, output_path)
                predictions.append(people_count)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Обработано {processed_count}/{len(test_image_ids)} изображений")
                
                print(f"📊 {img_id}: {people_count} человек")
            else:
                print(f"⚠️ Изображение не найдено: {img_path}")
                predictions.append(0)  # По умолчанию 0
        
        print(f"✅ Обработка завершена. Обработано {processed_count} изображений")
        return predictions
    
    def create_submission_file(self, test_dir):
        output_file = os.path.join(
            self.config.SUBMISSIONS_DIR,
            'final_submission.csv'
        )
        
        print("=" * 60)
        print("📝 СОЗДАНИЕ ФАЙЛА САБМИТА")
        print("=" * 60)
        
        predictions = self.process_test_dataset(test_dir, save_visualizations=True)
        
        # Создаем файл сабмита
        submission_template = pd.read_csv(self.config.SAMPLE_SUBMISSION)
        submission = submission_template.copy()
        submission['label'] = predictions
        
        # Анализ предсказаний
        unique_counts = submission['label'].value_counts().sort_index()
        print("\n📊 СТАТИСТИКА ПРЕДСКАЗАНИЙ:")
        for count, freq in unique_counts.items():
            print(f"  {count} человек: {freq} изображений")
        
        submission.to_csv(output_file, index=False)
        
        print(f"✅ Файл сабмита создан: {output_file}")
        
        return submission

def main():
    # Проверяем и создаем директории
    Config.setup_directories()
    
    predictor = OccupancyPredictor()
    
    test_images_dir = 'data/test'
    
    if not os.path.exists(test_images_dir):
        print(f"❌ Папка с тестовыми изображениями не найдена: {test_images_dir}")
        print("Создаем папку data/test и поместите туда тестовые изображения")
        return
    
    submission = predictor.create_submission_file(test_images_dir)

if __name__ == "__main__":
    main()