import streamlit as st
import os
import sys
from PIL import Image

# Настройки страницы
st.set_page_config(page_title="Occupancy AI", page_icon="🤖", layout="wide")

def main():
    # Заголовки
    st.title("🔎 Подсчет студентов в аудитории")

    # Виджет загрузки
    uploaded_file = st.file_uploader("Загрузите фото аудитории", type=['jpg', 'png', 'jpeg'])

    # Логика
    if uploaded_file is not None:
        # Сохраняем оригинал изображения
        temp_path = "temp_image.jpg"
        image = Image.open(uploaded_file)
        image.save(temp_path)
        
        # Вывод оригинал изображения
        st.image(image, caption="Исходное фото", use_container_width=True)
        
        # Кнопка запуска процесса
        if st.button("🕵️ Найти людей"):
            
            # Добавляем путь к скриптам
            if "src" not in sys.path:
                sys.path.append("src")
            
            try:
              
                from src.inference import OccupancyPredictor
                
                with st.spinner("Пожалуйста подождите..."):
                    # Инициализируем модель
                    predictor = OccupancyPredictor()
                    
                    # Готовим путь для сохранения результата
                    output_path = os.path.join("results", "temp_result.jpg")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Запускаем подсчет
                    count = predictor.predict_single_image(temp_path, output_path=output_path)
                
                # Вывод
                st.success(f"Найдено людей: {count}")
                
                # Вывод итоговую картинку
                if os.path.exists(output_path):
                    result_image = Image.open(output_path)
                    st.image(result_image, caption=f"Результат: {count} чел.", use_container_width=True)
                else:
                    st.warning("Итоговое изображение не создано, но подсчет выполнен")
                #Защита от ошибок
            except FileNotFoundError:
                st.error("Ошибка: Файл модели не найден.")
                st.info("Убедитесь, что модель обучена и лежит в папке models/occupancy_detector/weights/")
            except Exception as e:
                st.error(f"Ошибка: {e}")
                
if __name__ == "__main__":
    main()