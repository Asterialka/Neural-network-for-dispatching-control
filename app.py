import os
import cv2
import time
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import scrolledtext
import imageio

class RailwayDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Railway Detection App")

        self.label = tk.Label(master, text="Выберите фото и видео для обработки")
        self.label.pack()

        self.upload_photo_button = tk.Button(master, text="Загрузить фото", command=self.load_photo)
        self.upload_photo_button.pack()

        self.upload_video_button = tk.Button(master, text="Загрузить видео", command=self.load_video)
        self.upload_video_button.pack()

        self.process_button = tk.Button(master, text="Обработать", command=self.process_files)
        self.process_button.pack()

        self.text_area = scrolledtext.ScrolledText(master, width=60, height=20)
        self.text_area.pack()

        self.model = YOLO(r'C:\Users\plotn\Desktop\app\best.pt')  # Убедитесь, что модель находится в рабочей директории
        
        self.photo_path = None
        self.video_path = None

        # Путь для сохранения обработанных файлов
        self.output_dir = r'C:\Users\plotn\Desktop\app\output'
        os.makedirs(self.output_dir, exist_ok=True)  # Создание директории, если она не существует

    def load_photo(self):
        self.photo_path = filedialog.askopenfilename(title="Выберите фото", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if self.photo_path:
            messagebox.showinfo("Выбранное фото", f"Вы выбрали: {self.photo_path}")

    def load_video(self):
        self.video_path = filedialog.askopenfilename(title="Выберите видео", filetypes=[("Video Files", "*.mp4;*.avi")])
        if self.video_path:
            messagebox.showinfo("Выбранное видео", f"Вы выбрали: {self.video_path}")

    def process_files(self):
        if self.photo_path:
            self.process_image(self.photo_path)
        else:
            messagebox.showwarning("Ошибка", "Пожалуйста, загрузите фото.")

        if self.video_path:
            self.process_video(self.video_path)
        else:
            messagebox.showwarning("Ошибка", "Пожалуйста, загрузите видео.")

    def process_image(self, img_path):
        results = self.model(img_path)

        # Отключим показ классов, связанных с детекцией путей
        target_classes = ['railway-second', 'railway-third', 'railway-fourth', 'railway-fifth']
        target_class_ids = [cls_id for cls_id, name in results[0].names.items() if name in target_classes]
        
        # Фильтрация боксов, исключая целевые классы
        filtered_boxes = [box for box in results[0].boxes if int(box.cls) not in target_class_ids]
        results[0].boxes = filtered_boxes
        
        # Получаем классы и подсчитываем их (после фильтрации)
        labels = [results[0].names[int(cls)] for box in filtered_boxes for cls in [box.cls]]
        counter = {label: labels.count(label) for label in set(labels)}

        # Выводим результаты в текстовое поле
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, f"Обнаруженные метки: {' '.join([f'{k}: {v}' for k, v in counter.items()])}\n")

        # Определение классов для предупреждений
        alert_classes = ['loader', 'person']
        alerts = [cls for cls in alert_classes if cls in labels]

        if alerts:
            self.text_area.insert(tk.END, f"\nВнимание! Обнаружены объекты следующих классов: {', '.join(alerts)}\n")
        else:
            self.text_area.insert(tk.END, "\nНа изображении нет объектов из указанных классов.\n")

        # Сохранение обработанного изображения
        results_img = results[0].plot()
        img_output_path = os.path.join(self.output_dir, "processed_image.png")
        Image.fromarray(results_img).save(img_output_path)
        self.text_area.insert(tk.END, f"Обработанное изображение сохранено как '{img_output_path}'.\n")

        # Логирование
        self.log_results(counter)


    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть видео.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        log_interval = 5  # Логирование каждую секунду
        last_log_time = time.time()  # Начальное время для отслеживания интервала

        log_file_path = os.path.join(self.output_dir, "railway_logs.txt")
        with open(log_file_path, 'w') as log_file:
            log_file.write("Time | Free Paths | Detected Classes | Alerts\n")
            log_file.write("-------------------------------------------------\n")

        processed_frames = []  # Список для сохранения обработанных кадров
        target_paths = {'railway-second', 'railway-third', 'railway-fourth', 'railway-fifth'}
        alert_classes = ['loader', 'person']

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            detected_paths = set()
            detected_classes = set()
            alert_detected = []

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    class_name = result.names[cls_id]

                    if class_name in target_paths:
                        detected_paths.add(class_name)
                        continue
                    detected_classes.add(class_name)

                    if class_name in alert_classes:
                        alert_detected.append(class_name)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            free_paths_count = len(target_paths - detected_paths)
            free_paths_list = list(target_paths - detected_paths)
            alert_message = "Warning! Objects detected: " + ", ".join(alert_detected) if alert_detected else ""

            overlay = frame.copy()
            panel_height = 80
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            info_text = f"Free Paths: {free_paths_count} ({', '.join(free_paths_list)})"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if alert_message:
                cv2.putText(frame, alert_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Добавляем текущий обработанный кадр в список для GIF
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frames.append(frame_rgb)

            # Логируем данные, если прошел интервал log_interval
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_entry = f"{timestamp} | {free_paths_count} | {', '.join(detected_classes)}"
                log_entry += f" | ALERT: {', '.join(alert_detected)}\n" if alert_detected else " | No alerts\n"
                with open(log_file_path, 'a') as log_file:
                    log_file.write(log_entry)
                last_log_time = current_time

        cap.release()  # Закрываем объект VideoCapture

        # Сохраняем все кадры как GIF
        gif_output_path = os.path.join(self.output_dir, 'processed_railways_video.gif')
        imageio.mimsave(gif_output_path, processed_frames, duration=0.1)
        self.text_area.insert(tk.END, f"GIF успешно создан и сохранен как '{gif_output_path}'.\n")

    def create_gif(self, frames):
        if frames:
            # Преобразование всех кадров в формат PIL Image
            pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
            gif_output_path = os.path.join(self.output_dir, 'processed_railways_video.gif')  # Путь для сохранения GIF
            pil_frames[0].save(gif_output_path, save_all=True, append_images=pil_frames[1:], duration=100, loop=0)
            self.text_area.insert(tk.END, f"GIF успешно создан и сохранен как '{gif_output_path}'.\n")

    def log_results(self, counter):
        log_file_path = os.path.join(self.output_dir, "railway_logs.txt")
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Обработка {datetime.now()}:\n")
            for label, count in counter.items():
                log_file.write(f"{label}: {count}\n")
            log_file.write("\n")
        self.text_area.insert(tk.END, f"Логи сохранены как '{log_file_path}'.\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = RailwayDetectionApp(root)
    root.mainloop()
