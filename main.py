import os
import time
from ultralytics import YOLO
import cv2
import torch
from datetime import datetime
import tkinter as tk
import supervision as sv

screenshot_interval = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp = True

nubjarn_model = YOLO("best.pt")

box_annotator = sv.BoxCornerAnnotator()

label_annotator = sv.LabelAnnotator(text_scale=2)

window = tk.Tk()
window.title("Shushi Plate Estimation")

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

window_width = 700
window_height = 100

x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

welcome_message = "Welcome, Click 'START' to proceed."

welcome_label = tk.Label(window, text=welcome_message, font=("Arial", 14))
welcome_label.pack(side="top", pady=(window_height // 4, 0))

start_button = tk.Button(window, text="START", command=lambda: start_detection())
start_button.pack(side="top", pady=(10, 0))


def start_detection():
    welcome_label.destroy()
    start_button.destroy()
    main()


def send_alert(message, image_path):
    print(f"Alert: {message}")
    print(f"Image saved at: {image_path}")


def main():
    last_empty_time = time.time()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        nubjarn_results = nubjarn_model.predict(frame, device=device, amp=amp)
        nubjarn_result = nubjarn_results[0]

        nubjarn_detection = sv.Detections.from_ultralytics(nubjarn_result)

        plate_label = [f"{i}" for i in nubjarn_detection.data["class_name"]]

        frame = box_annotator.annotate(scene=frame, detections=nubjarn_detection)
        frame = label_annotator.annotate(
            scene=frame, detections=nubjarn_detection, labels=plate_label
        )

        class_counts = dict(
            {
                "red-dish": 0,
                "silver-dish": 0,
                "gold-dish": 0,
                "black-dish": 0,
                "other": 0,
                "Estimated Price": 0,
            }
        )
        for name in plate_label:
            if name in class_counts:
                class_counts[name] += 1
                if name == "red-dish":
                    class_counts["Estimated Price"] += 40
                elif name == "silver-dish":
                    class_counts["Estimated Price"] += 60
                elif name == "gold-dish":
                    class_counts["Estimated Price"] += 80
                elif name == "black-dish":
                    class_counts["Estimated Price"] += 120
            else:
                class_counts["other"] += 1

        y_offset = 20
        for cls, count in class_counts.items():
            text = f"{cls}: {count}"
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y_offset += 20

        cv2.imshow("Classroom Monitoring", frame)

        key = cv2.waitKey(30)
        if key == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    window.mainloop()
