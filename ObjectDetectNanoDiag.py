from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch
import psutil  # for RAM usage
import os

# Load model and check device
model = YOLO("yolov8n-seg.pt")
device = model.device
print(f"\nModel is using: {device}")

# If using CUDA, print GPU name
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("CUDA Memory Usage (MB):")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"  Reserved:  {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB\n")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    start_total = time.perf_counter()  # Start total frame timer

    ret, frame = cap.read()
    if not ret:
        break

    # Model inference timer
    start_model = time.perf_counter()
    results = model(frame)[0]
    end_model = time.perf_counter()

    overlay = frame.copy()

    # Mask processing timer
    start_mask = time.perf_counter()
    if results.masks is not None:
        for i, mask in enumerate(results.masks.data):
            mask = mask.cpu().numpy().astype("uint8") * 255
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            color = (0, 255, 0)

            for c in range(3):
                colored_mask[:, :, c] = (mask_resized / 255) * color[c]

            alpha = 0.4
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

            M = cv2.moments(mask_resized)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = frame.shape[1] // 2, frame.shape[0] // 2

            label = results.names[results.boxes.cls[i].item()]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            text_color = (255, 255, 255)
            outline_color = (30, 30, 30)

            cv2.putText(overlay, label, (cX - 30, cY), font, font_scale, outline_color, 4, cv2.LINE_AA)
            cv2.putText(overlay, label, (cX - 30, cY), font, font_scale, text_color, 2, cv2.LINE_AA)

    end_mask = time.perf_counter()

    # Show output
    cv2.imshow("Masked Segmentation", overlay)

    # System resource usage
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024**2  # in MB

    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated(0) / 1024**2
        gpu_reserved = torch.cuda.memory_reserved(0) / 1024**2
    else:
        gpu_alloc = gpu_reserved = 0

    # Total frame timing
    end_total = time.perf_counter()

    # Print timing and usage info
    print(f"Model inference time: {end_model - start_model:.4f} sec")
    print(f"Mask processing time: {end_mask - start_mask:.4f} sec")
    print(f"Total frame time: {end_total - start_total:.4f} sec")
    print(f"RAM Usage: {ram_usage:.2f} MB")
    print(f"GPU Usage - Allocated: {gpu_alloc:.2f} MB | Reserved: {gpu_reserved:.2f} MB\n")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
