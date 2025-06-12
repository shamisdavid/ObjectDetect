from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO("yolov8n-seg.pt")  # Use smaller model for Surface Pro

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]  # Get the first result from model inference
    overlay = frame.copy()  # Copy frame for transparent mask blending

    if results.masks is not None:
        for i, mask in enumerate(results.masks.data):
            # Convert mask to a binary mask
            mask = mask.cpu().numpy().astype("uint8") * 255
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            # Create empty color mask
            colored_mask = np.zeros_like(frame, dtype=np.uint8)

            # Get class ID and assign consistent color
            class_id = int(results.boxes.cls[i].item())
            np.random.seed(class_id)  # Ensure same color for same class
            color = tuple(np.random.randint(0, 256, size=3).tolist())

            # Apply color to the mask
            for c in range(3):
                colored_mask[:, :, c] = (mask_resized / 255) * color[c]

            # Blend the color mask with the original frame
            alpha = 0.4
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

            # Find mask center for label placement
            M = cv2.moments(mask_resized)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = frame.shape[1] // 2, frame.shape[0] // 2

            # Get class label
            label = results.names[class_id]

            # Font settings (slightly larger + thin outline)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.85  # Slightly larger
            text_color = (255, 255, 255)
            outline_color = (30, 30, 30)

            # Draw label with thin outline
            cv2.putText(overlay, label, (cX - 30, cY), font, font_scale, outline_color, 2, cv2.LINE_AA)
            cv2.putText(overlay, label, (cX - 30, cY), font, font_scale, text_color, 1, cv2.LINE_AA)

    # Show result with transparent masks
    cv2.imshow("Masked Segmentation", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
