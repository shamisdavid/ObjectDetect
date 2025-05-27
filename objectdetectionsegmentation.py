from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO("yolov8l-seg.pt")  # Or any other version you prefer

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
            color = (0, 255, 0)  # Default green color; can be randomized later

            # Apply the color to the mask
            for c in range(3):
                colored_mask[:, :, c] = (mask_resized / 255) * color[c]

            # Blend the color mask with the original frame using transparency
            alpha = 0.4  # Adjust transparency (0 = fully transparent, 1 = fully opaque)
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

            # Find the center of the mask to place the label closer to the mask
            M = cv2.moments(mask_resized)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = frame.shape[1] // 2, frame.shape[0] // 2  # fallback to center

            # Get the label from the results
            label = results.names[results.boxes.cls[i].item()]

            # Set label font and size
            font_scale = 1.0
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (255, 255, 255)  # white text
            outline_color = (30, 30, 30)  # dark outline

            # Text w/ outline for visibility
            cv2.putText(overlay, label, (cX - 30, cY), font, font_scale, outline_color, 4, cv2.LINE_AA)
            cv2.putText(overlay, label, (cX - 30, cY), font, font_scale, text_color, 2, cv2.LINE_AA)

    # Show result w/ transparent mask
    cv2.imshow("Masked Segmentation", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
