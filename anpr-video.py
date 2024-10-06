from ultralytics import YOLO
import cv2
import easyocr # type: ignore
import numpy as np


def filter_text(region, ocr_result, region_threshold=0.45):
    rectangle_size = region.shape[0] * region.shape[1]
    
    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if (length * height / rectangle_size) > region_threshold:
            plate.append(result[1])
            
    return plate

reader = easyocr.Reader(['en'])
def perform_ocr(image):
    ocr_result = reader.readtext(image)
    return ocr_result


def detect_license_plates(image, model, confidence_threshold=0.3):
    results = model(image)
    plate_detections = []
    
    for result in results:
        for detection in result.boxes:
            if detection.cls == 0 and detection.conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                plate_detections.append((x1, y1, x2, y2))
    
    return plate_detections


model = YOLO("Model\\anpr-best-v1.pt")

video_path = "Media\\video.mp4"  
cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  
    
    plate_regions = detect_license_plates(frame, model)
    
    for (x1, y1, x2, y2) in plate_regions:
        plate_roi = frame[y1:y2, x1:x2]
        
        # Perform OCR & apply filter_text function
        ocr_result = perform_ocr(plate_roi)
        filtered_text = filter_text(plate_roi, ocr_result)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f'Plate: {" ".join(filtered_text)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 0), 2)
    
    cv2.imshow('YOLO License Plate Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
