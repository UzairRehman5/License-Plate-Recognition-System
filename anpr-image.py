from ultralytics import YOLO
import cv2
import easyocr # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import os


def filter_text(region, ocr_result, region_threshold=0.45):
    rectangle_size = region.shape[0] * region.shape[1]
    
    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if (length*height / rectangle_size) > region_threshold:
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

image_path = "Media\img1.png"
image = cv2.imread(image_path)

output_dir = "Result"
os.makedirs(output_dir, exist_ok=True)

if image is None:
    print("Error loading image")
    
else:
    plate_regions = detect_license_plates(image, model)
    
    plate_counter = 0
    
    for (x1, y1, x2, y2) in plate_regions:
        plate_roi = image[y1:y2, x1:x2]
        
        print("")
        print(f"Detail related to License Plate {plate_counter + 1}")
        
        
        # Performing OCR & Applying filter_text function
        ocr_result = perform_ocr(plate_roi)
        filtered_text = filter_text(plate_roi, ocr_result)
        print(f"Filtered OCR result: {filtered_text}")
        
        
        print(f"License plate region in the image: {(x1, y1, x2, y2)}")
        
        # Saving license plate image
        plate_filename = os.path.join(output_dir, f'License_Plate_{plate_counter}.png')
        cv2.imwrite(plate_filename, plate_roi)
        
        # Displaying license plate image
        plt.imshow(cv2.cvtColor(plate_roi, cv2.COLOR_BGR2RGB))
        plt.title(f'License Plate {plate_counter+1}: {" ".join(filtered_text)}')
        plt.axis('off') 
        plt.show()  
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(image, f'Plate: {" ".join(filtered_text)}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)
        
        plate_counter += 1
    
    cv2.imshow('YOLO license plate detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("")
print(f"Total number of license plates detected: {plate_counter}")