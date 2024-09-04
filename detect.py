from ultralytics import YOLO
import easyocr
import cv2
import matplotlib.pyplot as plt

model = YOLO('best.pt')

image = cv2.imread('Images/car4.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = model.predict(source = image_rgb)

reader = easyocr.Reader(['en'])


def calculate_average_height(ocr_result):
    heights = []
    for (bbox, _, _) in ocr_result:
        print(f"BBox: {bbox}")
        (x0, y0), (x1, y1) = bbox[0], bbox[2]
        height = abs(y1 - y0)
        heights.append(height)
    if heights:
        return sum(heights) / len(heights)
    return 0

def filter_text_by_size(ocr_result, average_height, threshold=0.5):
    filtered_result = []
    for (bbox, text, ocr_conf) in ocr_result:
        (x0, y0), (x1, y1) = bbox[0], bbox[2]
        height = abs(y1 - y0)
        print(f"Height: {height},Text = {text}, Average Height: {average_height}")
        if height >= average_height * threshold:
            filtered_result.append((bbox, text, ocr_conf))
    return filtered_result


X1 = X2 = Y1 = Y2 = 0
MaxConfi = 0
Mplate = None

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = box.conf[0].item()

    number_plate = image_rgb[y1:y2, x1:x2]

    if confidence > MaxConfi:
        MaxConfi = confidence
        X1, X2, Y1, Y2 = x1, x2, y1, y2
        Mplate = number_plate

try:
    ocr_result = reader.readtext(Mplate)
    print(f"OCR Result: {ocr_result}")
    average_height = calculate_average_height(ocr_result)
    print(f"Average Height: {average_height}")
    filtered_result = filter_text_by_size(ocr_result, average_height)
    detected_text = ' '.join([text for _, text, _ in filtered_result])

    print(f"Detected text: {detected_text}")
    print(f"Detection confidence: {MaxConfi:.2f}")

    cv2.putText(image, f'{detected_text} ({MaxConfi:.2f})', (X1, Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
except Exception as e:
    print(f"Unable to detect text: {e}")
    
finally:
    cv2.rectangle(image, (X1, Y1), (X2, Y2), (255, 0, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
