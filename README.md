
# Limonates | ليمونات

## Overview | نظرة عامة
Limonates is a YOLO-based AI model specifically trained to detect fresh and rotten oranges.  
ليمونات هو نموذج ذكاء اصطناعي مبني على يولو ومدرب خصيصًا لاكتشاف البرتقال الطازج والفاسد.

---

## Features | الميزات
- Detects fresh oranges and rotten oranges with high precision.  
  يكتشف البرتقال الطازج والبرتقال الفاسد بدقة عالية.
- Generates annotated images and detailed detection metadata.  
  ينتج صورًا مشروحة وبيانات وصفية تفصيلية للكشف.
- Outputs detection results in both image and JSON formats.  
  يخرج نتائج الكشف بصيغتي الصور وJSON.

---

## Requirements | المتطلبات
- Python 3.8+
- Ultralytics YOLO library
- OpenCV
- NumPy

Install dependencies using the following command:  
قم بتثبيت المتطلبات باستخدام الأمر التالي:
```bash
pip install ultralytics opencv-python numpy
```

---

## Usage | الاستخدام

### Predict on an Image | الكشف عن صورة
1. Replace the `image_path` variable with the path to your image.  
   استبدل متغير `image_path` بمسار صورتك.
2. Run the script.  
   قم بتشغيل السكربت.
3. The annotated image and JSON file will be saved in the `output` directory.  
   سيتم حفظ الصورة المشروحة وملف JSON في دليل `output`.

```python
image_path = "images/mziana.jpeg"  # Example image  
output_dir = "output"  # Directory to save results
```

### Example Output | مثال على النتائج
- Annotated Image: `output/annotated_image.jpg`  
  الصورة المشروحة: `output/annotated_image.jpg`
- Detection Data: `output/detection_results.json`  
  بيانات الكشف: `output/detection_results.json`

---

## Code Walkthrough | شرح الكود

### Load Model | تحميل النموذج
```python
model = YOLO("models/Limonates.pt")  # Load Limonates model
```
```python
model = YOLO("models/Limonates.pt")  # تحميل نموذج ليمونات
```

### Perform Prediction | تنفيذ الكشف
```python
results = model(image_path, conf=0.25)  # Predict with confidence threshold
```
```python
results = model(image_path, conf=0.25)  # تنفيذ الكشف مع عتبة الثقة
```

### Save Results | حفظ النتائج
Annotated image and JSON data are saved for visualization and further analysis.  
يتم حفظ الصورة المشروحة وبيانات JSON للتصور والتحليل.

---

## License | الرخصة
This project is released under the MIT License.  
هذا المشروع مرخص بموجب رخصة MIT.

---

## Author
Developed by **ABDENNACER Elbasri** | Twitter: **@abdennacerelb** | Linkedin **@elbasri**.
