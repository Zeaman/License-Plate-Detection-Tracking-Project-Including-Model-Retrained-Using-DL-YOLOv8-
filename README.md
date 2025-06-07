# 🚗 License Plate Detection & Tracking with YOLOv8 + OCR

This project performs real-time **vehicle and license plate detection** using a **custom-trained YOLOv8 model**, tracks them using **SORT**, extracts license plate text using **EasyOCR**, and outputs **annotated videos** with **CSV logs**.

---

## 📽️ Demo

<img src="det_10.JPG" width="600">

---

## 🔧 Features

- ✅ Custom YOLOv8 license plate detector  
- ✅ Vehicle detection using pretrained YOLOv8  
- ✅ SORT tracking with unique ID per vehicle  
- ✅ OCR on detected license plates  
- ✅ Bounding box interpolation using SciPy (optional)  
- ✅ Annotated video output with auto-incremented filenames  
- ✅ CSV export with ID, frame number, timestamp, and plate number  

---

## 🧠 Tech Stack

| Component    | Tool/Library         |
|--------------|----------------------|
| Object Detection | YOLOv8 (Ultralytics) |
| Tracking     | SORT (Kalman + IOU Tracker) |
| OCR          | EasyOCR              |
| Video I/O    | OpenCV               |
| Scripting    | Python               |
| Interpolation| SciPy (for smoothing) |

---

## 🗂️ Project Structure

```bash
📂 license-plate-detector/
├── anpr_retrian_yolo.pt                 # Your trained YOLOv8 plate model
├── main_final.py                        # Main entry script
├── sort/
│   ├── sort.py                          # SORT tracking module
├── utils.py
├── input/                               # Input video files
├── output/                              # Annotated output videos
├── recognized_plates.csv                # CSV log of detections
└── README.md
```
## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License

Copyright (c) 2025 Amanuel Mihiret

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
In the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS ARE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OF OTHER DEALINGS IN THE
SOFTWARE.

## Contact

Amanuel Mihiret (MSc. in Mechatronics Engineering)
zeaman44@gmail.com,
amanmih@dtu.edu.et

