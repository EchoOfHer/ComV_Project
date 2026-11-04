# Food Image Classification Toolbox: Project Guide and Documentation

**Project Title**: Food Image Classification Model Using TensorFlow  
**Course Assignment Topic**: Topic 17 ("Smart Image Processing Toolbox for Real-World Applications")  
**Project Workspace Directory**: `D:\ComV_Project`  
**Target Environment**: MATLAB (R2020b or later) + Python 3.10-3.12 (TensorFlow / tf-keras)

---

## 1. Project Overview (อธิบายภาพรวมของงาน)

โปรเจกต์นี้เป็นการพัฒนา **Smart Image Processing Toolbox บน MATLAB** ควบคู่กับการเชื่อมต่อโมเดล Deep Learning **EfficientNetB0-Food101 บน TensorFlow** เพื่อแก้ปัญหาการประมวลผลและการจำแนกประเภทภาพอาหารในสถานการณ์จริง (Real-World Food Classification)

### 1.1 Architecture & Concept
ระบบเชื่อมโยงระหว่างการประมวลผลภาพแบบดั้งเดิม (Classical Digital Image Processing) และการเรียนรู้เชิงลึก (Deep Learning Inference) โดยแบ่งออกเป็น 3 โมดูลหลัก (5 เทคนิคสำคัญ) และ 1 ส่วนต่อประสานผู้ใช้ (Interactive GUI):

1. **Module 1: Image Enhancement & Filtering**
   - **Histogram Equalization (`enhance_histogram_eq.m`)**: ปรับปรุงคอนทราสต์ของภาพอาหารที่ถ่ายในสภาวะแสงไม่เอื้ออำนวย รองรับโหมด HSV (คงค่าสีเดิม ปรับเฉพาะความสว่าง V), YCbCr และ Grayscale
   - **Median Filtering (`filter_median.m`)**: กรองสัญญาณรบกวนประเภท Salt-and-Pepper หรือสัญญาณรบกวนจากเซนเซอร์กล้อง โดยยังคงรักษาขอบและความคมชัดของชิ้นอาหารไว้
2. **Module 2: Edge Detection & Color Segmentation**
   - **Canny Edge Detection (`edge_detect_canny.m`)**: ตรวจจับเส้นขอบโครงสร้างของจานและชิ้นอาหาร โดยคืนค่าเป็น Binary Edge Map ไบนารี พร้อมพารามิเตอร์ปรับขนาด Gaussian Sigma
   - **HSV Color Segmentation (`segment_hsv_color.m`)**: แยกส่วนองค์ประกอบวัตถุอาหารตามช่วงสีในปริภูมิสี HSV รองรับสีมาตรฐาน Red, Green และ Blue
3. **Module 3: Geometric Transformation & Integration**
   - **Bilinear Scaling (`transform_bilinear_resize.m`)**: ปรับขนาดมิติของภาพให้สอดคล้องกับขนาดอินพุตของโครงข่ายประสาทเทียม ($224 \times 224 \times 3$) ด้วยการประมาณค่าเชิงเส้นสองทิศทาง (Reverse Mapping)
   - **TensorFlow Model Bridge (`classify_food.m`)**: เชื่อมต่อไปยัง Python สคริปต์เพื่อส่งภาพอาหารเข้าโมเดลจำแนกประเภทและดึงผลลัพธ์ผ่านทาง JSON Subprocess
4. **Interactive GUI (`gui/FoodToolboxApp.m`)**
   - หน้าต่างแอปพลิเคชัน MATLAB (`uifigure`) ที่มีปุ่มอัปโหลดภาพ, ตัวเลือกโมดูลและเทคนิค, ปุ่มปรับพารามิเตอร์, ปุ่ม Apply, ปุ่ม Reset, และปุ่ม Classify Food
   - แสดงผลภาพต้นฉบับและภาพผลลัพธ์แบบเคียงข้างกัน (Side-by-Side)
   - แสดงรายการผลการทำนาย (Candidates List) ในรูปแบบ **คอลัมน์เดี่ยวแนวตั้ง (Single-Column Vertical Layout)** เรียงลำดับคะแนนความเชื่อมั่นสูงสุดลงมา

---

## 2. Directory and File Structure (โครงสร้างไฟล์ทั้งหมด)

```text
D:\ComV_Project\
|-- toolbox/                                # Core MATLAB Image Processing Functions
|   |-- enhance_histogram_eq.m              # Module 1: Histogram Equalization (HSV/YCbCr/Gray)
|   |-- filter_median.m                     # Module 1: Median Spatial Filter (Noise Reduction)
|   |-- edge_detect_canny.m                 # Module 2: Canny Edge Detection (Binary Edge Map)
|   |-- segment_hsv_color.m                 # Module 2: HSV Color Segmentation (Red/Green/Blue)
|   |-- transform_bilinear_resize.m         # Module 3: Bilinear Interpolation Image Resizing
|   `-- classify_food.m                     # Phase 4: MATLAB-to-Python TensorFlow Bridge
|-- gui/                                    # Graphical User Interface
|   `-- FoodToolboxApp.m                    # Interactive MATLAB GUI Application (uifigure)
|-- model/                                  # Pre-trained Deep Learning Model
|   |-- predict_food.py                     # Standalone Python TensorFlow/Keras Inference Script
|   `-- EfficientNetB0-Food101/             # HuggingFace Pre-trained Model Assets
|       |-- config.json                     # Model Architecture & 101 Food Classes Mapping
|       `-- tf_model.h5                     # Pre-trained Model Weights (~17.1 MB)
|-- data/                                   # Datasets and Test Media
|   `-- test_images/                        # Benchmark Food Images (img1.jpg to img15.jpg)
|-- tests/                                  # Verification and Quality Assurance
|   |-- run_automated_tests.m               # End-to-End Automated Test Runner (9 Suites, 40 Tests)
|   |-- test_gui_interactive.m             # Programmatic GUI Callback & Button Verification
|   |-- test_toolbox.m                      # Standalone Core Function Verification
|   |-- AUTOMATED_TEST_REPORT.md            # Generated Markdown Test Results Report
|   |-- automated_test_results.png          # Visual Verification Plot Grid
|   `-- toolbox_verification_results.png    # Standalone Verification Figure
|-- docs/                                   # Academic Reports and Presentation Outlines
|   |-- PROJECT_REPORT.md                   # Comprehensive Academic Report
|   `-- PRESENTATION_SLIDES_OUTLINE.md      # Presentation Slides Outline & Script
|-- README.md                               # Project Readme & Quick Reference
`-- PROJECT_GUIDE.md                        # This Comprehensive Guide
```

---

## 3. Installation and Setup Guide (แนวทางการโหลดไปใช้และการติดตั้ง)

### 3.1 Prerequisites (ความต้องการของระบบ)
1. **MATLAB**: Version R2020b หรือใหม่กว่า (ทดสอบผ่านบน MATLAB R2026a)
   - ต้องการ **Image Processing Toolbox**
2. **Python**: Version 3.10 - 3.12 (ทดสอบผ่านบน Python 3.12 64-bit)
   - สิทธิ์รัน Command Prompt / Terminal เพื่อติดตั้งแพ็กเกจ

### 3.2 Step 1: Clone or Copy Project Files
คัดลอกหรือแตกไฟล์ ZIP ทั้งหมดไว้ที่ไดเรกทอรีที่ต้องการ เช่น:
```text
D:\ComV_Project
```

### 3.3 Step 2: Install Python Dependencies
เปิด Command Prompt หรือ PowerShell แล้วรันคำสั่งติดตั้งไลบรารีที่จำเป็นสำหรับโมเดล TensorFlow:
```bash
pip install tensorflow tf-keras pillow numpy
```
*หมายเหตุ*: โมเดลใช้สถาปัตยกรรม Keras 2 (`tf_model.h5`) จึงต้องมีแพ็กเกจ `tf-keras` เพื่อความเข้ากันได้ 100% บน Python 3.12

### 3.4 Step 3: Verify MATLAB Working Directory
เปิดโปรแกรม MATLAB และกำหนด Current Folder ให้ชี้ไปที่โฟลเดอร์โปรเจกต์:
```matlab
cd('D:\ComV_Project');
```

---

## 4. How to Run and Test (วิธีการเรียกใช้งาน)

### 4.1 Launching the Interactive GUI (การเปิดใช้งาน GUI)
ใน MATLAB Command Window ให้รันคำสั่ง:
```matlab
cd('D:\ComV_Project\gui');
FoodToolboxApp;
```

**ขั้นตอนการใช้งานผ่านหน้าต่าง GUI**:
1. **Upload Food Image**: คลิกปุ่ม "Upload Food Image" เพื่อเลือกภาพอาหารจากโฟลเดอร์ `data/test_images/` (เช่น `img1.jpg`)
2. **Select Technique**: เลือกเทคนิคที่ต้องการประมวลผลจากรายการดรอปดาวน์:
   - 1. Histogram Equalization (เลือกโหมด HSV, YCbCr หรือ Grayscale)
   - 2. Median Filter (เลือกขนาด Kernel 3x3, 5x5 หรือ 7x7)
   - 3. Canny Edge Detection (เลือกค่า Sigma 1.0, 1.5 หรือ 2.0)
   - 4. HSV Color Segmentation (เลือกสี Red, Green หรือ Blue)
   - 5. Bilinear Scaling (เลือกขนาด 224x224 หรือ 128x128)
3. **Apply Technique**: คลิกปุ่ม "Apply Technique" ภาพผลลัพธ์จะแสดงผลทันทีในช่องขวา (Processed Image Result)
4. **Reset**: คลิกปุ่ม "Reset to Original" เมื่อต้องการยกเลิกผลการประมวลผลและกลับไปแสดงภาพต้นฉบับ
5. **Classify Food**: คลิกปุ่ม "Classify Food (Original Image)" ระบบจะเรียกโมเดล TensorFlow และแสดงผลลัพธ์ชื่ออาหารที่ทำนายได้พร้อมคะแนนความเชื่อมั่นสูงสุด และแสดงรายการ Candidates List แนวตั้งในช่องด้านล่าง

### 4.2 Running the Full Automated Test Suite (การรันชุดทดสอบอัตโนมัติ)
เพื่อตรวจสอบความถูกต้องของทั้ง 9 Suites (40 รายการทดสอบ) รันคำสั่งใน MATLAB:
```matlab
cd('D:\ComV_Project\tests');
run_automated_tests;
```
สคริปต์จะทำการ:
- ทดสอบโครงสร้างไฟล์และความพร้อมของโมเดล
- ตรวจสอบความถูกต้องของฟังก์ชันใน Module 1, Module 2, และ Module 3
- ทดสอบการทำงานร่วมกับโมเดล EfficientNetB0-Food101
- ทดสอบการกดปุ่มทุกปุ่มใน GUI (Upload, Apply, Reset, Classify)
- ตรวจสอบ Zero-Emoji Compliance
- สร้างรายงานสรุปผลที่ `tests/AUTOMATED_TEST_REPORT.md` และบันทึกภาพผลลัพธ์ที่ `tests/automated_test_results.png`

### 4.3 Running Standalone Interactive GUI Verification
เพื่อทดสอบการทำงานของปุ่ม GUI โดยละเอียดแบบแยกเฉพาะส่วน:
```matlab
cd('D:\ComV_Project\tests');
test_gui_interactive;
```

---

## 5. Scope Analysis and Clarification for Instructor (การวิเคราะห์ Scope และประเด็นที่ต้องปรึกษาอาจารย์)

### 5.1 สิ่งที่เอกสารระบุเทียบกับสิ่งที่นำมาใช้จริง

ในเอกสารคำสั่งของอาจารย์ (`Project.docx_(1).pdf`) ได้แบ่งโครงสร้างออกเป็น 3 โมดูลสำหรับนักศึกษา 3 คน:
- **Student 1**: Histogram Processing, Spatial Filtering, และ **Frequency Domain Filtering (Fourier)**
- **Student 2**: Edge Detection, Segmentation (Thresholding/Color), และ **Morphological Operations**
- **Student 3**: Geometric Transformations (Affine, Warping, Lens Distortion), Interpolation, และ **Image Registration**

ในโปรเจกต์ปัจจุบัน เราได้เลือก **Core Representative Techniques** จำนวน 5 เทคนิค (Histogram Equalization, Median Filter, Canny, HSV Color Segmentation, Bilinear Resize) ที่ตอบโจทย์การแก้ปัญหาภาพอาหารจริงและเชื่อมต่อเข้ากับโมเดลจำแนกประเภทภาพอาหารได้อย่างมีประสิทธิภาพสูงสุด

---

### 5.2 ประเด็นคำถามที่ควรนำไปสอบถามอาจารย์ให้แน่ชัด

เพื่อความชัดเจนก่อนการส่งงาน ขอแนะนำให้สอบถามอาจารย์ผู้สอนใน 3 ประเด็นสำคัญดังนี้:

#### คำถามที่ 1: ขอบเขตของเทคนิคที่ต้องส่งมอบ (Technique Scope)
> *"อาจารย์ครับ ในเอกสารโครงการมีการระบุเทคนิคย่อยไว้หลายรายการ (เช่น Frequency Domain/Fourier, Morphological operations, Affine transformation, Image registration) สำหรับหัวข้อ Food Image Classification จำเป็นต้องทำส่งครบทุกเทคนิคย่อยตามเอกสาร หรือสามารถคัดเลือกเฉพาะเทคนิคตัวแทนที่จำเป็นและสอดคล้องกับปัญหาของภาพอาหารจริง (เช่น การปรับคอนทราสต์, กรองสัญญาณรบกวน, ตรวจจับขอบ, แยกสีส่วนประกอบอาหาร, และการ Resize เข้าโมเดล) ได้ครับ?"*

#### คำถามที่ 2: เกณฑ์การประเมินสำหรับโครงงานเดี่ยวเทียบกับกลุ่ม 3 คน (Solo vs Group Evaluation)
> *"เนื่องจากโครงสร้างในเอกสารถูกแบ่งบทบาทสำหรับนักศึกษา 3 คน (Student 1, 2, 3) ในกรณีที่ทำคนเดียว เกณฑ์การประเมินจะพิจารณาจากความสมบูรณ์และการทำงานได้จริงของระบบ (End-to-End MVP: Image Processing + GUI + Classification Model) เป็นหลัก หรือจำเป็นต้องมีปริมาณเทคนิคครบเท่ากับกลุ่ม 3 คนครับ?"*

#### คำถามที่ 3: กรณีที่จำเป็นต้องมี Frequency Domain Filtering (Fourier Filter)
> *"หากใน Module 1 อาจารย์ต้องการให้มี Frequency Domain Filtering ด้วย ผมสามารถเพิ่มฟังก์ชัน Fourier Low-pass / High-pass Filter เข้าไปใน Toolbox ได้ครับ แต่อยากสอบถามว่าในขั้นตอนการส่งภาพเข้าโมเดลจำแนกประเภทอาหาร (Food Classification) อนุญาตให้ใช้ภาพ Spatial Domain ปกติ เพื่อไม่ให้สูญเสียรายละเอียดเชิงพื้นที่ของภาพอาหารได้ใช่ไหมครับ?"*
