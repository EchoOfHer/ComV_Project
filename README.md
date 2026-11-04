# Food Image Classification Toolbox (MATLAB)

Project Topic 17: **Food Image Classification Model Using TensorFlow**  
General Title: **Smart Image Processing Toolbox for Real-World Applications**  
Workspace: `D:\ComV_Project`

---

## 1. Project Directory Structure

```text
D:\ComV_Project\
|-- toolbox/                         # Core MATLAB Image Processing Functions
|   |-- enhance_histogram_eq.m       # Contrast Enhancement (HSV/YCbCr/Gray)
|   |-- filter_median.m              # Noise Reduction (Median Filter)
|   |-- edge_detect_canny.m          # Boundary Detection & Overlay (Canny)
|   |-- segment_hsv_color.m          # Color Segmentation (HSV Ranges & Presets)
|   |-- transform_bilinear_resize.m  # Bilinear Interpolation (TensorFlow input)
|   `-- classify_food.m              # MATLAB-to-TensorFlow Bridge
|-- tests/
|   |-- test_toolbox.m               # Standalone test script with visual comparisons
|   `-- toolbox_verification_results.png # Generated test output figure
|-- data/
|   `-- test_images/                 # Real food test images
|-- gui/
|   `-- FoodToolboxApp.m             # MATLAB Interactive GUI Application
|-- model/
|   |-- predict_food.py              # TensorFlow EfficientNetB0-Food101 Inference Script
|   `-- EfficientNetB0-Food101/      # Pre-trained Model Weights and 101 Food Labels
|-- docs/
|   |-- PROJECT_REPORT.md            # Comprehensive Academic Project Report
|   `-- PRESENTATION_SLIDES_OUTLINE.md # Presentation Slides & Script Outline
`-- README.md                        # Documentation and Quickstart Guide
```

---

## 2. Module Breakdown and Responsibilities

| Module | Technique | Purpose in Food Classification Context | Input and Output Interface |
|---|---|---|---|
| **Module 1**<br>Enhancement & Filtering | **Histogram Equalization** (`enhance_histogram_eq.m`) | Enhances contrast of food photos taken in poor lighting without distorting colors (equalizes V channel in HSV). | `[out_img, info] = enhance_histogram_eq(img, space)` |
| **Module 1**<br>Enhancement & Filtering | **Median Filtering** (`filter_median.m`) | Removes salt-and-pepper / sensor noise from camera photos while keeping food edges sharp. | `[out_img, info] = filter_median(img, kernel_size)` |
| **Module 2**<br>Segmentation & Edge Detection | **Canny Edge Detection** (`edge_detect_canny.m`) | Detects food boundaries and dish rims; outputs standard binary edge map. | `[edge_mask, info] = edge_detect_canny(img, thresh, sigma)` |
| **Module 2**<br>Segmentation & Edge Detection | **HSV Color Segmentation** (`segment_hsv_color.m`) | Isolates color regions in HSV space across primary colors (Red, Green, Blue). | `[seg_mask, segmented_img, info] = segment_hsv_color(img, color_preset)` |
| **Module 3**<br>Transformation & Integration | **Bilinear Scaling** (`transform_bilinear_resize.m`) | Resizes food images (e.g. to 224x224) with smooth interpolation for TensorFlow model input. | `[resized_img, info] = transform_bilinear_resize(img, target_size, use_custom)` |

---

## 3. How to Run

### Run the Verification Test Suite
In MATLAB:
```matlab
cd('D:/ComV_Project/tests');
test_toolbox
```
This script tests all 5 techniques with assertions, validates error handling, tests model classification, and saves the comparison figure to `tests/toolbox_verification_results.png`.

### Launch the Interactive GUI
In MATLAB:
```matlab
cd('D:/ComV_Project/gui');
FoodToolboxApp
```
The GUI allows you to upload food images, select and apply any of the 5 techniques with parameter controls, view original and processed images side-by-side, and classify food items with the vertical candidates list.
