# Automated Test Execution Report

**Project Title**: Food Image Classification Model Using TensorFlow
**Workspace**: `D:\ComV_Project`
**Timestamp**: 2026-11-04 12:21:47
**Execution Time**: 115.72 seconds
**Overall Status**: PASSED (40/40 Tests Passed)

## 1. Test Summary by Module

| Suite | Description | Total Tests | Passed | Failed | Status |
|---|---|:---:|:---:|:---:|:---:|
| Suite 1 | Project Structure and File Integrity | 13 | 13 | 0 | **PASS** |
| Suite 2 | Module 1: Histogram Equalization | 5 | 5 | 0 | **PASS** |
| Suite 3 | Module 1: Median Filtering | 3 | 3 | 0 | **PASS** |
| Suite 4 | Module 2: Canny Edge Detection | 2 | 2 | 0 | **PASS** |
| Suite 5 | Module 2: HSV Color Segmentation | 3 | 3 | 0 | **PASS** |
| Suite 6 | Module 3: Bilinear Scaling | 3 | 3 | 0 | **PASS** |
| Suite 7 | Phase 4: Classification Inference Verification | 3 | 3 | 0 | **PASS** |
| Suite 8 | Phase 3: GUI Lifecycle and Interactive Callbacks | 7 | 7 | 0 | **PASS** |
| Suite 9 | Codebase Zero-Emoji Compliance Check | 1 | 1 | 0 | **PASS** |

## 2. Detailed Test Item Breakdown

| # | Test Item Name | Status | Details |
|---|---|---|---|
| 1 | File exists: enhance_histogram_eq.m | **PASS** | Verified successfully |
| 2 | File exists: filter_median.m | **PASS** | Verified successfully |
| 3 | File exists: edge_detect_canny.m | **PASS** | Verified successfully |
| 4 | File exists: segment_hsv_color.m | **PASS** | Verified successfully |
| 5 | File exists: transform_bilinear_resize.m | **PASS** | Verified successfully |
| 6 | File exists: classify_food.m | **PASS** | Verified successfully |
| 7 | File exists: FoodToolboxApp.m | **PASS** | Verified successfully |
| 8 | File exists: predict_food.py | **PASS** | Verified successfully |
| 9 | File exists: config.json | **PASS** | Verified successfully |
| 10 | File exists: tf_model.h5 | **PASS** | Verified successfully |
| 11 | File exists: PROJECT_REPORT.md | **PASS** | Verified successfully |
| 12 | File exists: PRESENTATION_SLIDES_OUTLINE.md | **PASS** | Verified successfully |
| 13 | File exists: README.md | **PASS** | Verified successfully |
| 14 | HistEq: HSV mode preserves RGB dimensions and uint8 | **PASS** | Verified successfully |
| 15 | HistEq: YCbCr mode preserves dimensions and luminance | **PASS** | Verified successfully |
| 16 | HistEq: Gray mode returns single-channel equalized image | **PASS** | Verified successfully |
| 17 | HistEq: Double precision input preservation | **PASS** | Verified successfully |
| 18 | HistEq: Negative test rejects unsupported color space | **PASS** | Verified successfully |
| 19 | Median: 3x3 kernel noise reduction on RGB | **PASS** | Verified successfully |
| 20 | Median: 5x5 scalar kernel expansion and processing | **PASS** | Verified successfully |
| 21 | Median: Negative test rejects even kernel dimensions | **PASS** | Verified successfully |
| 22 | Canny: Logical binary edge map generation | **PASS** | Verified successfully |
| 23 | Canny: Grayscale image compatibility | **PASS** | Verified successfully |
| 24 | HSV Seg: Green preset mask and foreground isolation | **PASS** | Verified successfully |
| 25 | HSV Seg: Red and Blue presets support (RGB options) | **PASS** | Verified successfully |
| 26 | HSV Seg: Negative test rejects single-channel grayscale | **PASS** | Verified successfully |
| 27 | Bilinear: 224x224 target resizing for neural network | **PASS** | Verified successfully |
| 28 | Bilinear: 128x128 compact dimension resizing | **PASS** | Verified successfully |
| 29 | Bilinear: Algorithmic reverse-mapping fidelity (MSE: 0.1380) | **PASS** | Verified successfully |
| 30 | Inference: img1.jpg predicted as "Pad Thai" (96.04%) | **PASS** | Verified successfully |
| 31 | Inference: img10.jpg predicted as "Grilled Salmon" (99.83%) | **PASS** | Verified successfully |
| 32 | Inference: Candidate list strictly sorted in descending order | **PASS** | Verified successfully |
| 33 | GUI: Launch and window instantiation | **PASS** | Verified successfully |
| 34 | GUI: Upload callback loads image and updates axes | **PASS** | Verified successfully |
| 35 | GUI: Apply button processes Histogram Equalization | **PASS** | Verified successfully |
| 36 | GUI: Apply button processes Canny Edge Detection | **PASS** | Verified successfully |
| 37 | GUI: Reset button restores original image display | **PASS** | Verified successfully |
| 38 | GUI: Classify button triggers TensorFlow inference and updates list | **PASS** | Verified successfully |
| 39 | GUI: Clean termination and figure disposal | **PASS** | Verified successfully |
| 40 | Zero-Emoji: Verified across all source files (Found: 0 emojis) | **PASS** | Verified successfully |

## 3. Classification Inference Verification on Test Images

- **`img1.jpg`**: Predicted `Pad Thai` with **96.04%** confidence.
- **`img10.jpg`**: Predicted `Grilled Salmon` with **99.83%** confidence.
- **Candidates Sorting**: Validated descending score order.

## 4. Visual Verification Output

Generated visual proof saved to `tests/automated_test_results.png`.
