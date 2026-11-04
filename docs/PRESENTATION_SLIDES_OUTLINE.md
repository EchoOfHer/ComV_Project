# Presentation Outline: Food Image Processing Toolbox and Classification

**Time Allocation**: 5-10 Minutes  
**Project**: Topic 17 - Food Image Classification Model Using TensorFlow  
**Workspace**: `D:\ComV_Project`

---

### Slide 1: Title and Overview (1 min)
- **Title**: Food Image Processing Toolbox and TensorFlow Classification
- **Subtitle**: Computer Vision Real-World Application
- **Speaker**: Single-Developer Project
- **Key Message**: Combining classical image enhancement, segmentation, and filtering with deep learning food classification in an academic MATLAB interface.

---

### Slide 2: Real-World Motivation and Challenges (1.5 min)
- **The Food Vision Challenge**:
  - Poor or uneven dining lighting (low contrast).
  - High ISO sensor noise in kitchen environments.
  - Variable backgrounds and dish borders.
  - Requirement for fixed-size dimensions in Deep Learning architectures (224x224).
- **Solution Strategy**: A dedicated 5-technique Image Processing Toolbox serving as an analytical tool alongside a TensorFlow food classifier.

---

### Slide 3: The Five Core Image Processing Techniques (2.5 min)
- **1. Contrast Enhancement (HSV Histogram Equalization)**:
  - Preserves natural food color fidelity by equalizing only the Value (V) channel while keeping Hue and Saturation intact.
- **2. Noise Reduction (Median Filtering)**:
  - Eliminates impulse and sensor noise without blurring crisp food edges or dish boundaries.
- **3. Boundary Extraction (Canny Edge Detection)**:
  - Dual-threshold hysteresis and non-maximum suppression for sharp food contours, outputting the standard binary edge map.
- **4. Color Segmentation (HSV Color Segmentation)**:
  - Isolates color regions based on primary colors (Red, Green, Blue) using intrinsic Hue values independent of illumination.
- **5. Scaling for Machine Learning (Bilinear Interpolation)**:
  - Reverse mapping with weighted 4-neighbor interpolation to scale images to 224x224 for neural network input.

---

### Slide 4: Deep Learning Integration and System Architecture (1.5 min)
- **Model Choice**: EfficientNetB0-Food101 pre-trained and fine-tuned on the Food-101 dataset.
- **Why EfficientNetB0-Food101?**: Inverted bottleneck MBConv blocks, lightweight footprint (17.1 MB), fast CPU inference, and 101 dedicated food dish categories.
- **MATLAB-Python Bridge**: Decoupled subprocess architecture exchanging structured JSON payloads. Independent of Python C-API bindings.
- **Classification Input**: Evaluates the original RGB food image by default to ensure natural color feature representation.

---

### Slide 5: Live Demo and GUI Walkthrough (2.5 min)
- **Live Demo Flow**:
  1. Open MATLAB and launch `FoodToolboxApp`.
  2. Upload a sample food image from `data/test_images/` (e.g. `img1.jpg` or `img8.jpg`).
  3. Apply Histogram Equalization and Median Filtering to demonstrate visual enhancement.
  4. Showcase Canny Binary Edge Map and HSV Red/Green/Blue color segmentation.
  5. Click "Classify Food (Original Image)" to view instant prediction and the single-column vertical Candidates List (e.g. Hamburger 99.46%, Pulled Pork Sandwich 0.44%).

---

### Slide 6: Summary and Project Deliverables (1 min)
- **Deliverables Completed**:
  - Full MATLAB Toolbox functions with unit test suite.
  - Interactive MATLAB code-based GUI (programmatic uifigure).
  - Production-ready TensorFlow inference integration.
  - Comprehensive report and documentation.
- **Q&A**: Open the floor for questions.
