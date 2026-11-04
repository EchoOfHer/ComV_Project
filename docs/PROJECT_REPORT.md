# Project Report: Food Image Processing Toolbox and Classification

**Course / Project Title**: Smart Image Processing Toolbox for Real-World Applications  
**Selected Topic (#17)**: Food Image Classification Model Using TensorFlow  
**Workspace**: `D:\ComV_Project`  
**Development Approach**: Minimum Viable Project (MVP) Architecture

---

## 1. Project Objective and Scope

### 1.1 Objective
The primary goal of this project is to develop a MATLAB-based Image Processing Toolbox integrated with a pre-trained TensorFlow Deep Learning model. The system addresses practical challenges in food photography and dietary analysis, including contrast degradation under poor dining lighting, sensor noise, food and dish boundary extraction, color-based ingredient segmentation, and standardized image resizing for deep learning inference.

### 1.2 Scope and Definition of Done (DoD)
- **Scope Included**:
  - Implementation of five core image processing techniques covering Image Enhancement, Spatial Filtering, Edge Detection, Color Segmentation, and Geometric Transformation/Interpolation.
  - Interactive MATLAB code-based Graphical User Interface (`uifigure`) allowing image upload, technique selection, parameter adjustment, side-by-side visualization, and one-click food classification.
  - Pre-trained TensorFlow classification model (`EfficientNetB0-Food101`) integrated with MATLAB through a decoupled subprocess JSON bridge.
  - Single-column vertical layout for classification candidate predictions sorted by score.
  - Verification using real-world food test images (`img1.jpg` through `img15.jpg` in `data/test_images/`).
- **Scope Excluded (Per Specification)**:
  - Training custom deep learning models from scratch.
  - Accuracy or confidence benchmarking dashboards across alternative models.
  - Experimental or non-required image processing operations.

---

## 2. System Architecture and Workflow

```text
[User / Camera Image] (JPG/PNG/BMP)
         |
         v
[MATLAB GUI: FoodToolboxApp]
   |-- Display Original Image
   |
   |-- [Selectable Image Processing Techniques] (Interactive, Independent Selection)
   |    |-- 1. Histogram Equalization (HSV Value / YCbCr / Gray) -> Contrast Enhanced Image
   |    |-- 2. Median Filter (3x3 / 5x5 / 7x7) -------------------> Noise-Reduced Image
   |    |-- 3. Canny Edge Detection (Sigma Selection) ------------> Binary Edge Map
   |    |-- 4. HSV Color Segmentation (Red / Green / Blue) ------> Isolated Foreground
   |    `-- 5. Bilinear Interpolation (Target: 224x224) ----------> Normalized ML Dimensions
   |
   `-- [TensorFlow Classification Bridge: classify_food.m]
        | (Input: Original Image RGB; Subprocess Execution)
        v
   [Python 3.12 Engine: predict_food.py]
        |-- EfficientNetB0-Food101 (Fine-tuned on 101 Food Classes)
        |-- Bilinear Preprocessing (224x224 RGB)
        `-- Softmax Probabilities and Candidate Sorting
        | (Structured JSON Output)
        v
   [GUI Display: Food Classification Results]
        |-- Predicted Food: Top-1 Label with Confidence Score
        `-- Candidates List: Vertical Single-Column Display
```

---

## 3. Mathematical and Algorithmic Principles of the Five Techniques

### 3.1 Histogram Equalization in HSV Color Space (`enhance_histogram_eq.m`)
Food images captured in low-light dining environments often exhibit low contrast. Applying standard histogram equalization directly to individual Red, Green, and Blue channels alters color ratios and results in unnatural color shifts.  
To preserve natural food color fidelity, the image is converted to the **HSV (Hue-Saturation-Value)** color space:
$$V_{\text{eq}} = T(V) = (L - 1) \sum_{k=0}^{r_k} p_r(r_k)$$
Only the luminance/intensity channel ($V$) is equalized, while chrominance channels ($H$ and $S$) remain untouched. The result is transformed back to RGB.

### 3.2 Median Filtering for Impulse Noise Reduction (`filter_median.m`)
High ISO settings in kitchen photography introduce salt-and-pepper and sensor noise. Linear smoothing filters (such as box or Gaussian filters) blur sharp food boundaries.  
The **Median Filter** replaces each pixel with the median value in its $m \times n$ neighborhood:
$$y[i, j] = \text{median}\big(\{x[i+r, j+c] \mid (r, c) \in W\}\big)$$
This removes isolated noisy pixels while strictly maintaining sharp edges along dish rims and food contours.

### 3.3 Canny Edge Detection (`edge_detect_canny.m`)
The Canny edge detector provides boundary localization using:
1. **Gaussian Smoothing**: Convolves with a 2D Gaussian filter ($\sigma$).
2. **Gradient Magnitude and Direction**:
   $$M(x, y) = \sqrt{G_x^2 + G_y^2}, \quad \theta(x, y) = \arctan\left(\frac{G_y}{G_x}\right)$$
3. **Non-Maximum Suppression (NMS)**: Thinning edges along the gradient direction.
4. **Hysteresis Thresholding**: Dual thresholds ($T_{\text{low}}, T_{\text{high}}$) to trace continuous boundaries without spurious fragments.

### 3.4 HSV Color-Based Food Segmentation (`segment_hsv_color.m`)
In RGB space, illumination changes alter all three color components. In HSV space, Hue represents the intrinsic color independent of lighting intensity.  
A binary food ingredient mask $M(x, y)$ is generated by:
$$M(x, y) = (H_{\min} \le H(x, y) \le H_{\max}) \land (S_{\min} \le S(x, y) \le S_{\max}) \land (V_{\min} \le V(x, y) \le V_{\max})$$
Color selection ranges for primary RGB channels:
- **Red**: $H \in [0.94, 0.05]$ with circular wrap-around ($S \in [0.25, 1.00], V \in [0.20, 1.00]$)
- **Green**: $H \in [0.20, 0.45]$ ($S \in [0.20, 1.00], V \in [0.15, 1.00]$)
- **Blue**: $H \in [0.55, 0.75]$ ($S \in [0.20, 1.00], V \in [0.15, 1.00]$)

### 3.5 Bilinear Interpolation for Resizing (`transform_bilinear_resize.m`)
Deep learning classification models require standardized input dimensions ($224 \times 224$ pixels).  
**Bilinear Interpolation** uses reverse mapping to compute continuous coordinates $(x', y')$ in the source image, taking a weighted average of the four nearest neighbors:
$$f(x', y') \approx (1 - \Delta x)(1 - \Delta y) f(Q_{11}) + \Delta x (1 - \Delta y) f(Q_{21}) + (1 - \Delta x) \Delta y f(Q_{12}) + \Delta x \Delta y f(Q_{22})$$
This yields smooth gradients required for convolutional feature extraction.

---

## 4. TensorFlow Model Integration and Interoperability

### 4.1 Model Selection: EfficientNetB0-Food101
The classification model is **`ml-debi/EfficientNetB0-Food101`** (Hugging Face):
- **Domain-Specific Fine-Tuning**: Pre-trained and fine-tuned on the **Food-101** dataset (101 dedicated food dish categories), providing superior domain accuracy compared to generic 1000-class ImageNet models.
- **Model Efficiency**: EfficientNet-B0 utilizes compound scaling (balancing depth, width, and resolution with MBConv blocks) with a compact size of $\approx 17.1 \text{ MB}$ (`tf_model.h5`).
- **Input Specifications**: Standard $224 \times 224 \times 3$ RGB tensor, matching the Bilinear scaling toolbox output.

### 4.2 MATLAB-Python Bridge Architecture
To eliminate Python C-API version conflicts within MATLAB, the integration utilizes a decoupled command-line subprocess:
1. MATLAB's `classify_food.m` calls the standalone script `predict_food.py`.
2. Python executes model inference using `tf_keras` and prints an isolated JSON payload to stdout.
3. MATLAB's `classify_food.m` isolates the JSON boundary via character indexing and parses the structured response using `jsondecode`.
4. Classification uses the original RGB image as input to ensure natural color features are preserved for the neural network.

---

## 5. Experimental Test Results

The toolbox and pre-trained classification model were evaluated against real food test images located in `data/test_images/`:

| Image File | Actual Food Item | Top Predicted Class | Confidence (%) | Second Candidate |
|---|---|---|---|---|
| `img1.jpg` | Pad Thai / Noodles | **Pad Thai** | **96.04%** | Fried Rice (1.52%) |
| `img2.jpg` | Cheese / Bread Platter | **Cheese Plate** | **90.23%** | Pancakes (7.41%) |
| `img4.jpg` | Sushi Platter | **Sushi** | **99.98%** | Sashimi (0.01%) |
| `img5.jpg` | Glazed Chicken Wings | **Chicken Wings** | **98.23%** | Gnocchi (0.72%) |
| `img6.jpg` | Japanese Noodle Soup | **Ramen** | **100.00%** | Spaghetti Bolognese (0.00%) |
| `img7.jpg` | Berry Dessert | **Cheesecake** | **99.60%** | Strawberry Shortcake (0.24%) |
| `img8.jpg` | Beef Burger | **Hamburger** | **99.46%** | Pulled Pork Sandwich (0.44%) |
| `img9.jpg` | Mexican Taco | **Tacos** | **99.90%** | Club Sandwich (0.05%) |
| `img10.jpg` | Salmon Fillet | **Grilled Salmon** | **99.83%** | Steak (0.09%) |

---

## 6. Conclusion
The Food Image Processing Toolbox successfully meets all course rubric requirements and completes all six MVP phases. The system bridges classical computer vision techniques with modern deep learning inference inside a clean, academic MATLAB code-based GUI.
