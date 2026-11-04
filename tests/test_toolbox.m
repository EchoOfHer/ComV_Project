%% TEST_TOOLBOX - Validation and Verification Test Suite
% This script tests all 5 core functions of the Food Image Processing Toolbox,
% checks parameter validation, verifies assertions, and tests classification integration.
%
% Modules Covered:
% - Module 1: Histogram Equalization & Median Filtering
% - Module 2: Canny Edge Detection & HSV Color Segmentation
% - Module 3: Bilinear Scaling & TensorFlow Classification Integration

clear; clc; close all;

% Add toolbox and model paths
proj_root = fullfile(fileparts(mfilename('fullpath')), '..');
addpath(fullfile(proj_root, 'toolbox'));

fprintf('========================================================\n');
fprintf('  Food Image Processing Toolbox - Verification Suite\n');
fprintf('========================================================\n\n');

%% 1. Load Real Food Image or Generate Synthetic Image
fprintf('[1/7] Preparing test food image...\n');
test_files = dir(fullfile(proj_root, 'data', 'test_images', '*.jpg'));
if isempty(test_files)
    test_files = dir(fullfile(proj_root, 'data', 'test_images', '*.png'));
end

if ~isempty(test_files)
    real_sample_path = fullfile(test_files(1).folder, test_files(1).name);
    img = imread(real_sample_path);
    fprintf('      Loaded real test food image: "%s" (%dx%d RGB).\n', real_sample_path, size(img, 1), size(img, 2));
else
    % Create a synthetic food plate scene (salad dish) if no external image is found
    canvas_h = 400; canvas_w = 400;
    [X, Y] = meshgrid(1:canvas_w, 1:canvas_h);

    dist_plate = sqrt((X - 200).^2 + (Y - 200).^2);
    plate_mask = dist_plate <= 170;

    img = uint8(zeros(canvas_h, canvas_w, 3));
    img(:, :, 1) = 160; img(:, :, 2) = 110; img(:, :, 3) = 75;

    for c = 1:3
        ch = img(:, :, c);
        ch(plate_mask) = 230;
        img(:, :, c) = ch;
    end

    dist_salad = sqrt((X - 170).^2 + (Y - 180).^2);
    salad_mask = dist_salad <= 70 & plate_mask;
    for c = 1:3
        ch = img(:, :, c);
        if c == 1, ch(salad_mask) = 50; end
        if c == 2, ch(salad_mask) = 180; end
        if c == 3, ch(salad_mask) = 30; end
        img(:, :, c) = ch;
    end
    fprintf('      Generated synthetic food scene (400x400 RGB).\n');
end

% Add mild noise for filtering test
noisy_img = imnoise(img, 'salt & pepper', 0.02);
fprintf('      Test image prepared successfully (%dx%d RGB).\n\n', size(img, 1), size(img, 2));

%% 2. Test Module 1: Histogram Equalization (HSV, YCbCr, Gray)
fprintf('[2/7] Testing: enhance_histogram_eq...\n');
[hist_eq_img, hist_info] = enhance_histogram_eq(img, 'hsv');
assert(isequal(size(hist_eq_img), size(img)), 'Histogram eq size mismatch');
assert(isa(hist_eq_img, class(img)), 'Histogram eq data type mismatch');
assert(strcmp(hist_info.space, 'hsv'), 'Histogram eq space metadata mismatch');

% Test YCbCr and Gray modes
[hist_ycbcr, ~] = enhance_histogram_eq(img, 'ycbcr');
assert(isequal(size(hist_ycbcr), size(img)), 'YCbCr mode size mismatch');

[hist_gray, ~] = enhance_histogram_eq(img, 'gray');
assert(size(hist_gray, 3) == 1, 'Gray mode must return single channel image');
fprintf('      Passed! HSV, YCbCr, and Gray modes verified.\n\n');

%% 3. Test Module 1: Median Filtering
fprintf('[3/7] Testing: filter_median...\n');
[median_filtered_img, median_info] = filter_median(noisy_img, [3 3]);
assert(isequal(size(median_filtered_img), size(noisy_img)), 'Median filter size mismatch');
assert(isa(median_filtered_img, class(noisy_img)), 'Median filter data type mismatch');
assert(isequal(median_info.kernel_size, [3 3]), 'Kernel size metadata mismatch');

% Negative test: Even kernel must trigger error
even_kernel_caught = false;
try
    filter_median(noisy_img, [4 4]);
catch
    even_kernel_caught = true;
end
assert(even_kernel_caught, 'Median filter must reject even kernel sizes');
fprintf('      Passed! 3x3 filter and even-kernel validation verified.\n\n');

%% 4. Test Module 2: Canny Edge Detection
fprintf('[4/7] Testing: edge_detect_canny...\n');
[edge_mask, canny_info] = edge_detect_canny(img, [], 1.2);
assert(islogical(edge_mask), 'Edge mask must be of logical type');
assert(isequal(size(edge_mask), [size(img, 1), size(img, 2)]), 'Edge mask dimensions mismatch');
assert(canny_info.edge_pixel_count > 0, 'No edge pixels detected');
fprintf('      Passed! Detected %d edge pixels (binary edge map).\n\n', canny_info.edge_pixel_count);

%% 5. Test Module 2: HSV Color Segmentation
fprintf('[5/7] Testing: segment_hsv_color...\n');
[seg_mask, segmented_food, seg_info] = segment_hsv_color(img, 'green');
assert(islogical(seg_mask), 'Segmentation mask must be logical');
assert(isequal(size(seg_mask), [size(img, 1), size(img, 2)]), 'Segmentation mask dimensions mismatch');
assert(isequal(size(segmented_food), size(img)), 'Segmented foreground dimensions mismatch');

% Negative test: Grayscale image must trigger error
gray_input_caught = false;
try
    segment_hsv_color(rgb2gray(img), 'green');
catch
    gray_input_caught = true;
end
assert(gray_input_caught, 'HSV segmentation must reject single-channel grayscale input');
fprintf('      Passed! Food foreground ratio: %.2f%%.\n\n', seg_info.foreground_ratio * 100);

%% 6. Test Module 3: Bilinear Scaling
fprintf('[6/7] Testing: transform_bilinear_resize...\n');
target_tf_dim = [224, 224];
[resized_img, resize_info] = transform_bilinear_resize(img, target_tf_dim, true);
assert(isequal(size(resized_img), [target_tf_dim(1), target_tf_dim(2), size(img, 3)]), 'Resized dimensions mismatch');
assert(isa(resized_img, class(img)), 'Resized image data type mismatch');
assert(resize_info.target_size(1) == 224 && resize_info.target_size(2) == 224, 'Target size metadata mismatch');
fprintf('      Passed! Output dimensions verified as [224 x 224 x %d].\n\n', size(img, 3));

%% 7. Test Phase 4: Classification Integration
fprintf('[7/7] Testing: classify_food integration...\n');
if ~isempty(test_files)
    sample_to_test = fullfile(test_files(1).folder, test_files(1).name);
    [pred_label, conf, details] = classify_food(sample_to_test);
    assert(~isempty(pred_label), 'Predicted food label must not be empty');
    assert(conf > 0 && conf <= 100, 'Confidence score must be between 0 and 100');
    assert(isfield(details, 'candidates'), 'Classification details must contain candidates list');
    fprintf('      Passed! Prediction: %s (%.2f%% confidence).\n', pred_label, conf);
    fprintf('      Top Candidate: %s (Rank 1: %.2f%%).\n\n', details.candidates(1).label, details.candidates(1).confidence);
else
    fprintf('      Skipping classification test (no test images found).\n\n');
end

%% 8. Visual Comparison Grid Generation
fig = figure('Name', 'Food Image Processing Toolbox Demonstration', ...
             'Position', [100, 100, 1200, 750], ...
             'Visible', 'off');

subplot(2, 3, 1);
imshow(img);
title('(A) Original Food Image');

subplot(2, 3, 2);
imshow(hist_eq_img);
title('(B) [M1] Histogram Equalization (HSV)');

subplot(2, 3, 3);
imshow(median_filtered_img);
title('(C) [M1] Median Filtering (Noise Reduction)');

subplot(2, 3, 4);
imshow(edge_mask);
title('(D) [M2] Canny Binary Edge Map');

subplot(2, 3, 5);
imshow(segmented_food);
title('(E) [M2] HSV Segmentation (Green Preset)');

subplot(2, 3, 6);
imshow(resized_img);
title(sprintf('(F) [M3] Bilinear Scaling (%dx%d for TF)', target_tf_dim(1), target_tf_dim(2)));

% Save demonstration figure to disk
out_fig_path = fullfile(proj_root, 'tests', 'toolbox_verification_results.png');
saveas(fig, out_fig_path);
close(fig);
fprintf('      Saved demonstration figure: %s\n', out_fig_path);

fprintf('========================================================\n');
fprintf('  All verification tests passed successfully!\n');
fprintf('========================================================\n');
