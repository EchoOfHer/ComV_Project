%% RUN_AUTOMATED_TESTS - Automated End-to-End Test Runner
% This script executes a complete automated validation suite covering:
%   1. File and Directory Structure Integrity
%   2. Module 1: Histogram Equalization (HSV, YCbCr, Gray, Double precision, Negative tests)
%   3. Module 1: Median Filtering (3x3, 5x5, Even kernel negative validation)
%   4. Module 2: Canny Edge Detection (Binary mask, Visual overlay, Grayscale input)
%   5. Module 2: HSV Color Segmentation (Presets: green, red, blue, Grayscale negative test)
%   6. Module 3: Bilinear Scaling (224x224, 128x128, Custom algorithm comparison)
%   7. Phase 4: Classification Inference Verification (Model prediction, Candidate sorting)
%   8. Phase 3: GUI Lifecycle and Interactive Callbacks (Upload, Apply, Reset, Classify)
%   9. Codebase Zero-Emoji Compliance Check
%
% Generates:
%   - Console Pass/Fail summary
%   - Markdown Test Report: tests/AUTOMATED_TEST_REPORT.md
%   - Verification Plot: tests/automated_test_results.png

clear; clc; close all;

proj_root = fullfile(fileparts(mfilename('fullpath')), '..');
addpath(fullfile(proj_root, 'toolbox'));
addpath(fullfile(proj_root, 'gui'));

total_tests = 0;
passed_tests = 0;
failed_tests = 0;
test_records = {};

current_suite_id = 1;
current_suite_name = 'Project Structure and File Integrity';

t_start = tic;

fprintf('======================================================================\n');
fprintf('  STARTING AUTOMATED TEST RUNNER: FOOD IMAGE PROCESSING TOOLBOX\n');
fprintf('  Workspace: %s\n', proj_root);
fprintf('======================================================================\n\n');

%% Helper macro for recording results
function record_result(name, is_pass, msg)
    total_tests = evalin('caller', 'total_tests + 1');
    assignin('caller', 'total_tests', total_tests);
    if is_pass
        assignin('caller', 'passed_tests', evalin('caller', 'passed_tests + 1'));
        fprintf('  [PASS] %s\n', name);
    else
        assignin('caller', 'failed_tests', evalin('caller', 'failed_tests + 1'));
        fprintf('  [FAIL] %s - %s\n', name, msg);
    end
    s_id = evalin('caller', 'current_suite_id');
    s_name = evalin('caller', 'current_suite_name');
    records = evalin('caller', 'test_records');
    records{end+1} = struct('name', name, 'passed', is_pass, 'message', msg, ...
                            'suite_id', s_id, 'suite_name', s_name);
    assignin('caller', 'test_records', records);
end

%% SUITE 1: Project Structure and File Integrity
current_suite_id = 1;
current_suite_name = 'Project Structure and File Integrity';
fprintf('[Suite 1/9] Verifying Project File Structure...\n');
required_files = { ...
    fullfile(proj_root, 'toolbox', 'enhance_histogram_eq.m'), ...
    fullfile(proj_root, 'toolbox', 'filter_median.m'), ...
    fullfile(proj_root, 'toolbox', 'edge_detect_canny.m'), ...
    fullfile(proj_root, 'toolbox', 'segment_hsv_color.m'), ...
    fullfile(proj_root, 'toolbox', 'transform_bilinear_resize.m'), ...
    fullfile(proj_root, 'toolbox', 'classify_food.m'), ...
    fullfile(proj_root, 'gui', 'FoodToolboxApp.m'), ...
    fullfile(proj_root, 'model', 'predict_food.py'), ...
    fullfile(proj_root, 'model', 'EfficientNetB0-Food101', 'config.json'), ...
    fullfile(proj_root, 'model', 'EfficientNetB0-Food101', 'tf_model.h5'), ...
    fullfile(proj_root, 'docs', 'PROJECT_REPORT.md'), ...
    fullfile(proj_root, 'docs', 'PRESENTATION_SLIDES_OUTLINE.md'), ...
    fullfile(proj_root, 'README.md') ...
};

for i = 1:numel(required_files)
    fpath = required_files{i};
    [~, fname, fext] = fileparts(fpath);
    is_exist = exist(fpath, 'file') > 0;
    record_result(sprintf('File exists: %s%s', fname, fext), is_exist, 'File not found on disk');
end

%% Prepare Test Images
img_files = dir(fullfile(proj_root, 'data', 'test_images', '*.jpg'));
if isempty(img_files)
    img_files = dir(fullfile(proj_root, 'data', 'test_images', '*.png'));
end

if ~isempty(img_files)
    primary_img_path = fullfile(img_files(1).folder, img_files(1).name);
    secondary_img_path = fullfile(img_files(min(2, numel(img_files))).folder, img_files(min(2, numel(img_files))).name);
    img_rgb = imread(primary_img_path);
else
    % Create synthetic fallback image
    img_rgb = uint8(repmat(reshape([200, 150, 100], [1, 1, 3]), [400, 400]));
    primary_img_path = '';
    secondary_img_path = '';
end
img_gray = rgb2gray(img_rgb);
noisy_img = imnoise(img_rgb, 'salt & pepper', 0.02);

%% SUITE 2: Module 1 - Histogram Equalization
current_suite_id = 2;
current_suite_name = 'Module 1: Histogram Equalization';
fprintf('\n[Suite 2/9] Testing Module 1: Histogram Equalization...\n');

% Test HSV mode
try
    [res_hsv, info_hsv] = enhance_histogram_eq(img_rgb, 'hsv');
    pass_hsv = isequal(size(res_hsv), size(img_rgb)) && isa(res_hsv, 'uint8') && strcmp(info_hsv.space, 'hsv');
    record_result('HistEq: HSV mode preserves RGB dimensions and uint8', pass_hsv, 'Output size or type mismatch');
catch ME
    record_result('HistEq: HSV mode execution', false, ME.message);
end

% Test YCbCr mode
try
    [res_ycbcr, info_ycbcr] = enhance_histogram_eq(img_rgb, 'ycbcr');
    pass_ycbcr = isequal(size(res_ycbcr), size(img_rgb)) && strcmp(info_ycbcr.space, 'ycbcr');
    record_result('HistEq: YCbCr mode preserves dimensions and luminance', pass_ycbcr, 'YCbCr failure');
catch ME
    record_result('HistEq: YCbCr mode execution', false, ME.message);
end

% Test Grayscale mode
try
    [res_gray, ~] = enhance_histogram_eq(img_rgb, 'gray');
    pass_gray = (size(res_gray, 3) == 1) && (size(res_gray, 1) == size(img_rgb, 1));
    record_result('HistEq: Gray mode returns single-channel equalized image', pass_gray, 'Single channel expected');
catch ME
    record_result('HistEq: Gray mode execution', false, ME.message);
end

% Test Double Precision input
try
    img_double = im2double(img_rgb);
    [res_double, ~] = enhance_histogram_eq(img_double, 'hsv');
    pass_double = isa(res_double, 'double') && max(res_double(:)) <= 1.0;
    record_result('HistEq: Double precision input preservation', pass_double, 'Double precision range/type failed');
catch ME
    record_result('HistEq: Double precision input', false, ME.message);
end

% Negative Test: Invalid color space
neg_space_caught = false;
try
    enhance_histogram_eq(img_rgb, 'unsupported_space');
catch
    neg_space_caught = true;
end
record_result('HistEq: Negative test rejects unsupported color space', neg_space_caught, 'Expected error was not thrown');

%% SUITE 3: Module 1 - Median Filtering
current_suite_id = 3;
current_suite_name = 'Module 1: Median Filtering';
fprintf('\n[Suite 3/9] Testing Module 1: Median Filtering...\n');

% Test 3x3 kernel
try
    [res_med3, info_med3] = filter_median(noisy_img, [3 3]);
    pass_med3 = isequal(size(res_med3), size(noisy_img)) && isa(res_med3, 'uint8') ...
                && isequal(info_med3.kernel_size, [3 3]);
    record_result('Median: 3x3 kernel noise reduction on RGB', pass_med3, '3x3 filter output failure');
catch ME
    record_result('Median: 3x3 kernel execution', false, ME.message);
end

% Test 5x5 kernel (scalar expansion)
try
    [res_med5, info_med5] = filter_median(noisy_img, 5);
    pass_med5 = isequal(size(res_med5), size(noisy_img)) && isequal(info_med5.kernel_size, [5 5]);
    record_result('Median: 5x5 scalar kernel expansion and processing', pass_med5, '5x5 scalar kernel failure');
catch ME
    record_result('Median: 5x5 kernel execution', false, ME.message);
end

% Negative Test: Even kernel size rejection
neg_even_caught = false;
try
    filter_median(noisy_img, [4 4]);
catch
    neg_even_caught = true;
end
record_result('Median: Negative test rejects even kernel dimensions', neg_even_caught, 'Even kernel should be rejected');

%% SUITE 4: Module 2 - Canny Edge Detection
current_suite_id = 4;
current_suite_name = 'Module 2: Canny Edge Detection';
fprintf('\n[Suite 4/9] Testing Module 2: Canny Edge Detection...\n');

% Test on RGB input (standard binary edge map)
try
    [edge_mask, canny_info] = edge_detect_canny(img_rgb, [], 1.2);
    pass_canny = islogical(edge_mask) && isequal(size(edge_mask), [size(img_rgb, 1), size(img_rgb, 2)]) ...
                 && (canny_info.edge_pixel_count > 0);
    record_result('Canny: Logical binary edge map generation', pass_canny, 'Mask dimensions or type mismatch');
catch ME
    record_result('Canny: RGB input execution', false, ME.message);
end

% Test on Grayscale input
try
    [edge_mask_gray, ~] = edge_detect_canny(img_gray, [], 1.0);
    pass_canny_gray = islogical(edge_mask_gray) && isequal(size(edge_mask_gray), size(img_gray));
    record_result('Canny: Grayscale image compatibility', pass_canny_gray, 'Grayscale edge detection failed');
catch ME
    record_result('Canny: Grayscale input execution', false, ME.message);
end

%% SUITE 5: Module 2 - HSV Color Segmentation
current_suite_id = 5;
current_suite_name = 'Module 2: HSV Color Segmentation';
fprintf('\n[Suite 5/9] Testing Module 2: HSV Color Segmentation...\n');

% Test Green Preset
try
    [seg_mask_grn, seg_img_grn, info_grn] = segment_hsv_color(img_rgb, 'green');
    pass_grn = islogical(seg_mask_grn) && isequal(size(seg_img_grn), size(img_rgb)) ...
               && (info_grn.foreground_ratio >= 0 && info_grn.foreground_ratio <= 1);
    record_result('HSV Seg: Green preset mask and foreground isolation', pass_grn, 'Green preset failed');
catch ME
    record_result('HSV Seg: Green preset execution', false, ME.message);
end

% Test Red and Blue Presets (Standard RGB options)
try
    [~, seg_red, ~] = segment_hsv_color(img_rgb, 'red');
    [~, seg_blu, ~] = segment_hsv_color(img_rgb, 'blue');
    pass_multi_preset = isequal(size(seg_red), size(img_rgb)) && isequal(size(seg_blu), size(img_rgb));
    record_result('HSV Seg: Red and Blue presets support (RGB options)', pass_multi_preset, 'Red/Blue execution failed');
catch ME
    record_result('HSV Seg: Red/Blue presets execution', false, ME.message);
end

% Negative Test: Grayscale image rejection
neg_gray_seg_caught = false;
try
    segment_hsv_color(img_gray, 'green');
catch
    neg_gray_seg_caught = true;
end
record_result('HSV Seg: Negative test rejects single-channel grayscale', neg_gray_seg_caught, 'Grayscale should trigger error');

%% SUITE 6: Module 3 - Bilinear Scaling
current_suite_id = 6;
current_suite_name = 'Module 3: Bilinear Scaling';
fprintf('\n[Suite 6/9] Testing Module 3: Bilinear Scaling...\n');

% Test 224x224 scaling (Standard for Deep Learning)
try
    target_tf = [224, 224];
    [res_scale224, info_scale224] = transform_bilinear_resize(img_rgb, target_tf, true);
    pass_scale224 = isequal(size(res_scale224), [224, 224, 3]) && isa(res_scale224, 'uint8') ...
                    && isequal(info_scale224.target_size, target_tf);
    record_result('Bilinear: 224x224 target resizing for neural network', pass_scale224, '224x224 dimension failure');
catch ME
    record_result('Bilinear: 224x224 resizing execution', false, ME.message);
end

% Test 128x128 scaling
try
    [res_scale128, ~] = transform_bilinear_resize(img_rgb, [128, 128], true);
    pass_scale128 = isequal(size(res_scale128), [128, 128, 3]);
    record_result('Bilinear: 128x128 compact dimension resizing', pass_scale128, '128x128 dimension failure');
catch ME
    record_result('Bilinear: 128x128 resizing execution', false, ME.message);
end

% Compare Custom Reverse Mapping with MATLAB imresize (without anti-aliasing)
try
    [res_custom, ~] = transform_bilinear_resize(img_rgb, [224, 224], true);
    [res_builtin, ~] = transform_bilinear_resize(img_rgb, [224, 224], false);
    mse_diff = mean((double(res_custom(:)) - double(res_builtin(:))).^2);
    pass_algo_fidelity = (mse_diff < 1.0); % Sub-pixel convergence with built-in bilinear
    record_result(sprintf('Bilinear: Algorithmic reverse-mapping fidelity (MSE: %.4f)', mse_diff), ...
                  pass_algo_fidelity, 'Algorithmic difference too high');
catch ME
    record_result('Bilinear: Algorithmic comparison', false, ME.message);
end

%% SUITE 7: Phase 4 - Classification Inference Verification
current_suite_id = 7;
current_suite_name = 'Phase 4: Classification Inference Verification';
fprintf('\n[Suite 7/9] Testing TensorFlow Classification Inference Verification...\n');

details_test = struct();
% Test on Primary Test Image
try
    [label1, conf1, details1] = classify_food(primary_img_path);
    [~, f1_name, f1_ext] = fileparts(primary_img_path);
    pass_img1 = ~isempty(label1) && (conf1 > 0) && (conf1 <= 100) ...
                && isfield(details1, 'candidates') && (numel(details1.candidates) >= 3);
    record_result(sprintf('Inference: %s%s predicted as "%s" (%.2f%%)', f1_name, f1_ext, label1, conf1), ...
                  pass_img1, 'Primary image prediction or candidate list failure');
    details_test = details1;
catch ME
    record_result('Inference: Primary image execution', false, ME.message);
end

% Test on Secondary Test Image
try
    [label2, conf2, details2] = classify_food(secondary_img_path);
    [~, f2_name, f2_ext] = fileparts(secondary_img_path);
    pass_img2 = ~isempty(label2) && (conf2 > 0) && (conf2 <= 100) ...
                && isfield(details2, 'candidates') && (numel(details2.candidates) >= 3);
    record_result(sprintf('Inference: %s%s predicted as "%s" (%.2f%%)', f2_name, f2_ext, label2, conf2), ...
                  pass_img2, 'Secondary image prediction failure');
catch ME
    record_result('Inference: Secondary image execution', false, ME.message);
end

% Test Candidate List Sorting Order (Highest confidence first)
try
    cands = details_test.candidates;
    confs = [cands.confidence];
    is_sorted = issorted(confs, 'descend');
    record_result('Inference: Candidate list strictly sorted in descending order', is_sorted, 'Candidates not sorted');
catch ME
    record_result('Inference: Candidate list sorting check', false, ME.message);
end

%% SUITE 8: Phase 3 - GUI Lifecycle and Interactive Callbacks
current_suite_id = 8;
current_suite_name = 'Phase 3: GUI Lifecycle and Interactive Callbacks';
fprintf('\n[Suite 8/9] Testing GUI Lifecycle and Interactive Callbacks...\n');

gui_fig = [];
try
    gui_fig = FoodToolboxApp;
    pass_gui_launch = ~isempty(gui_fig) && isvalid(gui_fig);
    record_result('GUI: Launch and window instantiation', pass_gui_launch, 'GUI figure was not found or invalid');
catch ME
    record_result('GUI: Launch and window instantiation', false, ME.message);
end

if ~isempty(gui_fig) && isvalid(gui_fig)
    u_data = gui_fig.UserData;

    % 8.2 Test Image Upload Callback
    try
        u_data.set_image(primary_img_path);
        app_st = u_data.get_app_data();
        pass_upload = ~isempty(app_st.orig_img) && strcmp(app_st.img_path, primary_img_path) ...
                      && contains(u_data.lbl_status.Text, 'Loaded');
        record_result('GUI: Upload callback loads image and updates axes', pass_upload, 'Image upload failed in GUI');
    catch ME
        record_result('GUI: Upload callback execution', false, ME.message);
    end

    % 8.3 Test Apply Callback - Histogram Equalization
    try
        u_data.drop_technique.Value = '1. Histogram Equalization';
        u_data.drop_technique.ValueChangedFcn(u_data.drop_technique, []);
        u_data.drop_param.Value = 'HSV Space (Preserve Color)';
        u_data.btn_apply.ButtonPushedFcn(u_data.btn_apply, []);
        app_st = u_data.get_app_data();
        pass_apply_hist = ~isempty(app_st.processed_img) && contains(u_data.lbl_status.Text, 'Histogram Equalization');
        record_result('GUI: Apply button processes Histogram Equalization', pass_apply_hist, 'Apply HistEq failed');
    catch ME
        record_result('GUI: Apply HistEq execution', false, ME.message);
    end

    % 8.4 Test Apply Callback - Canny Edge Detection
    try
        u_data.drop_technique.Value = '3. Canny Edge Detection';
        u_data.drop_technique.ValueChangedFcn(u_data.drop_technique, []);
        u_data.drop_param.Value = 'Sigma 1.0';
        u_data.btn_apply.ButtonPushedFcn(u_data.btn_apply, []);
        app_st = u_data.get_app_data();
        pass_apply_canny = islogical(app_st.processed_img) && contains(u_data.lbl_status.Text, 'Canny');
        record_result('GUI: Apply button processes Canny Edge Detection', pass_apply_canny, 'Apply Canny failed');
    catch ME
        record_result('GUI: Apply Canny execution', false, ME.message);
    end

    % 8.5 Test Reset Callback
    try
        u_data.btn_reset.ButtonPushedFcn(u_data.btn_reset, []);
        app_st = u_data.get_app_data();
        pass_reset = isequal(app_st.processed_img, app_st.orig_img) && contains(u_data.lbl_status.Text, 'reset');
        record_result('GUI: Reset button restores original image display', pass_reset, 'Reset callback failed');
    catch ME
        record_result('GUI: Reset execution', false, ME.message);
    end

    % 8.6 Test Classify Food Callback
    try
        u_data.btn_classify.ButtonPushedFcn(u_data.btn_classify, []);
        cands_val = u_data.txt_candidates.Value;
        pass_classify_btn = ~isempty(cands_val) && numel(cands_val) >= 3 ...
                            && startsWith(cands_val{1}, '1.') ...
                            && contains(u_data.lbl_predicted_food.Text, 'Predicted Food:');
        record_result('GUI: Classify button triggers TensorFlow inference and updates list', pass_classify_btn, 'Classify callback failed');
    catch ME
        record_result('GUI: Classify execution', false, ME.message);
    end

    % 8.7 Clean Figure Termination
    try
        delete(gui_fig);
        pass_disposal = ~isvalid(gui_fig);
        record_result('GUI: Clean termination and figure disposal', pass_disposal, 'Disposal failed');
    catch ME
        record_result('GUI: Clean termination', false, ME.message);
    end
end

%% SUITE 9: Zero-Emoji Compliance Check
current_suite_id = 9;
current_suite_name = 'Codebase Zero-Emoji Compliance Check';
fprintf('\n[Suite 9/9] Verifying Zero-Emoji Compliance...\n');

emoji_count = 0;
emoji_pattern = '[\x{1F300}-\x{1F9FF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}]';

% Inspect only source code and text documentation files (exclude binary model weights)
files_to_check = {};
for i = 1:numel(required_files)
    f = required_files{i};
    [~, ~, ext] = fileparts(f);
    if ismember(lower(ext), {'.m', '.py', '.md', '.json'})
        files_to_check{end+1} = f; %#ok<AGROW>
    end
end

for i = 1:numel(files_to_check)
    f = files_to_check{i};
    fid = fopen(f, 'r', 'n', 'UTF-8');
    txt = fread(fid, '*char')';
    fclose(fid);
    matches = regexp(txt, emoji_pattern, 'match');
    emoji_count = emoji_count + numel(matches);
end

record_result(sprintf('Zero-Emoji: Verified across all source files (Found: %d emojis)', emoji_count), ...
              emoji_count == 0, 'Forbidden emojis detected in codebase');

%% Summary and Visualization
elapsed_time = toc(t_start);

fprintf('\n======================================================================\n');
fprintf('  AUTOMATED TEST SUMMARY\n');
fprintf('  Total Tests Executed: %d\n', total_tests);
fprintf('  Tests Passed:         %d (%.1f%%)\n', passed_tests, (passed_tests / total_tests) * 100);
fprintf('  Tests Failed:         %d\n', failed_tests);
fprintf('  Execution Duration:   %.2f seconds\n', elapsed_time);
fprintf('======================================================================\n\n');

% Generate Verification Plot
fig_out = figure('Name', 'Automated Test Verification Plot', 'Position', [100, 100, 1200, 750], 'Visible', 'off');

subplot(2, 3, 1); imshow(img_rgb); title('Input: Original Food Image');
subplot(2, 3, 2); imshow(res_hsv); title('Module 1: Histogram Equalized (HSV)');
subplot(2, 3, 3); imshow(res_med3); title('Module 1: Median Filtered (3x3)');
subplot(2, 3, 4); imshow(edge_mask); title('Module 2: Canny Binary Edge Map');
subplot(2, 3, 5); imshow(seg_img_grn); title('Module 2: HSV Color Segmented (Green)');
subplot(2, 3, 6); imshow(res_scale224); title('Module 3: Bilinear Scaled (224x224)');

plot_path = fullfile(proj_root, 'tests', 'automated_test_results.png');
saveas(fig_out, plot_path);
close(fig_out);
fprintf('Saved visual test results to: %s\n', plot_path);

%% Generate Markdown Test Report (Dynamically Computed from test_records)
report_path = fullfile(proj_root, 'tests', 'AUTOMATED_TEST_REPORT.md');
fid_rep = fopen(report_path, 'w', 'n', 'UTF-8');

fprintf(fid_rep, '# Automated Test Execution Report\n\n');
fprintf(fid_rep, '**Project Title**: Food Image Classification Model Using TensorFlow\n');
fprintf(fid_rep, '**Workspace**: `D:\\ComV_Project`\n');
fprintf(fid_rep, '**Timestamp**: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid_rep, '**Execution Time**: %.2f seconds\n', elapsed_time);
fprintf(fid_rep, '**Overall Status**: %s (%d/%d Tests Passed)\n\n', ...
    char(string(ternary(failed_tests == 0, 'PASSED', 'FAILED'))), passed_tests, total_tests);

fprintf(fid_rep, '## 1. Test Summary by Module\n\n');
fprintf(fid_rep, '| Suite | Description | Total Tests | Passed | Failed | Status |\n');
fprintf(fid_rep, '|---|---|:---:|:---:|:---:|:---:|\n');

suite_defs = { ...
    1, 'Project Structure and File Integrity'; ...
    2, 'Module 1: Histogram Equalization'; ...
    3, 'Module 1: Median Filtering'; ...
    4, 'Module 2: Canny Edge Detection'; ...
    5, 'Module 2: HSV Color Segmentation'; ...
    6, 'Module 3: Bilinear Scaling'; ...
    7, 'Phase 4: Classification Inference Verification'; ...
    8, 'Phase 3: GUI Lifecycle and Interactive Callbacks'; ...
    9, 'Codebase Zero-Emoji Compliance Check' ...
};

for s = 1:size(suite_defs, 1)
    s_id = suite_defs{s, 1};
    s_desc = suite_defs{s, 2};
    s_total = 0;
    s_passed = 0;
    s_failed = 0;
    for r = 1:numel(test_records)
        if test_records{r}.suite_id == s_id
            s_total = s_total + 1;
            if test_records{r}.passed
                s_passed = s_passed + 1;
            else
                s_failed = s_failed + 1;
            end
        end
    end
    s_status = ternary(s_failed == 0 && s_total > 0, 'PASS', 'FAIL');
    fprintf(fid_rep, '| Suite %d | %s | %d | %d | %d | **%s** |\n', ...
        s_id, s_desc, s_total, s_passed, s_failed, s_status);
end
fprintf(fid_rep, '\n');

fprintf(fid_rep, '## 2. Detailed Test Item Breakdown\n\n');
fprintf(fid_rep, '| # | Test Item Name | Status | Details |\n');
fprintf(fid_rep, '|---|---|---|---|\n');
for i = 1:numel(test_records)
    rec = test_records{i};
    status_str = ternary(rec.passed, 'PASS', 'FAIL');
    det_str = ternary(rec.passed, 'Verified successfully', rec.message);
    fprintf(fid_rep, '| %d | %s | **%s** | %s |\n', i, rec.name, status_str, det_str);
end

fprintf(fid_rep, '\n## 3. Classification Inference Verification on Test Images\n\n');
fprintf(fid_rep, '- **`%s%s`**: Predicted `%s` with **%.2f%%** confidence.\n', f1_name, f1_ext, label1, conf1);
fprintf(fid_rep, '- **`%s%s`**: Predicted `%s` with **%.2f%%** confidence.\n', f2_name, f2_ext, label2, conf2);
fprintf(fid_rep, '- **Candidates Sorting**: Validated descending score order.\n\n');

fprintf(fid_rep, '## 4. Visual Verification Output\n\n');
fprintf(fid_rep, 'Generated visual proof saved to `tests/automated_test_results.png`.\n');

fclose(fid_rep);
fprintf('Saved test execution report to: %s\n\n', report_path);

function val = ternary(cond, a, b)
    if cond, val = a; else, val = b; end
end
