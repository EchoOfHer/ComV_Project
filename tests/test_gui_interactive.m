%% TEST_GUI_INTERACTIVE - Interactive GUI Verification Script
% This script programmatically exercises every button and callback in FoodToolboxApp:
%   1. Window instantiation and component verification
%   2. Image upload simulation
%   3. Apply button across all 5 techniques
%   4. Reset button (state restoration)
%   5. Classify button (TensorFlow inference and candidates list population)
%   6. Clean figure disposal

clear; clc; close all;

proj_root = fullfile(fileparts(mfilename('fullpath')), '..');
addpath(fullfile(proj_root, 'toolbox'));
addpath(fullfile(proj_root, 'gui'));

fprintf('========================================================\n');
fprintf('  Interactive GUI Component & Callback Verification\n');
fprintf('========================================================\n\n');

% Find test image
img_files = dir(fullfile(proj_root, 'data', 'test_images', '*.jpg'));
if isempty(img_files)
    error('No test images found in data/test_images/');
end
test_img_path = fullfile(img_files(1).folder, img_files(1).name);
[~, img_name, img_ext] = fileparts(test_img_path);

%% 1. Launch GUI
fprintf('[1/6] Launching FoodToolboxApp...\n');
fig = FoodToolboxApp();
assert(~isempty(fig) && isvalid(fig), 'Failed to launch FoodToolboxApp');
u = fig.UserData;
fprintf('      Passed! GUI window instantiated successfully.\n\n');

%% 2. Test Image Upload
fprintf('[2/6] Testing Upload Action with "%s%s"...\n', img_name, img_ext);
u.set_image(test_img_path);
app_st = u.get_app_data();
assert(~isempty(app_st.orig_img), 'Original image is empty after upload');
assert(strcmp(app_st.img_path, test_img_path), 'Image path mismatch after upload');
assert(contains(u.lbl_status.Text, 'Loaded'), 'Status label did not update on image load');
fprintf('      Passed! Image loaded (%dx%d RGB), status: "%s".\n\n', ...
    size(app_st.orig_img, 1), size(app_st.orig_img, 2), u.lbl_status.Text);

%% 3. Test Apply Button Across Techniques
fprintf('[3/6] Testing Apply Button Across Techniques...\n');

% Technique 1: Histogram Equalization
u.drop_technique.Value = '1. Histogram Equalization';
u.drop_technique.ValueChangedFcn(u.drop_technique, []);
u.drop_param.Value = 'HSV Space (Preserve Color)';
u.btn_apply.ButtonPushedFcn(u.btn_apply, []);
app_st = u.get_app_data();
assert(~isempty(app_st.processed_img), 'Processed image empty after HistEq');
fprintf('      Passed: Apply 1. Histogram Equalization (HSV)\n');

% Technique 2: Median Filter
u.drop_technique.Value = '2. Median Filter (Noise Reduction)';
u.drop_technique.ValueChangedFcn(u.drop_technique, []);
u.drop_param.Value = '3x3 Kernel (Standard)';
u.btn_apply.ButtonPushedFcn(u.btn_apply, []);
fprintf('      Passed: Apply 2. Median Filter (3x3)\n');

% Technique 3: Canny Edge Detection
u.drop_technique.Value = '3. Canny Edge Detection';
u.drop_technique.ValueChangedFcn(u.drop_technique, []);
u.drop_param.Value = 'Sigma 1.0';
u.btn_apply.ButtonPushedFcn(u.btn_apply, []);
app_st = u.get_app_data();
assert(islogical(app_st.processed_img), 'Canny edge map must be logical binary mask');
fprintf('      Passed: Apply 3. Canny Edge Detection (Binary Map)\n');

% Technique 4: HSV Color Segmentation
u.drop_technique.Value = '4. HSV Color Segmentation';
u.drop_technique.ValueChangedFcn(u.drop_technique, []);
u.drop_param.Value = 'Red';
u.btn_apply.ButtonPushedFcn(u.btn_apply, []);
fprintf('      Passed: Apply 4. HSV Color Segmentation (Red)\n');

% Technique 5: Bilinear Scaling
u.drop_technique.Value = '5. Bilinear Scaling (224x224)';
u.drop_technique.ValueChangedFcn(u.drop_technique, []);
u.drop_param.Value = '224 x 224 (TensorFlow Standard)';
u.btn_apply.ButtonPushedFcn(u.btn_apply, []);
app_st = u.get_app_data();
assert(isequal(size(app_st.processed_img), [224, 224, 3]), 'Bilinear scaling dimension mismatch');
fprintf('      Passed: Apply 5. Bilinear Scaling (224x224)\n\n');

%% 4. Test Reset Button
fprintf('[4/6] Testing Reset Button...\n');
u.btn_reset.ButtonPushedFcn(u.btn_reset, []);
app_st = u.get_app_data();
assert(isequal(app_st.processed_img, app_st.orig_img), 'Reset failed to restore original image');
assert(contains(u.lbl_status.Text, 'reset'), 'Status label did not confirm reset');
fprintf('      Passed! Display and internal buffer reset to original image.\n\n');

%% 5. Test Classify Button
fprintf('[5/6] Testing Classify Food Button...\n');
u.btn_classify.ButtonPushedFcn(u.btn_classify, []);
cands = u.txt_candidates.Value;
assert(~isempty(cands) && numel(cands) >= 3, 'Candidates list empty or incomplete');
assert(startsWith(cands{1}, '1.'), 'Top candidate row must begin with rank 1');
assert(contains(u.lbl_predicted_food.Text, 'Predicted Food:'), 'Predicted food label not updated');
fprintf('      Passed! Prediction Label: "%s"\n', u.lbl_predicted_food.Text);
fprintf('      Candidates List Display:\n');
for i = 1:min(5, numel(cands))
    fprintf('        %s\n', cands{i});
end
fprintf('\n');

%% 6. Clean Figure Termination
fprintf('[6/6] Closing and disposing GUI window...\n');
delete(fig);
assert(~isvalid(fig), 'Figure was not disposed cleanly');
fprintf('      Passed! GUI window disposed cleanly.\n\n');

fprintf('========================================================\n');
fprintf('  All GUI interactive tests completed successfully!\n');
fprintf('========================================================\n');
