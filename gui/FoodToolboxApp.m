function fig_out = FoodToolboxApp()
% FOODTOOLBOXAPP Interactive GUI for Food Image Processing and Classification.
%
% Features:
%   - Upload food image (JPG, PNG, BMP)
%   - 5 Image Processing Techniques:
%       1. Histogram Equalization (HSV/YCbCr/Gray)
%       2. Median Filtering (Noise Reduction)
%       3. Canny Edge Detection (with green overlay option)
%       4. HSV Color Segmentation (Ingredient extraction)
%       5. Bilinear Interpolation (TensorFlow 224x224 preparation)
%   - Side-by-side comparison of Original vs Processed Image
%   - TensorFlow Pre-trained Food Classification (EfficientNetB0-Food101)
%     with single-column vertical candidates list.
%
% Usage:
%   In MATLAB command window:
%   cd('D:\ComV_Project\gui');
%   FoodToolboxApp;

    % Add toolbox path
    toolbox_path = fullfile(fileparts(mfilename('fullpath')), '..', 'toolbox');
    if exist(toolbox_path, 'dir')
        addpath(toolbox_path);
    end

    % Main Figure Window
    fig = uifigure('Name', 'Food Image Processing Toolbox and Classification', ...
                   'Position', [80, 60, 1200, 760], ...
                   'Color', [0.95, 0.96, 0.98]);

    % App State
    app_data = struct();
    app_data.orig_img = [];
    app_data.processed_img = [];
    app_data.img_path = '';

    % Layout Grid: Top header, main body, bottom status
    main_grid = uigridlayout(fig, [3, 1]);
    main_grid.RowHeight = {45, '1x', 28};
    main_grid.Padding = [15, 10, 15, 10];

    % Header Label (Text only, academic appearance)
    uilabel(main_grid, ...
        'Text', 'Food Image Processing Toolbox and TensorFlow Classification', ...
        'FontSize', 17, 'FontWeight', 'bold', ...
        'FontColor', [0.12, 0.23, 0.38], ...
        'HorizontalAlignment', 'center');

    % Body Grid: Left Controls (380px), Right Viewers ('1x')
    body_grid = uigridlayout(main_grid, [1, 2]);
    body_grid.ColumnWidth = {380, '1x'};
    body_grid.Padding = [0, 0, 0, 0];

    % --- Left Control Panel (Scrollable to prevent component overlap) ---
    ctrl_panel = uipanel(body_grid, 'Title', 'Control Panel', ...
                         'FontSize', 13, 'FontWeight', 'bold', ...
                         'BackgroundColor', [1, 1, 1], ...
                         'Scrollable', 'on');
    ctrl_grid = uigridlayout(ctrl_panel, [16, 1]);
    ctrl_grid.RowHeight = {34, 22, 32, 22, 32, 34, 32, 10, 18, 36, 22, 22, 110, 10, 20, 20};
    ctrl_grid.Padding = [10, 10, 10, 10];

    % 1. Upload Button
    btn_upload = uibutton(ctrl_grid, 'push', ...
        'Text', 'Upload Food Image', ...
        'FontSize', 12, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.20, 0.55, 0.90], ...
        'FontColor', [1, 1, 1]);

    % 2. Technique Selection
    uilabel(ctrl_grid, 'Text', 'Select Processing Technique:', 'FontWeight', 'bold');
    drop_technique = uidropdown(ctrl_grid, ...
        'Items', {'1. Histogram Equalization', ...
                  '2. Median Filter (Noise Reduction)', ...
                  '3. Canny Edge Detection', ...
                  '4. HSV Color Segmentation', ...
                  '5. Bilinear Scaling (224x224)'}, ...
        'Value', '1. Histogram Equalization');

    % 3. Parameter Controls
    lbl_param = uilabel(ctrl_grid, 'Text', 'Equalization Space:', 'FontWeight', 'bold');
    drop_param = uidropdown(ctrl_grid, ...
        'Items', {'HSV Space (Preserve Color)', 'YCbCr Space', 'Grayscale'}, ...
        'Value', 'HSV Space (Preserve Color)');

    % 4. Action Buttons (Apply & Reset)
    btn_apply = uibutton(ctrl_grid, 'push', ...
        'Text', 'Apply Technique', ...
        'FontSize', 12, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.18, 0.70, 0.44], ...
        'FontColor', [1, 1, 1]);

    btn_reset = uibutton(ctrl_grid, 'push', ...
        'Text', 'Reset to Original', ...
        'FontSize', 12, ...
        'BackgroundColor', [0.85, 0.88, 0.92]);

    % Divider
    uilabel(ctrl_grid, 'Text', '---------------------------------------------------', ...
            'FontColor', [0.75, 0.75, 0.75], 'HorizontalAlignment', 'center');

    % 5. Model Classification Section
    uilabel(ctrl_grid, 'Text', 'Classification Input: Original Image (RGB)', ...
            'FontSize', 11, 'FontColor', [0.4, 0.4, 0.4]);

    btn_classify = uibutton(ctrl_grid, 'push', ...
        'Text', 'Classify Food (Original Image)', ...
        'FontSize', 12, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.55, 0.28, 0.82], ...
        'FontColor', [1, 1, 1]);

    lbl_predicted_food = uilabel(ctrl_grid, ...
        'Text', 'Predicted Food: Waiting for input...', ...
        'FontSize', 12, 'FontWeight', 'bold', ...
        'FontColor', [0.12, 0.23, 0.38]);

    uilabel(ctrl_grid, 'Text', 'Candidates List:', 'FontWeight', 'bold', 'FontSize', 11);

    % Required Candidates List: Single-column vertical layout with scrolling
    txt_candidates = uitextarea(ctrl_grid, ...
        'Editable', 'off', ...
        'FontName', 'Courier New', ...
        'FontSize', 11, ...
        'Value', {'No classification performed yet.'});

    % Spacer / Empty cells
    uilabel(ctrl_grid, 'Text', '');
    uilabel(ctrl_grid, 'Text', '');
    uilabel(ctrl_grid, 'Text', '');

    % --- Right Display Panel (Original and Processed side-by-side) ---
    display_grid = uigridlayout(body_grid, [1, 2]);
    display_grid.ColumnWidth = {'1x', '1x'};
    display_grid.Padding = [10, 5, 10, 5];

    % Original Image Axes
    panel_orig = uipanel(display_grid, 'Title', 'Original Image', 'FontWeight', 'bold');
    grid_orig = uigridlayout(panel_orig, [1, 1]);
    ax_orig = uiaxes(grid_orig);
    ax_orig.XTick = []; ax_orig.YTick = [];
    title(ax_orig, 'No Image Loaded');

    % Processed Image Axes
    panel_proc = uipanel(display_grid, 'Title', 'Processed Image Result', 'FontWeight', 'bold');
    grid_proc = uigridlayout(panel_proc, [1, 1]);
    ax_proc = uiaxes(grid_proc);
    ax_proc.XTick = []; ax_proc.YTick = [];
    title(ax_proc, 'Result will appear here');

    % Bottom Status Label
    lbl_status = uilabel(main_grid, ...
        'Text', 'Status: Ready. Please upload an image to begin.', ...
        'FontSize', 11, 'FontColor', [0.35, 0.35, 0.35]);

    % --- Callbacks & Logic ---

    % Update Parameter dropdown depending on technique selected
    drop_technique.ValueChangedFcn = @(src, event) update_param_options();
    function update_param_options()
        tech = drop_technique.Value;
        if startsWith(tech, '1.')
            lbl_param.Text = 'Equalization Space:';
            drop_param.Items = {'HSV Space (Preserve Color)', 'YCbCr Space', 'Grayscale'};
            drop_param.Value = 'HSV Space (Preserve Color)';
        elseif startsWith(tech, '2.')
            lbl_param.Text = 'Median Kernel Window:';
            drop_param.Items = {'3x3 Kernel (Standard)', '5x5 Kernel (Stronger)', '7x7 Kernel (Heavy Noise)'};
            drop_param.Value = '3x3 Kernel (Standard)';
        elseif startsWith(tech, '3.')
            lbl_param.Text = 'Gaussian Sigma:';
            drop_param.Items = {'Sigma 1.0', 'Sigma 1.5', 'Sigma 2.0'};
            drop_param.Value = 'Sigma 1.0';
        elseif startsWith(tech, '4.')
            lbl_param.Text = 'Color Selection:';
            drop_param.Items = {'Red', 'Green', 'Blue'};
            drop_param.Value = 'Red';
        elseif startsWith(tech, '5.')
            lbl_param.Text = 'Target Resizing Dimension:';
            drop_param.Items = {'224 x 224 (TensorFlow Standard)', '128 x 128 (Compact)', '299 x 299 (Inception)'};
            drop_param.Value = '224 x 224 (TensorFlow Standard)';
        end
    end

    % Upload Callback
    btn_upload.ButtonPushedFcn = @(src, event) upload_callback();
    function upload_callback()
        % Default to test_images folder if exists
        default_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'test_images');
        if ~exist(default_dir, 'dir')
            default_dir = pwd;
        end

        [filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'}, ...
                                         'Select a Food Image', default_dir);
        if isequal(filename, 0)
            return; % User cancelled
        end

        full_path = fullfile(pathname, filename);
        try
            img = imread(full_path);
            app_data.orig_img = img;
            app_data.processed_img = img;
            app_data.img_path = full_path;

            % Display on axes
            imshow(img, 'Parent', ax_orig);
            title(ax_orig, sprintf('Original: %s (%dx%d)', filename, size(img, 1), size(img, 2)), 'Interpreter', 'none');

            imshow(img, 'Parent', ax_proc);
            title(ax_proc, 'Ready for processing');

            lbl_status.Text = sprintf('Loaded "%s" successfully.', filename);
            lbl_predicted_food.Text = 'Predicted Food: Ready to classify.';
            txt_candidates.Value = {'Ready. Click "Classify Food" to view candidates.'};
        catch ME
            uialert(fig, ['Could not open image: ' ME.message], 'File Read Error');
        end
    end

    % Apply Technique Callback
    btn_apply.ButtonPushedFcn = @(src, event) apply_callback();
    function apply_callback()
        if isempty(app_data.orig_img)
            uialert(fig, 'Please upload an image first.', 'No Image');
            return;
        end

        tech = drop_technique.Value;
        param = drop_param.Value;
        lbl_status.Text = ['Processing: Applying ' tech '...'];
        drawnow;

        try
            if startsWith(tech, '1.')
                % Histogram Equalization
                if contains(param, 'HSV')
                    sp = 'hsv';
                elseif contains(param, 'YCbCr')
                    sp = 'ycbcr';
                else
                    sp = 'gray';
                end
                [res, ~] = enhance_histogram_eq(app_data.orig_img, sp);
                imshow(res, 'Parent', ax_proc);
                title(ax_proc, sprintf('Hist Equalized (%s)', upper(sp)));
                app_data.processed_img = res;
                lbl_status.Text = sprintf('Histogram Equalization completed (%s mode).', upper(sp));

            elseif startsWith(tech, '2.')
                % Median Filter
                if startsWith(param, '3x3')
                    k = [3 3];
                elseif startsWith(param, '5x5')
                    k = [5 5];
                else
                    k = [7 7];
                end
                [res, ~] = filter_median(app_data.orig_img, k);
                imshow(res, 'Parent', ax_proc);
                title(ax_proc, sprintf('Median Filtered (%dx%d Kernel)', k(1), k(2)));
                app_data.processed_img = res;
                lbl_status.Text = sprintf('Median filter applied with %dx%d kernel.', k(1), k(2));

            elseif startsWith(tech, '3.')
                % Canny Edge Detection (standard binary edge map)
                sigma_val = 1.0;
                if contains(param, '1.5')
                    sigma_val = 1.5;
                elseif contains(param, '2.0')
                    sigma_val = 2.0;
                end
                [edge_mask, info] = edge_detect_canny(app_data.orig_img, [], sigma_val);
                imshow(edge_mask, 'Parent', ax_proc);
                title(ax_proc, sprintf('Canny Edge Map (Sigma %.1f)', sigma_val));
                app_data.processed_img = edge_mask;
                lbl_status.Text = sprintf('Canny edge detection completed (%d edge pixels detected).', info.edge_pixel_count);

            elseif startsWith(tech, '4.')
                % HSV Color Segmentation (RGB: Red, Green, Blue)
                color_choice = lower(param);
                [~, seg_img, info] = segment_hsv_color(app_data.orig_img, color_choice);
                imshow(seg_img, 'Parent', ax_proc);
                title(ax_proc, sprintf('HSV Segmented: %s (%.1f%% area)', param, info.foreground_ratio * 100));
                app_data.processed_img = seg_img;
                lbl_status.Text = sprintf('HSV color segmentation completed: %s.', param);

            elseif startsWith(tech, '5.')
                % Bilinear Resize
                if startsWith(param, '224')
                    ts = [224, 224];
                elseif startsWith(param, '128')
                    ts = [128, 128];
                else
                    ts = [299, 299];
                end
                [res, info] = transform_bilinear_resize(app_data.orig_img, ts, true);
                imshow(res, 'Parent', ax_proc);
                title(ax_proc, sprintf('Bilinear Resized to %dx%d (Scale: %.2fx)', ts(1), ts(2), info.scale_x));
                app_data.processed_img = res;
                lbl_status.Text = sprintf('Bilinear scaling applied: output size %dx%d.', ts(1), ts(2));
            end
        catch ME
            uialert(fig, ['Error executing technique: ' ME.message], 'Processing Error');
            lbl_status.Text = 'Processing encountered an error.';
        end
    end

    % Reset Callback
    btn_reset.ButtonPushedFcn = @(src, event) reset_callback();
    function reset_callback()
        if ~isempty(app_data.orig_img)
            app_data.processed_img = app_data.orig_img;
            imshow(app_data.orig_img, 'Parent', ax_proc);
            title(ax_proc, 'Reset to Original Image');
            lbl_status.Text = 'Display reset to original image.';
        end
    end

    % Classify Callback
    btn_classify.ButtonPushedFcn = @(src, event) classify_callback();
    function classify_callback()
        if isempty(app_data.orig_img)
            uialert(fig, 'Please upload a food image first.', 'No Image');
            return;
        end

        lbl_status.Text = 'Running TensorFlow Food Classification model... Please wait.';
        lbl_predicted_food.Text = 'Predicted Food: Classifying...';
        drawnow;

        try
            % Classify original image (by file path or matrix)
            if ~isempty(app_data.img_path) && exist(app_data.img_path, 'file')
                [label, conf, details] = classify_food(app_data.img_path);
            else
                [label, conf, details] = classify_food(app_data.orig_img);
            end

            lbl_predicted_food.Text = sprintf('Predicted Food: %s (%.1f%%)', label, conf);

            % Build vertical single-column candidates list
            cands_list = {};
            if isfield(details, 'candidates') && isstruct(details.candidates)
                c_array = details.candidates;
            elseif isfield(details, 'top3') && isstruct(details.top3)
                c_array = details.top3;
            else
                c_array = [];
            end

            if ~isempty(c_array)
                for k = 1:numel(c_array)
                    c_name = c_array(k).label;
                    c_conf = c_array(k).confidence;
                    % Formatted single-column row with uniform column alignment
                    row_str = sprintf('%d. %-20s %6.2f%%', k, c_name, c_conf);
                    cands_list{end+1} = row_str; %#ok<AGROW>
                end
                txt_candidates.Value = cands_list;
            else
                txt_candidates.Value = {sprintf('1. %-20s %6.2f%%', label, conf)};
            end

            lbl_status.Text = sprintf('Classification completed: %s (%.1f%% confidence).', label, conf);
        catch ME
            lbl_predicted_food.Text = 'Predicted Food: Classification error';
            txt_candidates.Value = {['Error: ' ME.message]};
            lbl_status.Text = ['Model execution error: ' ME.message];
            uialert(fig, ['TensorFlow classification failed: ' ME.message], 'Classification Error');
        end
    end

    % Programmatic helper for automated testing and validation
    function set_image_programmatic(img_or_path)
        if ischar(img_or_path) || isstring(img_or_path)
            full_path = char(img_or_path);
            img = imread(full_path);
            app_data.img_path = full_path;
            [~, fn, ext] = fileparts(full_path);
            filename = [fn, ext];
        else
            img = img_or_path;
            app_data.img_path = '';
            filename = 'programmatic_image';
        end
        app_data.orig_img = img;
        app_data.processed_img = img;

        imshow(img, 'Parent', ax_orig);
        title(ax_orig, sprintf('Original: %s (%dx%d)', filename, size(img, 1), size(img, 2)), 'Interpreter', 'none');

        imshow(img, 'Parent', ax_proc);
        title(ax_proc, 'Ready for processing');

        lbl_status.Text = sprintf('Loaded "%s" successfully.', filename);
        lbl_predicted_food.Text = 'Predicted Food: Ready to classify.';
        txt_candidates.Value = {'Ready. Click "Classify Food" to view candidates.'};
    end

    function res = get_live_app_data()
        res = app_data;
    end

    % Expose UI components and helpers via fig.UserData for automated verification
    fig.UserData = struct();
    fig.UserData.btn_upload = btn_upload;
    fig.UserData.btn_apply = btn_apply;
    fig.UserData.btn_reset = btn_reset;
    fig.UserData.btn_classify = btn_classify;
    fig.UserData.drop_technique = drop_technique;
    fig.UserData.drop_param = drop_param;
    fig.UserData.lbl_status = lbl_status;
    fig.UserData.lbl_predicted_food = lbl_predicted_food;
    fig.UserData.txt_candidates = txt_candidates;
    fig.UserData.ax_orig = ax_orig;
    fig.UserData.ax_proc = ax_proc;
    fig.UserData.set_image = @set_image_programmatic;
    fig.UserData.get_app_data = @get_live_app_data;

    if nargout > 0
        fig_out = fig;
    end
end
