function [seg_mask, segmented_img, info] = segment_hsv_color(img, h_range, s_range, v_range)
% SEGMENT_HSV_COLOR Segments food items based on color ranges in HSV color space.
%   [SEG_MASK, SEGMENTED_IMG, INFO] = SEGMENT_HSV_COLOR(IMG, H_RANGE, S_RANGE, V_RANGE)
%   [SEG_MASK, SEGMENTED_IMG, INFO] = SEGMENT_HSV_COLOR(IMG, COLOR_PRESET)
%
%   Inputs:
%       img         - Input RGB image (uint8 or double)
%       h_range     - [H_min, H_max] in [0, 1] OR preset string ('red', 'green', 'blue')
%       s_range     - [S_min, S_max] in [0, 1] (optional if preset is used)
%       v_range     - [V_min, V_max] in [0, 1] (optional if preset is used)
%
%   Outputs:
%       seg_mask      - Logical binary mask of segmented pixels
%       segmented_img - RGB image with background masked out (set to black)
%       info          - Struct with segmentation metrics:
%                       .h_range, .s_range, .v_range
%                       .foreground_ratio: Fraction of pixels in segmented food area
%
%   Part of Module 2: Image Segmentation (Student 2)

    if size(img, 3) ~= 3
        error('HSV color segmentation requires a 3-channel RGB image.');
    end

    rgb_img = im2uint8(img);
    hsv_img = rgb2hsv(img);
    H = hsv_img(:, :, 1);
    S = hsv_img(:, :, 2);
    V = hsv_img(:, :, 3);

    % Color preset handling (Standard RGB colors: Red, Green, Blue)
    if ischar(h_range) || isstring(h_range)
        preset = lower(char(h_range));
        switch preset
            case 'red'
                % Red wraps around 0 and 1 in HSV space
                h_range = [0.94, 0.05];
                s_range = [0.25, 1.00];
                v_range = [0.20, 1.00];
            case 'green'
                h_range = [0.20, 0.45];
                s_range = [0.20, 1.00];
                v_range = [0.15, 1.00];
            case 'blue'
                h_range = [0.55, 0.75];
                s_range = [0.20, 1.00];
                v_range = [0.15, 1.00];
            case 'yellow'
                h_range = [0.10, 0.18];
                s_range = [0.25, 1.00];
                v_range = [0.30, 1.00];
            otherwise
                error('Unknown color preset: %s. Use ''red'', ''green'', or ''blue''.', preset);
        end
    else
        % Default fallback ranges if omitted
        if nargin < 3 || isempty(s_range)
            s_range = [0.15, 1.00];
        end
        if nargin < 4 || isempty(v_range)
            v_range = [0.15, 1.00];
        end
    end

    % Thresholding in H channel (accounting for circular wrap-around if H_min > H_max like Red)
    if h_range(1) <= h_range(2)
        h_mask = (H >= h_range(1)) & (H <= h_range(2));
    else
        h_mask = (H >= h_range(1)) | (H <= h_range(2));
    end

    s_mask = (S >= s_range(1)) & (S <= s_range(2));
    v_mask = (V >= v_range(1)) & (V <= v_range(2));

    % Combined binary mask
    seg_mask = h_mask & s_mask & v_mask;

    % Cleanup: Remove small isolated noise dots using morphological opening
    seg_mask = bwareaopen(seg_mask, 50);

    % Apply mask to extract foreground
    segmented_img = rgb_img;
    for c = 1:3
        channel = segmented_img(:, :, c);
        channel(~seg_mask) = 0;
        segmented_img(:, :, c) = channel;
    end

    % Populate info struct
    info = struct();
    info.h_range = h_range;
    info.s_range = s_range;
    info.v_range = v_range;
    info.foreground_pixels = nnz(seg_mask);
    info.foreground_ratio = nnz(seg_mask) / (size(img, 1) * size(img, 2));
end
