function [out_img, info] = enhance_histogram_eq(img, space)
% ENHANCE_HISTOGRAM_EQ Enhances image contrast using Histogram Equalization.
%   [OUT_IMG, INFO] = ENHANCE_HISTOGRAM_EQ(IMG, SPACE)
%
%   Inputs:
%       img   - Input image (RGB uint8/double or Grayscale uint8/double)
%       space - Color space for equalization: 'hsv' (default), 'ycbcr', or 'gray'
%
%   Outputs:
%       out_img - Contrast-enhanced image (same data type as input)
%       info    - Struct containing metadata and histograms:
%                 .space: Color space used
%                 .orig_hist: Histogram of original intensity
%                 .eq_hist: Histogram of equalized intensity
%
%   Part of Module 1: Image Enhancement (Student 1)

    if nargin < 1 || isempty(img)
        error('Input image cannot be empty.');
    end

    if nargin < 2 || isempty(space)
        space = 'hsv';
    end

    info = struct();
    info.space = space;

    % Check if image is grayscale
    if size(img, 3) == 1
        out_img = histeq(img);
        [info.orig_hist, ~] = imhist(img);
        [info.eq_hist, ~] = imhist(out_img);
        return;
    end

    % For RGB Images: Avoid separate R, G, B histogram equalization to prevent color distortion.
    switch lower(space)
        case 'hsv'
            % Equalize only Value (V) channel to preserve Hue and Saturation
            hsv = rgb2hsv(img);
            v_channel = hsv(:, :, 3);
            [info.orig_hist, ~] = imhist(im2uint8(v_channel));
            
            v_eq = histeq(v_channel);
            [info.eq_hist, ~] = imhist(im2uint8(v_eq));
            
            hsv(:, :, 3) = v_eq;
            rgb_eq = hsv2rgb(hsv);
            if isa(img, 'uint8')
                out_img = im2uint8(rgb_eq);
            else
                out_img = rgb_eq;
            end

        case 'ycbcr'
            % Equalize only Luminance (Y) channel
            ycbcr = rgb2ycbcr(img);
            y_channel = ycbcr(:, :, 1);
            [info.orig_hist, ~] = imhist(y_channel);
            
            y_eq = histeq(y_channel);
            [info.eq_hist, ~] = imhist(y_eq);
            
            ycbcr(:, :, 1) = y_eq;
            rgb_eq = ycbcr2rgb(ycbcr);
            if isa(img, 'uint8')
                out_img = im2uint8(rgb_eq);
            else
                out_img = rgb_eq;
            end

        case 'gray'
            % Convert to grayscale then equalize
            gray_img = rgb2gray(img);
            [info.orig_hist, ~] = imhist(gray_img);
            out_img = histeq(gray_img);
            [info.eq_hist, ~] = imhist(out_img);

        otherwise
            error('Unsupported color space: %s. Use ''hsv'', ''ycbcr'', or ''gray''.', space);
    end
end
