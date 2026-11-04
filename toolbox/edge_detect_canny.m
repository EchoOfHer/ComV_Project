function [edge_mask, info] = edge_detect_canny(img, threshold, sigma)
% EDGE_DETECT_CANNY Detects object and food boundaries using the Canny operator.
%   [EDGE_MASK, INFO] = EDGE_DETECT_CANNY(IMG, THRESHOLD, SIGMA)
%
%   Inputs:
%       img        - Input image (RGB or Grayscale uint8/double)
%       threshold  - Sensitivity threshold [low, high] or [] for automatic (default: [])
%       sigma      - Standard deviation of Gaussian filter (default: 1.0)
%
%   Outputs:
%       edge_mask  - Logical binary edge map (1 for edge pixels, 0 for background)
%       info       - Struct containing:
%                    .threshold: Calculated or supplied [low, high] threshold
%                    .sigma: Sigma value used
%                    .edge_pixel_count: Total edge pixels detected
%
%   Part of Module 2: Edge Detection (Student 2)

    if nargin < 2
        threshold = [];
    end
    if nargin < 3 || isempty(sigma)
        sigma = 1.0;
    end

    % Convert RGB to grayscale for edge computation
    if size(img, 3) == 3
        gray_img = im2double(rgb2gray(img));
    else
        gray_img = im2double(img);
    end

    % Perform Canny Edge Detection (standard binary edge map)
    if isempty(threshold)
        [edge_mask, computed_thresh] = edge(gray_img, 'canny', [], sigma);
    else
        [edge_mask, computed_thresh] = edge(gray_img, 'canny', threshold, sigma);
    end

    % Metadata
    info = struct();
    info.threshold = computed_thresh;
    info.sigma = sigma;
    info.edge_pixel_count = nnz(edge_mask);
end
