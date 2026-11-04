function [out_img, info] = filter_median(img, kernel_size)
% FILTER_MEDIAN Reduces noise in food images using Median Filtering.
%   [OUT_IMG, INFO] = FILTER_MEDIAN(IMG, KERNEL_SIZE)
%
%   Inputs:
%       img         - Input image (RGB uint8 or Grayscale uint8/double)
%       kernel_size - Size of neighborhood window, e.g. 3 or [3 3] (default: [3 3])
%
%   Outputs:
%       out_img     - Noise-filtered image (preserves edges while removing impulse noise)
%       info        - Struct containing filter metadata:
%                     .kernel_size: 1x2 kernel dimensions
%                     .channels_processed: Number of color channels filtered
%
%   Part of Module 1: Spatial Filtering (Student 1)

    if nargin < 2 || isempty(kernel_size)
        kernel_size = [3 3];
    elseif isscalar(kernel_size)
        kernel_size = [kernel_size kernel_size];
    end

    % Verify odd kernel dimensions
    if any(mod(kernel_size, 2) == 0)
        error('Kernel dimensions must be odd integers (e.g. 3, 5, 7).');
    end

    info = struct();
    info.kernel_size = kernel_size;
    num_channels = size(img, 3);
    info.channels_processed = num_channels;

    out_img = zeros(size(img), 'like', img);

    % Apply 2D median filter channel by channel
    for c = 1:num_channels
        out_img(:, :, c) = medfilt2(img(:, :, c), kernel_size, 'symmetric');
    end
end
