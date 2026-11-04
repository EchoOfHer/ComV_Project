function [resized_img, info] = transform_bilinear_resize(img, target_size, use_custom_algorithm)
% TRANSFORM_BILINEAR_RESIZE Resizes an image using Bilinear Interpolation.
%   [RESIZED_IMG, INFO] = TRANSFORM_BILINEAR_RESIZE(IMG, TARGET_SIZE, USE_CUSTOM_ALGORITHM)
%
%   Inputs:
%       img                  - Input image (RGB or Grayscale uint8/double)
%       target_size          - [height, width] in pixels (e.g., [224 224] for TensorFlow models)
%                              or a scalar scaling factor (e.g., 0.5 or 1.5)
%       use_custom_algorithm - Boolean (default: true). If true, calculates bilinear
%                              weights explicitly via reverse-mapping for academic demonstration.
%                              If false, utilizes MATLAB built-in imresize.
%
%   Outputs:
%       resized_img          - Rescaled image
%       info                 - Struct containing scale factors, original and target dimensions
%
%   Part of Module 3: Geometric Transformations & Interpolation (Student 3)

    if nargin < 3 || isempty(use_custom_algorithm)
        use_custom_algorithm = true;
    end

    [orig_h, orig_w, num_channels] = size(img);

    % Parse target size
    if isscalar(target_size)
        target_h = round(orig_h * target_size);
        target_w = round(orig_w * target_size);
    else
        target_h = target_size(1);
        target_w = target_size(2);
    end

    info = struct();
    info.orig_size = [orig_h, orig_w];
    info.target_size = [target_h, target_w];
    info.scale_y = target_h / orig_h;
    info.scale_x = target_w / orig_w;
    info.method = 'Bilinear Interpolation';

    if ~use_custom_algorithm
        resized_img = imresize(img, [target_h, target_w], 'bilinear', 'Antialiasing', false);
        return;
    end

    % --- Custom Algorithmic Bilinear Interpolation (Reverse Mapping) ---
    img_double = double(img);
    resized_double = zeros(target_h, target_w, num_channels);

    % Coordinate grids in target image
    [X_target, Y_target] = meshgrid(1:target_w, 1:target_h);

    % Project target coordinates back to source coordinates
    % Continuous reverse mapping:
    X_src = (X_target - 0.5) * (orig_w / target_w) + 0.5;
    Y_src = (Y_target - 0.5) * (orig_h / target_h) + 0.5;

    % Clamp to image boundaries
    X_src = max(1, min(orig_w, X_src));
    Y_src = max(1, min(orig_h, Y_src));

    % Four nearest integer pixel coordinates in source
    x1 = floor(X_src);
    x2 = min(x1 + 1, orig_w);
    y1 = floor(Y_src);
    y2 = min(y1 + 1, orig_h);

    % Fractional offsets
    dx = X_src - x1;
    dy = Y_src - y1;

    % Bilinear weights
    w11 = (1 - dx) .* (1 - dy);
    w12 = (1 - dx) .* dy;
    w21 = dx .* (1 - dy);
    w22 = dx .* dy;

    % Compute interpolated values for each channel
    for c = 1:num_channels
        ch = img_double(:, :, c);
        
        Q11 = ch(sub2ind([orig_h, orig_w], y1, x1));
        Q12 = ch(sub2ind([orig_h, orig_w], y2, x1));
        Q21 = ch(sub2ind([orig_h, orig_w], y1, x2));
        Q22 = ch(sub2ind([orig_h, orig_w], y2, x2));

        interpolated = w11 .* Q11 + w12 .* Q12 + w21 .* Q21 + w22 .* Q22;
        resized_double(:, :, c) = interpolated;
    end

    if isa(img, 'uint8')
        resized_img = uint8(round(resized_double));
    else
        resized_img = resized_double;
    end
end
