function [pred_label, confidence, details] = classify_food(img_or_path, python_path)
% CLASSIFY_FOOD Classifies a food image using pre-trained TensorFlow model via Python.
%   [PRED_LABEL, CONFIDENCE, DETAILS] = CLASSIFY_FOOD(IMG_OR_PATH, PYTHON_PATH)
%
%   Inputs:
%       img_or_path - File path string (e.g., 'D:\ComV_Project\data\test_images\pizza.jpg')
%                     OR a MATLAB image matrix (uint8 RGB)
%       python_path - (Optional) Path to python executable. If omitted, uses dynamic lookup.
%
%   Outputs:
%       pred_label  - Top predicted food class name (e.g., 'Pizza')
%       confidence  - Confidence percentage (e.g., 90.38)
%       details     - Struct containing candidates list, metadata, and raw response
%
%   Part of Phase 4: Model Integration

    % Locate Python interpreter dynamically without hardcoded user paths
    if nargin < 2 || isempty(python_path)
        python_path = find_python_interpreter();
    end

    % Locate predict_food.py relative to toolbox directory
    script_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'model');
    script_path = fullfile(script_dir, 'predict_food.py');

    if ~exist(script_path, 'file')
        error('Inference script not found at: %s', script_path);
    end

    % Handle input: File path or Image matrix
    is_temp_file = false;
    if ischar(img_or_path) || isstring(img_or_path)
        target_image_path = char(img_or_path);
        if ~exist(target_image_path, 'file')
            error('Target image file does not exist: %s', target_image_path);
        end
    else
        % Save image matrix to a temporary PNG file for external Python process
        target_image_path = fullfile(tempdir, 'matlab_food_classify_temp.png');
        imwrite(img_or_path, target_image_path);
        is_temp_file = true;
    end

    % Construct command line with safe quotes
    cmd = sprintf('"%s" "%s" "%s"', python_path, script_path, target_image_path);

    % Execute inference command
    [status, cmdout] = system(cmd);

    % Clean up temporary file if created
    if is_temp_file && exist(target_image_path, 'file')
        delete(target_image_path);
    end

    if status ~= 0
        error('TensorFlow inference execution failed:\n%s', cmdout);
    end

    % Extract JSON block robustly (ignoring any warning or system banner headers)
    json_start = find(cmdout == '{', 1, 'first');
    json_end = find(cmdout == '}', 1, 'last');

    if isempty(json_start) || isempty(json_end) || json_start > json_end
        error('No valid JSON output received from Python inference:\n%s', cmdout);
    end

    json_str = cmdout(json_start:json_end);

    % Parse JSON response from Python
    try
        details = jsondecode(json_str);
    catch ME
        error('Failed to parse JSON from Python output:\n%s\nError: %s', json_str, ME.message);
    end

    if isfield(details, 'success') && ~details.success
        error('Python inference reported failure: %s', details.error);
    end

    pred_label = details.prediction;
    confidence = details.confidence_percent;
end

function py_path = find_python_interpreter()
% FIND_PYTHON_INTERPRETER Dynamically discovers a working Python executable.
    % 1. Check if 'python' on system PATH is callable and supports tf_keras
    [status, ~] = system('python -c "import tf_keras"');
    if status == 0
        py_path = 'python';
        return;
    end

    % 2. Check MATLAB pyenv configuration
    try
        pe = pyenv;
        if ~isempty(pe.Executable) && exist(pe.Executable, 'file')
            [status, ~] = system(sprintf('"%s" -c "import tf_keras"', pe.Executable));
            if status == 0
                py_path = pe.Executable;
                return;
            end
        end
    catch
        % pyenv query failed or unconfigured
    end

    % 3. Check standard Python Windows installation directories via %LOCALAPPDATA%
    local_app_data = getenv('LOCALAPPDATA');
    if ~isempty(local_app_data)
        python_versions = {'Python312', 'Python311', 'Python310'};
        for i = 1:numel(python_versions)
            cand = fullfile(local_app_data, 'Programs', 'Python', python_versions{i}, 'python.exe');
            if exist(cand, 'file')
                [status, ~] = system(sprintf('"%s" -c "import tf_keras"', cand));
                if status == 0
                    py_path = cand;
                    return;
                end
            end
        end
    end

    % 4. Check project-local virtual environments (.venv / venv)
    local_venvs = { ...
        fullfile(fileparts(mfilename('fullpath')), '..', '.venv', 'Scripts', 'python.exe'), ...
        fullfile(fileparts(mfilename('fullpath')), '..', 'venv', 'Scripts', 'python.exe') ...
    };
    for i = 1:numel(local_venvs)
        if exist(local_venvs{i}, 'file')
            py_path = local_venvs{i};
            return;
        end
    end

    % 5. Fallback to 'python' command
    py_path = 'python';
end
