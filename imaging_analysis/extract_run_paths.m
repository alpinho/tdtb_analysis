% List of folders containing SPM.mat files
spm_dirs = {
    '/cifs/diedrichsen/data/Cerebellum/music-sdtb/derivatives/sub-46/estimates/allmain_tasks/ffx_rwls_dbb_hrf128'
    % Add more subject/session paths here
};

% Loop over each directory
for i = 1:length(spm_dirs)
    spm_path = fullfile(spm_dirs{i}, 'SPM.mat');

    fprintf('\n=== Processing: %s ===\n', spm_path);

    if exist(spm_path, 'file')
        load(spm_path);  % Loads SPM struct
        
        % Check if the required field exists
        if isfield(SPM, 'xY') && isfield(SPM.xY, 'P')
            % Ensure scan list is a cell array
            if ischar(SPM.xY.P)
                all_paths = cellstr(SPM.xY.P);
            else
                all_paths = SPM.xY.P;
            end

            runs = SPM.nscan;
            start_idx = 1;

            fprintf('Run-level files for: %s\n', spm_dirs{i});
            for r = 1:length(runs)
                end_idx = start_idx + runs(r) - 1;
                run_paths = all_paths(start_idx:end_idx);

                % Remove volume index (e.g., ,1)
                clean_paths = regexprep(run_paths, ',\d+$', '');

                % Get just one file per run (usually all are from the same 4D file)
                run_file = clean_paths{1};

                fprintf('Run %d file: %s\n', r, run_file);

                start_idx = end_idx + 1;
            end
        else
            warning('SPM.xY.P not found in %s — possibly an incomplete model.', spm_dirs{i});
        end
    else
        warning('SPM.mat not found in %s', spm_dirs{i});
    end
end