function refresh_path(filepath, refreshed_dirs)
    % Get the directory of the file
    parent_dir = fileparts(filepath);
    
    % Define the starting folder (where we begin listing directories)
    start_folder = '/home/ROBARTS/agrilopi/tsclient';  
    
    % Ensure filepath contains the start_folder
    if ~startsWith(parent_dir, start_folder)
        warning('Path does not contain %s. Skipping refresh.', start_folder);
        return;
    end

    % Extract the portion of the path from "tsclient" onwards
    relative_path = erase(parent_dir, start_folder);
    
    % Split the relative path into parts
    folders = strsplit(relative_path, filesep);
    
    % Start from "tsclient" and progressively list directories
    current_path = start_folder;
    for i = 2:length(folders)  % Start at 2 to avoid empty first entry
        current_path = fullfile(current_path, folders{i});
        
        % Only refresh if this directory has NOT been refreshed before
        if exist(current_path, 'dir') == 7 && ~isKey(refreshed_dirs, current_path)
            system(['ls "', current_path, '"']);  % Force MATLAB to recognize the directory
            refreshed_dirs(current_path) = true;  % Mark directory as refreshed
            pause(0.05);  % Reduce delay for faster execution
        end
    end
end