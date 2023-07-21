function folder(new_folder)
if not(isfolder(new_folder))
    mkdir(new_folder);
% If derivatives folder already exists,
else
    % and it is not empty,
    content = dir(new_folder);
    if numel(content) > 2                        
        % delete all its content
        for iContent = 3 : numel(content)
            if ~content(iContent).isdir
                % remove files of folder
                delete(sprintf('%s/%s', new_folder, ...
                    content(iContent).name));
            end
        end
    end
end