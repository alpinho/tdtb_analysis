function psfiles(destination, suffix)         
% Delete former postscript file
if any(size(dir(fullfile(destination, 'spm_*.ps')), 1))
    delete(fullfile(destination, 'spm_*.ps'));
end

% Rename and move postscript file
old_psfile = dir('*.ps').name;
old_psname = old_psfile(1:end-3);
psfile = strcat(old_psname, ['_' suffix '.ps']);
movefile(old_psfile, destination);
movefile(fullfile(destination, old_psfile), fullfile(destination, psfile));