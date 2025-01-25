function spma_applyvdm(dataDir, subj_name, run, startTR, endTR, varargin)
% INPUT: 
%   dataDir:    Root directory for the imaging structure (needs directories 
%               imaging_data_raw and fieldmaps 
%   subj_name:  Name for subdirectory and file-prefix (i.e. 's05') 
%   run:        Cell array of identifiers for the run 
%               {'01' '02','03','04','05','06','07','08'} 
%   startTR:    First image to align 
%   endTR:      Last image to align (if INF, it will use all available)

prefix= '';
vararginoptions(varargin,{'prefix'});

A.roptions.pedir = 2;
A.roptions.which = [2 1];
A.roptions.rinterp = 4;
A.roptions.wrap = [0 1 0];
A.roptions.mask = 1;
A.roptions.prefix = 'u';

%_______images and fieldmap definition_________________________________
for j=1:numel(run)
    if (isinf(endTR))  
        % All available images: only works with 4d-nifits right now 
        V = nifti(fullfile(dataDir, [prefix, subj_name, '_', run{j}, ...
            '_bold.nii'])); 
        imageNumber=startTR:V.dat.dim(4);
    else 
        imageNumber= startTR:endTR;
    end
    scans = {};
    for i= 1:numel(imageNumber)
        scans{i, 1} = fullfile(dataDir, [prefix, subj_name, '_', ...
            run{j}, '_bold.nii,', num2str(imageNumber(i))]);       
    end
    A.data(j).scans = scans;
    A.data(j).vdmfile = {fullfile(dataDir, ...
        ['vdm5_sc', subj_name, '_', run{j}(1:6), '_phasediff_', ...
        run{j}(8:end), '.nii,1'])};
end

matlabbatch{1}.spm.tools.fieldmap.applyvdm = A;
spm_jobman('run',matlabbatch);
