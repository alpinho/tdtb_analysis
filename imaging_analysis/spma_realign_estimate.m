function spma_realign_(dataDir, subj_name, run, startTR, endTR, varargin)
% spma_realign_estimate(dataDir, subj_name, run, startTR, endTR)
% INPUT: 
%   dataDir:    Root directory for the imaging structure (needs directories 
%               imaging_data_raw and fieldmaos 
%   subj_name:  Name for subdirectory and file-prefix (i.e. 's05') 
%   run:        Cell array of identifiers for the run 
%               {'01' '02','03','04','05','06','07','08'} 
%   startTR:    First image to align 
%   endTR:      Last image to align (if INF, it will use all available)
% 

weight='';
prefix= 'a';
vararginoptions(varargin,{'prefix'}); 

%_______DEFAULTS_________________________________
A.eoptions.quality = 0.9;
A.eoptions.sep = 2; % default = 4;
A.eoptions.fwhm = 5;
A.eoptions.rtm = 1;
A.eoptions.interp = 2;
A.eoptions.wrap = [0 1 0];
A.eoptions.weight = {''};

%_______images definition_________________________________
for j=1:numel(run)
    if (isinf(endTR))  
        % All avaialble images: only works with 4d-nifits right now 
        V = nifti(fullfile(dataDir, [prefix, subj_name, '_', run{j}, ...
            '_bold.nii'])); 
        imageNumber=startTR:V.dat.dim(4);
    else 
        imageNumber= startTR:endTR;
    end;
    scans = {};
    for i= 1:numel(imageNumber)
        scans{i, 1} = fullfile(dataDir, [prefix, subj_name, '_', ...
            run{j}, '_bold.nii,', num2str(imageNumber(i))]);        
    end
    A.data{j,1} = scans;
end

matlabbatch{1}.spm.spatial.realign.estimate = A;
spm_jobman('run',matlabbatch);


