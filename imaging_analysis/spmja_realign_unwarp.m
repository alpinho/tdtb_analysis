function spmja_realign_unwarp(dataDir, subj_name, run, startTR, endTR, varargin)
% spmj_realign_unwarp(dataDir, subj_name, run, startTR, endTR)
% INPUT: 
%   dataDir:    Root directory for the imaging structure (needs directories 
%               imaging_data_raw and fieldmaps 
%   subj_name:  Name for subdirectory and file-prefix (i.e. 's05') 
%   run:        Cell array of identifiers for the run 
%               {'01' '02','03','04','05','06','07','08'} 
%   startTR:    First image to align 
%   endTR:      Last image to align (if INF, it will use all available)
% VARARGINOPTIONS: 
%   'prefix'            prefix for run name (default 'a'); 
%   'subfolderFieldmap' subfolder in the fieldmap directory

% Tobias Wiestler & Joern Diedrichsen
% 06/02/2012 subfolder option replaced with two options 'subfolderFieldmap' 'subfolderRawdata'
% 23/10/2012 added rawdataDir to be able to overwrite the standard naming convention
% 
prefix= 'a';
subfolderFieldmap='';
use3D=false;

vararginoptions(varargin,{'prefix', 'subfolderFieldmap','use3D'}); 

% displaying whether using 3D files or not
disp(['Using 3D: ' num2str(use3D)]);


%_______DEFAULTS_________________________________
J.eoptions.quality = 0.9;
J.eoptions.sep = 2;%4;                                                                                   
J.eoptions.fwhm = 5;                                                                                  
J.eoptions.rtm = 1; %why zero and not one                                                                                  
J.eoptions.einterp = 2;                                                                               
J.eoptions.ewrap = [0 1 0];     %  wrap-around in the [x y z] direction during the estimation (ewrap)  wrap of the front of the head to the back of the head                                                                       
J.eoptions.weight = {''};                                                                             
J.uweoptions.basfcn = [12 12];                                                                        
J.uweoptions.regorder = 1;                                                                            
J.uweoptions.lambda = 100000;                                                                         
J.uweoptions.jm = 0;                                                                                  
J.uweoptions.fot = [4 5];                                                                             
J.uweoptions.sot = [1];                                                                                 
J.uweoptions.uwfwhm = 4;                                                                              
J.uweoptions.rem = 1;                                                                                 
J.uweoptions.noi = 5;                                                                                 
J.uweoptions.expround = 'Average';                                                                    
J.uwroptions.uwwhich = [2 1]; %[2 1] with mean image without [2 0]
J.uwroptions.rinterp = 4;                                                                             
J.uwroptions.wrap = [0 1 0];  %  wrap-around in the [x y z] direction during the reslicing (wrap)                                                                                
J.uwroptions.mask = 1;                                                                                
J.uwroptions.prefix = 'u'; 


% if (isempty(rawdataDir))
%     rawdataDir=fullfile(dataDir, 'raw',subj_name,subfolderRawdata); 
% end; 

%_______images and fieldmap definition_________________________________
for j=1:numel(run)
    if (isinf(endTR))  
        % All avaialble images: only works with 4d-nifits right now 
        V = nifti(fullfile(dataDir, [prefix, subj_name, '_', run{j}, ...
            '_bold.nii'])); 
        imageNumber=startTR:V.dat.dim(4);
    else 
        imageNumber= startTR:endTR;
    end
    scans = {};
    for i= 1:numel(imageNumber)
        if use3D
            scans{i, 1} = fullfile(dataDir, [prefix, subj_name, '_', ...
                run{j}, '_', num2str(imageNumber(i)), '_bold.nii']);
        else
            scans{i, 1} = fullfile(dataDir, [prefix, subj_name, '_', ...
                run{j}, '_bold.nii,', num2str(imageNumber(i))]);
        end           
    end
    J.data(j).scans = scans;
    J.data(j).pmscan = {fullfile(dataDir, subfolderFieldmap, ...
        ['vdm5_sc', subj_name, '_', run{j}(1:6), '_phasediff_', ...
        run{j}(8:end), '.nii,1'])};
end 

matlabbatch{1}.spm.spatial.realignunwarp= J;
spm_jobman('run',matlabbatch);