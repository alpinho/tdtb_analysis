function spmja_makefieldmap(dataDir, subj_name, run, varargin)
% function spmj_makefieldmap(dataDir, subj_name, run, startTR, varargin)
%   dataDir: data directory of the project (see standard directory
%           structure)
%   run: string for run identifier: 
%           i.e. {'01'  '02','03','04','05','06','07','08'} 
%   subj_name: For directory and filenames, e.g.  's05'
% VARARGINOPTIONS 
%   prefix:  default 'a': Naming is <prefix><subjname>_run<runnumber>.nii
%   image: Number of the image in run to which fieldmap should be aligned
%               (default = 1) 
%   magnumber: number of magnitude file (when there are two)

% Tobias Wiestler 2010
% 2010 documentation Joern Diedrichsen
% 06.02.2012 subfolder option replaced with two options 'subfolderFieldmap'
% 'subfolderRawdata'

% 26/April/2012 - Modified by Naveed Ejaz
% Added support for 3D files while keeping backward compatibility for 4D
% files

prefix='a'; 
image=1; 
magnumber=1;
regularization=0.02;

use3D=false;

vararginoptions(varargin,{'prefix', 'image', 'use3D', 'magnumber', ...
    'regularization'}); 
spm_dir= fileparts(which('spm'));
spmVer=spm('Ver');


% displaying whether using 3D files or not
% disp(['Using 3D: ' num2str(use3D)])

%_______DEFAULTS_________________________________
J.defaults.defaultsval.et = [4.92 7.38];                                                                 
J.defaults.defaultsval.maskbrain = 1;                                                                   
J.defaults.defaultsval.blipdir = 1;

% TOTAL_EPI_READOUT_TIME = (EchoSpacing/IPatGRAPPA) * BaseResolution
% This is also equivalent to:
% TOTAL_EPI_READOUT_TIME = (EffectiveEchoSpacing*1000) * BaseResolution
% J.defaults.defaultsval.tert = 30.24; % tert = 0.72/2 * 84
% OR
% TOTAL_EPI_READOUT_TIME = (EchoSpacing/IPatGRAPPA) * NumberOfPhaseEncodingSteps
J.defaults.defaultsval.tert = 26.28; % tert = 0.72/2 * 73

J.defaults.defaultsval.epifm = 0;                                                                       
J.defaults.defaultsval.ajm = 0;                                                                         
J.defaults.defaultsval.uflags.method = 'Mark3D';                                                        
J.defaults.defaultsval.uflags.fwhm = 10;                                                                
J.defaults.defaultsval.uflags.pad = 0;                                                                  
J.defaults.defaultsval.uflags.ws = 1;   
switch (spmVer)
    case 'SPM12'
        J.defaults.defaultsval.mflags.template = {fullfile(spm_dir, ...
            'canonical','avg152T1.nii')}; 
    case 'SPM8'
        J.defaults.defaultsval.mflags.template = {fullfile(spm_dir, ...
            'templates','T1.nii')}; 
end; 
J.defaults.defaultsval.mflags.fwhm = 5;                                                                 
J.defaults.defaultsval.mflags.nerode = 2;                                                               
J.defaults.defaultsval.mflags.ndilate = 4;                                                              
J.defaults.defaultsval.mflags.thresh = 0.5;                                                             
J.defaults.defaultsval.mflags.reg = regularization;
J.matchvdm = 1;
J.sessname = 'run';
J.writeunwarped = 1;                                                                                    
J.anat = [];                  
J.matchanat = 1; 

% if (isempty(rawdataDir))
%     rawdataDir=fullfile(dataDir, 'raw', subj_name, subfolderRawdata); 
% end; 

%_______for multiple run with same fieldmap________________________________
for i=1:numel(run) 
    if use3D
        J.session(i).epi ={fullfile(dataDir, ...
            [prefix subj_name, '_', run{i}, '_', num2str(image), ...
            '_bold.nii'])};
    else
        J.session(i).epi ={fullfile(dataDir, ...
            [prefix, subj_name, '_', run{i}, '_bold.nii,', ...
            num2str(image)])};    
    end;
end

%_______change code here if we have 1 fieldmap for each run________________
J.phase ={fullfile(dataDir, [subj_name,'_phasediff.nii,1'])};
J.magnitude =  {fullfile(dataDir, [subj_name,'_magnitude', ...
    num2str(magnumber), '.nii,1'])};

matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj= J;

spm_jobman('run',matlabbatch);
