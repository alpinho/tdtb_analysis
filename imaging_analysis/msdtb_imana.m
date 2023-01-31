function [ output_args ] = msdtb_imana( what, varargin )

% %============================================================================
% PATH DEFINITIONS

% Add dependencies to path
if isdir('/srv/diedrichsen/data')
    workdir='/srv/diedrichsen/data';
    homedir = '/home/ROBARTS/agrilopi';
    addpath(sprintf('%s/../matlab/spm12', workdir));
    addpath(sprintf('%s/../matlab/spm12/toolbox/suit', workdir));
    addpath(sprintf('%s/../matlab/dataframe', workdir));
    addpath(sprintf('%s/../matlab/imaging/tools', workdir));
    addpath(sprintf('%s/spm12', homedir));
    addpath(sprintf('%s/suit', homedir));
    addpath(sprintf('%s/freesurfer', homedir));
elseif isdir('/home/analu/diedrichsen_data/data')
    workdir='/home/analu/diedrichsen_data/data';
else
    fprintf(...
        'Workdir not found. Mount or connect to server and try again.');
end

%% ----- Initialize suit toolbox -----
% check for SUIT installation
if isempty(which('suit_isolate_seg')) % this function is only visible while SPM is actually "running" (not just on the path). This needs to happen for SUIT to run.
    warning('Cannot find SUIT, starting SPM12.'); % this should not happen since checked for in the beginning. Still leaving this snippet for robustnes (e.g. if someone closes SPM between starting Lead and pressing run).
    spm fmri
    if isempty(which('suit_isolate_seg')) % still not found.
    error('SUIT toolbox not found. Please install SUIT toolbox for SPM12 first (http://www.diedrichsenlab.org/imaging/suit.htm).');
    end
end
suit_defaults;

%==============================================================================
global base_dir

base_dir = sprintf('%s/Cerebellum/music-sdtb', workdir);

%%% Freesurfer stuff
path1 = getenv('PATH');
path1 = [path1, ':/srv/software/freesurfer/6.0.0/bin'];
setenv('PATH', path1);
path1 = [path1, ':/srv/software/freesurfer/6.0.0/fsfast/bin'];
setenv('PATH', path1);
path1 = [path1, ':/srv/software/freesurfer/6.0.0/mni/bin'];
setenv('PATH', path1);
setenv('FREESURFER_HOME','/srv/software/freesurfer/6.0.0');
setenv(fullfile(base_dir, 'surfaceFreesurfer'));
setenv('SUBJECTS_DIR',fullfile(base_dir, 'surfaceFreesurfer'));
setenv('PATH', path1);

% defining the names of other directories
raw_dir = 'raw';
derivatives_dir = 'derivatives';
func_dir = 'func';
anat_dir = 'ses-01/anat';
fmap_dir = 'fmap';
est_dir  = 'estimates';
fs_dir   = 'surfaceFreeSurfer';
wb_dir   = 'surfaceWB';

% list of subjects
subj_n  = [3, 4, 7, 8, 10];
% subj_n  = [4, 7, 8, 10];
% subj_n  = [3];

subj_id = 1:length(subj_n);
for s=subj_id
    subj_str{s} = ['sub-' num2str(subj_n(s), '%02d')];
end

session_names = {'ses-01', 'ses-02'};
ses_id = 1:length(session_names);

% AC coordinates (non-symmetric ones)
loc_AC = {
          [1.0 23.6 -49.3], ...       %sub-03
          [1.1 28.6 -21.1], ...       %sub-04
          [-3.3 27.3 -54.7], ...      %sub-07
          [1.8 28.8 -45.5], ...       %sub-08
          [1.5 31.6 -51.10], ...      %sub-10
          };     
% loc_AC = {
%           [1.1 28.6 -21.1], ...       %sub-04
%           [-3.3 27.3 -54.7], ...      %sub-07
%           [1.5 31.6 -51.10], ...      %sub-10
%           };

numDummys = 0;
%==============================================================================

switch what
    
    case 'ANAT:reslice_lpi' 
        % Reslice anatomical image to set it within LPI-coordinate frames
        % Example usage: msdtb_imana('ANAT:reslice_lpi')
        
        sn = subj_id;       
        vararginoptions(varargin, {'sn'});
        for s = sn
            fprintf('- Reslicing %s anatomical to LPI\n', subj_str{s});
            
            % Get the directory of subjects anatomical
            raw_subj_dir = fullfile(base_dir, raw_dir, subj_str{s});
            deriv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s});
            
            % Get the anat folder
            if strcmp(subj_str{s}, 'sub-03') || ...
                    strcmp(subj_str{s}, 'sub-04') || ...
                    strcmp(subj_str{s}, 'sub-07') || ...
                    strcmp(subj_str{s}, 'sub-08')
                subj_anatraw_dir = fullfile(raw_subj_dir, 'ses-1/anat');
            else
                subj_anatraw_dir = fullfile(raw_subj_dir, anat_dir);
            end
            subj_anatderiv_dir = fullfile(deriv_subj_dir, anat_dir);
            
            % Create subject derivatives anat folders, if they don't exist
            if not(isfolder(subj_anatderiv_dir))
                mkdir(subj_anatderiv_dir);
            % Otherwise delete pre-existing nifti files
            else
                delete([subj_anatderiv_dir '/*.nii']);
            end
            
            % Name of the anatomical image
            if strcmp(subj_str{s}, 'sub-03') || ...
                    strcmp(subj_str{s}, 'sub-04') || ...
                    strcmp(subj_str{s}, 'sub-07') || ...
                    strcmp(subj_str{s}, 'sub-08')                
                anatr_name = sprintf('%s_ses-1_acq-MPRAGE_run-01_T1w', ...
                    subj_str{s});
            else
                anatr_name = sprintf('%s_ses-01_acq-MPRAGE_run-01_T1w', ...
                    subj_str{s});
            end
            anatd_name = sprintf('%s_T1w', subj_str{s});
            
            % Gunzip source file in localscratch
            gz_source  = fullfile(...
                subj_anatraw_dir, sprintf('%s.nii.gz', anatr_name));
            gunzip(gz_source, '/localscratch');
            gunz_source = fullfile(...
                '/localscratch', sprintf('%s.nii', anatr_name));
            dest = fullfile(subj_anatderiv_dir, ...
                sprintf('%s.nii', anatd_name));
            spmj_reslice_LPI(gunz_source, 'name', dest);
            
            % In the resliced image, set translation to zero
            V               = spm_vol(dest);
            dat             = spm_read_vols(V);
            % V.mat(1:3,4)    = [0 0 0];                     
            spm_write_vol(V,dat);
            
            % Delete unziped raw files from localscratch
            if any(size(dir('/localscratch/*.nii'), 1))
                delete('/localscratch/*.nii')
            end
        end % sn (subjects)

    case 'ANAT:center_ac' % recenter to AC (manually retrieve coordinates)
        % Example usage: msdtb_imana('ANAT:center_ac')
        % run spm display to get the AC coordinates
        fprintf('MANUALLY RETRIEVE AC COORDINATES')
        sn = subj_id;
        
        vararginoptions(varargin, {'sn'});
        
        for s = sn
            fprintf('- Centre AC for %s\n', subj_str{s});
            
            % Get the directory of subjects anatomical
            deriv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s});
            subj_anatderiv_dir = fullfile(deriv_subj_dir, 'ses-01/anat');
            
            % Get the name of the anatomical image
            anat_name = sprintf('%s_T1w.nii', subj_str{s});
            
            img             = fullfile(subj_anatderiv_dir, anat_name);
            V               = spm_vol(img);
            dat             = spm_read_vols(V);
            oldOrig         = V.mat(1:3,4);
            V.mat(1:3,4)    = oldOrig-loc_AC{s}.';
            % V.mat(1:3,4)    = -loc_AC{s}.';
            spm_write_vol(V,dat);
        end % s (subjects)

    case 'ANAT:segment' % segment the anatomical image
        % also saves the bias field estimated in SPM
        % ********IF YOU WANT TO APPLY SPM BIAS CORRECTION TO, USE
        % ANAT:T1w_bcorrect CASE***********************
        % Example usage: msdtb_imana('ANAT:segment')
        % check results when done
        sn = subj_id;

        vararginoptions(varargin, {'sn'});

        SPMhome = fileparts(which('spm.m'));
        J       = []; % spm jobman
        for s = sn
            fprintf('- Anatomical segmentation for %s\n', subj_str{s});
            % Get the directory of subjects anatomical
            deriv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s});
            subj_anatderiv_dir = fullfile(deriv_subj_dir, 'ses-01/anat');
            
            % Delete previous files produced during this step, if they exist
            if any(size(dir([subj_anatderiv_dir '/BiasField*']), 1))
                delete([subj_anatderiv_dir '/BiasField*']);
            end
            if any(size(dir([subj_anatderiv_dir '/c*']), 1))
                delete([subj_anatderiv_dir '/c*']);
            end
            if any(size(dir([subj_anatderiv_dir '/iy*']), 1))
                delete([subj_anatderiv_dir '/iy*']);
            end
            if any(size(dir([subj_anatderiv_dir '/y*']), 1))
                delete([subj_anatderiv_dir '/y*']);
            end
            if any(size(dir([subj_anatderiv_dir '/*.mat']), 1))
                delete([subj_anatderiv_dir '/*.mat']);
            end
            if any(size(dir([subj_anatderiv_dir '/*.ps']), 1))
                delete([subj_anatderiv_dir '/*.ps']);
            end
            
            % Get the name of the anatomical image
            anat_name = sprintf('%s_T1w.nii', subj_str{s});
            
            J.channel.vols     = {fullfile(subj_anatderiv_dir, anat_name)};
            J.channel.biasreg  = 0.001;
            J.channel.biasfwhm = 60;
            J.channel.write    = [1 0];
            J.tissue(1).tpm    = {fullfile(SPMhome,'tpm/TPM.nii,1')};
            J.tissue(1).ngaus  = 1;
            J.tissue(1).native = [1 0];
            J.tissue(1).warped = [0 0];
            J.tissue(2).tpm    = {fullfile(SPMhome,'tpm/TPM.nii,2')};
            J.tissue(2).ngaus  = 1;
            J.tissue(2).native = [1 0];
            J.tissue(2).warped = [0 0];
            J.tissue(3).tpm    = {fullfile(SPMhome,'tpm/TPM.nii,3')};
            J.tissue(3).ngaus  = 2;
            J.tissue(3).native = [1 0];
            J.tissue(3).warped = [0 0];
            J.tissue(4).tpm    = {fullfile(SPMhome,'tpm/TPM.nii,4')};
            J.tissue(4).ngaus  = 3;
            J.tissue(4).native = [1 0];
            J.tissue(4).warped = [0 0];
            J.tissue(5).tpm    = {fullfile(SPMhome,'tpm/TPM.nii,5')};
            J.tissue(5).ngaus  = 4;
            J.tissue(5).native = [1 0];
            J.tissue(5).warped = [0 0];
            J.tissue(6).tpm    = {fullfile(SPMhome,'tpm/TPM.nii,6')};
            J.tissue(6).ngaus  = 2;
            J.tissue(6).native = [0 0];
            J.tissue(6).warped = [0 0];

            J.warp.mrf     = 1;
            J.warp.cleanup = 1;
            J.warp.reg     = [0 0.001 0.5 0.05 0.2];
            J.warp.affreg  = 'mni';
            J.warp.fwhm    = 0;
            J.warp.samp    = 3;
            J.warp.write   = [1 1];
            matlabbatch{1}.spm.spatial.preproc=J;
            spm_jobman('run',matlabbatch);
        end % s (subject)
        
    case 'ANAT:run_all'
        % Example usage: msdtb_imana('ANAT:run_all')
        
        msdtb_imana('ANAT:reslice_lpi')
        msdtb_imana('ANAT:center_ac')
        msdtb_imana('ANAT:segment')        

    case 'ANAT:T1w_bcorrect' % bias correction for anatomical T1w (optional)
        % nishimoto_imana('ANAT:T1w_bcorrect')
        sn = subj_id;
        vararginoptions(varargin, {'sn'});
        
        for s = sn
            fprintf('- Bias correcting the anatomica image for %s\n', subj_str{s});
            % Get the directory of subjects anatomical and functional
            subj_anat_dir = fullfile(base_dir, subj_str{s}, anat_dir);
            % make copy of original mean epi, and work on that
            % Get the name of the anatomical image
            source  = fullfile(subj_anat_dir, sprintf('%s_T1w_lpi.nii', subj_str{s}));
            dest    = fullfile(subj_anat_dir, sprintf('b%s_T1w_lpi.nii', subj_str{s}));
            copyfile(source, dest);
            
            % bias correct mean image for grey/white signal intensities
            P{1}    = dest;
            spmj_bias_correct(P);
        end % s (sn)
        
    case 'FUNC:make_fieldmap' % Make fieldmap
        
        spm_figure('GetWin','Graphics'); % create SPM .ps file at the end
        
        sn = subj_id;
        ssn = ses_id; % list of sessions
        tasks = {'prod', 'percep', 'ntfd'};
        magnumber=1;
        prefix = '';
        vararginoptions(varargin,{'sn', 'ssn', 'tasks'});
        for s = sn
            for ses = ssn
                if strcmp(subj_str{s}, 'sub-03') || ...
                    strcmp(subj_str{s}, 'sub-04') || ...
                    strcmp(subj_str{s}, 'sub-07') || ...
                    strcmp(subj_str{s}, 'sub-08')  
                    rawses_str = ['ses-' num2str(ses, '%d')];
                    derivses_str = ['ses-' num2str(ses, '%02d')];
                else
                    rawses_str = ['ses-' num2str(ses, '%02d')];
                    derivses_str = rawses_str
                end
                func_folder = fullfile(base_dir, raw_dir, subj_str{s}, ...
                    rawses_str, func_dir);
                fmap_folder = fullfile(base_dir, raw_dir, subj_str{s}, ...
                    rawses_str, fmap_dir);
                mag_file = sprintf(...
                    '%s_%s_magnitude1.nii', subj_str{s}, rawses_str);
                phase_file = sprintf(...
                    '%s_%s_phasediff.nii', subj_str{s}, rawses_str);
                magnitude = fullfile(fmap_folder, [mag_file '.gz']);
                phasediff = fullfile(fmap_folder, [phase_file '.gz']);
                gunzip(magnitude, '/localscratch');
                gunzip(phasediff, '/localscratch');
                if strcmp(subj_str{s}, 'sub-03') || ...
                    strcmp(subj_str{s}, 'sub-04') || ...
                    strcmp(subj_str{s}, 'sub-07') || ...
                    strcmp(subj_str{s}, 'sub-08')
                    movefile(fullfile('/localscratch', mag_file), ...
                        fullfile('/localscratch', ...
                        sprintf('%s_ses-%02d_magnitude1.nii', ...
                        subj_str{s}, ses)));
                    movefile(fullfile('/localscratch', phase_file), ...
                        fullfile('/localscratch', ...
                        sprintf('%s_ses-%02d_phasediff.nii', ...
                        subj_str{s}, ses)));
                end
                rawtag = sprintf('%s_%s', subj_str{s}, rawses_str);
                derivtag = sprintf('%s_%s', subj_str{s}, derivses_str);
                run = {};
                run_tags = {};
                for tk=1:length(tasks)
                    if ses == 2 && strcmp(tasks{tk}, 'ntfd')
                        n_run = 4;
                    else
                        n_run = 2;
                    end
                    for r=1:n_run
                        func_files = sprintf(...
                            '%s_task-%s_run-%s_bold.nii', rawtag, ...
                            tasks{tk}, num2str(r, '%02d'));
                        func_data = fullfile(func_folder, ...
                            [func_files '.gz']);
                        gunzip(func_data, '/localscratch');
                        if strcmp(subj_str{s}, 'sub-03') || ...
                            strcmp(subj_str{s}, 'sub-04') || ...
                            strcmp(subj_str{s}, 'sub-07') || ...
                            strcmp(subj_str{s}, 'sub-08')
                            movefile(fullfile('/localscratch', ...
                                func_files), fullfile('/localscratch', ...
                                sprintf('%s_task-%s_run-%s_bold.nii', ...
                                derivtag, tasks{tk}, num2str(r, '%02d'))));
                        end
                        run_tags{tk}{r} = [...
                            'task-' tasks{tk} '_run-' num2str(r, '%02d')];
                    end
                end
                run = horzcat(run_tags{:});
                spmja_makefieldmap('/localscratch', derivtag, run, ...
                    'prefix', prefix, 'rawdataDir', '/localscratch');
                
                % Create if does not exist the derivatives folder
                fmap_deriv = fullfile(base_dir, derivatives_dir, ...
                    subj_str{s}, ['ses-' num2str(ses, '%02d')], fmap_dir);
                % Create/update destination folder
                folder(fmap_deriv);
                % Move files from "/localscratch" to derivatives folder
                movefile(['/localscratch/bmask' subj_str{s} '_ses-' ...
                    num2str(ses, '%02d') '_magnitude' ...
                    num2str(magnumber, '%d') '.nii'], fmap_deriv);
                movefile(['/localscratch/fpm_sc' subj_str{s} '_ses-' ...
                    num2str(ses, '%02d') '_phasediff.nii'], fmap_deriv);
                movefile(['/localscratch/m' subj_str{s} '_ses-' ...
                    num2str(ses, '%02d') '_magnitude' ...
                    num2str(magnumber, '%d') '.nii'], fmap_deriv);
                movefile(['/localscratch/sc' subj_str{s} '_ses-' ...
                    num2str(ses, '%02d') '_phasediff.nii'], fmap_deriv);
                movefile(['/localscratch/u' subj_str{s} '_ses-' ...
                    num2str(ses, '%02d') '_task-*_run-*_bold.nii'], ...
                    fmap_deriv);
                movefile(['/localscratch/vdm5_sc' subj_str{s} '_ses-' ...
                    num2str(ses, '%02d') '_phasediff.nii'], ...
                    fmap_deriv);
                for rn=1:length(run)
                    movefile(['/localscratch/vdm5_sc' subj_str{s} ...
                        '_ses-' num2str(ses, '%02d') '_phasediff_run' ...
                        num2str(rn, '%d') '.nii'], ...
                        fullfile(fmap_deriv, ['vdm5_sc' subj_str{s} ...
                        '_ses-' num2str(ses, '%02d') '_phasediff_' ...
                        run{rn} '.nii']));
                end
                movefile(['/localscratch/wfmag_' subj_str{s} '_ses-' ...
                    num2str(ses, '%02d') '_task-*_run-*_bold.nii'], ...
                    fmap_deriv);
                % Rename and move postscript file
                old_psfile = dir('*.ps').name;
                old_psname = old_psfile(1:end-3);
                psfile = strcat(old_psname, '_fieldmap.ps');
                movefile(old_psfile, fmap_deriv);
                movefile(fullfile(fmap_deriv, old_psfile), ...
                    fullfile(fmap_deriv, psfile));
                % Delete unziped raw files from localscratch
                if any(size(dir('/localscratch/*.nii'), 1))
                    delete('/localscratch/*.nii');
                end
            end % ses (sessions)
        end % s (sn)

    case 'FUNC:realign_unwarp' % realign functional images
        % SPM realigns all volumes to the first volume of first run
        % example usage: msdtb_imana('FUNC:realign', 'sn', 1)
        % Updated upstream
        
        spm_figure('GetWin','Graphics'); % create SPM .ps file at the end
        
        sn   = subj_id; % list of subjects
        ssn = ses_id; % list of sessions
        tasks = {'prod', 'percep', 'ntfd'};
        prefix = '';
        vararginoptions(varargin,{'sn', 'ssn', 'tasks'});
                
        for s = sn        
            raw_subjdir = fullfile(base_dir, raw_dir, subj_str{s});          
            deriv_subjdir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s});
            run = {};
            for ses = ssn
                if strcmp(subj_str{s}, 'sub-03') || ...
                    strcmp(subj_str{s}, 'sub-04') || ...
                    strcmp(subj_str{s}, 'sub-07') || ...
                    strcmp(subj_str{s}, 'sub-08')  
                    funcraw_folder = fullfile(raw_subjdir, ...
                        ['ses-' num2str(ses, '%d')], func_dir);
                else
                    funcraw_folder = fullfile(raw_subjdir, ...
                        ['ses-' num2str(ses, '%02d')], func_dir);
                end
                fmapderiv_folder = fullfile(deriv_subjdir, ...
                    ['ses-' num2str(ses, '%02d')], fmap_dir); 
                for tk=1:length(tasks)
                    if ses == 2 && strcmp(tasks{tk}, 'ntfd')
                        n_run = 4;
                    else
                        n_run = 2;
                    end
                    for r=1:n_run
                        if strcmp(subj_str{s}, 'sub-03') || ...
                            strcmp(subj_str{s}, 'sub-04') || ...
                            strcmp(subj_str{s}, 'sub-07') || ...
                            strcmp(subj_str{s}, 'sub-08')  
                            func_file = sprintf(...
                                '%s_ses-%d_task-%s_run-%s_bold.nii', ...
                                subj_str{s}, ses, tasks{tk}, ...
                                num2str(r, '%02d'));
                        else
                            func_file = sprintf(...
                                '%s_ses-%02d_task-%s_run-%s_bold.nii', ...
                                subj_str{s}, ses, tasks{tk}, ...
                                num2str(r, '%02d'));
                        end
                        func_data = fullfile(funcraw_folder, ...
                            [func_file '.gz']);
                        gunzip(func_data, '/localscratch');
                        if strcmp(subj_str{s}, 'sub-03') || ...
                            strcmp(subj_str{s}, 'sub-04') || ...
                            strcmp(subj_str{s}, 'sub-07') || ...
                            strcmp(subj_str{s}, 'sub-08')  
                            movefile(fullfile('/localscratch', ...
                                func_file), fullfile('/localscratch', ...
                                sprintf(...
                                '%s_ses-%02d_task-%s_run-%s_bold.nii', ...
                                subj_str{s}, ses, tasks{tk}, ...
                                num2str(r, '%02d'))));
                        end
                        fmap_fname = sprintf(...
                            'vdm5_sc%s_ses-%02d_phasediff_task-%s_run-%02d.nii', ...
                            subj_str{s}, ses, tasks{tk}, r);
                        fmap_file = fullfile(fmapderiv_folder, fmap_fname)
                        copyfile(fmap_file, '/localscratch');
                        run{ses}{tk}{r} = ['ses-' num2str(ses, '%02d') ...
                            '_task-' tasks{tk} '_run-' num2str(r, '%02d')];
                    end % r (n_run)
                end  % tk (tasks)
            end % ses (ses_id)
            run = horzcat(run{:});
            run = horzcat(run{:});
            % Load batch and run spm
            spmja_realign_unwarp('/localscratch', subj_str{s}, run, 1, Inf, ...
                'prefix', prefix, 'rawdataDir', '/localscratch');
             
            for ses = ssn
                % Create if does not exist the derivatives folder
                func_deriv_folder = fullfile(base_dir, derivatives_dir, ...
                    subj_str{s}, ['ses-' num2str(ses, '%02d')], func_dir);
                % Create/update destination folder
                folder(func_deriv_folder);
                % Move files from "/localscratch" to destination folder
                if ses == 1
                    % Move mean-unwarped EPI file
                    movefile(['/localscratch/meanu' subj_str{s} ...
                        '_ses-' num2str(ses, '%02d') ...
                        '_task-prod_run-01_bold.nii'], func_deriv_folder);
                    % Rename and move postscript file
                    old_psfile = dir('*.ps').name;
                    old_psname = old_psfile(1:end-3);
                    psfile = strcat(old_psname, '_realignunwarp.ps');
                    movefile(old_psfile, func_deriv_folder);
                    movefile(fullfile(func_deriv_folder, old_psfile), ...
                    fullfile(func_deriv_folder, psfile));
                end
                % Move motion-param .txt files
                movefile(['/localscratch/rp_' subj_str{s} '_ses-' ...
                    num2str(ses, '%02d') '_task-*_run-*_bold.txt'], ...
                    func_deriv_folder);
                
                % Move functional files w/ param estimation in their
                % headers, but not resliced only, as well as all .mat files
                % (i.e. those about realign and those about the unwarp
                movefile(['/localscratch/' subj_str{s} '_ses-' ...
                    num2str(ses, '%02d') '_task-*_run-*_bold*'], ...
                    func_deriv_folder);

                % Move unwarped images
                movefile(['/localscratch/u' subj_str{s} '_ses-' ...
                    num2str(ses, '%02d') '_task-*_run-*_bold*'], ...
                    func_deriv_folder);
            end % ses (ses_id)
            
            % Delete unziped raw files from localscratch
            if any(size(dir('/localscratch/*.nii'), 1))
                delete('/localscratch/*.nii');
            end
        end % s (sn)

    case 'FUNC:coreg' % coregistration with the anatomicals
        % (1) Manually seed the functional/anatomical registration
        % - Do "coregtool" on the matlab command window
        % - Select anatomical image and mean functional image to overlay
        % - Manually adjust mean functional image and save the results 
        %   ("r" will be added as a prefix)
        % Example usage: 
        % msdtb_imana('FUNC:coreg', 'sn', [1], 'prefix', 'r')ses-03
        
        spm_figure('GetWin','Graphics'); % create SPM .ps file at the end
        
        sn     = subj_id;   % list of subjects
        step   = 'manual';  % first 'manual' then 'auto'
        prefix = 'r'; % to use the bias corrected version, set it to 'rb'
        
        % ===================
        % After the manual registration, the mean functional image will be
        % saved with r as the prefix which will then be used in the
        % automatic registration
        vararginoptions(varargin, {'sn', 'step', 'prefix'});
        spm_jobman('initcfg')
        for s = sn   
            % Get the directory of subjects anatomical and functional
            deriv_folder = fullfile(base_dir, derivatives_dir, ...
                subj_str{s});
            anat_deriv = fullfile(deriv_folder, 'ses-01/anat');
            func_deriv = fullfile(deriv_folder, 'ses-01', func_dir);
            
            switch step
                case 'manual'
                    coregtool;
                    keyboard;
                case 'auto'
                    % do nothing
            end % switch step

            % (2) Automatically co-register functional and ...
            % anatomical images
            J.ref = {fullfile(anat_deriv, sprintf('%s_T1w.nii', subj_str{s}))};
            J.source = {fullfile(func_deriv, ...
                sprintf('%smeanu%s_ses-01_task-prod_run-01_bold.nii', ...
                prefix, subj_str{s}))};

            J.other             = {''};
            J.eoptions.cost_fun = 'nmi';
            J.eoptions.sep      = [4 2];
            J.eoptions.tol      = [0.02 0.02 0.02 0.001 0.001 0.001 ...
                0.01 0.01 0.01 0.001 0.001 0.001];
            J.eoptions.fwhm     = [7 7];
            matlabbatch{1}.spm.spatial.coreg.estimate=J;
            spm_jobman('run', matlabbatch);
            
            % Delete former postscript file
            if any(size(dir(fullfile(anat_deriv, 'spm_*.ps')), 1))
                delete(fullfile(anat_deriv, 'spm_*.ps'));
            end
            
            % Rename and move postscript file
            old_psfile = dir('*.ps').name;
            old_psname = old_psfile(1:end-3);
            psfile = strcat(old_psname, '_coregestimate.ps');
            movefile(old_psfile, anat_deriv);
            movefile(fullfile(anat_deriv, old_psfile), ...
                fullfile(anat_deriv, psfile));

            % (3) Manually check again
%               coregtool;
%               keyboard();
            % checking the affine matrix
%                 T1_vol = spm_vol(J.ref);
%                 T1_vol = T1_vol{1};
%                 T2_vol = spm_vol(J.source);
%                 T2_vol = T2_vol{1};
%                 x = spm_coreg(T2_vol, T1_vol);
%                 M = spm_matrix(x);
%                 display(M)

            % NOTE:
            % Overwrites meanepi, unless you update in step one, 
            % which saves it as rmeanepi.
            % Each time you click "update" in coregtool, it saves 
            % current alignment by appending the prefix 'r' to the 
            % current file. So if you continually update rmeanepi, 
            % you'll end up with a file called r...rrrmeanepi.
        end % s (sn)

    case 'FUNC:make_samealign' % align all the functionals
        % Aligns all functional images to rmean functional image
        % Example usage: 
        % msdtb_imana('FUNC:make_samealign', 'prefix', 'r', 'sn', [1])
        
        sn     = subj_id;  % subject list
        ssn = ses_id; % list of sessions
        tasks = {'prod', 'percep', 'ntfd'};
        prefix = 'r'; % prefix for the meanepi: r or rbb if bias corrected
        
        vararginoptions(varargin, {'sn', 'prefix'});
                
        for s = sn
            % Get the directory of subjects functional
            deriv_folder = fullfile(base_dir, derivatives_dir, ...
                subj_str{s});
            funcmean_deriv = fullfile(deriv_folder, 'ses-01', func_dir);
            Q = {};
            for ses = ssn
                func_deriv = fullfile(deriv_folder, ...
                    ['ses-' num2str(ses, '%02d')], func_dir);
                fprintf('- make_samealign  %s \n', subj_str{s})
                % Select image for reference 
                %%% note that functional images are aligned with the first
                %%% run from first session hence, the ref is always 
                %%% rmean<subj>_ses-01_run-01
                P{1} = fullfile(funcmean_deriv, sprintf(...
                    '%smeanu%s_ses-01_task-prod_run-01_bold.nii', ...
                    prefix, subj_str{s}));
                % Select images to be realigned
                for tk=1:length(tasks)
                    if ses == 2 && strcmp(tasks{tk}, 'ntfd')
                        n_run = 4;
                    else
                        n_run = 2;
                    end
                    for r=1:n_run
                        fpath = fullfile(func_deriv, sprintf(...
                            'u%s_ses-%02d_task-%s_run-%02d_bold.nii', ...
                            subj_str{s}, ses, tasks{tk}, r));
                        V = nifti(fpath);
                        imageNumber=1:V.dat.dim(4);
                        for i= 1:numel(imageNumber)
                            % for 'auto' mode in coregistration, remove prefix 
                            % and explicitly add 'r' prefix in the same place
    %                         Q{end+1} = fullfile(subj_func_dir, ...
    %                             sprintf('%s%s_ses-%02d_run-%02d.nii,%d', ...
    %                             prefix, subj_str{s}, ses, r, i)); 
                            Q{end+1} = [fpath ',' num2str(i, '%d')];
                        end % i (imageNumber)
                    end % r(runs)
                end % tk (tasks)
            end % ss (sess)
            spmj_makesamealign_nifti(char(P),char(Q));
        end % s (sn)

    case 'FUNC:make_maskImage' % make mask images (noskull and grey_only)
        % Make maskImage in functional space
        % Example usage: 
        % msdtb_imana('FUNC:make_maskImage', 'prefix', 'r', 'sn', 1)
        
        sn     = subj_id; % list of subjects
        prefix = 'r'; % prefix for the meanepi: r or rbb if bias corrected
        
        vararginoptions(varargin, {'sn', 'prefix'});
        
        
        for s = sn
            % Get the directory of subjects' derivatives
            deriv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s});
            
            % Path of anatomical folder
            subj_anatderiv_dir = fullfile(deriv_subj_dir, 'ses-01/anat');        
            
            % Path for mean EPI
            funcmean_deriv = fullfile(deriv_subj_dir, 'ses-01', func_dir);
            meanepi = fullfile(funcmean_deriv, sprintf(...
                '%smeanu%s_ses-01_task-prod_run-01_bold.nii', ...
                prefix, subj_str{s}));
            
            % Path of masks
            wholebrain_mask = fullfile(funcmean_deriv, ...
                'rmask_noskull.nii');
            graymatter_mask = fullfile(funcmean_deriv, 'rmask_gray.nii');

            % Delete old masks
            if any(size(dir(wholebrain_mask),1))
                delete(wholebrain_mask)
            end
            if any(size(dir(graymatter_mask),1))
                delete(graymatter_mask)
            end  
                
            % First mask: whole btain
            nam{1}  = meanepi;
            nam{2}  = fullfile(subj_anatderiv_dir, sprintf('c1%s_T1w.nii', ...
                subj_str{s}));
            nam{3}  = fullfile(subj_anatderiv_dir, sprintf('c2%s_T1w.nii', ...
                subj_str{s}));
            nam{4}  = fullfile(subj_anatderiv_dir, sprintf('c3%s_T1w.nii', ...
                subj_str{s}));
            spm_imcalc(nam, wholebrain_mask, 'i1>0 & (i2+i3+i4)>0.1');

            % Second mask: gray-matter mask
            nam     = {};
            nam{1}  = meanepi;
            nam{2}  = fullfile(subj_anatderiv_dir, sprintf('c1%s_T1w.nii', ...
                subj_str{s}));
            spm_imcalc(nam, graymatter_mask, 'i1>0 & i2>0.1');

        end % s (sn)
        
    case 'FUNC:run_all'
        % Example usage: msdtb_imana('FUNC:run_all')
        
        msdtb_imana('FUNC:make_fieldmap')
        msdtb_imana('FUNC:realign_unwarp')
        msdtb_imana('FUNC:coreg', 'prefix', '', 'step', 'auto')
        msdtb_imana('FUNC:make_samealign', 'prefix', '')
        msdtb_imana('FUNC:make_maskImage', 'prefix', '')
        
    case 'FUNC:meanepi_bcorrect' % bias correction for the mean image before coreg (optional)
        % uses the bias field estimated in SPM segmenttion
        % Example usage: nishimoto_imana('FUNC:meanepi_bcorrect')
        sn = subj_id;
        vararginoptions(varargin, {'sn'});
        
        for s = sn
            fprintf('- Bias correcting mean epi for %s\n', subj_str{s});
            % Get the directory of subjects anatomical and functional
            subj_func_dir = fullfile(base_dir, subj_str{s}, func_dir);
            % make copy of original mean epi, and work on that
            source  = fullfile(subj_func_dir, sprintf('mean%s_ses-01_run-01.nii', subj_str{s}));
            dest    = fullfile(subj_func_dir, sprintf('bmean%s_ses-01_run-01.nii', subj_str{s}));
            copyfile(source, dest);
            
            % bias correct mean image for grey/white signal intensities
            P{1}    = dest;
            spmj_bias_correct(P);
        end % s (sn)
        
    case 'FUNC:coregreslice'
        sn     = subj_id;   % list of subjects        
        step   = 'manual';  % first 'manual' then 'auto'
        prefix = 'r'; % to use the bias corrected version, set it to 'rb'

        vararginoptions(varargin, {'sn', 'step', 'prefix'});
        spm_jobman('initcfg')
        for s = sn
            % Get the directory of subjects anatomical and functional
            deriv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s})
            subj_anat_dir = fullfile(deriv_subj_dir, anat_dir);
            sbj_number = str2double((extractAfter(subj_str{s}, ...
                'sub-')))
            subsess = cellstr(sessmap.(['sub' num2str(sbj_number, ...
                '%02d')]));
            for smap = session_names
                sesstag = sessnum{find(contains(subsess,smap))};
                ses = sscanf(sesstag,'ses-%d');
                subj_func_dir = fullfile(deriv_subj_dir, func_dir, ...
                    ['ses-' num2str(ses, '%02d')]);
            
                % goes to subjects anatomical dir so that coreg tool ...
                % starts from that directory (just for convenience)
                cd(subj_anat_dir); 
            
                switch step
                    case 'manual'
                        coregtool;
                        keyboard;
                    case 'auto'
                        % do nothing
                end % switch step
                
                gunzip(sprintf(...
                    '%s_space-native_desc-resampled_T1w.nii.gz', ...
                    subj_str{s}));
                
                % Create copy of meanepi to be resliced
                if strcmp(smap,'mtt1') || strcmp(smap,'mtt2')
                    meanepi_file = fullfile(subj_func_dir, ...
                        sprintf('%smean%s_ses-%02d_run-03_bold.nii', ...
                        prefix, subj_str{s}, ses))
                    resliced_meanepi = fullfile(subj_func_dir, ...
                        sprintf(...
                        'resliced_%smean%s_ses-%02d_run-03_bold.nii', ...
                        prefix, subj_str{s}, ses))
                else
                    meanepi_file = fullfile(subj_func_dir, ...
                        sprintf('%smean%s_ses-%02d_run-01_bold.nii', ...
                        prefix, subj_str{s}, ses))
                    resliced_meanepi = fullfile(subj_func_dir, ...
                        sprintf(...
                        'resliced_%smean%s_ses-%02d_run-01_bold.nii', ...
                        prefix, subj_str{s}, ses))
                end
                copyfile(meanepi_file, resliced_meanepi)
                            
                J.ref = {fullfile(subj_anat_dir, sprintf(...
                    '%s_space-native_desc-resampled_T1w.nii', ...
                    subj_str{s}))};
                J.source = {resliced_meanepi};
                J.other             = {''};
                J.eoptions.cost_fun = 'nmi';
                J.eoptions.sep      = [4 2];
                J.eoptions.tol      = [0.02 0.02 0.02 0.001 0.001 0.001 ...
                    0.01 0.01 0.01 0.001 0.001 0.001];
                J.eoptions.fwhm     = [7 7];
                matlabbatch{1}.spm.spatial.coreg.estwrite=J;
                spm_jobman('run',matlabbatch);
            end
        end % s (sn)

    case 'FUNC:check_coreg' % prints out the transformation matrix for coreg

        % Run this case to get the transformation matrix and then use it
        % for translation/rotation to check the coreg.
        % the coreg case just estimates the transformation matrix, it
        % doesn't reslice! So you need to check it yourself!!!!
        % Input each subject separately
        % Example usage: nishimoto_imana('FUNC:check_coreg', 'sn', 1)
        
        sn = 1; 
        
        vararginoptions(varargin, {'sn'});
        
        anat_file = fullfile(base_dir, subj_str{sn}, 'anat', sprintf('%s_T1w_lpi.nii', subj_str{sn}));
        func_file = fullfile(base_dir, subj_str{sn}, 'func', sprintf('rmask_noskull.nii')); %???????????????
        
        T1_vol = spm_vol(anat_file);
        T2_vol = spm_vol(func_file);
        
        x = spm_coreg(T2_vol, T1_vol);
        M = spm_matrix(x);
        
        spm_get_space(T2, M * T2_vol.mat);

    case 'GLM:copy_paradigm-descriptors'
        % Example usage: msdtb_imana('GLM:copy_paradigm_descriptors')

        sn  = subj_id;
        ssn = ses_id; % list of sessions
        vararginoptions(varargin, {'sn'});
        
        if isdir('/srv/diedrichsen/data')
            source = fullfile(homedir, ...
                'analu/mygit/music-sdtb/music-sdtb_analysis/imaging_analysis/events/');
        else
            source = fullfile(...
                '/home/analu/mygit/music-sdtb/music-sdtb_analysis/imaging_analysis/events/');
        end
        destination = fullfile(workdir, ...
            'Cerebellum/music-sdtb/derivatives');

        for s = sn
            for ses = ssn
                sfiles = [source subj_str{s} '/ses-' ...
                    num2str(ses, '%02d') '/*.tsv'];
                dfolder = fullfile(destination, subj_str{s}, ...
                    ['ses-' num2str(ses, '%02d')], 'func');
                copyfile(sfiles, dfolder)
            end
        end
        
    case 'GLM:grand_design' % make the design matrix for the glm
        % models each condition as a separate regressors
        % For conditions with multiple repetitions, one regressor
        % represents all the instances
        % msdtb_imana('GLM:design', 'sn', [1])
        
        sn = subj_id;
        ssn = ses_id; % list of sessions
        tasks = {'prod', 'percep', 'ntfd'};
        hrf_cutoff = Inf;
        prefix = 'u'; % prefix of the preprocessed epi we want to use
        random = 0;
        vararginoptions(varargin, {'sn', 'hrf_cutoff', 'random'});
        
        % loop over subjects
        for s = sn
            deriv_subjdir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s});
            glms_folder = fullfile(deriv_subjdir, est_dir);
            
            if random == 1
                estimates_folder = fullfile(glms_folder, 'rand_ntfd');
            else
                estimates_folder = fullfile(glms_folder, 'ffx');
            end
            
            % Create estimates folder if does not exist or clean it
            folder(estimates_folder)
                
            J = []; % structure with SPM fields to make the design

            J.timing.units   = 'secs';
            J.timing.RT      = 1.2;
            J.timing.fmri_t  = 16;
            J.timing.fmri_t0 = 1;

            J.fact             = struct('name', {}, 'levels', {});
            J.bases.hrf.derivs = [0 0];
            J.bases.hrf.params = [4.5 11]; % set to [] if running wls
            J.volt             = 1;
            J.global           = 'None';
            J.mask             = {char(fullfile(deriv_subjdir, ...
                'ses-01', func_dir, 'rmask_noskull.nii'))};
            J.mthresh          = 1.;
            J.cvi_mask         = {char(fullfile(deriv_subjdir, ...
                'ses-01', func_dir,'rmask_gray.nii'))};
            J.cvi              = 'fast';

            J.dir = {estimates_folder};

            % loop over sessions
            count = 0;
            for ses = ssn
                funcderiv_folder = fullfile(deriv_subjdir, ...
                    ['ses-' num2str(ses, '%02d')], func_dir);
                
                % loop over tasks
                for tk=1:length(tasks)
                    if strcmp(tasks{tk}, 'prod')
                        ttag = 'production';
                    elseif strcmp(tasks{tk}, 'percep')
                        ttag = 'perception';
                    elseif strcmp(tasks{tk}, 'ntfd')
                        ttag = 'ntfd';
                    else
                        % do nothing
                    end                
                
                    % loop over runs
                    if random == 1
                        if ses == 2 && strcmp(tasks{tk}, 'ntfd')
                            start = 3;
                            n_run = 4;
                        else
                            continue
                        end
                    else
                        start = 1;
                        n_run = 2;
                    end
                    
                    for r=start:n_run
                        count = count + 1;
                        fpath = fullfile(funcderiv_folder, ...
                            sprintf(...
                            '%s%s_ses-%02d_task-%s_run-%02d_bold.nii', ...
                            prefix, subj_str{s}, ses, tasks{tk}, r));
                        V = niftiinfo(fpath);
                        numTRs = V.ImageSize(4);
                        % fill in nifti image names for the current run
                        N = cell(numTRs - numDummys, 1); % preallocating!
                        for i = 1:(numTRs-numDummys)
                            N{i} = fullfile(funcderiv_folder, ...
                                sprintf(...
                                '%s%s_ses-%02d_task-%s_run-%02d_bold.nii, %d', ...
                                prefix, subj_str{s}, ses, tasks{tk}, r, i));
                        end % i (image numbers)
                    
                        % Load the scans
                        J.sess(count).scans = N; % scans in the current runs
                    
                        J.sess(count).cond = struct('name', {}, ...
                            'onset', {}, 'duration', {}, 'tmod', {}, ...
                            'pmod', {}, 'orth', {});
                    
                        % Event Files
                        % get the path to the tsv file
                        tsv_file = fullfile(funcderiv_folder, sprintf(...
                            '%s_ses-%02d_task-%s_run-%02d_events.tsv', ...
                            subj_str{s}, ses, tasks{tk}, r));
                        
                        % get the tsvfile for the current run
                        D = struct([]); 
                        D = tdfread(tsv_file,'\t');
                    
                        trial_names = {};
                        trial_onsets = {};
                        trial_durations = {};
                        trial_names = cellstr(D.trial_type);
                        trial_onsets = num2cell(D.onset);
                        trial_durations = num2cell(D.duration);                    

                        unique_names = {};
                        unique_names = unique(trial_names).';
                        % Remove Rest, since it will be modelled implicitly
                        unique_names(ismember(unique_names, 'rest')) = [];
                        % Define paradigm descriptors
                        names = {};
                        onsets = {};
                        durations = {};
                        % Encoding condition goes first
                        names(2:2:length(unique_names))= unique_names(1:2:end);
                        names(1:2:length(unique_names))= unique_names(2:2:end);
                        % Create onsets and duration cells
                        for u = 1:length(names)
                            indexes = [];
                            indexes = find(strcmp(trial_names, names{u}));
                            for idx = 1:length(indexes)
                                onsets{u}(idx) = trial_onsets{indexes(idx)};
                                durations{u}(idx) = ...
                                    trial_durations{indexes(idx)};
                            end
                        end

                        save(...
                            sprintf(...
                            '/localscratch/%s_ses-%02d_task-%s_run-%02d_events.mat', ...
                            subj_str{s}, ses, tasks{tk}, r), ...
                            'names', 'onsets', 'durations'); 
                                       
                        J.sess(count).multi = {sprintf(...
                            '/localscratch/%s_ses-%02d_task-%s_run-%02d_events.mat', ...
                            subj_str{s}, ses, tasks{tk}, r)};
                    
                        J.sess(count).regress   = struct('name', {}, ...
                            'val', {});
                        J.sess(count).multi_reg = {''};
                        J.sess(count).hpf       = hrf_cutoff; % set to 0'inf' if using J.cvi = 'FAST'. SPM HPF not applied

                    end % r (n_run)
                end % tk (tasks)         
            end % ses (ssn)

            % FFX across all sessions for all tasks per participant
            spm_rwls_run_fmri_spec(J);

            % Remove *events.mat file from /localscratch
            if any(size(dir('/localscratch/*_events.mat'), 1))
                delete('/localscratch/*_events.mat');
            end
        end % sn (subject)     

    case 'GLM:estimate' % estimate beta values
        % Example usage: msdtb_imana('GLM:estimate', 'sn', [1], 'ses', {'archi'})
        
        sn       = subj_id; % subject list
        tasks = {'prod', 'percep', 'ntfd'};
        vararginoptions(varargin, {'sn'})
        
        for s = sn
            estderiv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s}, est_dir);            
            % loop over tasks
            for tk=1:length(tasks)
                if strcmp(tasks{tk}, 'prod')
                    ttag = 'production';
                elseif strcmp(tasks{tk}, 'percep')
                    ttag = 'perception';
                elseif strcmp(tasks{tk}, 'ntfd')
                    ttag = 'ntfd';
                else
                    continue
                end
                esttask_folder = fullfile(estderiv_subj_dir, ttag);
                % Delete previous estimates, if they exist
                if any(size(dir([esttask_folder '/*.nii']), 1))
                    delete([esttask_folder '/*.nii']);
                end
                % Load SPM.mat file with design and add path to store
                % the new estimates
                load(fullfile(esttask_folder, 'SPM.mat'));
                SPM.swd = esttask_folder;
                % Add as input SPM.mat file to the rwls GLM 
                spm_rwls_spm(SPM);
            end % tk (tasks)
        end % s (sn)    

    case 'GLM:individual_ffx_t'
        % Estimate ffx individual tmaps across runs and sessions
        
        % Go to the folder of script
        cd(fileparts(mfilename('fullpath')))
        
        sn       = subj_id; % subject list
        tasks = {'prod', 'percep', 'ntfd'};
        vararginoptions(varargin, {'sn'})
        
        contrasts = {'Enconding', [1 0 1 0 1 0 1 0]; ...
                     'Auditory Encoding', [1 0 1 0]; ...
                     'Visual Encoding', [0 0 0 0 1 0 1 0]; ...
                     'Auditory vs Visual Encoding', [1 0 1 0 -1 0 -1 0]; ...
                     'Visual vs Auditory Encoding', [-1 0 -1 0 1 0 1 0]; ...
                     'Beat vs Interval', [1 0 -1 0 1 0 -1 0]; ...
                     'Auditory Beat vs Auditory Interval', [1 0 -1 0]; ...
                     'Visual Beat vs Visual Interval', [0 0 0 0 1 0 -1 0]; ...
                     'Interval vs Beat', [-1 0 1 0 -1 0 1 0]; ...
                     'Auditory Interval vs Auditory Beat', [-1 0 1 0]; ...
                     'Visual Interval vs Visual Beat', [0 0 0 0 -1 0 1 0]; ...
                     'Decision', [0 1 0 1 0 1 0 1]; ...
                     };
        
        for s = sn
            estderiv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s}, est_dir);            
            % loop over tasks
            for tk=1:length(tasks)
                if strcmp(tasks{tk}, 'prod')
                    ttag = 'production';
                elseif strcmp(tasks{tk}, 'percep')
                    ttag = 'perception';
                elseif strcmp(tasks{tk}, 'ntfd')
                    ttag = 'ntfd';
                else
                    continue
                end
                esttask_folder = fullfile(estderiv_subj_dir, ttag);
                
                A = []; % structure with SPM fields to build the t-contrasts
                
                A.spmmat = {[esttask_folder '/SPM.mat']};
                
                for c=1:length(contrasts)
                    A.consess{c}.tcon.name = contrasts{c,1};
                    A.consess{c}.tcon.weights = contrasts{c,2};
                    A.consess{c}.tcon.sessrep = 'replsc';
                end
                % Delete existing contrasts
                A.delete = 1; % 1 yes, 0 no
                
                matlabbatch{1}.spm.stats.con=A;
                spm_jobman('run',matlabbatch);
            end % tk (tasks)
        end % s (subject)
        
    case 'GLM:run_all'
        % Example usage: msdtb_imana('GLM:run_all')

        % Note: Do not forget to copy tsv files to the server

        msdtb_imana('GLM:grand_design')
        msdtb_imana('GLM:estimate')
        msdtb_imana('GLM: individual_ffx_t') 
        
    case 'GLM:dmtx_unf' 
        % Saves a copy of SPM.mat prepared to be loaded in python       
        
        sn       = subj_id; % subject list
        ses = session_names; 
        vararginoptions(varargin, {'sn', 'ses'})
        
        for s = sn
            estderiv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s}, est_dir);
            
            sbj_number = str2double((extractAfter(subj_str{s},'sub-')));
            subsess = cellstr(sessmap.(['sub' num2str(sbj_number, ...
                '%02d')]));
            
            % loop over sessions
            for smap = ses
                % sesstag = sessnum{find(contains(subsess, smap))};
                smapstr = replace(smap{1}, '-', '');
                est_sess_dir = fullfile(estderiv_subj_dir, ...
                    ['ses-' smapstr]);
                
                % get the list of runs for the current session
                listing = dir(est_sess_dir);
                listitems = {listing.name};
                runtags = listitems(startsWith(listitems, 'run-'));
                
                for rn = 1:length(runtags)
                    estimates_dir = fullfile(est_sess_dir, ...
                        char(runtags(rn)));
                    if any(size(dir([estimates_dir '/design_matrix_unf.mat']),1))
                        delete([estimates_dir '/design_matrix_unf.mat'])
                    end
                    load(fullfile(estimates_dir, 'SPM.mat'));
                    X = SPM.xX.xKXs.X;
                    save(fullfile(estimates_dir, ...
                        'design_matrix.mat'), 'X');

                end % rn (runtags)
            end % ss (sessions)
        end % s (sn)  

    case 'GLM:F_contrast' % make F contrast
        %%% Calculating contrast images.
        % 'SPM_light' is created in this step (xVi is removed as it slows
        % down code for FAST GLM).
        % Example1: nishimoto_imana('GLM:F_contrast', 'sn', 1, 'glm', 1, 'ses', 1)
        
        sn       = returnSubjs;   %% list of subjects
        ses      = 2;
        glm      = 1;             %% The glm number
        
        vararginoptions(varargin, {'sn', 'glm', 'ses'})
        
        for s = sn
            % subject name
            fprintf('- Calculating F contrast for %s %s \n', ses_str{ses}, subj_str{s});
            glm_dir = fullfile(base_dir, subj_str{s}, est_dir, sprintf('glm%02d', glm), ses_str{ses});
            load(fullfile(glm_dir, 'SPM.mat'))
            
            SPM  = rmfield(SPM,'xCon');
            cd(fullfile(glm_dir))
            T    = load(fullfile(glm_dir, sprintf('%s_%s_reginfo.tsv', subj_str{s}, ses_str{ses})));
            
            % F contrast
            numConds = max(T.cond); 
            con = zeros(numConds,size(SPM.xX.X,2));
            for i=1:numConds
                con(i,T.cond==i)=1-1/numConds;
                con(i,T.cond>0 & T.cond~=i)=-1/numConds;
            end
            
            SPM.xCon(1) = spm_FcUtil('Set',name, 'F', 'c',con',SPM.xX.xKXs);
            SPM = spm_contrasts(SPM,1:length(SPM.xCon));
            save('SPM.mat', 'SPM','-v7.3');
            SPM = rmfield(SPM,'xVi'); % 'xVi' take up a lot of space and slows down code!
            save(fullfile(glmDir, subj_name, 'SPM_light.mat'), 'SPM');

        end % sn 
         
    case 'SURF:reconall' % Freesurfer reconall routine
        % Calls recon-all, which performs, all of the
        % FreeSurfer cortical reconstruction process
        % Example usage: msdtb_imana('SURF:reconall', 'sn', 1)
        
        sn   = subj_id; % subject list
        
        vararginoptions(varargin, {'sn'});
        % set freesurfer directory
        subj_fs_dir = fullfile(base_dir, fs_dir);
        
        % Parent dir of anatomical images    
        for s = sn
            fprintf('- recon-all %s\n', subj_str{s});
                        % Get the directory of subjects anatomical
            deriv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s});
            subj_anatderiv_dir = fullfile(deriv_subj_dir, 'ses-01/anat');
            
            % Get the name of the anatomical image
            anat_name = sprintf('%s_T1w.nii', subj_str{s});
            
            freesurfer_reconall(subj_fs_dir, subj_str{s}, ...
                fullfile(subj_anatderiv_dir, anat_name));
        end % s (sn)

    case 'SURF:xhemireg' % Cross-register surfaces left / right hem
        % surface-based interhemispheric registration
        % example: msdtb_imana('SURF:xhemireg', 'sn', [1, 2, 3, 4, 5])
        
        sn   = subj_id; % list of subjects

        vararginoptions(varargin, {'sn'})
        
        % set freesurfer directory
        fs_dir = fullfile(base_dir, 'surfaceFreeSurfer');
        
        for s = sn
            fprintf('- xhemiregl %s\n', subj_str{s});
            freesurfer_registerXhem(subj_str(s), fs_dir,'hemisphere', ...
                [1 2]); % For debug... [1 2] orig
        end % s (sn)
        
    case 'SURF:map_ico' % Align to the new atlas surface (map icosahedron)
        % Resamples a registered subject surface to a regular isocahedron
        % This allows things to happen in atlas space - each vertex number
        % corresponds exactly to an anatomical location
        % Makes a new folder, called ['x' subj] that contains the 
        % remapped subject
        % Uses function mri_surf2surf
        % mri_surf2surf: resamples one cortical surface onto another
        % Example usage: 
        % msdtb_imana('SURF:map_ico', 'sn', [1, 2, 3, 4, 5, 6])
        
        sn = subj_id; % list of subjects
        
        vararginoptions(varargin, {'sn'});
        
        % set freesurfer directory
        fs_dir = fullfile(base_dir, 'surfaceFreeSurfer');
        for s = sn
            fprintf('- map_ico %s\n', subj_str{s});
            freesurfer_mapicosahedron_xhem(subj_str{s}, fs_dir, ...
                'smoothing',1,'hemisphere',[1, 2]);
        end % s (sn)
        
    case 'SURF:fs2wb' % Resampling subject from freesurfer fsaverage to fs_LR
        % Example usage: ibc_imana('SURF:fs2wb', 'sn', [1], 'res', 32)
        
        sn   = subj_id; % list of subjects
        res  = 32;          % resolution of the atlas. options are: 32, 164
        hemi = [1, 2];      % list of hemispheres
        
        vararginoptions(varargin, {'sn', 'res', 'hemi'});
        
        % set freesurfer directory
        fs_dir = fullfile(base_dir, 'surfaceFreeSurfer');
        % set output directory
        wb_subj_dir  = fullfile(base_dir, wb_dir, 'data');
        
        for s = sn 
            fprintf('- fs2wb %s\n', subj_str{s});
            surf_resliceFS2WB(subj_str{s}, fs_dir, ...
                wb_subj_dir, 'hemisphere', hemi, 'resolution', ...
                sprintf('%dk', res))
        end % s (sn)

    case 'SURF:run_all'
        % Example usage: msdtb_imana('SURF:run_all')

        msdtb_imana('SURF:reconall')
        msdtb_imana('SURF:xhemireg')
        msdtb_imana('SURF:map_ico')
        cd(fileparts(mfilename('fullpath'))) % Go to the folder of script
        msdtb_imana('SURF:fs2wb', 'res', 32)        
        
    case 'SUIT:isolate_segment'  
        % Segment cerebellum into grey and white matter
        % Example usage: ibc_imana('SUIT:isolate_segment', 'sn', 1);
        
        sn = subj_id;
        
        vararginoptions(varargin, {'sn'});
        
        for s = sn
            fprintf('- Isolate and segment the cerebellum for %s\n', ...
                subj_str{s})
            spm_jobman('initcfg')
            
            % Get the directory of subjects anatomical
            raw_subj_dir = fullfile(base_dir, raw_dir, subj_str{s});
            anat_subj_dir = fullfile(raw_subj_dir, anat_dir);

            % Get the name of the anatomical image
            anat_name = sprintf('%s_T1w.nii', subj_str{s});
            % Define suit folder
            suit_dir = fullfile(raw_subj_dir, 'suit');
            % Create suit folder if it does not exist
            if ~exist(suit_dir, 'dir')
                mkdir (suit_dir)
            end
            
            % Copy T1w_lpi file to suit-anat folder
            source = fullfile(anat_subj_dir, anat_name);
            dest   = fullfile(suit_dir, anat_name);           
            copyfile(source, dest);
            
            % go to subject directory for suit and isolate segment
            suit_isolate_seg({dest}, 'keeptempfiles', 1);
        end % s (sn)

    case 'SUIT:normalise_dartel' % SUIT normalization using dartel
        % LAUNCH SPM FMRI BEFORE RUNNING!!!!!
        % example usage: ibc_imana('SUIT:normalise_dartel')
        sn = subj_id; %subjNum
        vararginoptions(varargin, 'sn');
        
        for s = sn
            suit_subj_dir = fullfile(base_dir, raw_dir, subj_str{s}, ...
                'suit');

            job.subjND.gray       = {fullfile(suit_subj_dir, ...
                sprintf('c_%s_T1w_seg1.nii', subj_str{s}))};
            job.subjND.white      = {fullfile(suit_subj_dir, ...
                sprintf('c_%s_T1w_seg2.nii', subj_str{s}))};
            job.subjND.isolation  = {fullfile(suit_subj_dir, ...
                sprintf('c_%s_T1w_pcereb_corr.nii', subj_str{s}))};
            suit_normalize_dartel(job);

        end % s (subjects)

    case 'SUIT:save_dartel_def'    
        % Saves the dartel flow field as a deformation file.
        sn = subj_id; %subjNum
        vararginoptions(varargin, 'sn');
        
        for s = sn
            suit_subj_dir = fullfile(base_dir, raw_dir, subj_str{s}, ...
                'suit');
            cd(suit_subj_dir);
            anat_name = sprintf('%s_T1w', subj_str{s});
            suit_save_darteldef(anat_name);
        end

    case 'SUIT:cerebellum_graymask'    
        % Conjunction of the pcereb_corr  and the cerebellar gray matter 
        % mask (in functional space) thresholded to 0.2

        sn = subj_id; %subjNum
        vararginoptions(varargin, 'sn');
        
        for s = sn
            suit_subj_dir = fullfile(base_dir, raw_dir, subj_str{s}, ...
                'suit');
            masks = {};
            % First image need to be in functional space to ensure 
            % correct space 
            masks{1} = fullfile(base_dir,'derivatives',subj_str{s},'estimates',...
                        'ses-archi','run-01','mask.nii')
            masks{2} = fullfile(suit_subj_dir, ...
                sprintf('c_%s_T1w_pcereb_corr.nii', subj_str{s}));
            masks{3} = fullfile(suit_subj_dir, ...
                sprintf('c_%s_T1w_seg1.nii', subj_str{s}));
            final_mask = fullfile(suit_subj_dir, 'maskbrainSUITGrey.nii');
            spm_imcalc(masks, final_mask, 'i2>0.1 & i3>0.1');
        end

    case 'SUIT:reslice' % Reslice stuff into suit space 
        % run the case with 'anatomical' to check the suit normalization
        % Example usage: nishimoto_imana('SUIT:reslice','type','ResMS', 'mask', 'pcereb_corr')
        % make sure that you reslice into 2mm^3 resolution
        
        sn   = subj_id;
        type = 'anatomical';  % 'betas' or 'con' or 'ResMS' or 'cerebellarGrey' or 'anatomical'
        mask = 'pcereb_corr'; % 'cereb_prob_corr_grey' or 'cereb_prob_corr' or 'dentate_mask' or 'pcereb'
        glm  = 1;             % glm number. Used for reslicing betas and contrasts 
        
        vararginoptions(varargin, {'sn', 'type', 'mask', 'glm'})
        
        for s = sn
            suit_dir = fullfile(base_dir, subj_str{s}, 'suit', 'anat');
            switch type
                case 'anatomical'
                    subj_dir = suit_dir;
                    % Get the name of the anatpmical image
                    files2map = sprintf('%s_T1w_lpi.nii', subj_str{s});
                    
                    job.subj.resample = {sprintf('%s,1', files2map)};                 
                case 'con'
                    subj_dir     = fullfile(base_dir, subj_str{s}, 'estimates', sprintf('glm%02d', glm), 'ses-01');
                    out_dir      = fullfile(base_dir, subj_str{s}, 'suit',sprintf('glm%02d',glm));
                    files2map    = dir(fullfile(subj_dir,sprintf('*con*'))); % find images to be resliced
                    
                    job.subj.resample     = {files2map.name};
                case 'ResMS'
                    subj_dir     = fullfile(base_dir, subj_str{s}, 'estimates', sprintf('glm%02d', glm), 'ses-01');
                    out_dir      = fullfile(base_dir, subj_str{s}, 'suit',sprintf('glm%02d',glm));
                    files2map    = dir(fullfile(subj_dir,sprintf('*ResMS*'))); % find images to be resliced
                    
                    job.subj.resample     = {files2map.name};
                    
            end
            
            cd(subj_dir);
            job.subj.affineTr  = {fullfile(suit_dir,sprintf('Affine_c_%s_T1w_lpi_seg1.mat', subj_str{s}))};
            job.subj.flowfield = {fullfile(suit_dir,sprintf('u_a_c_%s_T1w_lpi_seg1.nii', subj_str{s}))};
            job.subj.mask      = {fullfile(suit_dir, sprintf('c_%s_T1w_lpi_%s.nii', subj_str{s}, mask))};
            job.vox            = [2 2 2];
            suit_reslice_dartel(job);
            
            if ~strcmp(type,'anatomical')
                source=fullfile(subj_dir, '*wd*');
                dircheck(fullfile(out_dir));
                movefile(source,out_dir);
            end
            fprintf('- Resliced %s %s to SUIT\n', subj_str{s}, type);
        end % s (subject)
        
    case 'SUIT:map2flat' % Creates flatmaps
    % this case also creates group average for each task
    % First RUN nishimoto_iman
    sn    = subj_id;
    glm   = 1;
    type  = 'con'; % type of the image to be mapped to flatmap
    baseline = 'rest';
    group = 1;     % if this flag is set to 1, it bypasses the step that creates gifti files for each subject

    vararginoptions(varargin, {'sn', 'glm', 'type', 'group', 'baseline'});

    for s = sn
        suit_dir = fullfile(base_dir, subj_str{s}, 'suit', sprintf('glm%02d', glm));
        % file containing the giftis for the current subject
        filename{s} = fullfile(suit_dir, sprintf('%s.w%s.cerebellum.func.gii', subj_str{s}, type));

        switch type
            case 'con'
                % get all the contrasts
                files2map = dir(fullfile(suit_dir, sprintf('wd%s*%s', type, baseline)));
        end

        % map the files
        %%% get file paths and names of contrasts
        for i= 1:length(files2map)
            name{i} = fullfile(suit_dir, files2map(i).name);
            column_names{i} = files2map(i).name(7:end-4);
        end % i (contrast names)

        % if the subj specific gifti files have not been created, then
        % create them
        if group ~=1
            maps = suit_map2surf(name, 'stats', 'nanmean' );

            % map ResMS (will be used for univariate prewhitening)
            mapResMS = suit_map2surf(fullfile(suit_dir, 'wdResMS.nii'), 'stats', 'nanmean');
            Gres = surf_makeFuncGifti(mapResMS,'anatomicalStruct', 'Cerebellum', 'columnNames', {'ResMS'});

            % do univariate prewhitening
            data    = bsxfun(@rdivide, maps, mapResMS);

            % create one single gifti file containing all the contrasts
            data(:, length(files2map)+1) = Gres.cdata;
            column_names{length(files2map)+1} = 'ResMS';
            G = surf_makeFuncGifti(data,'anatomicalStruct', 'Cerebellum', 'columnNames', column_names);

            % save the single gifti file
            %%% a cell array containing the filenames.
            %%% will be used in creating group maps and summary
            save(G, filename{s});
            fprintf('- Done %s %s map2flat\n', subj_str{s}, type);                
        end % if not group  
    end % s (subject)

    % create group and group summary
    if group == 1
        suit_group_dir = fullfile(base_dir, 'suit', sprintf('glm%02d', glm), 'group');dircheck(suit_group_dir);
        cd(suit_group_dir)
        summaryname     = fullfile(suit_group_dir,sprintf('wgroup.%s.glm%02d.func.gii', type, glm));
        surf_groupGiftis(filename, 'groupsummary', summaryname, 'replaceNaNs', 1, 'outcolnames', subj_str);
    end

end

end

% Checking for directories and creating if not exist
function dircheck(dir)
if ~exist(dir,'dir')
    warning('%s doesn''t exist. Creating one now. You''re welcome! \n',dir);
    mkdir(dir);
end
end