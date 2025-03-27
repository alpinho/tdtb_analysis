function [ output_args ] = msdtb_imana( what, varargin )

% Go to the folder of script
cd(fileparts(mfilename('fullpath')))

% Add directory path of current script to the path
script_fullpath = mfilename('fullpath');
script_dirpath = script_fullpath(1:end-12);
addpath(script_dirpath);

% %============================================================================
% PATH DEFINITIONS

% Add dependencies to path
if isdir('/srv/diedrichsen/data')
    homedir = '/home/ROBARTS/agrilopi';
    workdir='/srv/diedrichsen/data';   
    localscratch = '/localscratch';
    addpath(sprintf('%s/../matlab/dataframe', workdir));
    addpath(sprintf('%s/../matlab/imaging/tools', workdir));
    addpath(sprintf('%s/../matlab/imaging/rwls', workdir));
    addpath(sprintf('%s/spm12', homedir));
    addpath(sprintf('%s/freesurfer', homedir));
elseif isdir('/home/analu/diedrichsen_data/data')
    homedir = '/home/analu';
    workdir='/home/analu/diedrichsen_data/data';
    localscratch = '/home/analu/localscratch';
    addpath(sprintf('%s/software/spm12', homedir));
    addpath(sprintf('%s/software/freesurfer', homedir));
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

% Initialize SPM defaults
spm('defaults', 'FMRI');
global defaults

% Set maxmem to 4 GB
defaults.stats.maxmem = 16 * 1024^3;  % 16 GB in bytes

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

% List of all subjects
% subj_n = [3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, ...
%     28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47];

% List of all subjects but pilot
subj_n = [3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, ...
    28, 29, 32, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47];

% subj_n = [3];

subj_id = 1:length(subj_n);
for s=subj_id
    subj_str{s} = ['sub-' num2str(subj_n(s), '%02d')];
end

% AC coordinates (non-symmetric ones)
% loc_AC = {
%           [1.0 23.6 -49.3], ...       %sub-03
%           [1.1 28.6 -21.1], ...       %sub-04
%           [-3.3 27.3 -54.7], ...      %sub-07
%           [1.8 28.8 -45.5], ...       %sub-08
%           [1.5 31.6 -51.10], ...      %sub-10
%           [-1.6 20.4 -42.8], ...      %sub-11
%           [-0.4 29.2 -21.9], ...      %sub-12
%           [3.3 34.6 -29.3], ...       %sub-13
%           [1.2 30.6 -46.4], ...       %sub-14
%           [5.4 17.2 23.4], ...        %sub-15
%           [0.7 22.8 -36.9], ...       %sub-16   
%           [2.2 29.3 19.0], ...        %sub-18
%           [1.8 30.4 -24.3], ...       %sub-20
%           [0.9 21.8 -10.5], ...       %sub-21
%           [0.9 32.4 -25.0], ...       %sub-22
%           [-2.5 25.9 -37.6], ...      %sub-23
%           [5.7 26.9 -13.7], ...       %sub-26
%           [3.5 25.9 -36.0], ...       %sub-28
%           [1.6 20.0 -1.1], ...        %sub-29
%           [4.5 22.3 -21.5], ...       %sub-32
%           [2.1 20.4 -11.5], ...       %sub-34
%           [2.8 23.7 -20.7], ...       %sub-35
%           [5.2 19.3 37.3], ...        %sub-38
%           [-3.0 28.9 -41.8], ...      %sub-39
%           [-0.5 20.0 7.2], ...        %sub-40
%           [2.1 20.3 -19.1], ...       %sub-41
%           [1.8 30.7 -62.8], ...       %sub-42
%           [0.8 35.2 -34.3], ...       %sub-43
%           [-1.3 22.4 -15.6], ...      %sub-44
%           [1.3 18.7 -21.7], ...       %sub-45
%           [-0.3 19.5 17.6], ...       %sub-46
%           [4.8 32.2 -10.0], ...       %sub-47
%           };

loc_AC = {
          [0.9 21.8 -10.5], ...         %sub-21
          [5.7 26.9 -13.7], ...         %sub-26
          };

numDummys = 0;

contrasts = {'Encoding', [1 1 1 1 0]; ...                            %1
             'Auditory Encoding', [1 1 0 0 0]; ...                   %2
             'Visual Encoding', [0 0 1 1 0]; ...                     %3
             'Auditory vs Visual Encoding', [1 1 -1 -1 0]; ...       %4
             'Visual vs Auditory Encoding', [-1 -1 1 1 0]; ...       %5
             'Beat', [1 0 1 0 0]; ...                                %6
             'Interval', [0 1 0 1 0]; ...                            %7
             'Beat vs Interval', [1 -1 1 -1 0]; ...                  %8
             'Interval vs Beat', [-1 1 -1 1 0]; ...                  %9
             'Auditory Beat', [1 0 0 0 0]; ...                       %10
             'Auditory Interval', [0 1 0 0 0]; ...                   %11
             'Auditory Beat vs Auditory Interval', [1 -1 0 0 0]; ... %12
             'Auditory Interval vs Auditory Beat', [-1 1 0 0 0]; ... %13
             'Visual Beat', [0 0 1 0 0]; ...                         %14
             'Visual Interval', [0 0 0 1 0]; ...                     %15
             'Visual Beat vs Visual Interval', [0 0 1 -1 0]; ...     %16            
             'Visual Interval vs Visual Beat', [0 0 -1 1 0]; ...     %17
             'Decision', [0 0 0 0 1]; ...                            %18
             };
         
contrasts_time = {'Encoding', [1 0 1 0 1 0 1 0 0 0] ; ...                           %1
                  'Auditory Encoding', [1 0 1 0 0 0 0 0 0 0]; ...                   %2
                  'Visual Encoding', [0 0 0 0 1 0 1 0 0 0]; ...                     %3
                  'Auditory vs Visual Encoding', [1 0 1 0 -1 0 -1 0 0 0]; ...       %4
                  'Visual vs Auditory Encoding', [-1 0 -1 0 1 0 1 0 0 0]; ...       %5
                  'Beat', [1 0 0 0 1 0 0 0 0 0]; ...                                %6
                  'Interval', [0 0 1 0 0 0 1 0 0 0]; ...                            %7
                  'Beat vs Interval', [1 0 -1 0 1 0 -1 0 0 0]; ...                  %8
                  'Interval vs Beat', [-1 0 1 0 -1 0 1 0 0 0]; ...                  %9
                  'Auditory Beat', [1 0 0 0 0 0 0 0 0 0]; ...                       %10
                  'Auditory Interval', [0 0 1 0 0 0 0 0 0 0]; ...                   %11
                  'Auditory Beat vs Auditory Interval', [1 0 -1 0 0 0 0 0 0 0]; ... %12
                  'Auditory Interval vs Auditory Beat', [-1 0 1 0 0 0 0 0 0 0]; ... %13
                  'Visual Beat', [0 0 0 0 1 0 0 0 0 0]; ...                         %14
                  'Visual Interval', [0 0 0 0 0 0 1 0 0 0]; ...                     %15
                  'Visual Beat vs Visual Interval', [0 0 0 0 1 0 -1 0 0 0]; ...     %16            
                  'Visual Interval vs Visual Beat', [0 0 0 0 -1 0 1 0 0 0]; ...     %17
                  'Decision', [0 0 0 0 0 0 0 0 1 0]; ...                            %18
                  };
              
contrasts_timedisp = {'Encoding', [1 0 0 1 0 0 1 0 0 1 0 0 0 0 0]; ...                            %1
                      'Auditory Encoding', [1 0 0 1 0 0 0 0 0 0 0 0 0 0 0]; ...                   %2
                      'Visual Encoding', [0 0 0 0 0 0 1 0 0 1 0 0 0 0 0]; ...                     %3
                      'Auditory vs Visual Encoding', [1 0 0 1 0 0 -1 0 0 -1 0 0 0 0 0]; ...       %4
                      'Visual vs Auditory Encoding', [-1 0 0 -1 0 0 1 0 0 1 0 0 0 0 0]; ...       %5
                      'Beat', [1 0 0 0 0 0 1 0 0 0 0 0 0 0 0]; ...                                %6
                      'Interval', [0 0 0 1 0 0 0 0 0 1 0 0 0 0 0]; ...                            %7
                      'Beat vs Interval', [1 0 0 -1 0 0 1 0 0 -1 0 0 0 0 0]; ...                  %8
                      'Interval vs Beat', [-1 0 0 1 0 0 -1 0 0 1 0 0 0 0 0]; ...                  %9
                      'Auditory Beat', [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; ...                       %10
                      'Auditory Interval', [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]; ...                   %11
                      'Auditory Beat vs Auditory Interval', [1 0 0 -1 0 0 0 0 0 0 0 0 0 0 0]; ... %12
                      'Auditory Interval vs Auditory Beat', [-1 0 0 1 0 0 0 0 0 0 0 0 0 0 0]; ... %13
                      'Visual Beat', [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]; ...                         %14
                      'Visual Interval', [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]; ...                     %15
                      'Visual Beat vs Visual Interval', [0 0 0 0 0 0 1 0 0 -1 0 0 0 0 0]; ...     %16            
                      'Visual Interval vs Visual Beat', [0 0 0 0 0 0 -1 0 0 1 0 0 0 0 0]; ...     %17
                      'Decision', [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]; ...                            %18
                      };
         
contrasts_random = {'Encoding', [1 1 1 1 1 1 0]; ...                              %1
                    'Auditory Encoding', [1 1 1 0 0 0 0]; ...                     %2
                    'Visual Encoding', [0 0 0 1 1 1 0]; ...                       %3
                    'Auditory vs Visual Encoding', [1 1 1 -1 -1 -1 0]; ...        %4
                    'Visual vs Auditory Encoding', [-1 -1 -1 1 1 1 0]; ...        %5
                    'Beat', [1 0 0 1 0 0 0]; ...                                  %6
                    'Interval', [0 1 0 0 1 0 0]; ...                              %7
                    'Random', [0 0 1 0 0 1 0]; ...                                %8
                    'Beat vs Interval', [1 -1 0 1 -1 0 0]; ...                    %9
                    'Interval vs Beat', [-1 1 0 -1 1 0 0]; ...                    %10
                    'Beat vs Random', [1 0 -1 1 0 -1 0]; ...                      %11
                    'Random vs Beat', [-1 0 1 -1 0 1 0]; ...                      %12
                    'Interval vs Random', [0 1 -1 0 1 -1 0]; ...                  %13
                    'Random vs Interval', [0 -1 1 0 -1 1 0]; ...                  %14
                    'Auditory Beat', [1 0 0 0 0 0 0]; ...                         %15
                    'Auditory Interval', [0 1 0 0 0 0 0]; ...                     %16
                    'Auditory Random', [0 0 1 0 0 0 0]; ...                       %17
                    'Auditory Beat vs Auditory Interval', [1 -1 0 0 0 0 0]; ...   %18
                    'Auditory Interval vs Auditory Beat', [-1 1 0 0 0 0 0]; ...   %19
                    'Auditory Beat vs Auditory Random', [1 0 -1 0 0 0 0]; ...     %20
                    'Auditory Random vs Auditory Beat', [-1 0 1 0 0 0 0]; ...     %21
                    'Auditory Interval vs Auditory Random', [0 1 -1 0 0 0 0]; ... %22
                    'Auditory Random vs Auditory Interval', [0 -1 1 0 0 0 0]; ... %23
                    'Visual Beat', [0 0 0 1 0 0 0]; ...                           %24
                    'Visual Interval', [0 0 0 0 1 0 0]; ...                       %25
                    'Visual Random', [0 0 0 0 0 1 0]; ...                         %26
                    'Visual Beat vs Visual Interval', [0 0 0 1 -1 0 0]; ...       %27
                    'Visual Interval vs Visual Beat', [0 0 0 -1 1 0 0]; ...       %28
                    'Visual Beat vs Visual Random', [0 0 0 1 0 -1 0]; ...         %29
                    'Visual Random vs Visual Beat', [0 0 0 -1 0 1 0]; ...         %30                    
                    'Visual Interval vs Visual Random', [0 0 0 0 1 -1 0]; ...     %31
                    'Visual Random vs Visual Interval', [0 0 0 0 -1 1 0]; ...     %32
                    'Decision', [0 0 0 0 0 0 1]; ...                              %33
                    };
                
contrasts_random_time = {'Encoding', [1 0 1 0 1 0 1 0 1 0 1 0 0 0]; ...                              %1
                         'Auditory Encoding', [1 0 1 0 1 0 0 0 0 0 0 0 0 0]; ...                     %2
                         'Visual Encoding', [0 0 0 0 0 0 1 0 1 0 1 0 0 0]; ...                       %3
                         'Auditory vs Visual Encoding', [1 0 1 0 1 0 -1 0 -1 0 -1 0 0 0]; ...        %4
                         'Visual vs Auditory Encoding', [-1 0 -1 0 -1 0 1 0 1 0 1 0 0 0]; ...        %5
                         'Beat', [1 0 0 0 0 0 1 0 0 0 0 0 0 0]; ...                                  %6
                         'Interval', [0 0 1 0 0 0 0 0 1 0 0 0 0 0]; ...                              %7
                         'Random', [0 0 0 0 1 0 0 0 0 0 1 0 0 0]; ...                                %8
                         'Beat vs Interval', [1 0 -1 0 0 0 1 0 -1 0 0 0 0 0]; ...                    %9
                         'Interval vs Beat', [-1 0 1 0 0 0 -1 0 1 0 0 0 0 0]; ...                    %10
                         'Beat vs Random', [1 0 0 0 -1 0 1 0 0 0 -1 0 0 0]; ...                      %11
                         'Random vs Beat', [-1 0 0 0 1 0 -1 0 0 0 1 0 0 0]; ...                      %12
                         'Interval vs Random', [0 0 1 0 -1 0 0 0 1 0 -1 0 0 0]; ...                  %13
                         'Random vs Interval', [0 0 -1 0 1 0 0 0 -1 0 1 0 0 0]; ...                  %14
                         'Auditory Beat', [1 0 0 0 0 0 0 0 0 0 0 0 0 0]; ...                         %15
                         'Auditory Interval', [0 0 1 0 0 0 0 0 0 0 0 0 0 0]; ...                     %16
                         'Auditory Random', [0 0 0 0 1 0 0 0 0 0 0 0 0 0]; ...                       %17
                         'Auditory Beat vs Auditory Interval', [1 0 -1 0 0 0 0 0 0 0 0 0 0 0]; ...   %18
                         'Auditory Interval vs Auditory Beat', [-1 0 1 0 0 0 0 0 0 0 0 0 0 0]; ...   %19
                         'Auditory Beat vs Auditory Random', [1 0 0 0 -1 0 0 0 0 0 0 0 0 0]; ...     %20
                         'Auditory Random vs Auditory Beat', [-1 0 0 0 1 0 0 0 0 0 0 0 0 0]; ...     %21
                         'Auditory Interval vs Auditory Random', [0 0 1 0 -1 0 0 0 0 0 0 0 0 0]; ... %22
                         'Auditory Random vs Auditory Interval', [0 0 -1 0 1 0 0 0 0 0 0 0 0 0]; ... %23
                         'Visual Beat', [0 0 0 0 0 0 1 0 0 0 0 0 0 0]; ...                           %24
                         'Visual Interval', [0 0 0 0 0 0 0 0 1 0 0 0 0 0]; ...                       %25
                         'Visual Random', [0 0 0 0 0 0 0 0 0 0 1 0 0 0]; ...                         %26
                         'Visual Beat vs Visual Interval', [0 0 0 0 0 0 1 0 -1 0 0 0 0 0]; ...       %27
                         'Visual Interval vs Visual Beat', [0 0 0 0 0 0 -1 0 1 0 0 0 0 0]; ...       %28
                         'Visual Beat vs Visual Random', [0 0 0 0 0 0 1 0 0 0 -1 0 0 0]; ...         %29
                         'Visual Random vs Visual Beat', [0 0 0 0 0 0 -1 0 0 0 1 0 0 0]; ...         %30                    
                         'Visual Interval vs Visual Random', [0 0 0 0 0 0 0 0 1 0 -1 0 0 0]; ...     %31
                         'Visual Random vs Visual Interval', [0 0 0 0 0 0 0 0 -1 0 1 0 0 0]; ...     %32
                         'Decision', [0 0 0 0 0 0 0 0 0 0 0 0 1 0]; ...                              %33
                        };
                    
contrasts_random_timedisp = {'Encoding', [1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 0 0 0]; ...                              %1
                             'Auditory Encoding', [1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; ...                     %2
                             'Visual Encoding', [0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0]; ...                       %3
                             'Auditory vs Visual Encoding', [1 0 0 1 0 0 1 0 0 -1 0 0 -1 0 0 -1 0 0 0 0 0]; ...        %4
                             'Visual vs Auditory Encoding', [-1 0 0 -1 0 0 -1 0 0 1 0 0 1 0 0 1 0 0 0 0 0]; ...        %5
                             'Beat', [1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]; ...                                  %6
                             'Interval', [0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]; ...                              %7
                             'Random', [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0]; ...                                %8
                             'Beat vs Interval', [1 0 0 -1 0 0 0 0 0 1 0 0 -1 0 0 0 0 0 0 0 0]; ...                    %9
                             'Interval vs Beat', [-1 0 0 1 0 0 0 0 0 -1 0 0 1 0 0 0 0 0 0 0 0]; ...                    %10
                             'Beat vs Random', [1 0 0 0 0 0 -1 0 0 1 0 0 0 0 0 -1 0 0 0 0 0]; ...                      %11
                             'Random vs Beat', [-1 0 0 0 0 0 1 0 0 -1 0 0 0 0 0 1 0 0 0 0 0]; ...                      %12
                             'Interval vs Random', [0 0 0 1 0 0 -1 0 0 0 0 0 1 0 0 -1 0 0 0 0 0]; ...                  %13
                             'Random vs Interval', [0 0 0 -1 0 0 1 0 0 0 0 0 -1 0 0 1 0 0 0 0 0]; ...                  %14
                             'Auditory Beat', [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; ...                         %15
                             'Auditory Interval', [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; ...                     %16
                             'Auditory Random', [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; ...                       %17
                             'Auditory Beat vs Auditory Interval', [1 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; ...   %18
                             'Auditory Interval vs Auditory Beat', [-1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; ...   %19
                             'Auditory Beat vs Auditory Random', [1 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; ...     %20
                             'Auditory Random vs Auditory Beat', [-1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; ...     %21
                             'Auditory Interval vs Auditory Random', [0 0 0 1 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; ... %22
                             'Auditory Random vs Auditory Interval', [0 0 0 -1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; ... %23
                             'Visual Beat', [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]; ...                           %24
                             'Visual Interval', [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]; ...                       %25
                             'Visual Random', [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]; ...                         %26
                             'Visual Beat vs Visual Interval', [0 0 0 0 0 0 0 0 0 1 0 0 -1 0 0 0 0 0 0 0 0]; ...       %27
                             'Visual Interval vs Visual Beat', [0 0 0 0 0 0 0 0 0 -1 0 0 1 0 0 0 0 0 0 0 0]; ...       %28
                             'Visual Beat vs Visual Random', [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 -1 0 0 0 0 0]; ...         %29
                             'Visual Random vs Visual Beat', [0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 1 0 0 0 0 0]; ...         %30                    
                             'Visual Interval vs Visual Random', [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 -1 0 0 0 0 0]; ...     %31
                             'Visual Random vs Visual Interval', [0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 1 0 0 0 0 0]; ...     %32
                             'Decision', [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]; ...                              %33
                             };

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
            gunzip(gz_source, localscratch);
            gunz_source = fullfile(...
                localscratch, sprintf('%s.nii', anatr_name));
            dest = fullfile(subj_anatderiv_dir, ...
                sprintf('%s.nii', anatd_name));
            spmj_reslice_LPI(gunz_source, 'name', dest);
            
            % In the resliced image, set translation to zero
            V               = spm_vol(dest);
            dat             = spm_read_vols(V);
            % V.mat(1:3,4)    = [0 0 0];                     
            spm_write_vol(V,dat);
            
            % Delete unziped raw files from localscratch
            if any(size(dir([localscratch '/*.nii']), 1))
                delete([localscratch '/*.nii'])
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
            V.mat(1:3,4)    = oldOrig - loc_AC{s}.';
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
        
    case 'ANAT:t1_normalization'
        % Example usage: msdtb_imana('ANAT:normalization')
        
        sn       = subj_id; % subject list
        vararginoptions(varargin, {'sn'});
        
        for s = sn
            % Get the directory of subjects anatomical
            deriv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s});
            subj_anatderiv_dir = fullfile(deriv_subj_dir, 'ses-01/anat');
            
            % Get the name of the anatomical image
            anat_name = sprintf('%s_T1w.nii', subj_str{s});
            subj_anatfile = fullfile(subj_anatderiv_dir, anat_name)
                
            % Deformation-Field file
            deffield_file = [subj_anatderiv_dir '/y_' subj_str{s} '_T1w.nii'];
                
            % Apply normalization
            spmja_normalization_write(deffield_file, subj_anatfile, ...
                'voxel_size', [1.0 1.0 1.0])
        end
        
    case 'ANAT:run_all'
        % Example usage: msdtb_imana('ANAT:run_all')
        
        % msdtb_imana('ANAT:reslice_lpi')
        % msdtb_imana('ANAT:center_ac')
        msdtb_imana('ANAT:segment')
        msdtb_imana('ANAT:t1_normalization') 
        % msdtb_imana('GROUP:mean_t1')
        
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
            deriv_subj_dir = fullfile(base_dir, derivatives_dir, subj_str{s});
            anat_subj_dir = fullfile(deriv_subj_dir, anat_dir);

            % Get the name of the anatomical image
            anat_name = sprintf('%s_T1w.nii', subj_str{s});
            % Define suit folder
            suit_dir = fullfile(deriv_subj_dir, 'ses-01/suit');
            % Create suit folder if it does not exist
            if ~exist(suit_dir, 'dir')
                mkdir (suit_dir)
            end
            
            % Copy T1w_lpi file to suit folder
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
            suit_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s}, 'ses-01/suit');

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
            suit_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s}, 'ses-01/suit');
            cd(suit_subj_dir);
            anat_name = sprintf('%s_T1w', subj_str{s});
            suit_save_darteldef(anat_name);
        end
        
    case 'FUNC:realign_estimate'
        % SPM realigns all volumes to the first volume of first run
        % example usage: msdtb_imana('FUNC:realign', 'sn', 1)
        % Updated upstream
        
        spm_figure('GetWin','Graphics'); % create SPM.ps file at the end
        
        sn   = subj_id; % list of subjects
        prefix = '';
        vararginoptions(varargin,{'sn'});
                
        for s = sn
                   
            deriv_subjdir = fullfile(base_dir, derivatives_dir, subj_str{s});
            
            run = {};
            
            [raw_sestag, preproc_sestag, raw_sesrun, preproc_sesrun] = ...
                datamap(subj_str{s});
            
            for ses = 1:length(raw_sestag)
                funcraw_folder = fullfile(base_dir, raw_dir, ...
                    subj_str{s}, ...
                    convertStringsToChars(raw_sestag{ses}), ...
                    func_dir);
                funcderiv_folder = fullfile(base_dir, derivatives_dir, ...
                    subj_str{s}, ...
                    convertStringsToChars(preproc_sestag{ses}), ...
                    func_dir);
                
                rawtag = sprintf('%s_%s', subj_str{s}, ...
                    raw_sestag{ses});                
                preproctag = sprintf('%s_%s', subj_str{s}, ...
                    preproc_sestag{ses});
                
                % Empty localscratch and functional derivatives folder
                if ses == 1
                    folder(localscratch);
                    folder(funcderiv_folder);
                end

                % This 'for loop' routine has to be done in this order, i.e.
                % gunzip in one folder and then rename in another and not in 
                % the opposite direction in order to avoid problems when 
                % renaming files, as for instance in sub-43 ses-03, ntfd run 1 
                % is renamed into run-02 when the original run-02 has not been 
                % renamed yet.
                for r=1:length(raw_sesrun{ses})
                    func_raw_fname = sprintf('%s_%s_bold.nii', ...
                        rawtag, raw_sesrun{ses}{r});
                    func_raw_file = fullfile(funcraw_folder, ...
                        [func_raw_fname '.gz']);
                    func_preproc_fname = sprintf('%s_%s_bold.nii', ...
                        preproctag, preproc_sesrun{ses}{r});
                    
                    % Gunzip EPI in functional derivatives folder
                    gunzip(func_raw_file, funcderiv_folder);

                    % Move gunzipped EPI files to localscratch with ...
                    % renamed session tag whenever necessary
                    movefile(fullfile(funcderiv_folder, func_raw_fname), ...
                        fullfile(localscratch, func_preproc_fname));
                    
                    run{ses}{r} = [...
                        convertStringsToChars(preproc_sestag{ses}), '_', ...
                        convertStringsToChars(preproc_sesrun{ses}{r})];                    
                end % r (length(raw_sesrun{ses}))
            end % ses (ses_id)
            
            run = horzcat(run{:});           
            % Load batch and run spm
            spma_realign_estimate(localscratch, subj_str{s}, run, 1, Inf, ...
                'prefix', prefix);
            
            for ses = 1:length(preproc_sestag)
                % Set path of the derivatives folder
                func_deriv_folder = fullfile(base_dir, derivatives_dir, ...
                    subj_str{s}, preproc_sestag{ses}, func_dir);
                % Move files from localscratch to destination folder
                if ses == 1
                    % Rename and move postscript file 
                    psfiles(func_deriv_folder, 'realignestimate')
                end
                
                % Move motion-param .txt files
                movefile([localscratch '/rp_' subj_str{s} '_' ...
                    convertStringsToChars(preproc_sestag{ses}) ...
                    '_task-*_run-*_bold.txt'], func_deriv_folder);
                
                % Move functional files w/ param estimation in their
                % headers, but not resliced only, as well as all .mat files
                movefile([localscratch '/' subj_str{s} '_' ...
                    convertStringsToChars(preproc_sestag{ses}) ...
                    '_task-*_run-*_bold*'], func_deriv_folder);

            end % ses (ses_id)
            
            % Delete unziped raw files from localscratch
            if any(size(dir([localscratch '/*.nii']), 1))
                delete([localscratch '/*.nii']);
            end
        end % s (sn)
        
    case 'FUNC:calculate_vdm' % Make fieldmap
        
        spm_figure('GetWin','Graphics'); % create SPM .ps file at the end
        
        sn = subj_id;
        magnumber=1;
        epi_prefix = '';
        vararginoptions(varargin,{'sn'});
        for s = sn
            [raw_sestag, preproc_sestag, ~, preproc_sesrun] = ...
                datamap(subj_str{s});
            
            for ses = 1:length(raw_sestag)
                funcderiv_folder = fullfile(base_dir, derivatives_dir, ...
                    subj_str{s}, ...
                    convertStringsToChars(preproc_sestag{ses}), ...
                    func_dir);
                fmapraw_folder = fullfile(base_dir, raw_dir, ...
                    subj_str{s}, ...
                    convertStringsToChars(raw_sestag{ses}), ...
                    fmap_dir);
                fmapderiv_folder = fullfile(base_dir, derivatives_dir, ...
                    subj_str{s}, ...
                    convertStringsToChars(preproc_sestag{ses}), ...
                    fmap_dir);
                
                rawtag = sprintf('%s_%s', subj_str{s}, ...
                    raw_sestag{ses});
                preproctag = sprintf('%s_%s', subj_str{s}, ...
                    preproc_sestag{ses});
                
                mag_fname = sprintf('%s_magnitude1.nii', rawtag);
                phase_fname = sprintf('%s_phasediff.nii', rawtag);
                
                magnitude = fullfile(fmapraw_folder, [mag_fname '.gz']);
                phasediff = fullfile(fmapraw_folder, [phase_fname '.gz']);
                
                % Unzip magnitude and phase files to be loaded by SPM
                gunzip(magnitude, localscratch);
                gunzip(phasediff, localscratch);

                % Create/update destination folders
                folder(fmapderiv_folder);
                
                % Move gunzipped magnitude and phase files to ...
                % ... destination folder and rename session tag ...
                % ... from one to two digits, if that is the case
                movefile(fullfile(localscratch, mag_fname), ...
                    fullfile(fmapderiv_folder, ...
                    sprintf('%s_magnitude1.nii', preproctag)));
                movefile(fullfile(localscratch, phase_fname), ...
                    fullfile(fmapderiv_folder, ...
                    sprintf('%s_phasediff.nii', preproctag)));
 
                % Copy EPI files w/ the realignment already estimated to...
                % ... fmap dir
                for r=1:length(preproc_sesrun{ses})
                    func_files = sprintf(...
                        '%s_%s_bold.nii', preproctag, preproc_sesrun{ses}{r});
                    func_data = fullfile(funcderiv_folder, func_files);
                    
                    % Move files
                    copyfile(func_data, fullfile(fmapderiv_folder, ...
                        sprintf('%s_%s_bold.nii', preproctag, ...
                        preproc_sesrun{ses}{r})));
                end                

                % Load SPM Batch
                if strcmp(subj_str{s}, 'sub-18')
                    spmja_makefieldmap(fmapderiv_folder, preproctag, ...
                        preproc_sesrun{ses}, 'prefix', epi_prefix, ...
                        'regularization', 0.03);
                else
                    spmja_makefieldmap(fmapderiv_folder, preproctag, ...
                        preproc_sesrun{ses}, 'prefix', epi_prefix);
                end
                
                % Add task tag to fieldmap files
                for r=1:length(preproc_sesrun{ses})
                    movefile([fmapderiv_folder '/vdm5_sc' preproctag ...
                        '_phasediff_run' num2str(r, '%d') '.nii'], ...
                        fullfile(fmapderiv_folder, ['vdm5_sc' preproctag ...
                        '_phasediff_' ...
                        preproc_sesrun{ses}{r} '.nii']));
                end
                
                % Rename and move postscript file
                psfiles(fmapderiv_folder, 'fieldmap')
                
                % Delete unziped raw files from localscratch
                if any(size(dir([localscratch '/*.nii']), 1))
                    delete([localscratch '/*.nii']);
                end
                
            end % ses (length(raw_sestag))
        end % s (sn)
        
    case 'FUNC:apply_vdm'
        spm_figure('GetWin','Graphics'); % create SPM .ps file at the end
        
        sn   = subj_id; % list of subjects
        prefix = '';
        vararginoptions(varargin,{'sn'});
        
        for s = sn
                   
            deriv_subjdir = fullfile(base_dir, derivatives_dir, subj_str{s});
            
            run = {};
            
            [~, preproc_sestag, ~, preproc_sesrun] = datamap(subj_str{s});
            
            for ses = 1:length(preproc_sestag)
                funcderiv_folder = fullfile(base_dir, derivatives_dir, ...
                    subj_str{s}, ...
                    convertStringsToChars(preproc_sestag{ses}), ...
                    func_dir);
                fmapderiv_folder = fullfile(deriv_subjdir, ...
                    convertStringsToChars(preproc_sestag{ses}), fmap_dir);
                
                preproctag = sprintf('%s_%s', subj_str{s}, ...
                    preproc_sestag{ses});
                
                % Empty localscratch
                if ses == 1
                    folder(localscratch);
                end

                for r=1:length(preproc_sesrun{ses})
                    % Copy EPI files to localscratch
                    func_fname = sprintf('%s_%s_bold.nii', ...
                        preproctag, preproc_sesrun{ses}{r});
                    func_file = fullfile(funcderiv_folder, func_fname);
                    copyfile(func_file, localscratch);
                    % Copy fieldmap files to localscratch
                    fmap_fname = sprintf('vdm5_sc%s_phasediff_%s.nii', ...
                        preproctag, preproc_sesrun{ses}{r});
                    fmap_file = fullfile(fmapderiv_folder, fmap_fname);
                    copyfile(fmap_file, localscratch);
                    
                    run{ses}{r} = [...
                        convertStringsToChars(preproc_sestag{ses}), '_', ...
                        convertStringsToChars(preproc_sesrun{ses}{r})];                    
                end % r (length(raw_sesrun{ses}))
            end % ses (ses_id)
            
            run = horzcat(run{:});           
            % Load batch and run spm
            spma_applyvdm(localscratch, subj_str{s}, run, 1, Inf);
        
            for ses = 1:length(preproc_sestag)
                func_deriv_folder = fullfile(base_dir, derivatives_dir, ...
                    subj_str{s}, ...
                    convertStringsToChars(preproc_sestag{ses}), ...
                    func_dir);
                % Move files from localscratch to destination folder
                if ses == 1
                    % Delete previous mean-unwarped EPI file from
                    % destination folder
                    delete(fullfile(func_deriv_folder, 'meanu*.nii'))
                    % Move mean-unwarped EPI file
                    movefile([localscratch '/meanu' subj_str{s} ...
                        '_ses-01_task-prod_run-01_bold.nii'], ...
                        func_deriv_folder);
                end
                
                % Delete previous unwarped EPI files from destination folder
                delete(fullfile(func_deriv_folder, 'u*.nii'))

                % Move unwarped images
                movefile([localscratch '/u' subj_str{s} '_' ...
                    convertStringsToChars(preproc_sestag{ses}) ...
                    '_task-*_run-*_bold*'], func_deriv_folder);
            end % ses (ses_id)
            
            % Delete unziped raw files from localscratch
            if any(size(dir([localscratch '/*.nii']), 1))
                delete([localscratch '/*.nii']);
            end
            
        end % s (sn)
        
    case 'FUNC:make_fieldmap' % Make fieldmap
        
        spm_figure('GetWin','Graphics'); % create SPM .ps file at the end
        
        sn = subj_id;
        magnumber=1;
        epi_prefix = '';
        vararginoptions(varargin,{'sn'});
        for s = sn
            [raw_sestag, preproc_sestag, raw_sesrun, preproc_sesrun] = ...
                datamap(subj_str{s});
            
            for ses = 1:length(raw_sestag)
                funcraw_folder = fullfile(base_dir, raw_dir, ...
                    subj_str{s}, ...
                    convertStringsToChars(raw_sestag{ses}), ...
                    func_dir);
                funcderiv_folder = fullfile(base_dir, derivatives_dir, ...
                    subj_str{s}, ...
                    convertStringsToChars(preproc_sestag{ses}), ...
                    func_dir);
                fmapraw_folder = fullfile(base_dir, raw_dir, ...
                    subj_str{s}, ...
                    convertStringsToChars(raw_sestag{ses}), ...
                    fmap_dir);
                fmapderiv_folder = fullfile(base_dir, derivatives_dir, ...
                    subj_str{s}, ...
                    convertStringsToChars(preproc_sestag{ses}), ...
                    fmap_dir);
                
                rawtag = sprintf('%s_%s', subj_str{s}, ...
                    raw_sestag{ses});
                preproctag = sprintf('%s_%s', subj_str{s}, ...
                    preproc_sestag{ses});
                
                mag_fname = sprintf('%s_magnitude1.nii', rawtag);
                phase_fname = sprintf('%s_phasediff.nii', rawtag);
                
                magnitude = fullfile(fmapraw_folder, [mag_fname '.gz']);
                phasediff = fullfile(fmapraw_folder, [phase_fname '.gz']);
                
                % Unzip magnitude and phase files to be loaded by SPM
                gunzip(magnitude, localscratch);
                gunzip(phasediff, localscratch);

                % Create/update destination folders
                folder(funcderiv_folder);
                folder(fmapderiv_folder);
                
                % Move gunzipped magnitude and phase files to ...
                % ... destination folder and rename session tag ...
                % ... from one to two digits, if that is the case
                movefile(fullfile(localscratch, mag_fname), ...
                    fullfile(fmapderiv_folder, ...
                    sprintf('%s_magnitude1.nii', preproctag)));
                movefile(fullfile(localscratch, phase_fname), ...
                    fullfile(fmapderiv_folder, ...
                    sprintf('%s_phasediff.nii', preproctag)));
 
                % Do the same for the EPI files
                for r=1:length(raw_sesrun{ses})
                    func_files = sprintf(...
                        '%s_%s_bold.nii', rawtag, raw_sesrun{ses}{r});
                    func_data = fullfile(funcraw_folder, [func_files '.gz']);
                    
                    % Gunzip EPI files in localscratch to be loaded by SPM
                    gunzip(func_data, localscratch);
                    
                    % Move files to fmapderiv
                    % Rename EPI files if session tag or task-run number
                    % are not equal to their corresponding defaults
                    % Note: This has to be done in this order, i.e.
                    % gunzip in one folder and then rename in another and not in 
                    % the opposite direction in order to avoid problems when 
                    % renaming files, as for instance in sub-43 ses-03, ntfd run 1 
                    % is renamed into run-02 when the original run-02 has not been 
                    % renamed yet.
                    movefile(fullfile(localscratch, func_files), ...
                        fullfile(fmapderiv_folder, ...
                        sprintf('%s_%s_bold.nii', ...
                        preproctag, preproc_sesrun{ses}{r})));
                end                

                % Load SPM Batch
                if strcmp(subj_str{s}, 'sub-18')
                    spmja_makefieldmap(fmapderiv_folder, preproctag, ...
                        preproc_sesrun{ses}, 'prefix', epi_prefix, ...
                        'regularization', 0.03);
                else
                    spmja_makefieldmap(fmapderiv_folder, preproctag, ...
                        preproc_sesrun{ses}, 'prefix', epi_prefix);
                end
                
                % Add task tag to fieldmap files
                for r=1:length(preproc_sesrun{ses})
                    movefile([fmapderiv_folder '/vdm5_sc' preproctag ...
                        '_phasediff_run' num2str(r, '%d') '.nii'], ...
                        fullfile(fmapderiv_folder, ['vdm5_sc' preproctag ...
                        '_phasediff_' ...
                        preproc_sesrun{ses}{r} '.nii']));
                end
                
                % Rename and move postscript file
                psfiles(fmapderiv_folder, 'fieldmap')
                
                % Delete unziped raw files from localscratch
                if any(size(dir([localscratch '/*.nii']), 1))
                    delete([localscratch '/*.nii']);
                end
                
            end % ses (length(raw_sestag))
        end % s (sn)           

    case 'FUNC:realign_unwarp' % realign functional images
        % SPM realigns all volumes to the first volume of first run
        % example usage: msdtb_imana('FUNC:realign', 'sn', 1)
        % Updated upstream
        
        spm_figure('GetWin','Graphics'); % create SPM .ps file at the end
        
        sn   = subj_id; % list of subjects
        prefix = '';
        vararginoptions(varargin,{'sn'});
                
        for s = sn
                   
            deriv_subjdir = fullfile(base_dir, derivatives_dir, subj_str{s});
            
            run = {};
            
            [~, preproc_sestag, ~, preproc_sesrun] = datamap(subj_str{s});
            
            for ses = 1:length(preproc_sestag)
                
                fmapderiv_folder = fullfile(deriv_subjdir, ...
                    convertStringsToChars(preproc_sestag{ses}), fmap_dir);
                
                preproctag = sprintf('%s_%s', subj_str{s}, ...
                    preproc_sestag{ses});
                
                % Empty localscratch
                if ses == 1
                    folder(localscratch);
                end

                for r=1:length(preproc_sesrun{ses})
                    % Copy EPI files to localscratch
                    func_fname = sprintf('%s_%s_bold.nii', ...
                        preproctag, preproc_sesrun{ses}{r});
                    func_file = fullfile(fmapderiv_folder, func_fname);
                    copyfile(func_file, localscratch);
                    % Copy fieldmap files to localscratch
                    fmap_fname = sprintf('vdm5_sc%s_phasediff_%s.nii', ...
                        preproctag, preproc_sesrun{ses}{r});
                    fmap_file = fullfile(fmapderiv_folder, fmap_fname);
                    copyfile(fmap_file, localscratch);
                    
                    run{ses}{r} = [...
                        convertStringsToChars(preproc_sestag{ses}), '_', ...
                        convertStringsToChars(preproc_sesrun{ses}{r})];                    
                end % r (length(raw_sesrun{ses}))
            end % ses (ses_id)
            
            run = horzcat(run{:});           
            % Load batch and run spm
            spmja_realign_unwarp(localscratch, subj_str{s}, run, 1, Inf, ...
                'prefix', prefix);
            
            for ses = 1:length(preproc_sestag)
                % Set path of the derivatives folder
                func_deriv_folder = fullfile(base_dir, derivatives_dir, ...
                    subj_str{s}, preproc_sestag{ses}, func_dir);
                % Create/update destination folder
                folder(func_deriv_folder);
                % Move files from localscratch to destination folder
                if ses == 1
                    % Move mean-unwarped EPI file
                    movefile([localscratch '/meanu' subj_str{s} ...
                        '_ses-01_task-prod_run-01_bold.nii'], ...
                        func_deriv_folder);
                    % Rename and move postscript file 
                    psfiles(func_deriv_folder, 'realignunwarp')
                end
                % Move motion-param .txt files
                movefile([localscratch '/rp_' subj_str{s} '_' ...
                    convertStringsToChars(preproc_sestag{ses}) ...
                    '_task-*_run-*_bold.txt'], func_deriv_folder);
                
                % Move functional files w/ param estimation in their
                % headers, but not resliced only, as well as all .mat files
                % (i.e. those about realign and those about the unwarp
                movefile([localscratch '/' subj_str{s} '_' ...
                    convertStringsToChars(preproc_sestag{ses}) ...
                    '_task-*_run-*_bold*'], func_deriv_folder);

                % Move unwarped images
                movefile([localscratch '/u' subj_str{s} '_' ...
                    convertStringsToChars(preproc_sestag{ses}) ...
                    '_task-*_run-*_bold*'], func_deriv_folder);
            end % ses (ses_id)
            
            % Delete unziped raw files from localscratch
            if any(size(dir([localscratch '/*.nii']), 1))
                delete([localscratch '/*.nii']);
            end
        end % s (sn)

    case 'FUNC:coreg' % coregistration with the anatomicals
        % (1) Manually seed the functional/anatomical registration
        % - Do "coregtool" on the matlab command window
        % - Select anatomical image and mean functional image to overlay
        % - Manually adjust mean functional image and save the results 
        %   ("r" will be added as a prefix)
        % Example usage: 
        % msdtb_imana('FUNC:coreg', 'sn', [1], 'prefix', 'r')
        
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
            % Create a copy of the original meanepi where
            
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
            
            % Run coregegistration
            matlabbatch{1}.spm.spatial.coreg.estimate=J;
            spm_jobman('run', matlabbatch);
            
            % Rename and move postscript file
            psfiles(func_deriv, 'coregestimate')
         
            % Display the affine matrix of the transformation
            % mean_epi = spm_vol(J.source);
            % t1 = spm_vol(J.ref);
            % x = spm_coreg(mean_epi{1}.fname, t1{1}.fname); % computes coreg again but without storing parameters in the header of source image
            % M = spm_matrix(x);
            % display(M)
            
            % Delete former postscript file
            if any(size(dir(fullfile(anat_deriv, 'spm_*.ps')), 1))
                delete(fullfile(anat_deriv, 'spm_*.ps'));
            end

            % (3) Manually check again
            % coregtool;
            % keyboard();

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
        prefix = 'r'; % prefix for the meanepi: r or rbb if bias corrected
        
        vararginoptions(varargin, {'sn', 'prefix'});
                
        for s = sn
            % Get the directory of subjects functional
            deriv_folder = fullfile(base_dir, derivatives_dir, ...
                subj_str{s});
            funcmean_deriv = fullfile(deriv_folder, 'ses-01', func_dir);
            
            [~, preproc_sestag, ~, preproc_sesrun] = datamap(subj_str{s});
            
            Q = {};
            for ses=1:length(preproc_sestag)
                func_deriv = fullfile(deriv_folder, preproc_sestag{ses}, ...
                    func_dir);
                fprintf('- make_samealign  %s \n', subj_str{s})
                % Select image for reference 
                %%% note that functional images are aligned with the first
                %%% run from first session hence, the ref is always 
                %%% mean<subj>_ses-01_run-01
                P{1} = fullfile(funcmean_deriv, sprintf(...
                    '%smeanu%s_ses-01_task-prod_run-01_bold.nii', ...
                    prefix, subj_str{s}));

                for r=1:length(preproc_sesrun{ses})
                    fpath = convertStringsToChars(fullfile(func_deriv, ...
                        sprintf('u%s_%s_%s_bold.nii', subj_str{s}, ...
                        preproc_sestag{ses}, preproc_sesrun{ses}{r})));
                    V = nifti(fpath);
                    imageNumber=1:V.dat.dim(4);
                    for i= 1:numel(imageNumber)
                        % for 'auto' mode in coregistration, remove prefix 
                        % and explicitly add 'r' prefix in the same place
                        Q{end+1} = [fpath ',' num2str(i, '%d')];
                    end % i (imageNumber)
                end % r(runs)
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
                
            % First mask: whole brain
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
        
    case 'FUNC:mask_normalization'
        % Example usage: msdtb_imana('FUNC:mask_normalization')
        
        sn       = subj_id; % subject list
        vararginoptions(varargin, {'sn'});
        
        for s = sn
            % Get the directory of subjects anatomical
            deriv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s});
            subj_anatderiv_dir = fullfile(deriv_subj_dir, 'ses-01/anat');
                
            % Deformation-Field file
            deffield_file = [subj_anatderiv_dir '/y_' subj_str{s} '_T1w.nii'];
            
            % Path of masks
            funcmean_deriv = fullfile(deriv_subj_dir, 'ses-01', func_dir);
            wholebrain_mask = fullfile(funcmean_deriv, ...
                'rmask_noskull.nii');
            graymatter_mask = fullfile(funcmean_deriv, 'rmask_gray.nii');
                
            % Apply normalization
            spmja_normalization_write(deffield_file, wholebrain_mask, ...
                'voxel_size', [2.5 2.5 2.5])
            spmja_normalization_write(deffield_file, graymatter_mask, ...
                'voxel_size', [2.5 2.5 2.5])
        end 
        
    case 'FUNC:run_all'
        % Example usage: msdtb_imana('FUNC:run_all')
        
        msdtb_imana('FUNC:make_fieldmap')
        msdtb_imana('FUNC:realign_unwarp')
        
%         msdtb_imana('FUNC:realign_estimate')
%         msdtb_imana('FUNC:calculate_vdm')
%         msdtb_imana('FUNC:apply_vdm')
        
        msdtb_imana('FUNC:coreg', 'prefix', '', 'step', 'auto')
        msdtb_imana('FUNC:make_samealign', 'prefix', '')
        msdtb_imana('FUNC:make_maskImage', 'prefix', '')
        msdtb_imana('FUNC:mask_normalization')

    case 'GLM:copy_paradigm-descriptors'
        % Example usage: msdtb_imana('GLM:copy_paradigm-descriptors', ...
        %                            'paradigm_type', 'drbb')

        sn = subj_id;

        paradigm_type = ''; % default: dbb
        % paradigm_type = 'drbb';

        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        vararginoptions(varargin, {'sn', 'paradigm_type'});
        
        if isdir('/srv/diedrichsen/data')
            source = fullfile(homedir, ...
                'tsclient/analu/mygit/music_sdtb/music-sdtb_analysis/imaging_analysis/events');
        else
            source = fullfile(...
                '/home/analu/mygit/music_sdtb/music-sdtb_analysis/imaging_analysis/events');
        end
        destination = fullfile(workdir, ...
            'Cerebellum/music-sdtb/derivatives');

        if isempty(paradigm_type)
            suffix = paradigm_type;
        else
            suffix = ['_' paradigm_type];
        end
        
        % Initialize an empty cache to track refreshed directories
        refreshed_dirs = containers.Map();

        for s = sn
            
            [original_events, renamed_events] = eventsmap(subj_str{s});
            
            for ses = 1:length(original_events)
                for r = 1:length(original_events{ses})
                    sfname = [subj_str{s} '_' original_events{ses}{r} ...
                        suffix '_events.tsv'];
                    sfile = convertStringsToChars(fullfile(...
                        source, ...
                        subj_str{s}, ...
                        original_events{ses}{r}(1:6), ...
                        sfname));
                    dfolder = convertStringsToChars(fullfile(...
                        destination, ...
                        subj_str{s}, ...
                        renamed_events{ses}{r}(1:6), ...
                        'func'));
                    
                    % Delete previous tsv files from destination folder
%                     if r == 1
%                         dold_files = fullfile(dfolder, ...
%                             [subj_str{s} '*_events.tsv']);
%                         delete(dold_files);
%                     end

                    % Refresh only new directories
                    refresh_path(sfile, refreshed_dirs);
                    
                    % Copy new tsv files
                    system(['cp ' sfile ' ' dfolder]);

                    % Rename copied tsv file in destination folder if it
                    % has to be renamed
                    current_file = fullfile(dfolder, sfname);
                    nfname = [subj_str{s} '_' renamed_events{ses}{r} ...
                        suffix '_events.tsv'];
                    new_file = fullfile(dfolder, nfname);
                    if ~strcmp(current_file, new_file)
                        movefile(current_file, new_file);
                    end

                end % r (original_events{ses})
            end % ses (original_events)
        end % s (sn)
        
    case 'GLM:grand_design_rwls' % make the design matrix for the glm
        % models each condition as a separate regressors
        % For conditions with multiple repetitions, one regressor
        % represents all the instances

        % Example usage:
        % msdtb_imana('GLM:grand_design_rwls', 'sn', [1], ...
        %             'design', {'prod', 'percep', 'ntfd', 'allmain_tasks'}, ...
        %             'hrf_cutoff', 42.3, ...
        %             'covariance_model', 'fast', ...
        %             'events_file_tag', 'drbb_events', ...
        %             'output_folder', 'ffx_rwls_drbb_hrf42')
        
        sn = subj_id;
        design = {'prod', 'percep', 'ntfd', 'rand_ntfd', 'allmain_tasks'};
        
        % hrf_cutoff = Inf; 
        hrf_cutoff = 128; 
        % hrf_cutoff = 42.3;
        
        % To compute the covariance structure. If 'wls', SPM will
        % estimate the noise variance in the model-estimation module and then
        % re-weight each image by the inverse of the variance. Make sure to use 
        % the rwls model-estimation module in SPM and not the standard
        % model-estimation module. If covariance_model is 'fast', standard
        % serial-correlation correction will be employed
        covariance_model = 'wls'; % 'wls' or 'fast'
        
        prefix = 'u'; % prefix of the preprocessed epi we want to use
        events_file_tag = 'events'; % with dbb model (this is the default)
        output_folder = 'ffx';
        
        vararginoptions(varargin, {'sn', 'design', 'hrf_cutoff', ...
            'covariance_model', 'events_file_tag', 'output_folder'});
        
        spm_figure('GetWin','Graphics'); % create SPM .ps file at the end
        
        % loop over subjects
        for s = sn
            deriv_subjdir = fullfile(base_dir, derivatives_dir, subj_str{s});
            glms_folder = fullfile(deriv_subjdir, est_dir);
            
            [~, preproc_sestag, ~, preproc_sesrun] = datamap(subj_str{s});
            
            % loop over design
            for dg=1:length(design)
                estimates_folder = fullfile(glms_folder, design{dg}, ...
                    output_folder);

                % Create estimates folder if does not exist or clean it
                folder(estimates_folder);

                J = []; % structure with SPM fields to make the design

                J.timing.units   = 'secs';
                J.timing.RT      = 1.2;
                J.timing.fmri_t  = 16;
                J.timing.fmri_t0 = 8;

                J.fact           = struct('name', {}, 'levels', {});
                
                % Using time and dispersion derivatives in the HRF. 
                % This allows the model to adjust the HRF shape and timing ... 
                % ... on a voxel-wise basis. Useful for whole-brain ...
                % ... analysis when we suspect regional variability.
                % [0, 0] no derivatives
                % [1, 0] time derivatives
                % [1, 1] time and dispersion derivatives
                J.bases.hrf.derivs = [0 0];
                
                % p(1)	Delay of response (secs)	6
                % p(2)	Delay of undershoot (secs)	16
                % p(3)	Dispersion of response	1
                % p(4)	Dispersion of undershoot	1
                % p(5)	Ratio of response to undershoot	6
                % p(6)	Onset (secs)	0
                % p(7)	Length of HRF kernel (secs)	32
                % J.bases.hrf.params = [6 16 1 1 6 0 32]; % default for canonical HRF in SPM
                
                J.volt             = 1;
                J.global           = 'None';
                J.mask             = {char(fullfile(deriv_subjdir, ...
                    'ses-01', func_dir, 'rmask_noskull.nii'))};
                J.mthresh          = 1.;
                J.cvi_mask         = {char(fullfile(deriv_subjdir, ...
                    'ses-01', func_dir,'rmask_gray.nii'))}; % only for rwls
                J.cvi              = covariance_model;

                J.dir = {estimates_folder};
                
                % Define tasks to be included in the design
                if strcmp(design{dg}, 'allmain_tasks')
                    tasks = {'prod', 'percep', 'ntfd'};
                    ssn = 1:length(preproc_sestag); % list of sessions
                elseif strcmp(design{dg}, 'rand_ntfd')
                    tasks = {'ntfd'};
                    ssn = length(preproc_sestag);
                else
                    tasks = {design{dg}};
                    ssn = 1:length(preproc_sestag); % list of sessions
                end

                % loop over sessions
                count = 0;
                for ses = 1:length(preproc_sestag)
                    funcderiv_folder = fullfile(deriv_subjdir, ...
                        ['ses-' num2str(ses, '%02d')], func_dir);

                    % loop over tasks
                    for tk=1:length(tasks)              
                        % loop over runs
                        if strcmp(design{dg}, 'rand_ntfd')
                            start = 3;
                            n_run = 4;
                        else
                            start = 1;
                            n_run = 2;
                        end
                        
                        %List runs and tsv files
                        runs_list = {};
                        tsv_fnames_list = {};
                        for rf=start:n_run
                            run_fname = sprintf(...
                                '%s%s_ses-%02d_task-%s_run-%02d_bold.nii', ...
                                prefix, subj_str{s}, ses, tasks{tk}, rf);
                            tsv_fname = sprintf(...
                                '%s_ses-%02d_task-%s_run-%02d_%s.tsv', ...
                                subj_str{s}, ses, tasks{tk}, rf, ...
                                events_file_tag);
                            run_path = fullfile(funcderiv_folder, run_fname);
                            if isfile(run_path)
                                runs_list = [runs_list, run_path];
                                tsv_fnames_list = [tsv_fnames_list, tsv_fname];
                            end
                        end
                        
                        for r=1:length(runs_list)
                            count = count + 1;                            
                            V = niftiinfo(runs_list{r});
                            numTRs = V.ImageSize(4);
                            % fill in nifti image names for the current run
                            N = cell(numTRs - numDummys, 1); % preallocating!
                            for i = 1:(numTRs-numDummys)
                                N{i} = [runs_list{r} ', ' num2str(i, '%d')];
                            end % i (image numbers)

                            % Load the scans
                            J.sess(count).scans = N; %scans in the current runs

                            J.sess(count).cond = struct('name', {}, ...
                                'onset', {}, 'duration', {}, 'tmod', {}, ...
                                'pmod', {}, 'orth', {});

                            % get the tsvfile for the current run
                            tsv_path = fullfile(funcderiv_folder, ...
                                tsv_fnames_list{r})
                            D = struct([]); 
                            D = tdfread(tsv_path,'\t');

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

                            % %%%%%%% Reorder regressors %%%%%%%
                            names = reorder_regressors(unique_names, ...
                                events_file_tag, design{dg});
                            
                            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            
                            % Create onsets and duration cells
                            for u = 1:length(names)
                                indexes = [];
                                indexes = find(strcmp(trial_names, names{u}));
                                for idx = 1:length(indexes)
                                    onsets{u}(idx) = ...
                                        trial_onsets{indexes(idx)};
                                    durations{u}(idx) = ...
                                        trial_durations{indexes(idx)};
                                end
                            end

                            save(fullfile(localscratch, ...
                                sprintf('%s.mat', tsv_fnames_list{r})), ...
                                'names', 'onsets', 'durations');

                            J.sess(count).multi = {fullfile(localscratch, ...
                                sprintf('%s.mat', tsv_fnames_list{r}))};

                            J.sess(count).regress   = struct('name', {}, ...
                                'val', {});
                            J.sess(count).multi_reg = {''};
                            J.sess(count).hpf       = hrf_cutoff;
                        end % r (n_run)
                    end % tk (tasks)         
                end % ses (ssn)
                
                % FFX across specified runs and sessions            
                spm_rwls_run_fmri_spec(J); % rwls
                
            end % dg (design)            

            % Remove *events.mat file from localscratch
            if any(size(dir([localscratch '/*_events.mat']), 1))
                delete([localscratch '/*_events.mat']);
            end
        end % sn (subject)     

    case 'GLM:estimate_rwls' % estimate beta values
        % Example usage:
        % msdtb_imana('GLM:estimate_rwls', 'sn', [1], ...
        %             'design', {'prod', 'percep', 'ntfd', 'allmain_tasks'}, ...
        %             'output_folder', 'ffx_rwls_drbb_hrf42')       
        
        sn = subj_id; % subject list
        design = {'prod', 'percep', 'ntfd', 'rand_ntfd', 'allmain_tasks'};
        output_folder = 'ffx';
        vararginoptions(varargin, {'sn', 'design', 'output_folder'});
        
        for s = sn
            estderiv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s}, est_dir);            
            % loop over designs
            for dg=1:length(design)
                estdesign_folder = fullfile(estderiv_subj_dir, design{dg}, ...
                    output_folder)
                % Delete previous estimates, if they exist
                if any(size(dir([estdesign_folder '/*.nii']), 1))
                    delete([estdesign_folder '/*.nii']);
                end
                % Load SPM.mat file with design and add path to store
                % the new estimates
                load(fullfile(estdesign_folder, 'SPM.mat'));
                SPM.swd = estdesign_folder;
                % Add as input SPM.mat file to the rwls GLM 
                spm_rwls_spm(SPM);
            end % dg (designs)
        end % s (sn)    
        
    case 'PLOT:covariance_structure'
        % Step 1: Load the covariance matrix from SPM
        load('SPM.mat');  % Load the SPM.mat file
        covariance_matrix = SPM.xX.Bcov;  % Extract the covariance matrix

        % Step 2: Plot the covariance matrix
        figure;
        imagesc(covariance_matrix);  % Plot the covariance matrix
        colormap('hot');  % Use the 'hot' colormap, similar to SPM's default
        colorbar;  % Display the color bar
        title('Covariance Matrix of Parameter Estimates');
        xlabel('Parameters');
        ylabel('Parameters');
        axis square;  % Make the plot square

        % Step 3: Add grid lines for clarity
        grid on;
        set(gca, 'XTick', 1:length(covariance_matrix), 'YTick', 1:length(covariance_matrix));  % Set tick marks

        % Step 4: Optionally add parameter labels
        labels = {'Param1', 'Param2', 'Param3', 'Param4'};  % Replace with your labels
        set(gca, 'XTickLabel', labels, 'YTickLabel', labels, 'TickLabelInterpreter', 'none');
        xtickangle(45);  % Rotate the x-axis labels for readability

        % Step 5: Adjust colorbar limits if needed
        caxis([min(covariance_matrix(:)), max(covariance_matrix(:))]);  % Adjust color scaling
        
    case 'PLOT:timeseries'
        
        s = 42;
        design = 'allmain_tasks'; % design = {'prod', 'percep', 'ntfd', 'rand_ntfd', 'allmain_tasks'}
        
        % Define the center coordinate and radius in native space
        centerCoord = [-21.7, 3.3, 5.3];  % Replace with your native space coordinates (in mm)
        roi_name = 'lPutamen';       % name of ROI
        radius = 8;               % Replace with your desired radius in millimeters
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Go to the folder of script
        cd(fileparts(mfilename('fullpath')))
        
        % Load SPM.mat
        task_estimates_dir = fullfile(base_dir, derivatives_dir, ...
            sprintf('sub-%02d', s), est_dir, design, 'ffx_rwls_dbb_hrf128')
        SPM = load(fullfile(task_estimates_dir, 'SPM.mat')); 
        SPM=SPM.SPM;
        
        % Ensure that SPM.xY.P exists
        if ~isfield(SPM, 'xY') || ~isfield(SPM.xY, 'P')
            error('SPM.xY.P is not available. Please ensure that SPM.mat contains the design matrix with specified images.');
        end
        
        % Get the paths to your fMRI images from SPM.xY.P
        VY = SPM.xY.VY;  % Should be an N x 1 cell array of file paths
        
        % Load first fMRI volume of first 4D NIfTI file to get dimensions and transformation matrix
        fname = [VY(1).fname ',' num2str(1)];
        V = spm_vol(fname);
        dims = V.dim(1:3);  % Dimensions of the volume
        mat = V.mat;        % Transformation matrix (voxel to native space coordinates)
        
        % Generate a grid of voxel coordinates
        [xGrid, yGrid, zGrid] = ndgrid(1:dims(1), 1:dims(2), 1:dims(3));
        
        % Convert voxel indices to native space coordinates
        voxelCoords = [xGrid(:), yGrid(:), zGrid(:), ones(numel(xGrid), 1)]';
        worldCoords = mat * voxelCoords;
        worldCoords = worldCoords(1:3, :)';  % Exclude the homogeneous coordinate
        
        % Compute Euclidean distances from each voxel to the center coordinate
        distances = sqrt(sum((worldCoords - centerCoord).^2, 2));

        % Create the spherical mask
        sphereMask = distances <= radius;
        sphereMask = reshape(sphereMask, dims);

        % Find the indices of voxels within the sphere
        roiIndices = find(sphereMask);

        % Initialize array to hold the timeseries
        numVolumes = length(VY);
        y_raw = zeros(numVolumes, 1);

        % Loop through each fMRI volume to extract the ROI timeseries
        for i = 1:numVolumes
            Vs = spm_vol(VY(i));  % Load the fMRI volume
            Vs(i).dat = spm_read_vols(Vs);    % Read the volume data
            Vs(i).dt = [64 0];    % Set number format to double 
            Vs(i).pinfo = [1 0 0]'; % set scaling to zero becuase done 
            Vs(i).dat(isnan(Vs(i).dat)) = 0;
            roiData = Vs(i).dat(roiIndices); % Extract data within the spherical ROI
            % A(:,i)=spm_sample_vol(roiData, worldCoords(:, 1), worldCoords(:, 2), worldCoords(:, 3),0); % Sampling of ROI with nearest neighbor interpolation
            y_raw(i) = mean(roiData); % Compute mean signal in the ROI
            % y_raw(i) = mean(A(:,i));
        end
        
        % Do the regression and compute y_hat
        y_filt = spm_filter(SPM.xX.K,y_raw);
        SPM.xX.xKXs.X = spm_filter(SPM.xX.K,SPM.xX.X);
        SPM.xX.pKX = pinv(SPM.xX.xKXs.X);
        B = SPM.xX.pKX*y_filt; %-Parameter estimates
        y_res   = y_filt - SPM.xX.xKXs.X*B;             %-Residuals
        reg_interest=[SPM.xX.iH SPM.xX.iC]; % Get the Design Matrix (only for regressors-of-interest)
        y_hat = SPM.xX.xKXs.X(:,reg_interest)*B(reg_interest,:); %- predicted values
        y_adj  = y_hat + y_res; % The time series adjusted for nuiscance regressors 

        % Create a time vector based on TR
        TR = 1.2;  % Repetition time
        timeVector = (0:length(y_raw)-1) * TR;
        
        % Plot the extracted timeseries
        figure;
        for n=1:10
            subplot(10,1,n);
            if n==1
                plot(timeVector(1, 1:626), y_adj(1:626, 1), 'b-', 'LineWidth', 1.5);
                hold on;
                plot(timeVector(1, 1:626), y_hat(1:626, 1), 'r-', 'LineWidth', 1.5);
                title('Overlapping Timeseries');
            else
                plot(timeVector(1, 626*(n-1):626*n), y_adj(626*(n-1):626*n, 1), 'b-', 'LineWidth', 1.5);
                hold on;
                plot(timeVector(1, 626*(n-1):626*n), y_hat(626*(n-1):626*n, 1), 'r-', 'LineWidth', 1.5);
                if n==5
                    ylabel('Signal Intensity');
                elseif n==10
                    xlabel('Time (seconds)');
                end
            end           
            legend('Raw Signal', 'Predicted Signal');
            grid on;
            hold off;
        end
        
    case 'SUIT:cerebellum_graymask'    
        % Conjunction of the pcereb_corr  and the cerebellar gray matter 
        % mask (in functional space) thresholded to 0.2
        
        % %%%%%%%%%%%%%%%%%% DEFAULT VALUES OF VARARGIN %%%%%%%%%%%%%%%%%%%%%%%
        sn = subj_id; %subjNum
        design = 'allmain_tasks';
        model = 'ffx_rwls';
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        vararginoptions(varargin, {'sn', 'design', 'model'});
        
        for s = sn
            suit_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s}, 'ses-01/suit');
            masks = {};
            % First image need to be in functional space to ensure 
            % correct space 
            masks{1} = fullfile(base_dir, 'derivatives', subj_str{s}, ...
                est_dir, design, model, 'mask.nii')
            masks{2} = fullfile(suit_subj_dir, ...
                sprintf('c_%s_T1w_pcereb_corr.nii', subj_str{s}));
            masks{3} = fullfile(suit_subj_dir, ...
                sprintf('c_%s_T1w_seg1.nii', subj_str{s}));
            final_mask = fullfile(suit_subj_dir, 'maskbrainSUITGrey.nii');
            spm_imcalc(masks, final_mask, 'i2>0.1 & i3>0.1');
        end

    case 'GLM:individual_ffx_t'
        % Estimate ffx individual tmaps across runs and sessions
        
        % Example usage:
        % msdtb_imana('GLM:individual_ffx_t', ...
        %             'design', {'rand_ntfd'}, ...
        %             'model_type', 'drbb', ...
        %             'output_folder', 'ffx_rwls_drbb_hrf42')
 
        contrasts_list = {};
        % Go to the folder of script
        cd(fileparts(mfilename('fullpath')))
        
        % %%%%%%%%%%%%%%%%%% DEFAULT VALUES OF VARARGIN %%%%%%%%%%%%%%%%%%%%%%%
        
        sn = subj_id; % subject list
        
        design = {'prod', 'percep', 'ntfd', 'allmain_tasks'};
        % design = {'rand_ntfd'};
        
        model_type = 'dbb';
        % model_type = 'drbb';
        
        output_folder = 'ffx';
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        vararginoptions(varargin, {'sn', 'design', 'model_type', ...
            'output_folder'});
        
        contrasts_list = {};
        if isequal(design, {'prod', 'percep', 'ntfd', 'allmain_tasks'})
            contrast_prefix = {'Production: ', 'Perception: ', 'NTFD: ', ...
                'AllTasks: '};
            if strcmp(model_type, 'dbb')
                contrasts_list = contrasts;
            elseif strcmp(model_type, 'drbb')
                contrasts_list = contrasts_drbb;
            end
        elseif isequal(design, {'rand_ntfd'})
            contrast_prefix = {'Random NTFD: '};
            if strcmp(model_type, 'dbb')
                contrasts_list = contrasts_random;
            elseif strcmp(model_type, 'drbb')
                contrasts_list = contrasts_random_drbb;
            end
        end
        
        for s = sn
            estderiv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s}, est_dir);  
            for dg=1:length(design)          
                estdesign_folder = fullfile(estderiv_subj_dir, design{dg}, ...
                    output_folder)

                A = []; % structure with SPM fields to build the t-contrasts
                A.spmmat = {[estdesign_folder '/SPM.mat']};
                for c=1:length(contrasts_list)
                    A.consess{c}.tcon.name = [contrast_prefix{dg} ...
                        contrasts_list{c,1}];
                    A.consess{c}.tcon.weights = contrasts_list{c,2};
                    A.consess{c}.tcon.sessrep = 'replsc';
                end

                % Delete existing contrasts
                A.delete = 1; % 1 yes, 0 no
                matlabbatch{1}.spm.stats.con=A;
                spm_jobman('run', matlabbatch);
            end
        end % s (subject)
        
    case 'GLM:calc_PSC'                           
        % Calculate percent signal change for selected contrasts - based on betas
        
        % Example usage:
        % msdtb_imana('GLM:calc_PSC', ...
        %             'design', {'rand_ntfd'}, ...
        %             'model_derivatives', 'time', ...
        %             'output_folder', 'ffx_rwls_drbb_hrf42_timederiv')
        
        % %%%%%%%%%%%%%%%%%% DEFAULT VALUES OF VARARGIN %%%%%%%%%%%%%%%%%%%%%%%
        
        sn = subj_id; % subject list
        
        design = {'prod', 'percep', 'ntfd', 'allmain_tasks'};
        % design = {'rand_ntfd'};
        
        model_derivatives = 'no_deriv'
        % model_derivatives = 'time'
        % model_derivatives = 'timedisp'
        
        output_folder = 'ffx';
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        vararginoptions(varargin, {'sn', 'design', 'model_derivatives', ...
            'output_folder'});

        for s = sn
            % Go to subject’s directory and load SPM info
            estderiv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s}, est_dir);
            
            for dg=1:length(design)
                estdesign_folder = fullfile(estderiv_subj_dir, design{dg}, ...
                    output_folder);
                cd(fullfile(estdesign_folder));
                
                % Delete pre-existing psc* files in the folder
                if any(size(dir(...
                        [estdesign_folder '/psc*.nii']), 1))
                    delete([estdesign_folder '/psc*.nii']);
                end
                
                load SPM;
                X=(SPM.xX.X(:,SPM.xX.iC)); % Design matrix - raw
                                
                P={};
                numB=length(SPM.xX.iB);     % Partitions - runs
                for p=SPM.xX.iB
                    P{end+1,1}=fullfile(estdesign_folder, ...
                        sprintf('beta_%4.4d.nii',p));  % get the intercepts and use them to calculate the baseline (mean images)
                end
                
                t_con_name = extractfield(SPM.xCon, 'name');
                % Create string with formula
                con_div_intercepts = '';
                for r=1:numB
                    if r == numB
                        con_div_intercepts = sprintf('i%d./((%si%d)/%d)', ...
                            r+1, con_div_intercepts, r, numB);
                    else
                        con_div_intercepts = sprintf('%si%d+', ...
                            con_div_intercepts, r);
                    end
                end
                
                for con=1:length(t_con_name)  % all contrasts
                    
                    % Height of response (peak) according to regressors
                    % of the contrast
                    if strcmp(design{dg},'rand_ntfd')
                        step=7;
                    else
                        step=5;
                    end
                    maxX=[];
                    maxX=max(X);
                    if strcmp(model_derivatives, 'time')
                        maxX = maxX(1:2:end); % remove peaks of time derivatives
                    elseif strcmp(model_derivatives, 'timedisp')
                        maxX = maxX(1:3:end); % remove peaks of time and dispersion derivatives
                    end
                    if con < length(t_con_name)
                        maxX(step:step:end)=[] % remove 'decision' peaks for the encoding constrasts
                        h=min(maxX);
                    else
                        decision_max=[];
                        decision_max=maxX(step:step:end) % retrieve the decision peaks for the decision contrast
                        h=min(decision_max);
                    end
                        
                    P{numB+1,1}=fullfile(estdesign_folder, ...
                        sprintf('con_%04d.nii', con));
                    outname=fullfile(estdesign_folder, ...
                        sprintf('psc_%04d.nii', con));
                    
                    formula=sprintf('100.*%f.*%s', h, con_div_intercepts);
                    
                    A = [];
                    A.input = P;
                    A.output = outname;
                    A.outdir = {output_folder};
                    A.expression = formula;
                    A.var = struct('name', {}, 'value', {});
                    A.options.dmtx = 0;
                    A.options.mask = 0;
                    A.options.interp = 1;
                    A.options.dtype = 4;               

                    matlabbatch{1}.spm.util.imcalc=A;
                    spm_jobman('run', matlabbatch);

                    fprintf('Contrast: %s\n', t_con_name{con});
                end
                fprintf('%s - Done\n', subj_str{s});
            end
        end
        
    case 'CON:norm_smooth'
        % Normalize and smooth individual contrasts or t-maps
        % Example usage: msdtb_imana(
        %                   'CON:norm_smooth', ...
        %                   'design', {'rand_ntfd'}, ...
        %                   'input_folder', 'ffx_rwls_drbb_hrf42', ...
        %                   'file_type', 'spmT')     
        
        sn = subj_id; % subject list
        
        % %%%%%%%%%%%%%%%%%% DEFAULT VALUES OF VARARGIN %%%%%%%%%%%%%%%%%%%%%%%
        
        design = {'prod', 'percep', 'ntfd', 'allmain_tasks'};
        % design = {'rand_ntfd'};
        
        input_folder = 'ffx_rwls_dbb_hrf128';
        
        file_type = 'con'; % options: 'con, spmT, ResMS, psc, beta' (beta refers to the prewhitened files)
        smoothing_kernel = [8 8 8];
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        vararginoptions(varargin, {'sn', 'design', 'input_folder', ...
            'file_type', 'smoothing_kernel'});
        
        for s = sn
            estderiv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s}, est_dir); 
            for dg=1:length(design)           
                estdesign_folder = fullfile(estderiv_subj_dir, design{dg}, ...
                    input_folder);
                
                % Delete pre-existing w* files of the same type in the
                % folder
                if any(size(dir(...
                        [estdesign_folder '/w' file_type '*.nii']), 1))
                    delete([estdesign_folder '/w' file_type '*.nii']);
                end
                % Delete pre-existing sw* files of the same type in the
                % folder
                if any(size(dir(...
                        [estdesign_folder '/sw' file_type '*.nii']), 1))
                    delete([estdesign_folder '/sw' file_type '*.nii']);
                end
                
                confiles = {};
                if strcmp(file_type, 'ResMS')
                    cname = sprintf('%s.nii', file_type);
                    confiles = {fullfile(estdesign_folder, cname)};
                elseif strcmp(file_type, 'beta')
                    beta_files = dir([estdesign_folder '/' file_type ...
                        '_*_desc-prewhitened.nii']);
                    beta_paths = fullfile(estdesign_folder, {beta_files.name});
                    confiles = beta_paths(:);  % Convert to an n x 1 cell array
                else
                    % List of contrasts in source folder
                    n_contrasts = numel(dir([estdesign_folder '/' file_type ...
                        '_*.nii']));
                    for c=1:n_contrasts
                        cname = sprintf('%s_%04d.nii', file_type, c);
                        confiles{c,1} = fullfile(estdesign_folder, cname);
                    end
                end
                
                % Deformation-Field file
                deffield_folder = fullfile(base_dir, derivatives_dir, ...
                    subj_str{s}, 'ses-01', 'anat');
                deffield_file = [deffield_folder '/y_' subj_str{s} '_T1w.nii'];
                
                % Apply normalization
                spmja_normalization_write(deffield_file, confiles, ...
                    'voxel_size', [2.5 2.5 2.5])
                
                % List normalized contrasts in ffx folder
                cd(estdesign_folder)
                wsource_list = dir(fullfile(estdesign_folder, ...
                    ['w' file_type '*.nii']));
                w_values=struct2cell(wsource_list);
                wsource_files = w_values(1,1:end)';
                
                % Smooth normalized contrasts
                S = [];
                S.data = wsource_files;
                S.fwhm = smoothing_kernel;
                S.dtype = 0;
                S.im = 0;
                S.prefix = 's';
                
                matlabbatch{1} = {};
                matlabbatch{1}.spm.spatial.smooth=S;
                spm_jobman('run',matlabbatch);
                
            end % dg (design)           
        end % s (sn)
        
    case 'GROUP:mask'
        % Example usage: msdtb_imana('GROUP:mask', 'mask_type', 'wrmask_gray')
        
        sn       = subj_id; % subject list
        mask_type = 'wrmask_noskull'; % whole-brain
        % mask_type = 'wrmask_gray'   % gray-matter
        vararginoptions(varargin, {'sn', 'mask_type'});
        
        group_dir = fullfile(base_dir, derivatives_dir, 'group/anat')
        gmask_name = sprintf('group_%s.nii', mask_type(3:end));
        group_mask_path = fullfile(group_dir, gmask_name);
        if not(isfolder(group_dir))
            % Create group dir if does not exist
            mkdir(group_dir);
        elseif isfile(group_mask_path)
            % Delete previous group map if it exists
            delete(group_mask_path);
        end

        normalized_masks = {};
        formula = '';
        for s = sn
            % Get the directory of subjects' functional data
            deriv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s});
            
            % Path of individual masks
            funcmean_deriv = fullfile(deriv_subj_dir, 'ses-01', func_dir);
            normalized_masks{s,1} = fullfile(funcmean_deriv, ...
                [mask_type '.nii']);
            
            % Create string with formula
            if s == length(sn)
                formula = sprintf('(%si%d)/%d >= 0.8', formula, s, length(sn))
            else
                formula = sprintf('%si%d+', formula, s);
            end
        end
        
        A = [];
        A.input = normalized_masks;
        A.output = gmask_name;
        A.outdir = {group_dir};
        A.expression = formula;
        A.var = struct('name', {}, 'value', {});
        A.options.dmtx = 0;
        A.options.mask = 0;
        A.options.interp = 1;
        A.options.dtype = 4;               

        matlabbatch{1}.spm.util.imcalc=A;
        spm_jobman('run', matlabbatch); 
        
    case 'CON:masking'
        % Apply mask to the smoothed contrasts
        % Example usage: msdtb_imana(
        %                   'CON:masking', 
        %                   'input_folder', 'ffx_rwls_drbb_hrf42', ...
        %                   'output_folder', 'masked_derivatives_rwls_drbb_hrf42', ...
        %                   'file_type', 'spmT', ...
        %                   'masktag', 'gmmasked')     
        
        sn = subj_id; % subject list
        
        % %%%%%%%%%%%%%%%%%% DEFAULT VALUES OF VARARGIN %%%%%%%%%%%%%%%%%%%%%%%
        
        design = {'prod', 'percep', 'ntfd', 'allmain_tasks'};
        % design = {'rand_ntfd'};
        
        input_folder = 'ffx_rwls_dbb_hrf128';
        output_folder = 'masked_derivatives_rwls_dbb_hrf128';
        
        file_type = 'con'; % the another one is 'spmT, ResMS or psc'
        smoothing_kernel = [8 8 8];
        
        masktag = 'wbmasked'; % whole-brain masking
        % masktag = 'gmmasked'; % gray-matter masking
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        vararginoptions(varargin, {'sn', 'design', 'input_folder', ...
            'output_folder', 'file_type', 'smoothing_kernel', 'masktag'});
        
        if strcmp(masktag, 'wbmasked')
            group_mask = fullfile(base_dir, derivatives_dir, ...
                'group/anat/group_mask_noskull.nii');
        elseif strcmp(masktag, 'gmmasked')
            group_mask = fullfile(base_dir, derivatives_dir, ...
                'group/anat/group_mask_gray.nii');
        end
        
        for s = sn
            estderiv_subj_dir = fullfile(base_dir, derivatives_dir, ...
                subj_str{s}, est_dir); 
            for dg=1:length(design)           
                estdesign_folder = fullfile(estderiv_subj_dir, design{dg}, ...
                    input_folder);
                
                confiles = {};
                if strcmp(file_type, 'ResMS')
                    cname = sprintf('%s.nii', file_type);
                    confiles = {fullfile(estdesign_folder, cname)};
                else
                    % List of contrasts in source folder
                    n_contrasts = numel(dir([estdesign_folder '/' file_type ...
                        '_*.nii']));
                    for c=1:n_contrasts
                        cname = sprintf('%s_%04d.nii', file_type, c);
                        confiles{c,1} = fullfile(estdesign_folder, cname);
                    end
                end
                
                % List normalized contrasts in ffx folder
                cd(estdesign_folder)
                wsource_list = dir(fullfile(estdesign_folder, ...
                    ['w' file_type '*.nii']));
                w_values=struct2cell(wsource_list);
                wsource_files = w_values(1,1:end)';
                
                % List smoothed, normalized contrasts in ffx folder
                swsource_list = dir(fullfile(estdesign_folder, ...
                    ['sw' file_type '*.nii']));
                sw_values=struct2cell(swsource_list);
                swsource_files = sw_values(1,1:end)';
                
                % Create destination folder if does not exist...
                destination_dir = fullfile(estderiv_subj_dir, ...
                    design{dg}, output_folder);
                if not(isfolder(destination_dir))
                    mkdir(destination_dir);
                end
                
                % ... or delete pre-existing files... 
                % ... w/ the same type of masking from destination folder
                if any(size(dir([destination_dir '/w' file_type '*' masktag ...
                        '.nii']), 1))
                    delete([destination_dir '/w' file_type '*' masktag ...
                        '.nii']);
                end
                
                % Mask non-smoothed normalized contrasts with 
                % the group-level whole-brain mask                
                for w = 1:length(wsource_files) 
                    M = [];
                    M.input = {group_mask; wsource_files{w}};
                    M.output = [wsource_files{w}(1:end-4) '_desc-' ...
                        masktag '.nii'];
                    M.outdir = {destination_dir};
                    M.expression = 'i2.*(i1>0.99)';
                    M.var = struct('name', {}, 'value', {});
                    M.options.dmtx = 0;
                    M.options.mask = 0;
                    M.options.interp = 1;
                    M.options.dtype = 4;               

                    matlabbatch{1} = {};
                    matlabbatch{1}.spm.util.imcalc=M;
                    spm_jobman('run', matlabbatch);
                    
                    % Delete non-masked file
                    % delete(sprintf('%s', wsource_files{w}));
                end
                
                % Mask smoothed normalized contrasts with 
                % the group-level whole-brain mask      
                for sm = 1:length(swsource_files)
                    MS = [];
                    MS.input = {group_mask; swsource_files{sm}};
                    MS.output = [swsource_files{sm}(2:end-4) '_desc-sm' ...
                        num2str(smoothing_kernel(1)) masktag '.nii'];
                    MS.outdir = {destination_dir};
                    MS.expression = 'i2.*(i1>0.99)';
                    MS.var = struct('name', {}, 'value', {});
                    MS.options.dmtx = 0;
                    MS.options.mask = 0;
                    MS.options.interp = 1;
                    MS.options.dtype = 4;               

                    matlabbatch{1} = {};
                    matlabbatch{1}.spm.util.imcalc=MS;
                    spm_jobman('run', matlabbatch);
                    
                    % Delete non-masked file
                    % delete(sprintf('%s', swsource_files{sm}));
                end
                
            end % dg (design)           
        end % s (sn)       
        
    case 'GLMCON:run_all'
        % Example usage: msdtb_imana('GLMCON:run_all')

        % Note: Do not forget to copy tsv files to the server
        
        sbj = subj_id; % subject list
        vararginoptions(varargin, 'sbj')
        
        glm_type = 'rwls';
        % glm_type = 'standard';
        
        hrf_cutoff = 'hrf128';
        % hrf_cutoff = 'hrf128_timederiv';
        % hrf_cutoff = 'hrf128_timedispderiv';
        % hrf_cutoff = 'hrf42';

%         msdtb_imana('GLM:grand_design_rwls', 'sn', sbj)
%         msdtb_imana('GLM:estimate_rwls', 'sn', sbj, 'output_folder', 'ffx')
%         
%         msdtb_imana('GLM:individual_ffx_t', 'sn', sbj, ...
%             'output_folder', 'ffx')
%         msdtb_imana('GLM:individual_ffx_t', 'sn', sbj, ...
%             'design', {'rand_ntfd'}, 'output_folder', 'ffx')
%                 
%         msdtb_imana('GLM:calc_PSC', 'sn', sbj, ...
%             'output_folder', 'ffx')
%         msdtb_imana('GLM:calc_PSC', 'sn', sbj, 'design', {'rand_ntfd'}, ...
%             'output_folder', 'ffx')
%                 
%         msdtb_imana('CON:norm_smooth', 'sn', sbj, ...
%             'input_folder', 'ffx')
%         msdtb_imana('CON:norm_smooth', 'sn', sbj, 'design', {'rand_ntfd'}, ...
%             'input_folder', 'ffx')
%         msdtb_imana('CON:norm_smooth', 'sn', sbj, 'file_type', 'spmT', ...
%             'input_folder', 'ffx')
%         msdtb_imana('CON:norm_smooth', 'sn', sbj, 'design', {'rand_ntfd'}, 'file_type', 'spmT', ...
%             'input_folder', 'ffx')
%         msdtb_imana('CON:norm_smooth', 'sn', sbj, 'file_type', 'psc', ...
%             'input_folder', 'ffx')
%         msdtb_imana('CON:norm_smooth', 'sn', sbj, 'design', {'rand_ntfd'}, 'file_type', 'psc', ...
%             'input_folder', 'ffx')
        
        msdtb_imana('GROUP:mask', 'sn', subj_id) % whole-brain mask
        msdtb_imana('GROUP:mask', 'sn', subj_id, 'mask_type', 'wrmask_gray') % gray-matter mask
                
        msdtb_imana('CON:masking', 'sn', sbj, ...
            'input_folder', 'ffx', ...
            'output_folder', 'masked_derivatives')
        msdtb_imana('CON:masking', 'sn', sbj, 'design', {'rand_ntfd'}, ...
            'input_folder', 'ffx', ...
            'output_folder', 'masked_derivatives')
        msdtb_imana('CON:masking', 'sn', sbj, 'file_type', 'spmT', ...
            'input_folder', 'ffx', ...
            'output_folder', 'masked_derivatives')
        msdtb_imana('CON:masking', 'sn', sbj, 'design', {'rand_ntfd'}, 'file_type', 'spmT', ...
            'input_folder', 'ffx', ...
            'output_folder', 'masked_derivatives')
        msdtb_imana('CON:masking', 'sn', sbj, 'file_type', 'psc', ...
            'input_folder', 'ffx', ...
            'output_folder', 'masked_derivatives')
        msdtb_imana('CON:masking', 'sn', sbj, 'design', {'rand_ntfd'}, 'file_type', 'psc', ...
            'input_folder', 'ffx', ...
            'output_folder', 'masked_derivatives')
        
        msdtb_imana('CON:masking', 'sn', sbj, 'masktag', 'gmmasked', ...
            'input_folder', 'ffx', ...
            'output_folder', 'masked_derivatives')
        msdtb_imana('CON:masking', 'sn', sbj, 'design', {'rand_ntfd'}, 'masktag', 'gmmasked', ...
            'input_folder', 'ffx', ...
            'output_folder', 'masked_derivatives')
        msdtb_imana('CON:masking', 'sn', sbj, 'file_type', 'spmT', 'masktag', 'gmmasked', ...
            'input_folder', 'ffx', ...
            'output_folder', 'masked_derivatives')
        msdtb_imana('CON:masking', 'sn', sbj, 'design', {'rand_ntfd'}, 'file_type', 'spmT', 'masktag', 'gmmasked', ...
            'input_folder', 'ffx', ...
            'output_folder', 'masked_derivatives')
        msdtb_imana('CON:masking', 'sn', sbj, 'file_type', 'psc', 'masktag', 'gmmasked', ...
            'input_folder', 'ffx', ...
            'output_folder', 'masked_derivatives')
        msdtb_imana('CON:masking', 'sn', sbj, 'design', {'rand_ntfd'}, 'file_type', 'psc', 'masktag', 'gmmasked', ...
            'input_folder', 'ffx', ...
            'output_folder', 'masked_derivatives')
        
    case 'GROUP:mean_t1'
        % Example usage: msdtb_imana('GROUP:mean_t1')
        
        sn = subj_id; % subject list
        vararginoptions(varargin, {'sn'});

        deriv_folder = fullfile(base_dir, derivatives_dir)
        group_anatdir = fullfile(deriv_folder, 'group/anat')
        gt1_name = 'group_t1.nii';
        
        maps={};
        mean_formula = '';
        for s = sn
            % Get the directory of subjects anatomical
            deriv_subj_dir = fullfile(deriv_folder, subj_str{s});
            subj_anatderiv_dir = fullfile(deriv_subj_dir, 'ses-01/anat');
            
            % Get the name and fullpath of the anatomical image
            normanat_name = sprintf('w%s_T1w.nii', subj_str{s});
            maps{s,1} = fullfile(subj_anatderiv_dir, normanat_name);
            
            % Create string with formula for mean of T1 images
            if s == length(sn)
                mean_formula = sprintf('(%si%d)/%d', ...
                    mean_formula, s, length(sn))
            else
                mean_formula = sprintf('%si%d+', mean_formula, s);
            end           
        end        
                 
        % Compute mean of T1 images across subjects
        spma_imcalc(maps, gt1_name, group_anatdir, mean_formula, 0)      
        
    case 'GROUP:ffx_t'
        % Estimate ffx group tmaps       
        % Example usage: msdtb_imana(
        %                   'GROUP:ffx_t', ...
        %                   'input_folder', 'snorm_standard', ...
        %                   'output_folder', 'ffx_onesample_t_standard')
        
        sn = subj_id; %subjNum
        
        design = {'prod', 'percep', 'ntfd', 'allmain_tasks'};
        % design = {'rand_ntfd'};
        input_folder = 'snorm_maps_rwls';
        output_folder = 'ffx_onesample_t_rwls';
        
        % %%%%%%%%%%% CHANGE DIRECTLY HERE FOR SPLIT DESIGN %%%%%%%%%%%%%%%
        contrasts_list = {};
        contrasts_list = contrasts;
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        vararginoptions(varargin, {'sn', 'input_folder', 'output_folder'});
        
        % spm_figure('GetWin','Graphics'); % create SPM .ps file at the end
        
        if strcmp(design, 'allmain_tasks')
            n_runs = 12
        elseif strcmp(design, 'rand_ntfd')
            n_runs = 2
        else
            n_runs = 4
        end
        
        for con = 1:length(contrasts_list)
            for dg=1:length(design)
                output_dir = fullfile(base_dir, derivatives_dir, 'group', ...
                    design{dg}, output_folder);
                contrast = strrep(contrasts_list{con,1}, ' ', '_');
                contrast_dir = fullfile(output_dir, ...
                    sprintf('con_%02d_%s', con, contrast));
                folder(contrast_dir);
                maps = {};
                resms = {};
                mean_formula = '';
                for s = sn               
                    subj_dir = fullfile(base_dir, derivatives_dir, ...
                        subj_str{s}, est_dir, design{dg}, input_folder);
                    map_name = sprintf('swcon_%04d_masked.nii', con);
                    maps{s,1} = fullfile(subj_dir, map_name);
                    resms{s,1} = fullfile(subj_dir, 'swResMS_masked.nii');
                    % Create string with formula for mean contrasts
                    if s == length(sn)
                        mean_formula = sprintf('(%si%d)/%d', ...
                            mean_formula, s, length(sn))
                    else
                        mean_formula = sprintf('%si%d+', mean_formula, s);
                    end
                end
                gcon_name = sprintf('group_swcon_%04d.nii', con);
                gresms_name = sprintf('group_swResMS.nii', con);
                 
                % Compute mean of contrasts across subjects
                spma_imcalc(maps, gcon_name, contrast_dir, mean_formula, 0)
                
                % Compute mean of ResMS across subjects
                spma_imcalc(resms, gresms_name, contrast_dir, mean_formula, 0)
                
                % Compute ffx t-statistic across subjects
                tstat_formula = sprintf('i1./sqrt(i2/(%d*%d))', length(sn), ...
                    n_runs)
                tstat_inputs = {fullfile(contrast_dir, gcon_name);
                                fullfile(contrast_dir, gresms_name)};
                tstat_outname = sprintf('groupffx_spmT_%04d', con);           
                spma_imcalc(tstat_inputs, tstat_outname, contrast_dir, ...
                    tstat_formula, 0)                
            end
        end
        
    case 'GROUP:onesample_t_design'
        % Example usage: msdtb_imana('GROUP:onesample_t_design', ...
        %                'design', {'rand_ntfd'}, ...
        %                'input_folder', 'masked_derivatives_rwls_drbb_hrf42', ...
        %                'output_folder', 'rfx_onesample_t_rwls_gm_drbb_hrf42', ...
        %                'suffix', 'sm8gmmasked')
        
        sn = subj_id; % subject list
        
        % %%%%%%%%%%%%%%%%%% DEFAULT VALUES OF VARARGIN %%%%%%%%%%%%%%%%%%%%%%%
        
        design = {'prod', 'percep', 'ntfd', 'allmain_tasks'};
        % design = {'rand_ntfd'};
        
        input_folder = 'masked_derivatives';
        output_folder = 'rfx_onesample_t';
        
        suffix = 'sm8wbmasked';
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        vararginoptions(varargin, {'sn', 'design', 'input_folder', ...
            'output_folder', 'suffix'});
        
        if strcmp(suffix, 'sm8wbmasked')
            group_mask = fullfile(base_dir, derivatives_dir, ...
                'group/anat/group_mask_noskull.nii');
        elseif strcmp(suffix, 'sm8gmmasked')
            group_mask = fullfile(base_dir, derivatives_dir, ...
                'group/anat/group_mask_gray.nii');
        end
        
        contrasts_list = {};
        if isequal(design, {'prod', 'percep', 'ntfd', 'allmain_tasks'})
            contrasts_list = contrasts;
        elseif isequal(design, {'rand_ntfd'})
            contrasts_list = contrasts_random;
        end
        
        spm_figure('GetWin','Graphics'); % create SPM .ps file at the end
        
        derivgroup_dir = fullfile(base_dir, derivatives_dir, 'group');        
        for dg=1:length(design)
            ost_dir = fullfile(derivgroup_dir, design{dg}, output_folder);
            if isfolder(ost_dir)
                rmdir(ost_dir, 's');
            end
            if not(isfolder(ost_dir))
                mkdir(ost_dir);
            end
            for sc=1:length(contrasts_list)
                contrast = strrep(contrasts_list{sc,1}, ' ', '_');
                condir = fullfile(ost_dir, ...
                    sprintf('con_%02d_%s', sc, contrast));
                folder(condir);
                swname = sprintf('wcon_%04d_desc-%s.nii', sc, suffix);
                icon = {};
                for s = sn
                    snormcon_folder = fullfile(base_dir, derivatives_dir, ...
                        subj_str{s}, est_dir, design{dg}, input_folder);
                    icon{s,1} = fullfile(snormcon_folder, swname);
                end % s (sn)
                A = [];
                A.dir = {condir};
                A.des.t1.scans = icon;
                A.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
                A.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
                A.masking.tm.tm_none = 1;
                A.masking.im = 1;
                A.masking.em = {group_mask};
                A.globalc.g_omit = 1;
                A.globalm.gmsca.gmsca_no = 1;
                A.globalm.glonorm = 1;

                matlabbatch{1}.spm.stats.factorial_design=A;
                spm_jobman('run',matlabbatch);                                  
            end % sc (n_smoothcon)
        end % dg (design)
        
    case 'GROUP:estimation'
        % Example usage: msdtb_imana('GROUP:estimation', ...
        %                            'design', {'rand_ntfd'}, ...
        %                            'model', {'rfx_onesample_t_rwls_wb_drbb_hrf42'})
        
        sn = subj_id; % subject list
        
        % %%%%%%%%%%%%%%%%%% DEFAULT VALUES OF VARARGIN %%%%%%%%%%%%%%%%%%%%%%%
        
        design = {'prod', 'percep', 'ntfd', 'allmain_tasks'};
        % design = {'rand_ntfd'};
        
        model = {'rfx_onesample_t'};

        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        vararginoptions(varargin, {'sn', 'design', 'model'})
        
        contrasts_list = {};
        if isequal(design, {'prod', 'percep', 'ntfd', 'allmain_tasks'})
            contrasts_list = contrasts;
        elseif isequal(design, {'rand_ntfd'})
            contrasts_list = contrasts_random;
        end
        
        spm_figure('GetWin','Graphics'); % create SPM .ps file at the end
        
        derivgroup_dir = fullfile(base_dir, derivatives_dir, 'group'); 
        for dg=1:length(design)
            designgroup_dir = fullfile(derivgroup_dir, design{dg});
            for m=1:length(model)
                modelgroup_dir = fullfile(designgroup_dir, model{m});
                for c=1:length(contrasts_list)
                    contrast = strrep(contrasts_list{c,1}, ' ', '_');
                    condir = fullfile(modelgroup_dir, ...
                        sprintf('con_%02d_%s', c, contrast));
                    % Delete any pre-existing nifit file
                    if any(size(dir([condir '/*.nii']), 1))
                        delete([condir '/*.nii']);
                    end
                    % Add as input SPM.mat file
                    A = [];
                    A.spmmat = {fullfile(condir, 'SPM.mat')};
                    A.write_residuals = 0;
                    A.method.Classical = 1;
                
                    matlabbatch{1}.spm.stats.fmri_est=A;
                    spm_jobman('run',matlabbatch);
                end % c (contrasts_list)
            end % m (model)
        end % dg (design)
        
    case 'GROUP:rfx_t'
        % Estimate rfx group tmaps
        % Example usage: msdtb_imana('GROUP:rfx_t', ...
        %                            'design', {'rand_ntfd'}, ...
        %                            'model', {'rfx_onesample_t_rwls_wb_drbb_hrf42'})
        
        sn = subj_id; % subject list
        
        % %%%%%%%%%%%%%%%%%% DEFAULT VALUES OF VARARGIN %%%%%%%%%%%%%%%%%%%%%%%
        
        design = {'prod', 'percep', 'ntfd', 'allmain_tasks'};
        % design = {'rand_ntfd'};
        
        model = {'rfx_onesample_t'}; % or 'rfx_onesample_t_rwls_gm_dbb_hrf128'

        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        vararginoptions(varargin, {'sn', 'design', 'model'})
        
        contrasts_list = {};
        if isequal(design, {'prod', 'percep', 'ntfd', 'allmain_tasks'})
            contrast_prefix = {'Production: ', 'Perception: ', 'NTFD: ', ...
                'AllTasks: '};
            contrasts_list = contrasts;
        elseif isequal(design, {'rand_ntfd'})
            contrast_prefix = {'Random NTFD: '};
            contrasts_list = contrasts_random;
        end
        
        spm_figure('GetWin','Graphics'); % create SPM .ps file at the end
        
        derivgroup_dir = fullfile(base_dir, derivatives_dir, 'group'); 
        for dg=1:length(design)
            designgroup_dir = fullfile(derivgroup_dir, design{dg});
            for m=1:length(model)
                modelgroup_dir = fullfile(designgroup_dir, model{m});                
                for c=1:length(contrasts_list)
                    contrast = strrep(contrasts_list{c,1}, ' ', '_');
                    condir = fullfile(modelgroup_dir, ...
                        sprintf('con_%02d_%s', c, contrast));
                    A = []; % structure with SPM fields to build the t-contrast                
                    A.spmmat = {[condir '/SPM.mat']};
                    A.consess{1}.tcon.name = [contrast_prefix{dg} ...
                        contrasts_list{c,1}];
                    A.consess{1}.tcon.weights = [1];
                    A.consess{1}.tcon.sessrep = 'none';

                    % Delete existing contrasts
                    A.delete = 1; % 1 yes, 0 no
                    matlabbatch{1}.spm.stats.con=A;
                    spm_jobman('run', matlabbatch);
                end % c (contrasts_list)
            end % m (model)
        end % dg (design)
        
    case 'GROUP:second_level_run_all'
        % Example usage: msdtb_imana('GROUP:second_level_run_all')
        
                % Example usage: msdtb_imana('GROUP:estimation', ...
        %                            'design', {'rand_ntfd'}, ...
        %                            'model', {'rfx_onesample_t_rwls_wb_drbb_hrf42'})
        
        sbj = subj_id; % subject list
        vararginoptions(varargin, 'sbj')
        
        glm_type = 'rwls';
        % glm_type = 'standard';
        
        hrf_cutoff = 'hrf128';
        % hrf_cutoff = 'hrf128_timederiv';
        % hrf_cutoff = 'hrf128_timedispderiv';
        % hrf_cutoff = 'hrf42';
        
        msdtb_imana('GROUP:onesample_t_design', 'sn', sbj, ...
            'input_folder', 'masked_derivatives_rwls_dbb_hrf128', ...
            'output_folder', 'rfx_onesample_t_rwls_dbb_hrf128_wb')
        msdtb_imana('GROUP:onesample_t_design', 'sn', sbj, ...
            'design', {'rand_ntfd'}, ...
            'input_folder', 'masked_derivatives_rwls_dbb_hrf128', ...
            'output_folder', 'rfx_onesample_t_rwls_dbb_hrf128_wb')
        msdtb_imana('GROUP:onesample_t_design', 'sn', sbj, ...
            'input_folder', 'masked_derivatives_rwls_dbb_hrf128', ...
            'output_folder', 'rfx_onesample_t_rwls_dbb_hrf128_gm', ...
            'suffix', 'sm8gmmasked')
        msdtb_imana('GROUP:onesample_t_design', 'sn', sbj, ...
            'design', {'rand_ntfd'}, ...
            'input_folder', 'masked_derivatives_rwls_dbb_hrf128', ...
            'output_folder', 'rfx_onesample_t_rwls_dbb_hrf128_gm', ...
            'suffix', 'sm8gmmasked')

        msdtb_imana('GROUP:estimation', 'sn', sbj, ...
            'model', {'rfx_onesample_t_rwls_dbb_hrf128_wb'})
        msdtb_imana('GROUP:estimation', 'sn', sbj, ...
            'design', {'rand_ntfd'}, ...
            'model', {'rfx_onesample_t_rwls_dbb_hrf128_wb'})
        msdtb_imana('GROUP:estimation', 'sn', sbj, ...
            'model', {'rfx_onesample_t_rwls_dbb_hrf128_gm'})
        msdtb_imana('GROUP:estimation', 'sn', sbj, ...
            'design', {'rand_ntfd'}, ...
            'model', {'rfx_onesample_t_rwls_dbb_hrf128_gm'})

        msdtb_imana('GROUP:rfx_t', 'sn', sbj, ...
            'model', {'rfx_onesample_t_rwls_dbb_hrf128_wb'})
        msdtb_imana('GROUP:rfx_t', 'sn', sbj, ...
            'design', {'rand_ntfd'}, ...
            'model', {'rfx_onesample_t_rwls_dbb_hrf128_wb'})
        msdtb_imana('GROUP:rfx_t', 'sn', sbj, ...
            'model', {'rfx_onesample_t_rwls_dbb_hrf128_gm'})
        msdtb_imana('GROUP:rfx_t', 'sn', sbj, ...
            'design', {'rand_ntfd'}, ...
            'model', {'rfx_onesample_t_rwls_dbb_hrf128_gm'})        
                
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