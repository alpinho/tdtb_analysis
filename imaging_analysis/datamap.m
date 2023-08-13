function [raw_ses, preproc_ses, raw_runs, preproc_runs] = datamap(subject)


switch true
    case strcmp(subject, 'sub-03')
        
        % Sessions
        raw_ses = {"ses-1", "ses-2"};
        preproc_ses = {"ses-01", "ses-02"};
        
        % Runs
        runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        runs_ses2 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02", "task-ntfd_run-03", "task-ntfd_run-04"
            ];
        raw_runs = {runs_ses1, runs_ses2};
        preproc_runs = raw_runs;

        
    case strcmp(subject, 'sub-04')
        
        % Sessions
        raw_ses = {"ses-1", "ses-2"};
        preproc_ses = {"ses-01", "ses-02"};
        
        % Runs
        runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        runs_ses2 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02", "task-ntfd_run-03", "task-ntfd_run-04"
            ];
        raw_runs = {runs_ses1, runs_ses2};
        preproc_runs = raw_runs;

 
    case strcmp(subject, 'sub-07')
        
        % Sessions
        raw_ses = {"ses-1", "ses-2"};
        preproc_ses = {"ses-01", "ses-02"};
        
        % Runs
        runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        runs_ses2 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02", "task-ntfd_run-03", "task-ntfd_run-04"
            ];
        raw_runs = {runs_ses1, runs_ses2};
        preproc_runs = raw_runs;


    case strcmp(subject, 'sub-08')
        
        % Sessions
        raw_ses = {"ses-1", "ses-2"};
        preproc_ses = {"ses-01", "ses-02"};
        
        % Runs
        runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        runs_ses2 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02", "task-ntfd_run-03", "task-ntfd_run-04"
            ];
        raw_runs = {runs_ses1, runs_ses2};
        preproc_runs = raw_runs;


    case strcmp(subject, 'sub-14') % 3 sessions
        
        % Sessions
        raw_ses = {"ses-01", "ses-02", "ses-03"};
        preproc_ses = raw_ses;
        
        % Runs
        raw_runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-03", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        preproc_runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        runs_ses2 = [
            "task-prod_run-01",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        raw_runs_ses3 = [
            "task-prod_run-01",   ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        preproc_runs_ses3 = [
            "task-prod_run-02",   ...
            "task-ntfd_run-03",   "task-ntfd_run-04"
            ];
        raw_runs = {raw_runs_ses1, runs_ses2, raw_runs_ses3};
        preproc_runs = {preproc_runs_ses1, runs_ses2, preproc_runs_ses3};

        
    case strcmp(subject, 'sub-18')
        
        % Sessions
        raw_ses = {"ses-01", "ses-02"};
        preproc_ses = raw_ses;
        
        % Runs
        raw_runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-03",   "task-ntfd_run-04"
            ];
        preproc_runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        runs_ses2 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02", "task-ntfd_run-03", "task-ntfd_run-04"
            ];
        raw_runs = {raw_runs_ses1, runs_ses2};
        preproc_runs = {preproc_runs_ses1, runs_ses2};

 
    case strcmp(subject, 'sub-32')
        
        % Sessions
        raw_ses = {"ses-01", "ses-02"};
        preproc_ses = raw_ses;
        
        % Runs
        raw_runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-06",   "task-ntfd_run-07"
            ];
        preproc_runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        runs_ses2 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02", "task-ntfd_run-03",   "task-ntfd_run-04"
            ];
        raw_runs = {raw_runs_ses1, runs_ses2};
        preproc_runs = {preproc_runs_ses1, runs_ses2};
        

    case strcmp(subject, 'sub-34')
        
        % Sessions
        raw_ses = {"ses-01", "ses-02"};
        preproc_ses = raw_ses;
        
        % Runs
        raw_runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-02", "task-percep_run-03", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        preproc_runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        runs_ses2 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02", "task-ntfd_run-03",   "task-ntfd_run-04"
            ];
        raw_runs = {raw_runs_ses1, runs_ses2};
        preproc_runs = {preproc_runs_ses1, runs_ses2};


    case strcmp(subject, 'sub-38')
        
        % Sessions
        raw_ses = {"ses-01", "ses-02"};
        preproc_ses = raw_ses;
        
        % Runs
        raw_runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-03", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        preproc_runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        raw_runs_ses2 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-02", "task-percep_run-03", ...
            "task-ntfd_run-01",   "task-ntfd_run-02", "task-ntfd_run-03", "task-ntfd_run-04"
            ];
        preproc_runs_ses2 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02", "task-ntfd_run-03", "task-ntfd_run-04"
            ];

        raw_runs = {raw_runs_ses1, raw_runs_ses2};
        preproc_runs = {preproc_runs_ses1, preproc_runs_ses2};

        
    case strcmp(subject, 'sub-40')
        
        % Sessions
        raw_ses = {"ses-01", "ses-02"};
        preproc_ses = raw_ses;
        
        % Runs
        raw_runs_ses1 = [
            "task-prod_run-02",   "task-prod_run-03",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-02",   "task-ntfd_run-03"
            ];
        preproc_runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        runs_ses2 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02", "task-ntfd_run-03", "task-ntfd_run-04"
            ];

        raw_runs = {raw_runs_ses1, runs_ses2};
        preproc_runs = {preproc_runs_ses1, runs_ses2};
        
    
    case strcmp(subject, 'sub-42') % 3 sessions
        
        % Sessions
        raw_ses = {"ses-01", "ses-02", "ses-03"};
        preproc_ses = raw_ses;
        
        % Runs
        runs_ses1 = [
            "task-prod_run-01",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",
            ];
        raw_runs_ses2 = [
            "task-prod_run-01",   ...
            "task-ntfd_run-01"
            ];
        preproc_runs_ses2 = [
            "task-prod_run-02",   ...
            "task-ntfd_run-02"
            ];
        raw_runs_ses3 = [
            "task-prod_run-02",   "task-prod_run-03", ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02", "task-ntfd_run-03", "task-ntfd_run-04"
            ];
        preproc_runs_ses3 = [
            "task-prod_run-01",   "task-prod_run-02", ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02", "task-ntfd_run-03", "task-ntfd_run-04"
            ];
        raw_runs = {runs_ses1, raw_runs_ses2, raw_runs_ses3};
        preproc_runs = {runs_ses1, preproc_runs_ses2, preproc_runs_ses3};
        
 
    case strcmp(subject, 'sub-43') % 3 sessions
        
        % Sessions
        raw_ses = {"ses-01", "ses-02", "ses-03"};
        preproc_ses = raw_ses;
        
        % Runs
        runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02", ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02",
            ];
        runs_ses2 = [
            "task-prod_run-01", ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01"
            ];
        raw_runs_ses3 = [
            "task-prod_run-01",   ...
            "task-ntfd_run-01", "task-ntfd_run-02", "task-ntfd_run-03"
            ];
        preproc_runs_ses3 = [
            "task-prod_run-02",   ...
            "task-ntfd_run-02", "task-ntfd_run-03", "task-ntfd_run-04"
            ];
        raw_runs = {runs_ses1, runs_ses2, raw_runs_ses3};
        preproc_runs = {runs_ses1, runs_ses2, preproc_runs_ses3};


    case strcmp(subject, 'sub-45')
        
        % Sessions
        raw_ses = {"ses-01", "ses-02"};
        preproc_ses = raw_ses;
        
        % Runs
        runs_ses1 = [
            "task-prod_run-01", "task-prod_run-02", ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01", "task-ntfd_run-02",
            ];
        raw_runs_ses2 = [
            "task-prod_run-01", "task-prod_run-02", ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-03", "task-ntfd_run-04", "task-ntfd_run-05", "task-ntfd_run-06"
            ];
        preproc_runs_ses2 = [
            "task-prod_run-01", "task-prod_run-02", ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01", "task-ntfd_run-02", "task-ntfd_run-03", "task-ntfd_run-04"
            ];
        raw_runs = {runs_ses1, raw_runs_ses2};
        preproc_runs = {runs_ses1, preproc_runs_ses2};


    otherwise
        
        % Sessions
        raw_ses = {"ses-01", "ses-02"};
        preproc_ses = raw_ses;
        
        % Runs
        runs_ses1 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02"
            ];
        runs_ses2 = [
            "task-prod_run-01",   "task-prod_run-02",   ...
            "task-percep_run-01", "task-percep_run-02", ...
            "task-ntfd_run-01",   "task-ntfd_run-02", "task-ntfd_run-03",   "task-ntfd_run-04"
            ];
        raw_runs = {runs_ses1, runs_ses2};
        preproc_runs = raw_runs;


end