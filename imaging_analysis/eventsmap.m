function [original_events, renamed_events] = eventsmap(subject)

switch true
    case strcmp(subject, 'sub-14')
        % Event Files
        events_ses1 = [            
            "ses-01_task-prod_run-01",   "ses-01_task-prod_run-02",   ...
            "ses-01_task-percep_run-01", "ses-01_task-percep_run-02", ...
            "ses-01_task-ntfd_run-01",   "ses-01_task-ntfd_run-02"
            ];

        events_ses2 = [
            "ses-02_task-prod_run-01",   ...
            "ses-02_task-percep_run-01", "ses-02_task-percep_run-02", ...
            "ses-02_task-ntfd_run-01",   "ses-02_task-ntfd_run-02"
            ];

        original_events_ses3 = [
            "ses-02_task-prod_run-02",   ...
            "ses-02_task-ntfd_run-03",   "ses-02_task-ntfd_run-04"
            ];

        renamed_events_ses3 = [
            "ses-03_task-prod_run-02",   ...
            "ses-03_task-ntfd_run-03",   "ses-03_task-ntfd_run-04"
            ];

        original_events = {events_ses1, events_ses2, original_events_ses3};
        renamed_events = {events_ses1, events_ses2, renamed_events_ses3};

    otherwise
        % Event Files
        events_ses1 = [            
            "ses-01_task-prod_run-01",   "ses-01_task-prod_run-02",   ...
            "ses-01_task-percep_run-01", "ses-01_task-percep_run-02", ...
            "ses-01_task-ntfd_run-01",   "ses-01_task-ntfd_run-02"
            ];

        events_ses2 = [
            "ses-02_task-prod_run-01",   "ses-02_task-prod_run-02", ...
            "ses-02_task-percep_run-01", "ses-02_task-percep_run-02", ...
            "ses-02_task-ntfd_run-01",   "ses-02_task-ntfd_run-02", ...
            "ses-02_task-ntfd_run-03",   "ses-02_task-ntfd_run-04"
            ];

        original_events = {events_ses1, events_ses2};
        renamed_events = original_events;
end
