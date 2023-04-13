function rnames = reorder_regressors(unames, tag, task)

rnames = {};

switch true
    case strcmp(tag, 'mr_drbb_events') && ~strcmp(task, 'rand_ntfd')
        rnames(1:2)=unames(1:2);
        rnames(3:4)=unames(5:6);
        rnames(5)=unames(3);
        rnames(6)=unames(4);
    case strcmp(tag, 'mr_drbb_events') && strcmp(task, 'rand_ntfd')
        rnames(1:3)=unames(1:3);
        rnames(4:6)=unames(6:8);
        rnames(7)=unames(4);
        rnames(8)=unames(5);
    case strcmp(tag, 'mr_dbb_events') && ~strcmp(task, 'rand_ntfd')
        rnames(1:2)=unames(1:2);
        rnames(3:4)=unames(4:5);
        rnames(5)=unames(3);
    case strcmp(tag, 'mr_dbb_events') && strcmp(task, 'rand_ntfd')
        rnames(1:3)=unames(1:3);
        rnames(4:6)=unames(5:7);
        rnames(7)=unames(4);
    case strcmp(tag, 'mr_brbb_events') && ~strcmp(task, 'rand_ntfd')
        rnames(1:2)=unames(1:2);
        rnames(3:4)=unames(4:5);
        rnames(5)=unames(3);
    case strcmp(tag, 'mr_brbb_events') && strcmp(task, 'rand_ntfd')
        rnames(1:3)=unames(1:3);
        rnames(4:6)=unames(5:7);
        rnames(7)=unames(4);
    case strcmp(tag, 'splitdesign_events') && ~strcmp(task, 'rand_ntfd')
        rnames(1)=unames(2);
        rnames(2)=unames(1);
        rnames(3)=unames(4);
        rnames(4)=unames(3);
        rnames(5)=unames(7);
        rnames(6)=unames(6);
        rnames(7)=unames(9);
        rnames(8)=unames(8);
        rnames(9)=unames(5);
    case strcmp(tag, 'splitdesign_events') && strcmp(task, 'rand_ntfd')
        rnames(1)=unames(2);
        rnames(2)=unames(1);
        rnames(3)=unames(4);
        rnames(4)=unames(3);
        rnames(5)=unames(5);
        rnames(6)=unames(8);
        rnames(7)=unames(7);
        rnames(8)=unames(10);
        rnames(9)=unames(9);
        rnames(10)=unames(11);
        rnames(11)=unames(6);
    case ~strcmp(tag, 'splitdesign_events') && strcmp(task, 'rand_ntfd')
        rnames(1:3)=unames(1:3);
        rnames(4:6)=unames(5:7);
        rnames(7)=unames(4);
    case ~strcmp(tag, '') && ~strcmp(task, 'rand_ntfd')
        rnames(1:2)=unames(1:2);
        rnames(3:4)=unames(4:5);
        rnames(5)=unames(3);
end

% % Non-merged decision model
% if strcmp(...
%         events_file_tag, 'splitdesign_events') ...
%         && ~strcmp(design{dg}, 'rand_ntfd')
%     % as well as low condition for the split design
%     names(3:3:length(unique_names))= ...
%         unique_names(1:3:end);
%     names(1:3:length(unique_names))= ...
%         unique_names(3:3:end);
%     names(2:3:length(unique_names))= ...
%         unique_names(2:3:end);
% else
%     names(2:2:length(unique_names))= ...
%         unique_names(1:2:end);
%     names(1:2:length(unique_names))= ...
%         unique_names(2:2:end);
% end