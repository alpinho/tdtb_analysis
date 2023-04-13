function reorder_regressors(unique_names, tag, task)

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

names = {};

switch true
    case strcmp(tag, 'splitdesign_events') && ~strcmp(task, 'rand_ntfd')
    
end
% Merged decision model
if strcmp(tag, 'splitdesign_events') ...
        && ~strcmp(task, 'rand_ntfd')
    names(1)=unique_names(2);
    names(2)=unique_names(1);
    names(3)=unique_names(4);
    names(4)=unique_names(3);
    names(5)=unique_names(7);
    names(6)=unique_names(6);
    names(7)=unique_names(9);
    names(8)=unique_names(8);
    names(9)=unique_names(5);
elseif strcmp(tag, 'splitdesign_events') ...
        && strcmp(task, 'rand_ntfd')
    names(1)=unique_names(2);
    names(2)=unique_names(1);
    names(3)=unique_names(4);
    names(4)=unique_names(3);
    names(5)=unique_names(5);
    names(6)=unique_names(8);
    names(7)=unique_names(7);
    names(8)=unique_names(10);
    names(9)=unique_names(9);
    names(10)=unique_names(11);
    names(11)=unique_names(6);
else
    names(1:2)=unique_names(1:2);
    names(3:4)=unique_names(4:5);
    names(5)=unique_names(3);
end

if ~strcmp(tag, 'splitdesign_events') ...
        && strcmp(task, 'rand_ntfd')
    names(1:3)=unique_names(1:3);
    names(4:6)=unique_names(5:7);
    names(7)=unique_names(4);
elseif strcmp(tag, 'mr_drbb_events') ...
        && ~strcmp(task, 'rand_ntfd')
    names(1:2)=unique_names(1:2)
    names(3:4)=unique_names(5:6)
    names(5)=unique_names(3)
    names(6)=unique_names(4)
elseif strcmp(tag, 'mr_drbb_events') ...
        && strcmp(task, 'rand_ntfd')
    names(1:3)=unique_names(1:3)
    names(4:6)=unique_names(6:8)
    names(7)=unique_names(4)
    names(8)=unique_names(5)
end