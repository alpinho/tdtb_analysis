function spmja_normalization_write(deformation_file, images,varargin)
% function spmj_normalization_write(deformation_file, images)
% INPUTS:
%   deformation_file:     the *_sn.mat file for the deformation
%   images:               a cell array of the images you want to resmape
% VARARGIN:
%   'outimages',cell:  names (and path) under which you want to store
%                         the resampled images. Otherwise they will be
%                         stored under the same directory with prefix 'w'
%   'voxel_size', double: voxel size of the normalized image(s)

%12.09.2012 TW: change bounding box from [-78 -112 -50 78   76  85] to [-78 -112 -50 78   76  100]
if ischar(images)
    images=cellstr(images);
end;
outimages=[];

voxel_size = [2 2 2];

vararginoptions(varargin,{'outimages', 'voxel_size'});

J.subj.def = {deformation_file};
J.subj.resample = images;
J.woptions.bb = [-78 -112 -70
    78   76  85];
J.woptions.vox = voxel_size;
J.woptions.interp = 4;
J.woptions.prefix = 'w';

matlabbatch{1}.spm.spatial.normalise.write=J;
spm_jobman('run',matlabbatch);
