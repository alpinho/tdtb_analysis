function spma_imcalc(input_paths, output_name, output_dir, formula, mask_val)
% 'input_paths', cell
% 'output_name', string
% 'output_dir', string
% 'formula', string
% 'mask_val', double

A = [];
A.input = input_paths;
A.output = output_name;
A.outdir = {output_dir};
A.expression = formula;
A.var = struct('name', {}, 'value', {});
A.options.dmtx = 0;
A.options.mask = mask_val;
A.options.interp = 1;
A.options.dtype = 4;               

matlabbatch{1} = {};
matlabbatch{1}.spm.util.imcalc=A;
spm_jobman('run', matlabbatch);