
%%% This function cuts an aligned pair of styled and plain exemplars into patches.
%%% Input parameters:
% styled: path to styled exemplar (currently doesn't deal with transparent background images)
% plain: path to plain exemplar (currently doesn't deal with transparent background images)
% output_dir: the patches will be saved to this folder, such that styled
%             patches will be saved to a sub-folder titled 'styled' and plain
%             patches to a sub-folder titled 'plain'
% patchsize: required patch size (we used 64 in most of our experiments)
% scale: resize exemplar according to provided scale (our rule of thumb is
%        that patches should contain visible parts of strokes, not too zoomed in
%        so that the stroke is blown up and not too zoomed out so that intricate
%        geometries are visible within the patch)
% freq: patch sampling frequency/density (if -1 the script will estimate
%       the frequency based on a desired amount of ~100000 patches)
% is_white_bg: 0 for black background exemplar and 1 for a white background exemplar
% is_white_bg_output: required background of output patches, 0 for black
%                     and 1 for white (when training a grayscale exemplar use 0, 
%                     and when training an RGB exemplar use 1)


function cut_aligned_exemplar(styled, plain, output_dir, patchsize, scale, freq, is_white_bg, is_white_bg_output)

mkdir(output_dir);
soutput_dir = [output_dir filesep 'styled'];
poutput_dir = [output_dir filesep 'plain'];
mkdir(soutput_dir);
mkdir(poutput_dir);

nump = 100000;
nump_per_im = nump;

plainim = imread(plain);

if(freq == -1)
    im = imresize(plainim, scale);
    if(length(size(im)) == 3)
        gim = rgb2gray(im);
    else
        gim = im;
    end

    if(is_white_bg)
        gim = 255 - gim;
    end
    cim = crop_im(gim);
    sz = size(cim);

    stp3 = (sz(1) - patchsize) * (sz(2) - patchsize) * 360 / nump_per_im;
    stp = stp3 ^ (1/3);
    rcim = imrotate(255*cim,45);
    rcim = crop_im(rcim);
    rsz = size(rcim);
    stp3 = (rsz(1) - patchsize) * (rsz(2) - patchsize) * 360 / nump_per_im;
    rstp = stp3 ^ (1/3);

    stp = round((stp + rstp) / 2);
else
    stp = round(freq);
end


stlim = imread(styled);

plainim = imresize(plainim, scale);
stlim = imresize(stlim, scale);
if(length(size(plainim)) == 3)
    maskp = rgb2gray(plainim);
else
    maskp = plainim;
end
if(length(size(stlim)) == 3)
    masks = rgb2gray(stlim);
else
    masks = stlim;
end

if(is_white_bg)
    maskp = 255 - maskp;
    masks = 255 - masks;
end
[~, px1, px2, py1, py2] = crop_im(maskp);
[~, sx1, sx2, sy1, sy2] = crop_im(masks);
x1 = min(px1,sx1);
x2 = max(px2,sx2);
y1 = min(py1,sy1);
y2 = max(py2,sy2);
cmask = maskp(x1:x2,y1:y2);
pim = plainim(x1:x2,y1:y2,:);
sim = stlim(x1:x2,y1:y2,:);

[filepath,name,ext] = fileparts(styled);
ssavename = [soutput_dir filesep name];
psavename = [poutput_dir filesep name];
ims_cut_patches(sim,pim,cmask,patchsize,stp,is_white_bg,is_white_bg_output,ssavename,psavename);

end


function ims_cut_patches(sim,pim,mask,patchsize,stp,is_white_bg,is_white_bg_output,ssavename,psavename)

fp = patchsize ^ 2;

if(is_white_bg)
    fill_val = 255;
else
    fill_val = 0;
end


for ang=0:stp:359
    cs = cos(deg2rad(ang));
    sn = sin(deg2rad(ang));
    rotmat = [cs sn 0; -sn cs 0; 0 0 1];
    tform = maketform('affine',rotmat);
    sim2 = imtransform(sim,tform,'bicubic','fill',fill_val);
    pim2 = imtransform(pim,tform,'bicubic','fill',fill_val);
    mask2 = imtransform(mask,tform,'bicubic','fill',0);
    sim2 = pad_im(sim2,is_white_bg,patchsize);
    pim2 = pad_im(pim2,is_white_bg,patchsize);
    mask2 = pad_im(mask2,0,patchsize);
    sz = size(sim2);
    
    for x=1:stp:(sz(1)-patchsize+1)
        for y=1:stp:(sz(2)-patchsize+1)
            gp = mask2(x:x+patchsize-1, y:y+patchsize-1);
            nz = find(gp ~= 0);
            ratio = length(nz) / fp;
            if(ratio > 0.9 || ratio < 0.1)
                continue;
            end
            ps = sim2(x:x+patchsize-1, y:y+patchsize-1, :);
            pp = pim2(x:x+patchsize-1, y:y+patchsize-1, :);
            ssavename2 = sprintf('%s_%d_%d_%d.png', ssavename, ang, x, y);
            psavename2 = sprintf('%s_%d_%d_%d.png', psavename, ang, x, y);
            if(is_white_bg ~= is_white_bg_output)
                ps = 255-ps;
                pp = 255-pp;
            end
            imwrite(ps, ssavename2);
            imwrite(pp, psavename2);
        end
    end
    
end


end


function pim = pad_im(im, is_white_bg, patchsize)

hpsz = round(patchsize / 2);

if(is_white_bg)
    padval = 255;
else
    padval = 0;
end

pim = padarray(im,[hpsz, hpsz], padval, 'both');

end


function im = threshold_im(im)

% assumes image is grayscale with dark bg
im(im < 50) = 0;
im(im >= 50) = 255;

end


function [im2,x1,x2,y1,y2] = crop_im(im)

% assumes image is grayscale with dark bg
% first need to threshold
im = threshold_im(im);
nz = find(im);
[x,y] = ind2sub(size(im),nz);
x1 = min(x);
x2 = max(x);
y1 = min(y);
y2 = max(y);
im2 = im(x1:x2,y1:y2);

end

