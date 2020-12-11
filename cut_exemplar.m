
%%% This function cuts an exemplar into patches.
%%% Input parameters:
% ex: path to exemplar, could be either a single image exemplar, or a folder of
%     images featuring the same style (currently doesn't deal with transparent 
%     background images)
% output_dir: the patches will be saved to this folder
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


function cut_exemplar(ex, output_dir, patchsize, scale, freq, is_white_bg, is_white_bg_output)

exs = dir(ex);

if(isdir(ex))
    exs = exs(3:end);
    idir = ex;
else
    idir = '.';
end

mkdir(output_dir);

numim = length(exs);
nump = 100000;
nump_per_im = nump / numim;

test = [idir filesep exs(1).name];
testim = imread(test);

if(freq == -1)
    im = imresize(testim, scale);
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


for i=1:length(exs)
    cex = [idir filesep exs(i).name];
    im = imread(cex);
    im = imresize(im, scale);
    if(length(size(im)) == 3)
        gim = rgb2gray(im);
    else
        gim = im;
    end
    
    if(is_white_bg)
        gim = 255 - gim;
    end
    [cgim, x1, x2, y1, y2] = crop_im(gim);
    cim = im(x1:x2,y1:y2,:);
    nm = exs(i).name;
    [filepath,name,ext] = fileparts(nm);
    savename = [output_dir filesep name];
    im_cut_patches(cim,cgim,patchsize,stp,is_white_bg,is_white_bg_output,savename);
end

end


function im_cut_patches(im,gim,patchsize,stp,is_white_bg,is_white_bg_output,savename)

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
    im2 = imtransform(im,tform,'bicubic','fill',fill_val);
    gim2 = imtransform(gim,tform,'bicubic','fill',0);
    im2 = pad_im(im2,is_white_bg,patchsize);
    gim2 = pad_im(gim2,0,patchsize);
    
    sz = size(im2);
    
    for x=1:stp:(sz(1)-patchsize+1)
        for y=1:stp:(sz(2)-patchsize+1)
            gp = gim2(x:x+patchsize-1, y:y+patchsize-1);
            nz = find(gp ~= 0);
            ratio = length(nz) / fp;
            if(ratio > 0.9 || ratio < 0.1)
                continue;
            end
            p = im2(x:x+patchsize-1, y:y+patchsize-1,:);
            savename2 = sprintf('%s_%d_%d_%d.png', savename, ang, x, y);
            if(is_white_bg ~= is_white_bg_output)
                p = 255-p;
            end
            imwrite(p, savename2);
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

