function roidb = roidb_from_vg(imdb)
    roidb.name = imdb.name;
    
    for i = 1:length(imdb.image_ids)
       mask = (imdb.obj_img_idxs_ == i);       
       gt_boxes = imdb.obj_bboxes_(mask, :);
       gt_classes = imdb.obj_classes_(mask);
       regions = imdb.region_proposals_{i};
      
       % fprintf('%s %d\n', imdb.image_ids{i}, sum(mask));
       roidb.rois(i) = attach_regions(gt_boxes, gt_classes, ...,
                            imdb.class_to_id, regions, imdb.sizes(i, :), i);
    end
end

function rec = attach_regions(gt_boxes, gt_classes, class_to_id, regions, im_size, img_idx)
    % Convert regions from [x y w h] to [x1 y1 x2 y2]
    regions(:, 3) = regions(:, 1) + regions(:, 3);
    regions(:, 4) = regions(:, 2) + regions(:, 4);
    regions = double(regions);
    
    x1 = gt_boxes(:, 1); y1 = gt_boxes(:, 2);
    x2 = gt_boxes(:, 3); y2 = gt_boxes(:, 4);
    im_height = im_size(1); im_width = im_size(2);
    bad_gt = (x1 > im_width | x2 > im_width | y1 > im_height | y2 > im_height);

    if any(bad_gt)
        gt_boxes = gt_boxes(~bad_gt, :);
        gt_classes = gt_classes(~bad_gt);
        warning('Image %d has %d bad ground-truth objects', img_idx, sum(bad_gt));
    end
    
    gt_classes = class_to_id.values(gt_classes);
    gt_classes = cat(1, gt_classes{:});
    
    num_gt_boxes = size(gt_boxes, 1);
    num_regions = size(regions, 1);
    all_boxes = cat(1, gt_boxes, regions);
    
    rec.gt = cat(1, true(num_gt_boxes, 1), false(num_regions, 1));
    rec.overlap = zeros(num_gt_boxes + num_regions, class_to_id.Count, 'single');
    for i = 1:num_gt_boxes
        rec.overlap(:, gt_classes(i)) = ...
            max(rec.overlap(:, gt_classes(i)), boxoverlap_(all_boxes, gt_boxes(i, :)));
    end
    rec.boxes = single(all_boxes);
    rec.feat = [];
    rec.class = uint16(cat(1, gt_classes, zeros(num_regions, 1)));
end

function o = boxoverlap_(a, b)
% Compute the symmetric intersection over union overlap between a set of
% bounding boxes in a and a single bounding box in b.
%
% a  a matrix where each row specifies a bounding box
% b  a single bounding box

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2011-2012 Ross Girshick
% Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------
    x1 = max(a(:,1), b(1));
    y1 = max(a(:,2), b(2));
    x2 = min(a(:,3), b(3));
    y2 = min(a(:,4), b(4));

    w = x2-x1+1;
    h = y2-y1+1;
    inter = w.*h;
    aarea = (a(:,3)-a(:,1)+1) .* (a(:,4)-a(:,2)+1);
    barea = (b(3)-b(1)+1) * (b(4)-b(2)+1);
    % intersection over union overlap
    o = inter ./ (aarea+barea-inter);
    % set invalid entries to 0 overlap
    o(w <= 0) = 0;
    o(h <= 0) = 0;
end