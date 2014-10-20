function res = imdb_eval_vg(cls, boxes, imdb, suffix, nms_thresh)

    if ~exist('suffix', 'var') || isempty(suffix) || strcmp(suffix, '')
      suffix = '';
    else
      if suffix(1) ~= '_'
        suffix = ['_' suffix];
      end
    end

    if ~exist('nms_thresh', 'var') || isempty(nms_thresh)
      nms_thresh = 0.3;
    end

    roidb = imdb.roidb_func(imdb);
    class_id = imdb.class_to_id(cls);
    num_images = length(imdb.image_ids);
    dets = cell(num_images, 1);
    gt = cell(num_images, 1);
    for i = 1:num_images
        rois = roidb.rois(i);
%         I = find(rois.gt && rois.class == class_id);
        gt{i} = rois.boxes(rois.gt & rois.class == class_id, :);
        keep = nms(boxes{i}, nms_thresh);
        dets{i} = boxes{i}(keep, :);
    end
    
    [precision, recall, ap] = evaluate_detections(dets, gt, 0.5);
    ap_auc = xVOCap(recall, precision);
    
    res.recall = recall;
    res.precision = precision;
    res.ap = ap;
    res.ap_auc = ap_auc;
end