function [precision, recall, ap] = evaluate_detections(dets, gt, thresh)
    % Evaluate detections for a single class by computing precision,
    % recall, and average precision.
    %
    % INPUTS
    % dets: Cell array where dets{i} gives the detections for the ith
    %       image. dets{i} is a matrix where each row is
    %       [x1 y1 x2 y2 score].
    % gt: Cell array where gt{i} gives the ground-truth bounding boxes for
    %     the ith image. Each row of gt{i} has the form [x1 y1 x2 y2].
    % thresh: Threshold on intersection over union for detections to be
    %         considered true positives.

    assert(length(dets) == length(gt));
    
    num_images = length(dets);
    num_positive = 0;
    ious = cell(num_images, 1);
    % Compute IOU between all detections and all objects in all images.
    % Also append a column to each detections matrix with the image index
    % and another with the index of that detection within that images.
    for i = 1:num_images
        ious{i} = boxoverlap_all(dets{i}, gt{i});
        num_dets = size(dets{i}, 1);
        dets{i} = [dets{i}, i * ones(num_dets, 1), (1:num_dets)'];
        num_positive = num_positive + size(gt{i}, 1);
    end
    dets_stacked = cat(1, dets{:});
    num_total_dets = size(dets_stacked, 1);
    
    [~, ord] = sort(dets_stacked(:, 5), 'descend');
    fp = zeros(num_total_dets, 1);
    tp = zeros(num_total_dets, 1);
    for i = 1:num_total_dets
        det_idx = ord(i);
        img_idx = dets_stacked(det_idx, 6);
        ii = dets_stacked(det_idx, 7);
        [t, j] = max(ious{img_idx}(ii, :));
        if t >= thresh
            disp(dets_stacked(det_idx, :));
            tp(i) = 1;
            % Zero out the IOU with the ground truth box and all other
            % detections to ensure that no other detections match to this
            % ground truth box.
            ious{img_idx}(:, j) = 0;
        else
            fp(i) = 1;
        end
    end
    
    % The rest is taken straight from VOCevaldet so it ought to work.
    fp = cumsum(fp);
    tp = cumsum(tp);
    disp(tp);
    disp(fp);
    disp(num_positive);
    recall = tp / num_positive;
    precision = tp ./ (fp + tp);
    
    ap = 0;
    for t = 0:0.1:1
        p = max(precision(recall >= t));
        if isempty(p)
            p = 0;
        end
        ap = ap + p / 11;
    end
end