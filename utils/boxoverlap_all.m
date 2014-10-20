function iou = boxoverlap_all(a, b)
% Compute symmetric overlap over union between all pairs of bounding boxes.
% a - N x 4 array where each row is [x1 y1 x2 y2]
% b - M x 4 array where each row is [x1 y1 x2 y2]
% 
% If a and b have drastically different sizes this will be much faster if
% b is smaller.
% 
% Returns: N x M array of intersections over unions.

    N = size(a, 1);
    M = size(b, 1);
    iou = zeros(N, M);
    for j = 1:M
        iou(:, j) = boxoverlap(a, b(j, :));
    end

end