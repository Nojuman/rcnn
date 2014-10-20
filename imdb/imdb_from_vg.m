function imdb = imdb_from_vg(data_path)
    load(data_path, 'imdb', 'obj_bboxes', 'obj_classes', 'obj_img_idxs', ...
                    'region_proposals');
    
    % Attach extra stuff to imdb that is easier to do in matlab
    imdb.num_classes = length(imdb.classes);
    imdb.class_to_id = ...
        containers.Map(imdb.classes, 1:imdb.num_classes);
    imdb.class_ids = 1:imdb.num_classes;
    
    imdb.image_at = @(i) ...
        sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
    
    for i = 1:length(imdb.image_ids)
        info = imfinfo(imdb.image_at(i));
        imdb.sizes(i, :) = [info.Height, info.Width];
    end
    
    imdb.roidb_func = @roidb_from_vg;
    
    % Private VG details
    imdb.obj_bboxes_ = obj_bboxes;
    imdb.obj_classes_ = obj_classes;
    imdb.obj_img_idxs_ = obj_img_idxs;
    imdb.region_proposals_ = region_proposals;
end