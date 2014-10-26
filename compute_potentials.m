function potentials = compute_potentials(rcnn_model, imdb)

  conf = rcnn_config('sub_dir', imdb.name);
  image_ids = imdb.image_ids;
  num_images = length(image_ids);
  feat_opts = rcnn_model.training_opts;
  num_classes = length(rcnn_model.classes);

  aboxes = cell(num_classes, 1);
  box_inds = cell(num_classes, 1);
  for i = 1:num_classes
    aboxes{i} = cell(length(image_ids), 1);
    box_inds{i} = cell(length(image_ids), 1);
  end

  if ~isfield(rcnn_model, 'folds')
    folds{1} = 1:length(image_ids);
  else
    folds = rcnn_model.folds;
  end
  if length(folds) > 1
    error('Model must have one fold');
  end

  potentials = struct();
  potentials.image_ids = image_ids;
  potentials.classes = rcnn_model.classes;
  potentials.boxes = cell(num_images, 1);
  potentials.scores = cell(num_images, 1);
  potentials.overlap = cell(num_images, 1);
  
  potentials.image_id_to_idx = containers.Map();
  for i = 1:length(image_ids)
      potentials.image_id_to_idx(image_ids{i}) = i;
  end
  
  potentials.class_to_idx = containers.Map();
  for i = 1:length(potentials.classes)
      potentials.class_to_idx(potentials.classes{i}) = i;
  end

  count = 0;
  for f = 1:length(folds)
    for i = folds{f}
      count = count + 1;
      fprintf('%s: extracting (%s) %d/%d\n', procid(), imdb.name, count, ...
              length(image_ids));
      d = rcnn_load_cached_pool5_features(feat_opts.cache_name, ...
            imdb.name, image_ids{i});
      if isempty(d.feat)
        continue;
      end
      d.feat = rcnn_pool5_to_fcX(d.feat, feat_opts.layer, rcnn_model);
      d.feat = rcnn_scale_features(d.feat, feat_opts.feat_norm_mean);
      zs = bsxfun(@plus, d.feat*rcnn_model.detectors(f).W, ...
                  rcnn_model.detectors(f).B);
      % zs is [num_boxes x num_obj_classes]
      potentials.boxes{i} = d.boxes(~d.gt, :);
      potentials.scores{i} = zs(~d.gt, :);
      potentials.overlap{i} = d.overlap(~d.gt, :);
    end
  end

end
