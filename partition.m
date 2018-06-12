function [train tl valid vl] = partition (tset, tlab, cf)
% paritions tset into training and validation parts
% cf - coefficient of samples in the train set

  shuffle_idx = randperm(rows(tset))';
  train_idx = shuffle_idx(1:rows(tset)*cf);
  valid_idx = shuffle_idx(rows(train_idx)+1:end);
  train = tset(train_idx, :);
  tl = tlab(train_idx);
  valid = tset(valid_idx, :);
  vl = tlab(valid_idx); 

endfunction
