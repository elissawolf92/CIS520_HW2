function [fidx val max_ig] = dt_choose_feature_multi(X, Z, Xrange, colidx)
% DT_CHOOSE_FEATURE_MULTI - Selects feature with maximum multi-class IG.
%
% Usage:
% 
%   [FIDX FVAL MAX_IG] = dt_choose_feature(X, Z, XRANGE, COLIDX)
%
% Given N x D data X and N x K indicator labels Z, where X(:,j) can take on values in XRANGE{j}, chooses
% the split X(:,FIDX) <= VAL to maximize information gain MAX_IG. I.e., FIDX is
% the index (chosen from COLIDX) of the feature to split on with value
% FVAL. MAX_IG is the corresponding information gain of the feature split.
%
% Note: The relationship between Y and Z is that Y(i) = find(Z(i,:)).
% Z is the categorical representation of Y: Z(i,:) is a vector of all zeros
% except for a one in the Y(i)'th column.
% 
% Hint: It is easier to compute entropy, etc. when using Z instead of Y.
%
% SEE ALSO
%    DT_TRAIN_MULTI

% YOUR CODE GOES HERE

% 
label_sums = sum(Z);
total_labels = sum(Z(:));
label_probs = label_sums./total_labels;

H = multi_entropy(label_probs');

ig = zeros(numel(Xrange), 1);
split_vals = zeros(numel(Xrange), 1);

t = CTimeleft(numel(colidx));
fprintf('Evaluating features on %d examples: ', size(X,1));
for i = colidx
    t.timeleft();
    
   if numel(Xrange{i}) == 1
       ig(i) = 0; split_vals(i) = 0;
       continue;
   end
   
   r = linspace(double(Xrange{i}(1)), double(Xrange{i}(end)), min(10, numel(Xrange{i})));
   split_f = bsxfun(@le, X(:,i), r(1:end-1));
   
   px = mean(split_f);
   
   y_given_x_probs = zeros(size(Z,2), size(split_f,2));
   y_given_notx_probs = zeros(size(Z,2), size(split_f,2));
   
   for y_label = 1:size(Z,2) % for each possible label of Y
      Y = Z(:,y_label); % n x 1 vector, Y(i) = 1 if Xi is in class y_label
      y_given_x = bsxfun(@and, Y, split_f);
      y_given_notx = bsxfun(@and, Y, ~split_f);
      y_given_x_counts = sum(y_given_x);
      y_given_x_probs(y_label,:) = (y_given_x_counts./sum(split_f));
      y_given_notx_counts = sum(y_given_notx); 
      y_given_notx_probs(y_label,:) = (y_given_notx_counts./sum(~split_f));
   end
   
   cond_H = px.*multi_entropy(y_given_x_probs) + (1-px).*multi_entropy(y_given_notx_probs);
    
   [ig(i) best_split] = max(H-cond_H);
   split_vals(i) = r(best_split);
    
end

% Choose feature with best split.
[max_ig fidx] = max(ig);
val = split_vals(fidx);