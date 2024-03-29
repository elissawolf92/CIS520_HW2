function root = build_dt(X, Y, depth_limit)
% DT_TRAIN_MULTI - Trains a multi-class decision tree classifier.
%
% Usage:
%
%    tree = dt_train(X, Y, DEPTH_LIMIT)
%
% Returns a tree of maximum depth DEPTH_LIMIT learned using the ID3
% algorithm. Assumes X is a N x D matrix of N examples with D features. Y
% must be a N x 1 {1, ..., K} vector of labels. 
%
% Returns a linked hierarchy of structs with the following fields:
%
%   node.terminal - whether or not this node is a terminal (leaf) node
%   node.fidx, node.fval - the feature based test (is X(fidx) <= fval?)
%                          associated with this node
%   node.value - 1 x K vector of P(Y==K) as predicted by this node
%   node.left - child node struct on left branch (f <= val)
%   node.right - child node struct on right branch (f > val)
%
% SEE ALSO
%    DT_CHOOSE_FEATURE_MULTI, DT_VALUE

% YOUR CODE GOES HERE

% Pre-compute the range of each feature.
for i = 1:size(X, 2)
    Xrange{i} = unique(X(:,i));
end

% Starting Z matrix
Z = zeros(size(X,1), max(Y));
for i = 1:numel(Y)
    col = Y(i);
    Z(i,col) = 1;
end

y_sums = sum(Z);
default_probs = y_sums./sum(Z(:));

% Recursively split data to build tree.
root = split_node(X, Z, Xrange, default_probs, 1:size(Xrange, 2), 0, depth_limit);

function [node] = split_node(X, Z, Xrange, default_value, colidx, depth, depth_limit)
% Utility function called recursively; returns a node structure.
%    
%  [node] = split_node(X, Z, Xrange, default_value, colidx, depth, depth_limit)
%  
%  inputs: 
%    Xrange - cell array containing the range of values for each feature
%    default_value - the default value of the node if Y is empty
%    colidx - the indices of features (columns) under consideration
%    depth - current depth of the tree
%    depth_limit - maximum depth of the tree

% The various cases at which we will return a terminal (leaf) node:
%    - we are at the maximum depth
%    - all 1's in Z are in the same column
%    - we have only a single (or no) examples left
%    - we have no features left to split on

[r,c] = find(Z);

if depth == depth_limit || numel(unique(c))==1 || size(Z,1) <= 1 || numel(colidx) == 0
    node.terminal = true;
    node.fidx = [];
    node.fval = [];
    if numel(Z) == 0
        node.value = default_value;
    else
        %Calculate probabilities for this node
        y_sums = sum(Z);
        node.value = y_sums./sum(Z(:));
    end
    node.left = []; node.right = [];

%     fprintf('depth %d [%d/%d]: Leaf node: = %s\n', depth, sum(Y==0), sum(Y==1), ...
%         mat2str(node.value));
    return;
end

node.terminal = false;

% Choose a feature to split on using information gain.
[node.fidx node.fval max_ig] = dt_choose_feature_multi(X, Z, Xrange, colidx);

% Remove this feature from future consideration.
colidx(colidx==node.fidx) = [];

% Split the data based on this feature.
leftidx = find(X(:,node.fidx)<=node.fval);
rightidx = find(X(:,node.fidx)>node.fval);

% Store the value of this node in case we later wish to use this node as a
% terminal node (i.e. pruning.)
y_sums = sum(Z);
node.value = y_sums./sum(Z(:));

% fprintf('depth %d [%d]: Split on feature %d <= %.2f w/ IG = %.2g (L/R = %d/%d)\n', ...
%     depth, numel(Y), node.fidx, node.fval, max_ig, numel(leftidx), numel(rightidx));

% Recursively generate left and right branches.
node.left = split_node(X(leftidx, :), Z(leftidx,:), Xrange, node.value, colidx, depth+1, depth_limit);
node.right = split_node(X(rightidx, :), Z(rightidx,:), Xrange, node.value, colidx, depth+1, depth_limit);