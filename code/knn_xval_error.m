function [error] = knn_xval_error(K, X, Y, part, distFunc)
% KNN_XVAL_ERROR - KNN cross-validation error.
%
% Usage:
%
%   ERROR = knn_xval_error(K, X, Y, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the K-NN algorithm on the 
% given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used (see KNN_TEST).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KNN_TEST

% FILL IN YOUR CODE HERE

n = max(part);
total_error = 0;
% n is the number of folds
for i = 1:n
    % get error for each fold, then average
    
    % All Xs that are in this fold
    % This slices X so that row vectors remain in X
    % if part(<this row of X>) == i <current fold>
    % These will be test points
    testPointIndices = find(part(find(X(:,1))) == i);
    testPoints = X((part(find(X(:,1))) == i),:);
    
    % Get corresponding labels for these Xs
    % X(find(Y),:) gives an X corresponding to that index of Y
    %testTrueLabels = Y(ismember(X(find(Y)), testPoints) > 0);
    testTrueLabels = Y(ismember(find(Y), testPointIndices) > 0);
    
    % All Xs that are NOT in this fold
    % These will be training points
    trainPointIndices = find(part(find(X(:,1))) ~= i);
    trainPoints = X((part(find(X(:,1))) ~= i),:);
    % Corresponding Ys
    trainLabels = Y(ismember(find(Y), trainPointIndices) > 0);
    
    test_knn_labels = knn_test(K, trainPoints, trainLabels, testPoints, distFunc);
    % If the product is <0, the two labels have different signs and thus
    % are in different classes
    fold_error = sum(testTrueLabels.* test_knn_labels < 0);
    fold_error = fold_error/size(testTrueLabels,1);
    total_error = total_error + fold_error;
end

error = total_error/n;
    
    

