%% Script/instructions on how to submit plots/answers for question 2.
% Put your textual answers where specified in this script and run it before
% submitting.

% Loading the data: this loads X, X_noisy, and Y.
load('../data/breast-cancer-data-fixed.mat');

%% 2.1
answers{1} = 'Error for the noisy dataset has higher averages, but smaller standard deviations relative to the data.  Results are about the same for different numbers of folds.  In both, true error is slightly lower than n-fold error.';

K = 1;
N_vals = [2,4,8,16];
distFunc = 'l2';
nfold_errs = zeros(100, 16); % Set up the matrices where we will store our results
nfold_errs_noisy = zeros(100,16);
true_errs = zeros(100,1); 
true_errs_noisy = zeros(100,1);

for i=1:100
   % Randomly split the data into testing and training sets, using 400 samples for training.
   % Randomly select 400 indices that will be our training set
   training_indices = randsample(size(X,1), 400);
   trainPoints = X(ismember(find(X(:,1)),training_indices) > 0,:);
   trainLabels = Y(ismember(find(Y(:,1)),training_indices) > 0,:);
   
   trainPointsNoisy = X_noisy(ismember(find(X(:,1)),training_indices) > 0,:);
   
   testPoints = X(ismember(find(X(:,1)),training_indices) == 0,:);
   testPointsNoisy = X_noisy(ismember(find(X(:,1)),training_indices) == 0,:);
   testLabels = Y(ismember(find(Y(:,1)),training_indices) == 0,:);
   
   % Test for each of the four N values
   for j=1:4
       N = N_vals(j);
       % Compute n-fold error on the training set
       % First get our random partition
       part = make_xval_partition(size(trainPoints,1), N);
       
       % Now get nfold error
       nfold_error = knn_xval_error(K, trainPoints, trainLabels, part, distFunc);
       nfold_error_noisy = knn_xval_error(K, trainPointsNoisy, trainLabels, part, distFunc);
       % Store it in our matrix
       nfold_errs(i,N_vals(j)) = nfold_error;
       nfold_errs_noisy(i,N_vals(j)) = nfold_error_noisy;
       
   end
   
   % Compute the true test error
   % First get the test labels
   true_test_labels = knn_test(K, trainPoints, trainLabels, testPoints, distFunc);
   true_test_labels_noisy = knn_test(K, trainPointsNoisy, trainLabels, testPointsNoisy, distFunc);
   % Now calculate error
   test_error = sum(testLabels.*true_test_labels < 0);
   test_error = test_error/size(testLabels,1);
   true_errs(i) = test_error;
   
   test_error_noisy = sum(testLabels.*true_test_labels_noisy < 0);
   test_error_noisy = test_error_noisy/size(testLabels,1);
   true_errs_noisy(i) = test_error_noisy;
    
end

% Plots for 2.1

% Plot of nfold error for standard dataset
y = mean(nfold_errs);
e = std(nfold_errs);
%x = [2,4,8,16];
x = 1:16;
errorbar(x,y,e);

hold on;

% Stats for test error for standard dataset
test_y = mean(true_errs);
test_e = std(true_errs);
test_x = 1;
errorbar(test_x,test_y,test_e);

xlabel('Number of folds')
title('N-Fold error of Standard dataset (2.1)')
print -djpeg plot_2.1.jpg

hold off;

% Plot of nfold error for noisy dataset
y = mean(nfold_errs_noisy);
e = std(nfold_errs_noisy);
%x = [2,4,8,16];
x = 1:16;
errorbar(x,y,e);

hold on;

% Stats for test error for noisy dataset
test_y_noisy = mean(true_errs_noisy);
test_e_noisy = std(true_errs_noisy);
test_x_noisy = 1;
errorbar(test_x_noisy,test_y_noisy,test_e_noisy);

xlabel('Number of folds')
title('N-Fold error of Noisy dataset (2.1)')
print -djpeg plot_2.1-noisy.jpg

% Plotting with error bars: first, arrange your data in a matrix as
% follows:
%
%  nfold_errs(i,j) = nfold error with n=j of i'th repeat
%  
% Then we want to plot the mean with error bars of standard deviation as
% follows: y = mean(nfold_errs), e = std(nfold_errs), x = [2 4 8 16].
% 
% >> errorbar(x, y, e);
%
% To add labels to the graph, use xlabel('X axis label') and ylabel
% commands. To add a title, using the title('My title') command.
% See the class Matlab tutorial wiki for more plotting help.
% 
% Once your plot is ready, save your plot to a jpg by selecting the figure
% window and running the command:
%
% >> print -djpg plot_2.1-noisy.jpg % (for noisy version of data)
% >> print -djpg plot_2.1.jpg  % (for regular version of data)
%
% YOU MUST SAVE YOUR PLOTS TO THESE EXACT FILES.

%% 2.2
answers{2} = 'The best k value is 2.  The best sigma value is 5.  I choose 5 because 4 yielded the lowest error for standard data, while 6 yielded the lowest error for noisy data.';

% K = {1,3,5,...,15}
% sigma = {1,2,3,...,8}
% Run with 10 folds

K_nfold_errors = zeros(100,15);
K_nfold_errors_noisy = zeros(100,15);
K_test_errors = zeros(100,15);
K_test_errors_noisy = zeros(100,15);

kern_nfold_errors = zeros(100,8);
kern_nfold_errors_noisy = zeros(100,8);
kern_test_errors = zeros(100,8);
kern_test_errors_noisy = zeros(100,8);

N = 10;

for i = 1:100
    training_indices = randsample(size(X,1), 400);
    trainPoints = X(ismember(find(X(:,1)),training_indices) > 0,:);
    trainLabels = Y(ismember(find(Y(:,1)),training_indices) > 0,:);
   
    trainPointsNoisy = X_noisy(ismember(find(X(:,1)),training_indices) > 0,:);
   
    testPoints = X(ismember(find(X(:,1)),training_indices) == 0,:);
    testPointsNoisy = X_noisy(ismember(find(X(:,1)),training_indices) == 0,:);
    testLabels = Y(ismember(find(Y(:,1)),training_indices) == 0,:);
    
    % Random partition
    part = make_xval_partition(size(trainPoints,1), N);
    
    for K = 1:15
       
        % Get nfold error
        nfold_error = knn_xval_error(K, trainPoints, trainLabels, part, distFunc);
        nfold_error_noisy = knn_xval_error(K, trainPointsNoisy, trainLabels, part, distFunc);
        % Store it in our matrix
        K_nfold_errors(i,K) = nfold_error;
        K_nfold_errors_noisy(i,K) = nfold_error_noisy;
        
        % Get test error
        % First get the test labels
        true_test_labels = knn_test(K, trainPoints, trainLabels, testPoints, distFunc);
        true_test_labels_noisy = knn_test(K, trainPointsNoisy, trainLabels, testPointsNoisy, distFunc);
        % Now calculate error
        test_error = sum(testLabels.*true_test_labels < 0);
        test_error = test_error/size(testLabels,1);
        K_test_errors(i,K) = test_error;
   
        test_error_noisy = sum(testLabels.*true_test_labels_noisy < 0);
        test_error_noisy = test_error_noisy/size(testLabels,1);
        K_test_errors_noisy(i,K) = test_error_noisy;     
    end
    
    for sigma = 1:8
        % 10 Fold error
        kern_error = kernreg_xval_error(sigma, trainPoints, trainLabels, part, distFunc);
        kern_error_noisy = kernreg_xval_error(sigma, trainPointsNoisy, trainLabels, part, distFunc);
        
        kern_nfold_errors(i,sigma) = kern_error;
        kern_nfold_errors_noisy(i,sigma) = kern_error_noisy;
        
        % Test error
        true_test_labels = kernreg_test(sigma, trainPoints, trainLabels, testPoints, distFunc);
        true_test_labels_noisy = kernreg_test(sigma, trainPointsNoisy, trainLabels, testPointsNoisy, distFunc);
        % Now calculate error
        test_error = sum(testLabels.*true_test_labels < 0);
        test_error = test_error/size(testLabels,1);
        kern_test_errors(i,sigma) = test_error;
   
        test_error_noisy = sum(testLabels.*true_test_labels_noisy < 0);
        test_error_noisy = test_error_noisy/size(testLabels,1);
        kern_test_errors_noisy(i,sigma) = test_error_noisy;     
        
    end
end

hold off;

% Plots for k-nn, standard
x_knn = 1:15;
e_knn = std(K_nfold_errors);
y_knn = mean(K_nfold_errors);
h1 = errorbar(x_knn, y_knn, e_knn, 'b');

hold on;

x_knn_test = 1:15;
e_knn_test = std(K_test_errors);
y_knn_test = mean(K_test_errors);
h2 = errorbar(x_knn_test, y_knn_test, e_knn_test, 'r');

legend([h1 h2], 'Nfold error', 'Test error');

xlabel('K');
title('10-Fold error of Standard dataset using knn (2.2)');
print -djpeg plot_2.2-k.jpg

hold off;

% Plots for k-nn, noisy
x_knn_noisy = 1:15;
e_knn_noisy = std(K_nfold_errors_noisy);
y_knn_noisy = mean(K_nfold_errors_noisy);
h1 = errorbar(x_knn_noisy, y_knn_noisy, e_knn_noisy, 'b');

hold on;

x_knn_noisy_test = 1:15;
e_knn_noisy_test = std(K_test_errors_noisy);
y_knn_noisy_test = mean(K_test_errors_noisy);
h2 = errorbar(x_knn_noisy_test, y_knn_noisy_test, e_knn_noisy_test, 'r');

legend([h1 h2], 'Nfold error', 'Test error');

xlabel('K');
title('10-Fold error of Noisy dataset using k-nn (2.2)');
print -djpeg plot_2.2-k-noisy.jpg

hold off;

% Plots for kern, standard
x_kern = 1:8;
e_kern = std(kern_nfold_errors);
y_kern = mean(kern_nfold_errors);
h1 = errorbar(x_kern, y_kern, e_kern, 'b');

hold on;

x_kern_test = 1:8;
e_kern_test = std(kern_test_errors);
y_kern_test = mean(kern_test_errors);
h2 = errorbar(x_kern_test, y_kern_test, e_kern_test, 'r');

legend([h1 h2], 'Nfold error', 'Test error');

xlabel('Sigma');
title('10-Fold error of Standard dataset using kernreg (2.2)');
print -djpeg plot_2.2-sigma.jpg

hold off;

% Plots for kern, noisy
x_kern_noisy = 1:8;
e_kern_noisy = std(kern_nfold_errors_noisy);
y_kern_noisy = mean(kern_nfold_errors_noisy);
h1 = errorbar(x_kern_noisy, y_kern_noisy, e_kern_noisy, 'b');

hold on;

x_kern_test_noisy = 1:8;
e_kern_test_noisy = std(kern_test_errors_noisy);
y_kern_test_noisy = mean(kern_test_errors_noisy);
h2 = errorbar(x_kern_test_noisy, y_kern_test_noisy, e_kern_test_noisy, 'r');

legend([h1 h2], 'Nfold error', 'Test error');

xlabel('Sigma');
title('10-Fold error of Noisy dataset using kernreg (2.2)');
print -djpeg plot_2.2-sigma-noisy.jpg

hold off;


% Save your plots as follows:
%
%  noisy data, k-nn error vs. K --> plot_2.2-k-noisy.jpg
%  noisy data, kernreg error vs. sigma --> plot_2.2-sigma-noisy.jpg
%  regular data, k-nn error vs. K --> plot_2.2-k.jpg
%  regular data, kernreg error vs. sigma --> plot_2.2-sigma.jpg

%% Finishing up - make sure to run this before you submit.
save('problem_2_answers.mat', 'answers');