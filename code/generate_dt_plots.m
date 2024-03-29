%% Script/instructions on how to submit plots/answers for question 3.
% Put your textual answers where specified in this script and run it before
% submitting.

% Loading the data
data = load('../data/mnist_all.mat');

% Running a training set for binary decision tree classifier
% [X Y] = get_digit_dataset(data, {'7','9'}, 'train');
% 
% %% Train a depth 4 binary decision tree
% dt = dt_train(X, Y, 4);
% 
% %%
% [Xtest Ytest] = get_digit_dataset(data, {'7','9'}, 'test');
% Yhat = zeros(size(Ytest));
% for i = 1:size(Xtest,1)
%     Yhat(i) = dt_value(dt, Xtest(i,:)) >= 0.5;
% end
% 
% mean(Yhat ~= Ytest)

%% 3.1
answers{1} = 'The model begins to be overfit when depth = 6; at this point, the lines cross and test error is higher than training error.';

training_errors = zeros(1,6);
test_errors = zeros(1,6);
max_depth_vals = 1:6;

for max_depth = max_depth_vals
    % Get training set
    [X Y] = get_digit_dataset(data, {'1','3','7'}, 'train');
    % Train a tree
    dt = dt_train_multi(X, Y, max_depth);
    % Get test set
    [Xtest Ytest] = get_digit_dataset(data, {'1','3','7'}, 'test');
    
    % Get training error
    Yhat_training = zeros(size(Y));
    for i = 1:size(X,1)
        value = dt_value(dt, X(i,:));
        [label, label_ind] = max(value);
        Yhat_training(i) = label_ind;
    end
    training_errors(max_depth) = mean(Yhat_training ~= Y);
    
    % Get test error
    Yhat_testing = zeros(size(Ytest));
    for i = 1:size(Xtest,1)
        value = dt_value(dt, Xtest(i,:));
        [label, label_ind] = max(value);
        Yhat_testing(i) = label_ind;
    end
    test_errors(max_depth) = mean(Yhat_testing ~= Ytest);
end

h1 = plot(max_depth_vals, training_errors,'b');
hold on;
h2 = plot(max_depth_vals, test_errors, 'r');
xlabel('Tree depth')
legend([h1 h2], 'Training error', 'Test error');
title('Error vs. tree depth')
print -djpeg plot_3.1.jpg
hold off;

% Saving your plot: once you have succesfully plotted your data; e.g.,
% something like:
% >> plot(depth, [train_err test_err]);
% Remember: You can save your figure to a .jpg file as follows:
% >> print -djpg plot_3.1.jpg

%% 3.2
answers{2} = 'The most commonly confused digits are 3/5 and 4/9.';

% Get training set
[X, Y] = get_digit_dataset(data, {'0','1','2','3','4','5','6','7','8','9'}, 'train');
% Train a tree
dt = dt_train_multi(X, Y, 6);
% Get test set
[Xtest, Ytest] = get_digit_dataset(data, {'0','1','2','3','4','5','6','7','8','9'}, 'test');
Ytest = Ytest - 1; % get values in range 0-9 to match with digits
% this way the value of ytest is the digit it is classified as

Yhat = zeros(size(Ytest));
for i = 1:size(Xtest,1)
    value = dt_value(dt, Xtest(i,:));
    [label, label_ind] = max(value);
    Yhat(i) = label_ind-1; %subtract one because index 1 corresponds to digit 0, and so on
end

M = zeros(10,10);
%M_ij = P(y=i-1, h(x)=j-1)
%M_11 = P(y=0, h(x)=0)
sample_chosen = 0;
for sample = 1:size(Yhat,1)
    r = Ytest(sample)+1;
    c = Yhat(sample)+1;
    M(r,c) = M(r,c) + 1;
    % I want to pick one where 3 and 5 got confused
    if r == 4 & c == 6 & sample_chosen == 0
       sample_chosen = sample; 
    end
end
M = M./numel(Ytest);

plotnumeric(M);
xlabel('Predicted digit');
ylabel('Actual digit');
title('Digit classication confusion matrix - 3.2');
save -ascii confusion.txt M
print -djpeg plot_3.2.jpg

% Saving your plot: once you've computed M, plot M with the plotnumeric.m
% command we've provided. e.g:
% >> plotnumeric(M);
%
% Save your file to plot_3.2.jpg
%
% ***** ALSO *******
% Save your confusion matrix M to a .txt file as follows:
% >> save -asci confusion.txt M

%% 3.3
answers{3} = 'In this example, the 3 was misclassified as a 5, with p(y=3) = .34 and p(y=5) = .39 (so close!).  Looking at the pixels used, this makes sense; I think all of these pixels would look the same if the number had been a 5.  All are located in the parts of the 3 that are very similar to a 5.';

sample = Xtest(sample_chosen,:);
H = plot_dt_digit(dt, sample);
print -djpeg plot_3.3.jpg

% E.g., if Xtest(i,:) is an example your method fails on, call:
% >> plot_dt_digit(tree, Xtest(i,:));
%
% Save your file to plot_3.3.jpg

%% Finishing up - make sure to run this before you submit.
save('problem_3_answers.mat', 'answers');