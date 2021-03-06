% Generate .mat file from data

function [] = generate(path1, path2)

    fprintf('Reading %s \n', path1);
    master = csvread(path1, 1,0);
    
    % Shuffling the train data.
    fprintf('Shuffling the training data set.\n');
    ordering = randperm(size(master,1));
    M = master(ordering,:);
    fprintf('Shuffling success!!!\n');
    
    % Splitting train data.
    fprintf('Splitting the training data set into test and CV data.\n');
    split = 0.9*size(M,1);
    train = M(1:split,:);
    cv = M(split+1:end,:);
    
    % Saving training data.
    fprintf('Saving training and CV data.......\n');
    X_train = train(:,2:(end-2));
    Y_train = train(:,end);
    weight_train = train(:,end-1);
    save('data\train\train.mat', 'X_train', 'Y_train', 'weight_train');
    
    % Saving CV data.
    X_cv = cv(:, 2:(end-2));
    Y_cv = cv(:, end);
    weight_cv = cv(:,end-1);
    save('data\train\cv.mat', 'X_cv', 'Y_cv', 'weight_cv');
   
    % Saving test data.
    fprintf('Reading %s \n', path2);
    master_test = csvread(path2, 1, 0);
	test_id = master_test(:,1);
	X_test = master_test(:, 2:end);
	fprintf('Saving test data...\n');
    save('data\test\test.mat', 'X_test', 'test_id');
	fprintf('Maka success!!! \n');
    
end


