function weight = svm_learn_rank(data, CVAL)
% notice: for size(data,1) represent feature dim
% size(data, 2) represent sample number
% apply chi-squire non-linear kernel map
if nargin < 2
    CVAL = 1; % C value for the ranking function or SVR
end

data_map = vl_homkermap(data, 2, 'kernel', 'kchi2');
data = data_map;

data = data';
[num_sample, num_dim] = size(data);
temp = cumsum(data,1)./repmat((1:num_sample)',1, num_dim);
temp = sqrt(abs(temp));
w_fow = liblinearsvr(temp,CVAL,2); 

data = flipud(data);
temp = cumsum(data,1)./repmat((1:num_sample)',1, num_dim);
temp = sqrt(abs(temp));
w_rev = liblinearsvr(temp,CVAL,2); 

weight = [w_fow;w_rev];
