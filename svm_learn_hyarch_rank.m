function weight = svm_learn_hyarch_rank(data, CVAL)
% notice: for size(data,1) represent feature dim
% size(data, 2) represent sample number
% apply chi-squire non-linear kernel map
% num_clip: overlap divide frames into num clip
if nargin < 2
    CVAL = 1; % C value for the ranking function or SVR
end

data_map = vl_homkermap(data, 2, 'kernel', 'kchi2');
data = data_map;

data = data';
[num_frame, num_dim] = size(data);

%% flip frame to learn reversed feature
%data = flipud(data);

%% notice: parameter for hyarchiical rank model
%num_clip = floor(num_frame*2/3);
num_clip = 40;
rate = 0.8;
% num_clip = floor((sqrt(double(8*num_frame+1))-1)/2);  % n_clip =clip_lenth/2
% assume overlap 0.5
 % increase rate, overlap (1-rate)
clip_len = num_frame/(num_clip*rate - rate + 1);
% clip_len = 4*num_frame/(num_clip+1);

fea = zeros(num_clip,num_dim);
for clip_idx = 1:num_clip
    sdt = (clip_idx-1)*clip_len*rate + 1;
    sp_frm = floor(sdt:(sdt+clip_len));
    sp_frm = min(sp_frm, num_frame);
    data_tmp = data(sp_frm,:);
    wei = rank_learn_foward(data_tmp, CVAL);
    fea(clip_idx,:) = wei';
end

weight = rank_learn_foward(fea, CVAL);

%% add reverse processing
data = flipud(data);
fea = zeros(num_clip,num_dim);
for clip_idx = 1:num_clip
    sdt = (clip_idx-1)*clip_len*rate + 1;
    sp_frm = floor(sdt:(sdt+clip_len));
    sp_frm = min(sp_frm, num_frame);
    data_tmp = data(sp_frm,:);
    wei = rank_learn_foward(data_tmp, CVAL);
    fea(clip_idx,:) = wei';
end
weight_rev = rank_learn_foward(fea, CVAL);
weight = [weight;weight_rev];

function w_flow = rank_learn_foward(data, CVAL)
    [num, num_dim] = size(data);
    temp = cumsum(data,1)./repmat((1:num)',1, num_dim);
    temp = sqrt(abs(temp));
    w_flow = liblinearsvr(temp,CVAL,2); 
