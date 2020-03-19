function run_fitGaussian(subj_id, mentat)



%% GENERATIVE MODEL 
    
%     model = 'graphridge';
    model = 'ridge';
    RF = 'gaussian';
if nargin == 0
    path = [];
    randseed = 1107;
    rs = RandStream('mt19937ar', 'Seed', randseed); 
     RandStream.setGlobalStream(rs); %So the data is always the same
%    RandStream.setDefaultStream(rs); %So the data is always the same
    
    fun_gxy = @(x, y, mux, muy, sd, amp) amp./(2.*pi.*sd.^2).*exp( ((x-mux).^2 + (y-muy).^2) ./ (-2.*sd.^2) );
    %    fun_gxy = @(x, y, mux, muy, sd, amp) amp.*exp( ((x-mux).^2 + (y-muy).^2) ./ (-2.*sd.^2) );
    
    fun_gxy_anisotripic = @(x, y, mux, muy, a,b,c, amp) amp*exp( - (a*(x-mux).^2 - 2*b*(x-mux).*(y-muy) + c*(y-muy).^2)) ;
    
    % load the stimulus (Also in dropbox, change path in case)
    load('P:\3019003.01\PercLearn\ANAT\S03\Stimuli\images_files.mat')
    %load('images_files.mat')
    
    imagesN = nan([101 101 size(images,3)]);
    for i = 1:size(images,3)
        imagesN(:,:,i) = imresize(images(:,:,i), [101 101]);
    end
    imagesN(:,:,2) = imagesN(:,:,1);
    
    nrPixels = size(imagesN,1) * size(imagesN, 2);
    time = size(imagesN, 3);
    stim = reshape(imagesN, nrPixels, size(imagesN, 3));
       
    % Load the hemodynamic response function (Also in dropbox, change path in case)
    load('HRF.mat', 'HIRF')
    HIRF = HIRF';
    conv_stim = [];
    for i = 1:nrPixels; 
        conv_stim(i, :) = conv(stim(i,:), HIRF); 
    end %
    
    SH = conv_stim(:,1:time); % now we have it convolved. time is in rows, pixels in column
    
    %Add intercept to the model
    X = [SH; ones(1, time)]';

    % Open a grid for pRFs
    degree = 9;
    imdim =  sqrt(nrPixels);
    xx = linspace((-degree) , degree , imdim);
    [X2, Y] = meshgrid(xx);
    
    % Create receptive fields (for now gaussian)
    nrVox = 10;
    
    mux = rand(1,nrVox)*20-10;
    muy = rand(1,nrVox)*20-10;
%     mux = rand(1,nrVox)*16-8;
%     muy = rand(1,nrVox)*16-8;
    amp = rand(1,nrVox)*1.5+0.5;

    W = [];

    sd = 0.5 + (1.5-.5).*rand(1, nrVox);
    
    for i = 1:nrVox ; pRFs = fun_gxy(X2, Y, mux(i), muy(i), sd(i), amp(i)) ; W(:,i) = pRFs(:); end 
    
    % Simulate time-series
    nrRuns = 4;
    TR = 146;

    % Create noise. Each voxels has a different sd draw from a normal
    % distribution with mean = 0;
    sds = 1 + ((0.2)-1) .*rand(1,nrVox);

    
    % Add baseline activity to the weight matrix (prf matrix)
%     baseline = 500 + ((0)- 500).*rand(nrRuns,nrVox);
    baseline = 10 + ((0)- 10).*rand(nrRuns,nrVox);

    B = [];
    B_noNoise = [];
    
    for i = 1:nrRuns
        
        W1 = [W; baseline(i,:)];
        e = bsxfun(@times, randn(time, nrVox), sds);
        
        B = [B; X*W1 + e];
        B_noNoise = [B_noNoise; X*W1];
    end
        
else

    if mentat == 1
        % With the real data we use a grid of 101x101 (original size)
        path = ['/project/3019003.01/PercLearn/FUNC/', subj_id , filesep];
        load([path, 'Samples/Samples_', subj_id, '.mat'], 'TimeSeries_pRF')
        load('/project/3019003.01/PercLearn/ANAT/S03/Stimuli/images_files.mat')
    else
%         path = ['P:\3019003.01\PercLearn\ANAT\', subj_id , '\'];
%         load([path, 'Rawts_PRFsRuns_',num2str(subj_id),'.mat']);
        
        path = ['P:\3019003.01\PercLearn\FUNC\', subj_id , '\'];
        load([path, 'Samples\Samples_', subj_id, '.mat'], 'TimeSeries_pRF')
        load('P:\3019003.01\PercLearn\ANAT\S03\Stimuli\images_files.mat')
    end
    B = TimeSeries_pRF(:,1:10);
%     B = bsxfun(@minus, B, mean(B)); 
%     B = bsxfun(@rdivide, B, std(B));
%     B = tsAllruns;
    
    imagesN = nan([51 51 size(images,3)]);
    for i = 1:size(images,3)
        imagesN(:,:,i) = imresize(images(:,:,i), [51 51]);
    end
    imagesN(:,:,2) = imagesN(:,:,1);
    
    nrPixels = size(imagesN,1) * size(imagesN, 2);
    time = size(imagesN, 3);
    nrVox = size(B,2);
    nrRuns = size(B,1)/time;
    TR = time * nrRuns;
    
    stim = reshape(imagesN, nrPixels, size(imagesN, 3));
    
    % Load the hemodynamic response function (Also in dropbox, change path in case)
    load('HRF.mat', 'HIRF')
    HIRF = HIRF';
    conv_stim = [];
    
    for i = 1:nrPixels;
        conv_stim(i, :) = conv(stim(i,:), HIRF);
    end 
    
    SH = conv_stim(:,1:time);
    
    X = repmat(SH, 1, nrRuns);
    
       % Open a grid for pRFs
    degree = 9;
    imdim =  sqrt(nrPixels);
    xx = linspace((-degree) , degree , imdim);
    [X2, Y] = meshgrid(xx);
end
%% 
 true_parameters = [mux; muy; sd ;amp ];
% [est_prfs] = fitGaussian_realData(B, SH, X2, Y, nrVox, HIRF, true_parameters, baseline,B_noNoise);
 [est_prfs] = fitGaussian_gridRegress(B, SH, X2, Y, nrVox, true_parameters,  baseline,B_noNoise, TR);
% [est_prfs] = fitGaussian_gridRegress_RealData(B, X, X2, Y, nrVox, TR, nrRuns, subj_id);












end