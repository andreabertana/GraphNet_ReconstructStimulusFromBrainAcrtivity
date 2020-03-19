function [EstimatedpRFs, noiseVar] = pRF_regress(B, path, subj_id, mentat)


%% Steps that are needed:
%   - Define the stimulus over time
%   - Load the ts (Create manually or give as input of the function)
%   - Find the best pRFs (expressed in pixel values) that predictes the data 
%       How? With least-square-fit that is regualised via L2 norm.
%
%       The solution in ordet to predict the for ridge regression
%       ts = (X' * X + lambda*(I'))^-1 * X' * y
%       
%       for graphridge
%       sol = (X'*X + lambda*(1-alpha)*G)\X'*y;
%
% %%%%%%%%%%%%%%%%%%% NOTICE %%%%%%%
%
%  - A normalization with respect to the number of pixels x degree is not
%  implemented. As result, R2 change(is biased) with respect to degrees.
%
% - Now it does an Exaustive Search
%

%% GENERATIVE MODEL 
    
    model = 'graphridge';
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
    
    % Downsample images and vectorize
    
    imagesN = images(1:4:end, 1:4:end, :); %RB: Is this the best way to downsample?
    
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
    amp = rand(1,nrVox)*1+0.5;
    %amp = rand(1,nrVox)*1000;
    W = [];
    
%     RF = 'gaussian';
    if strcmp(RF, 'gaussian')
       
        sd = rand(1,nrVox)*1.5;
        
        for i = 1:nrVox ; pRFs = fun_gxy(X2, Y, mux(i), muy(i), sd(i), amp(i)) ; W(:,i) = pRFs(:); end %RB: You don't need a for-loop here %
        
    elseif strcmp(RF, 'anisotropicGauss')
        
        sdy = rand(1,nrVox)*2+1;
        sdx = rand(1,nrVox)*2+1;
        theta = Shuffle(0:pi/nrVox:pi);
        for i = 1:nrVox 
            
            a(i) = cos(theta(i))^2/2/sdx(i)^2 + sin(theta(i))^2/2/sdy(i)^2;
            b(i) = -sin(2*theta(i))/4/sdx(i)^2 + sin(2*theta(i))/4/sdy(i)^2 ;
            c(i) = sin(theta(i)).^2/2/sdx(i)^2 + cos(theta(i))^2/2/sdy(i)^2;
            pRFs = fun_gxy_anisotripic(X2, Y, mux(i), muy(i), a(i), b(i), c(i), amp(i)) ;
            W(:,i) = pRFs(:); 
        end
    else
        disp('Specify one tipe of RF please')
    end
    
    % Simulate time-series
    nrRuns = 4;
    
    % Create noise. Each voxels has a different sd draw from a normal
    % distribution with mean = 0;
    sds = 3 + ((0.2)-3).*rand(1,nrVox);
     sds = 0.1 + ((0.01)-0.1).*rand(1,nrVox);
%     sds = rand(1,nrVox)*10^-5;
    noise = bsxfun(@times, randn(nrVox, time), sds');
    
    % Add baseline activity to the weight matrix (prf matrix)
    baseline = 500 + ((0)- 500).*rand(nrRuns,nrVox);
    B = [];
    
    for i = 1:nrRuns
        
        W1 = [W; baseline(i,:)];
        e = bsxfun(@times, randn(time, nrVox), sds);
        
        %B = repmat(X, nrRuns, 1)*W + e;
        B = [B; X*W1 + e];
    end
    
    % Now remove the intercept and create and X with 1 intercept per run
    X = X(:,1:end-1);
    X = repmat(X, nrRuns, 1);
    
    intercept =zeros(nrRuns,time*nrRuns);
    durationXrun = 1:time:time*nrRuns;
    for i = 1:nrRuns
        intercept(i, durationXrun(i): time*i) = 1;
    end
    
    % Now we have one intercept per run
    X = [X, intercept'];
        
% %%  If we want to compare it with just fitting a gaussian ...
%     pRFparams = [mux; muy; sd; amp]';
%     fitGaussian(B, stim, pRFparams, HIRF, X2, Y, nrVox);  % Comment out...not ready yet...
    
else
    if mentat == 1
        % With the real data we use a grid of 101x101 (original size)
        load('/project/3019003.01/PercLearn/ANAT/S03/Stimuli/images_files.mat')
    else
        load('P:\3019003.01\PercLearn\ANAT\S03\Stimuli\images_files.mat')
    end
    
    B = bsxfun(@minus, B, mean(B));
    B = bsxfun(@rdivide, B, std(B));
    
    
%     B = B + randn(size(B))*800;
    
%     imagesN = images(1:4:end, 1:4:end, :);
    imagesN = nan([51 51 size(images,3)]);
    for i = 1:size(images,3)
        imagesN(:,:,i) = imresize(images(:,:,i), [51 51]);
    end
    imagesN(:,:,2) = imagesN(:,:,1);
%     imagesN = imagesN>0.5;
    
    nrPixels = size(imagesN,1) * size(imagesN, 2);
    time = size(imagesN, 3);
    nrVox = size(B,2);
    nrRuns = size(B,1)/time;
    
    stim = reshape(imagesN, nrPixels, size(imagesN, 3));
    
    % Load the hemodynamic response function (Also in dropbox, change path in case)
    load('HRF.mat', 'HIRF')
    HIRF = HIRF';
    conv_stim = [];
    
    for i = 1:nrPixels;
        conv_stim(i, :) = conv(stim(i,:), HIRF);
    end 
    
    SH = conv_stim(:,1:time)';
    
    X = repmat(SH, nrRuns, 1);
    
    intercept =zeros(nrRuns,time*nrRuns);
    durationXrun = 1:time:time*nrRuns;
    for i = 1:nrRuns
        intercept(i, durationXrun(i): time*i) = 1;
    end
    
    % Now we have one intercept per run
    X = [X, intercept'];
    
    % Open a grid for pRFs
    degree = 9;
    imdim =  sqrt(nrPixels);
    xx = linspace((-degree) , degree , imdim);
    [X2, Y] = meshgrid(xx);
   
end
%% 
%[est_prfs] = fitGaussian_realData(B, SH, HIRF, X2, Y, nrVox);

 %%
cvInd = reshape(1:time*nrRuns, time, nrRuns);
    
    %% TRAINING STAGE
t = tic;
a = 1e-10;
opts = optimset('MaxFunEvals', 1e12, 'MaxIter', 1e10, 'Display', 'off');

% Define range of lambda  and alpha. Here alpha is 0 as we
lambda_range = 10.^(-5:0.5:10);
alpha_range = 0;

% Define the identity matrix that will be needed for the solution
if strcmp(model, 'ridge')    
    I = eye(nrPixels);
%     Ix = [I zeros(nrPixels,1)];
    Iplus = [[I zeros(nrPixels,nrRuns)]; zeros(nrRuns, nrPixels+nrRuns)];    
    IplusTrain = Iplus(1:end-1, 1:end-1);
elseif strcmp(model,'graphridge')  
    [xn, yn] = meshgrid(1:51);
    xd = bsxfun(@minus, xn(:), xn(:)');
    yd = bsxfun(@minus, yn(:), yn(:)');
    D = sqrt(xd.^2 + yd.^2);
    N = D==1;
    G = diag(sum(N));
    G(N) = -1;
    Iplus = [[G zeros(nrPixels,nrRuns)]; zeros(nrRuns, nrPixels+nrRuns)];
    IplusTrain = [[G zeros(nrPixels,nrRuns - 1)]; zeros(nrRuns - 1, nrPixels+ (nrRuns - 1))];
end

% GRID SEARCH : DEFINE PARAMETERS
[l_grid, a_grid] = meshgrid(lambda_range, alpha_range);
Ncomb = numel(l_grid);
Ngridsearch = ceil(Ncomb/1);
gridSpacing = round(Ncomb/Ngridsearch);
gridInd = round(gridSpacing/2):gridSpacing:Ncomb;

Ncoeff = size(X, 2);
%R2 = nan(Ngridsearch, 1);
R2 = nan(Ngridsearch, nrVox);
used_lambda = nan(Ngridsearch,1);
fprintf('\n--GRID /  EXAHUSTIVE SEARCH--');
idxrun = fliplr(0:nrRuns -1);

% RB: because all the runs are the same, the design matrices are also
% always the same
train_X = X(1:size(X,1)/nrRuns*(nrRuns-1), 1:end-1); 
test_X = X(1:size(X,1)/nrRuns, 1:end-nrRuns);

for i = 1:Ngridsearch
   
    lambda = l_grid(gridInd(i));
    
    fprintf('\nGrid search iteration %2d/%2d, Lambda: %3.2g\n', i, Ngridsearch, lambda);
    
    %this_fval = nan(max(cvInd),1);
    pred = nan(time*nrRuns, nrVox);
    %fprintf('\n%3d/%3d - lambda: %7.2e, alpha: %3.2f', i, Ngridsearch, l_grid(gridInd(i)), a_grid(gridInd(i)));
    
    divmat = (train_X'*train_X + lambda*IplusTrain)\train_X';
    
    for testset = 1:nrRuns
        
        % Define paramters
        
        
        partition = 1:nrRuns; partition(partition == testset) = [];
        testInd = reshape(cvInd(:, partition), 1, time * (nrRuns - 1));
        
        %%%%%% Estimate %%%%%%
%         this_X = X(testInd, :);
%         this_X(:, end - idxrun(testset)) = [];
        this_B = B(testInd,:);
        
        % the solution (W) is analityc
%         sol = (this_X'*this_X + lambda*IplusTrain)\this_X'*this_B;
        sol = divmat*this_B;
        
        %%%%%% Validate %%%%%%
%         test_X = X(cvInd(:,testset), 1:end-nrRuns); 
%         test_X(:, end - idxrun(testset)) = [];
        pred(cvInd(:,testset),:) = test_X*sol(1:end-nrRuns+1,:);                        
    end
    used_lambda(i) = lambda;  
    
    % Regress the intercept per run
    res = B - pred;
    run_ints = intercept'\res;
    res = res - intercept'*run_ints;
    B_noint = B - intercept'*run_ints;
    
    % Compute R2
    R2(i, :) = 1 - sum(bsxfun(@minus, res, mean(res)).^2) ./ sum(bsxfun(@minus, B_noint, mean(B_noint)).^2);    
%     R2(i,:) = sum(res.^2, 1);
  
%     EstimatedpRFs = []; noiseVar = [];
%     return
end


fprintf('\n\n--Takiing Best R2, lambdas and creating Estimated pRFs --');

[bestR2, bestInd] = max(R2);
% [bestR2, bestInd] = min(R2);
best_lambda = used_lambda(bestInd);

%% Use the best lambda to fit the whole dataset
X_allruns = X;
EstimatedpRFs = nan(Ncoeff, nrVox);

%for i = 1: nrVox; EstimatedpRFs(:,i) = (X_allruns'*X_allruns + best_lambda(i)*Iplus)\X_allruns'*B(:,i); end

unique_bl = unique(best_lambda);
for j = 1:length(unique_bl)
    vox_ind = best_lambda==unique_bl(j);
    EstimatedpRFs(:, vox_ind) = (X_allruns'*X_allruns + unique_bl(j)*Iplus)\X_allruns'*B(:, vox_ind);            
end

ests = nan(Ncoeff, length(lambda_range));
for j =1:length(lambda_range)
    lambda = lambda_range(j);
    ests(:, j) =(X_allruns'*X_allruns + lambda*Iplus)\X_allruns'*B(:, 1);            
end
preds = X_allruns*ests;


% all_prf_ests = nan([size(EstimatedpRFs), length(unique_bl)]);
% for j = 1:length(unique_bl)
% 
%     all_prf_ests(:,:,j) = (X_allruns'*X_allruns + unique_bl(j)*Iplus)\X_allruns'*B;
% 
% end
% for i = 1:nrVox
% 
%     EstimatedpRFs(:,i) = all_prf_ests(:, i, unique_bl==best_lambda(i));
% 
% end

%Recreate best predicted time_series
PredictedTs = X_allruns*EstimatedpRFs;

% Noise variances
noiseVar = mean(power(B' - EstimatedpRFs' * X_allruns', 2),2);

etime = toc(t);

disp(['Time  =  ', num2str(etime/60), ' Minutes']);

Pix = sqrt(nrPixels);
Intercept = EstimatedpRFs(end - nrRuns : end, :);
EstimatedpRFs = reshape(EstimatedpRFs(1: end - nrRuns, :), [Pix Pix nrVox] );

% save only if we are not in simulation mode
if nargin ~= 0 
    save([path,filesep, subj_id, '_pRFestimates.mat'],'Iplus', 'imdim', 'nrVox' , 'degree', 'noiseVar', 'EstimatedpRFs', 'best_lambda', 'bestR2', 'nrRuns', 'Intercept', 'Pix', 'used_lambda', 'PredictedTs')
end
%%
%load('GraphRidge_500Voxels.mat');
%pRF_Testing(Iplus, imdim, nrVox, W' , degree, noiseVar, sds, EstimatedpRFs', baseline)
if nargin == 0
    %% Plot/store/check results
    if nargin == 0 ; OriginalpRFs = reshape(W1(1: end-1, :), [Pix Pix nrVox] ); end
    %%
    figure;
    plot(baseline, Intercept, 'o')
    xlabel('Original baseline')
    ylabel('Fitted intercept')
    title('Fitted intercept vs original baseline ')
    axis([-10 10 -10 10])
    
    %rand; %AB: breakpoint here so that you can plot all voxels that you want by changing few things. I run the paragraph with crt + enter
    
    %% Plot lambda choose vs R2
    figure;
    plot(bestR2, best_lambda, 'o')
    %axis([0 1 min(used_lambda(:,1)) max(used_lambda(:,1))])
    %axis([0 1  min(unique(best_lambda)) - 10000 max(unique(best_lambda)) + 10000])
    xlabel('R2')
    ylabel('Best lambda')
    %% plot lambdas in used lambdas baseline
    [~, idx_lambdas] = ismember(best_lambda, used_lambda);
    figure;
    plot(used_lambda)
    hold on
    plot(idx_lambdas, best_lambda, 'o')
    %% Compute the best R2 we can predict from the data with noise
    B_NoNoise = repmat(X, nrRuns, 1)*W;
    res = B-B_NoNoise;
    R2_NoNoise = 1 - sum(bsxfun(@minus, res, mean(res)).^2) ./ sum(bsxfun(@minus, B, mean(B)).^2);
    
    figure;
    % subplot(2,1,1)
    plot(bestR2,R2_NoNoise, 'o')
    xlabel('Best R2 explained by the model')
    ylabel('Maximum R2 given the noise')
    
    
    %% Plot histogram of the best R2
    
    figure;
    hist(bestR2, sqrt(nrVox))
    xlabel('R2')
    ylabel(['Nr of Voxels out of ', num2str(nrVox)])
    title('Histogram R2')
    axis([-0.3 1 0 sqrt(nrVox) *2])
    
    %% Plot voxels with the best, mean and worst fit
    
    [~, R2_idx] = sort(bestR2, 'descend');
    
    BestVoxR2 = R2_idx(1:4);
    MediumVoxR2 = R2_idx((nrVox/2)-1 : nrVox/2 + 2);
    WorstVoxR2 = R2_idx(end-3 : end);
    
    %
    %rand;%AB: breakpoint here so that you can plot all voxels that you want by changing few things. I run the paragraph with crt + enter
    %
    for j = 1:3
        if j == 1; voxIdx = BestVoxR2; titlePart = 'BestVox R2 = ';
        elseif j == 2; voxIdx = MediumVoxR2; titlePart = 'MediumVox R2 = ';
        elseif j == 3; voxIdx = WorstVoxR2; titlePart = 'WorstVox R2 = ';
        end
        for i = 1:4
            figure;
            
            subplot(2,2,1)
            hold on
            %imagesc(xx,xx,OriginalpRFs(:,:,voxIdx(i)))
            xlabel('Degree')
            ylabel('Degree')
            axis([-degree degree -degree degree])
            axis square
            colorbar
            ca = caxis;
            caxis([-max(abs(ca)), max(abs(ca))]);
            title('Original pRF')
            
            subplot(2,2,2)
            hold on
            imagesc(xx,xx,EstimatedpRFs(:,:,voxIdx(i)))
            colorbar
            %caxis([-max(abs(ca)), max(abs(ca))]);
            xlabel('Degree')
            ylabel('Degree')
            title('Estimated pRF')
            axis([-degree degree -degree degree])
            axis square
            
            subplot(2,2,[3 4])
            plot(B(:, voxIdx(i)))
            hold on
            plot(PredictedTs(:, voxIdx(i)), '--r')
            title([titlePart, num2str(bestR2(voxIdx(i))) ])
            xlabel('Time in TR')
            ylabel('Bold')
            
            
        end
    end
end
end

