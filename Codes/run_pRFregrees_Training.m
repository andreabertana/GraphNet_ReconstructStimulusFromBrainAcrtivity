function run_pRFregrees_Training(subj_id, mentat)

if mentat == 1
    path = ['/project/3019003.01/PercLearn/FUNC/', subj_id , filesep];
    load([path, 'Samples/Samples_', subj_id, '.mat'], 'TimeSeries_pRF')
else
    path = ['P:\3019003.01\PercLearn\FUNC\', subj_id , '\'];
    load([path, 'Samples\Samples_', subj_id, '.mat'], 'TimeSeries_pRF')
end

% if ~exist([path, 'pRF'], 'dir'); mkdir([path, 'pRF']); end

% [EstimatedpRFs, noiseVar] = pRF_regress(TimeSeries_pRF(:,1060:1070), [path, 'pRF'], subj_id, mentat);
[EstimatedpRFs, noiseVar] = pRF_regress_singlelambda(TimeSeries_pRF(:,1060:1070), [path, 'pRF'], subj_id, mentat);

% load tmp
% [EstimatedpRFs, noiseVar] = pRF_regress(B_sim, [path, 'pRF'], subj_id, mentat);



end