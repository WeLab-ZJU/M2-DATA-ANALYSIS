
clear;
% Matrix provided by REST-META-MDD
load('Stat_Sub_Info_MDDvsNC.mat');  
load('FC_MDDvsNC.mat');            

X = [ones(size(Dx)), Dx, Age, Sex, Edu, Motion];
Z = {ones(size(Dx)), Dx};
G = {Site, Site};

ROIlen = 116;
Nsub = size(res, 3);

TMatrix = zeros(ROIlen, ROIlen);
PMatrix = zeros(ROIlen, ROIlen);

for i = 1:ROIlen
    for j = i+1:ROIlen
        y = squeeze(res(i, j, :));
        lme = fitlmematrix(X, y, Z, G);
        TMatrix(i, j) = lme.Coefficients{2, 'tStat'};
        PMatrix(i, j) = lme.Coefficients{2, 'pValue'};
    end
end

p_flat = PMatrix(triu(true(ROIlen), 1));
p_adj = mafdr(p_flat, 'BHFDR', true);
sig_idx = find(p_adj < 0.05);

[rows, cols] = ind2sub([ROIlen, ROIlen], sig_idx);
sig_mask = false(ROIlen, ROIlen);
sig_mask(triu(true(ROIlen), 1)) = (p_adj < 0.05);
[roi_i, roi_j] = find(sig_mask);
