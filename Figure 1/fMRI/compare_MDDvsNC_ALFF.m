
%%
clear;
% Matrix provided by REST-META-MDD
load('Stat_Sub_Info_MDDvsNC.mat');
load('ALFF_MDDvsNC.mat');

X=[ones(size(Dx)),Dx,Age,Sex,Edu,Motion];
Z={ones(size(Dx)),Dx};
G={Site,Site};
k = length(Site);

ROIlen = 116;
TMatrix=zeros(ROIlen,1);
PMatrix=zeros(ROIlen,1);

for i=1:ROIlen
    y = res(i,:);
    lme = fitlmematrix(X,y,Z,G);
    TMatrix(i)=lme.Coefficients{2,4}; 
    PMatrix(i)=lme.Coefficients{2,6};
end

ans = find(mafdr(PMatrix,'BHFDR', true)<0.05);

