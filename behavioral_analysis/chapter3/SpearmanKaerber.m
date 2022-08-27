function SpearmanKaerber
%%provides PSE, SD, DL, and CE derived by the Spearman-Kaerber method
%%also gives bootstrap estimates of these parameters including SE and CI

%% Input
c = [300 350 400 450 500 550 600 650 700];  %comparison duration
n1 = [14 11 12 5 3 0 1 0 0]; %number of R1 responses
n2 = [1 4 2 10 9 15 14 15 14]; %number of R2 responses
s=500; %standard duration

[nTrials, fi, fiMono] = monotonize(n1, n2);

%% Spearman-Kaerber estimate
DLc = 0.6745; %75th percentile of standard normal
fiMono2 = diff([0, fiMono, 1]);
SKcLevels = [c(1) - diff(c(1:2)), c, c(end) + diff(c(end-1:end))];

PSE_SK   = 1/2*sum(fiMono2.*diff(SKcLevels.^2)./diff(SKcLevels));
M_SK     = 1/3*sum(fiMono2.*diff(SKcLevels.^3)./diff(SKcLevels));
Sigma_SK = sqrt(M_SK - PSE_SK^2);
DL_SK    = Sigma_SK*DLc;
CE_SK    = PSE_SK-s;

%% Bootstrap for estimating SE
nSamples = 1000;

for j=1:length(fi)
    BSn1(:,j)=sum(rand(nSamples, nTrials(j)) > fi(j),2);
end

for j=1:nSamples
    [~, BSfi(j,:), BSfiMono(j,:)] = monotonize(BSn1(j,:), nTrials-BSn1(j,:));
    BSfiMono2(j,:) = diff([0, BSfiMono(j,:), 1]);
    
    BS_PSE_SK(j) = 1/2*sum(BSfiMono2(j,:).*diff(SKcLevels.^2)./diff(SKcLevels));
    BS_M_SK(j)   = 1/3*sum(BSfiMono2(j,:).*diff(SKcLevels.^3)./diff(SKcLevels));
    BS_Sigma_SK(j) = sqrt(BS_M_SK(j) - BS_PSE_SK(j)^2);
    BS_DL_SK(j)  = BS_Sigma_SK(j)*DLc;
    BS_CE_SK(j)  = BS_PSE_SK(j)-s;
end

BootstrapMeans=[mean(BS_PSE_SK), mean(BS_Sigma_SK), mean(BS_DL_SK), mean(BS_CE_SK)];
BootstrapSEMs=[std(BS_PSE_SK), std(BS_Sigma_SK), std(BS_DL_SK), std(BS_CE_SK)];
BootstrapCIs=[prctile(BS_PSE_SK, 2.5), prctile(BS_Sigma_SK, 2.5),...
    prctile(BS_DL_SK, 2.5), prctile(BS_CE_SK, 2.5);...
    prctile(BS_PSE_SK, 97.5), prctile(BS_Sigma_SK, 97.5), ...
    prctile(BS_DL_SK, 97.5), prctile(BS_CE_SK, 97.5)];

%% Output
RelFreq = [fi; fiMono]
SKEstimates_PSE_SD_DL_CE = [PSE_SK, Sigma_SK, DL_SK, CE_SK]
BootstrapEstimates_PSE_SD_DL_CE = [BootstrapMeans; BootstrapSEMs; BootstrapCIs]

%% True and monotonized function and PSE estimate
plot(c,fi,'blacko',c,fiMono,'blackx-',[PSE_SK PSE_SK],get(gca, 'ylim'),'k:')
xlabel(' Comparison Level c'); ylabel(' Relative Frequency of R_2');
ylim([-0.03 1.03])
legend('Observed relative frequency of R_2',...
    'Monotonized relative frequency of R_2',...
    'SK estimate of PSE','Location','SouthEast')


%% Monotonize data
function [nTrials, fi, fiMono] = monotonize(n1, n2)

nTrials = n1+n2;    %number of trials per comparison duration
fi = n2./nTrials;   %relative frequency of "S2" responses
fiMono = fi;        %start value for monotonized data

while any(fiMono(1:end-1) > fiMono(2:end))  %as long as there is any non-monotonicity
    i=1;
    while i <= length(fiMono) - 1      %check for all c-levels until the second-to-last
        if fiMono(i) <= fiMono(i+1)    % if fi_c(i) <= fi_c(i+1)  (i.e., monotonous)
            i=i+1;                              % do nothing but increase i
        else                           % but if not monotonous
            k=1;                                % start a counter k
            while true
                tempfi = sum(fiMono(i:i+k) .* nTrials(i:i+k)) ...  % compute temporary fi for fi_c(i) to fi_c(i+k)
                    ./ sum(nTrials(i:i+k));
                if i+k+1 > length(fiMono); break                   % stop if i+k+1 < nclev
                elseif fiMono(i+k+1) > tempfi; break               % stop when next fi would be larger than temporary mean (i.e., monotonous)
                else
                    k=k+1;                                         % otherwise increase counter and start again, averaging over one more level
                end
            end
            fiMono(i:i+k) = repmat(tempfi,1,k+1);                  % after exit, replace the fis from i:k with the monotonized fi
            i=i+k+1;                                               % and just proceed at clevel i+k+1
        end
    end
end

