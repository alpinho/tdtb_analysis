%%estimates the parameters PSE and sigma according to the Waveform Moment Analysis
%%also gives CE, and provides bootstrap estimates of all parameters including SE and CI

%%Input 
c=[300 350 400 450 500 550 600 650 700];  %comparison duration
ns=[0 5 6 11 8 4 1 0 0]; %number of Rsame responses
nd=[15 10 8 4 4 11 14 15 14]; %number of Rdiffernt responses
standard=500; %standard duration

%% Waveform Moment Analysis
fiSame = ns./(ns+nd);      %relative frequency of "same" responses
fiStar = fiSame./sum(fiSame);
% check=sum(fiStar)  %should be == 1

PSE_WMA = sum(fiStar.*c);
Sigma_WMA = sqrt(sum((c-PSE_WMA).^2.*fiStar));
CE_WMA = PSE_WMA-standard;

%% Bootstrap for estimating SE
nSamples = 1000;
nTrials = ns+nd;
for j=1:length(fiSame)
    BSns(:,j)=sum(rand(nSamples, nTrials(j)) < fiSame(j),2);
end

for j=1:nSamples
    BSfiSame(j,:) = BSns(j,:)./nTrials;      %rel freq of "same" responses
    BSfiStar(j,:) = BSfiSame(j,:)./sum(BSfiSame(j,:));
    % check(j)=sum(BSfiStar(j,:));  %all values should be == 1
    
    BS_PSE_WMA(j)   = sum(BSfiStar(j,:).*c);
    BS_Sigma_WMA(j) = sqrt(sum((c-BS_PSE_WMA(j)).^2.*BSfiStar(j,:)));
    BS_CE_WMA(j)    = BS_PSE_WMA(j)-standard;
end

BootstrapMeans=[mean(BS_PSE_WMA), mean(BS_Sigma_WMA), mean(BS_CE_WMA)];
BootstrapSEMs =[std(BS_PSE_WMA), std(BS_Sigma_WMA), std(BS_CE_WMA)];
BootstrapCIs=[prctile(BS_PSE_WMA, 2.5), prctile(BS_Sigma_WMA,  2.5),...
    prctile(BS_CE_WMA,  2.5); prctile(BS_PSE_WMA, 97.5), ...
    prctile(BS_Sigma_WMA, 97.5), prctile(BS_CE_WMA, 97.5)];

%% Output
WMAEstimatesPSE_SD_CE = [PSE_WMA, Sigma_WMA, CE_WMA]
BootstrapEstimates = [BootstrapMeans; BootstrapSEMs; BootstrapCIs]

%% Observed function and PSE estimate
plot(c,fiSame,'blacko-',[PSE_WMA PSE_WMA],get(gca, 'ylim'),'k:')
xlabel(' Comparison Level c'); 
ylabel(' Relative Frequency of R_{same}');
ylim([-0.03 1.03])
legend('Observed relative frequency of R_{same}', ...
    'WMA estimate of PSE','Location','NorthWest')

