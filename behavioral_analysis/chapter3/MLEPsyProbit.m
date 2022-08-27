%%Probit analysis
%estimates PSE and DL assuming an ogive psychometric function, 
%modeled as a cumulative density function of a normal distribution.
%also provides CE, and gives SEs and CIs for all parameters

%%Input 
c=[300 350 400 450 500 550 600 650 700];  %comparison duration
n1=[14 11 12 5 3 0 1 0 0]; %number of R1 responses
n2=[1 4 2 10 9 15 14 15 14]; %number of R2 responses
standard=500; %standard duration

%% Estimate mu and sigma
f=@(y) -sum(n2.*log(normcdf(c,y(1),y(2))))-sum(n1.*log(1-normcdf(c,y(1),y(2))));
[xmin,value,flag,output]=fminsearch(f,[mean(c),mean(c)*0.1]);
PSEest=xmin(1); DLest=xmin(2)*norminv(0.75); CEest=PSEest-standard;

%% Fisher Information
m=xmin(1);s=xmin(2); h=0.00001;
I(1,1) =  -(f([m+h,s])-2*f([m,s])+f([m-h,s]))/h^2;
I(2,2) =  -(f([m,s+h])-2*f([m,s])+f([m,s-h]))/h^2;
I(1,2) =  -(f([m+h,s+h])+f([m-h,s-h])-f([m-h,s+h])-f([m+h,s-h]))/(4*h^2);
I(2,1) =  I(1,2);
Cov=abs(inv(I)); SE_PSE=sqrt(Cov(1,1)); SE_DL=sqrt(Cov(2,2))*norminv(0.75);
r=I(1,2)/sqrt(I(1,1)*I(2,2));

%% Output
ProbitEstimates_PSE_DL_CE = [PSEest, DLest,CEest; SE_PSE SE_DL SE_PSE]
conf95_PSE = [PSEest-1.96*SE_PSE, PSEest+1.96*SE_PSE]
conf95_DL = [DLest-1.96*SE_DL, DLest+1.96*SE_DL]
conf95_CE = [CEest-1.96*SE_PSE, CEest+1.96*SE_PSE]
r %estimated correlation between the two estimates

%% True and estimated function
figure(1)
psyest = @(x) normcdf(x,xmin(1),xmin(2)); x=min(c):max(c); F=psyest(x);
plot(x,F,'black-',c,n2./(n1+n2),'blacko')
xlabel(' Comparison Level c'), ylabel(' Relative Frequency of R_2')
ylim([-0.03 1.03])
legend('Estimated psychometric function','Observed relative frequency',...
    'Location','SouthEast')


% %% Fitting the Logistic Function
% standard=500;
% %% LogLikelihood function and estimation of a and b
% logL = @(y) -sum(n2.*log(1./(1+exp(-(c-y(1))./y(2)))))... 
%             -sum(n1.*log(1./(1+exp( (c-y(1))./y(2)))));  
% [xmin,value,flag,output]=fminsearch(logL,[mean(c),mean(c)*0.1]);
% PSEest=xmin(1);DLest=xmin(2)*log(3); CEest=PSEest-standard;
% 
% %% Fisher Information
% m=xmin(1);s=xmin(2); h=0.00001;
% I(1,1) =  -(logL([m+h,s]) - 2*logL([m,s]) + logL([m-h,s])) / h^2;
% I(2,2) =  -(logL([m,s+h]) - 2*logL([m,s]) + logL([m,s-h])) / h^2;
% I(1,2) =  -(logL([m+h,s+h]) + logL([m-h,s-h]) ...
%             - logL([m-h,s+h]) - logL([m+h,s-h])) / (4*h^2);
% I(2,1) =  I(1,2); 
% Cov=abs(inv(I)); SE_PSE=sqrt(Cov(1,1)); SE_DL=sqrt(Cov(2,2))*log(3);
% r=I(1,2)/sqrt(I(1,1)*I(2,2));
% 
% conf95_PSE = [PSEest-1.96*SE_PSE, PSEest+1.96*SE_PSE];
% conf95_DL = [DLest-1.96*SE_DL, DLest+1.96*SE_DL];
% conf95_CE = [CEest-1.96*SE_PSE, CEest+1.96*SE_PSE];
% 
% %% Output
% LogisticEstimates_PSE_DL_CE = [PSEest,DLest, CEest; SE_PSE SE_DL SE_PSE]
% conf95_PSE
% conf95_DL
% conf95_CE
% r
% 
% %% True and Estimated function
% figure(2)
% psyest = @(x) 1./(1+exp(-(x-xmin(1))./xmin(2)));
% x=min(c):max(c); F=psyest(x);
% plot(x,F,'black-',c,n2./(n1+n2),'blacko')
% xlabel(' Comparison Level c'); ylabel(' Relative Frequency of R_2');
% ylim([-0.03 1.03])
% legend('Estimated psychometric function',...
%        'Observed relative frequency of R_2','Location','SouthEast')
