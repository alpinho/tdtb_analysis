%%PseudoGaussian
%estimates epsilon (CE), w, PSE, and DL for the Pseudo-Gaussian function.
%also provides SEs and CIs for all parameters

%%Input 
c=[300 350 400 450 500 550 600 650 700];  %comparison duration
n1=[14 11 12 5 3 0 1 0 0]; %number of R1 responses
n2=[1 4 2 10 9 15 14 15 14]; %number of R2 responses
s=500; %standard duration

%% LogLikelihood function and estimation of a and b
logL = @(y) -sum(n2.*log(  normcdf(c, y(1)+s, y(2).*c )))... 
            -sum(n1.*log(1-normcdf(c, y(1)+s, y(2).*c )));
[xmin,value,flag,output]=fminsearch(logL,[0.0,0.1]);
CEest=xmin(1); west=xmin(2); PSEest=s+CEest; DLest=s*west;

%% Fisher Information
a=xmin(1);b=xmin(2); h=0.00001;
I(1,1) =  -(logL([a+h,b]) - 2*logL([a,b]) + logL([a-h,b])) / h^2;
I(2,2) =  -(logL([a,b+h]) - 2*logL([a,b]) + logL([a,b-h])) / h^2;
I(1,2) =  -(logL([a+h,b+h]) + logL([a-h,b-h]) ...
            - logL([a-h,b+h]) - logL([a+h,b-h])) / (4*h^2);
I(2,1) =  I(1,2); 
Cov=abs(inv(I)); SE_CE=sqrt(Cov(1,1)); SE_w=sqrt(Cov(2,2)); 
SE_PSE=SE_CE; SE_DL = s*SE_w;

conf95_PSE = [PSEest-1.96*SE_PSE, PSEest+1.96*SE_PSE];
conf95_w = [west-1.96*SE_w, west+1.96*SE_w];
conf95_CE = [CEest-1.96*SE_PSE, CEest+1.96*SE_PSE];

%% Output
PSE_w_CE = [PSEest,west, CEest; SE_PSE SE_w SE_PSE]
conf95_PSE
conf95_w
conf95_CE

z=norminv(0.75,0,1);
DLest=(s+CEest) * west*z/(1-(west*z)^2)

%% True and Estimated function
psyest = @(x) normcdf(x, xmin(1)+s, xmin(2).*x);
x=min(c):max(c); F=psyest(x);
plot(x,F,'black-',c,n2./(n1+n2),'blacko')
xlabel(' Comparison Level c'); ylabel(' Relative Frequency of R_2');
ylim([-0.03 1.03])
legend('Estimated psychometric function',...
       'Observed relative frequency of R_2','Location','SouthEast')
