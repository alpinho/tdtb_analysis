%%estimates the parameters gamma, eta, sigma of the same-different model 
%%under the assumption that the standard deviation ? increases with comparison level 
%%(cf. Birngruber et al., 2014).
%% also provides SEs and CIs for these parameters 

%%Input 
c=[300 350 400 450 500 550 600 650 700];  %comparison duration
ns=[0 5 6 11 8 4 1 0 0]; %number of Rsame responses
nd=[15 10 8 4 4 11 14 15 14]; %number of Rdiffernt responses
s=500; %standard duration

%% Estimate gamma, eta, sigma
f = @(y) -sum(ns.*log(normcdf(y(1),    c-s-y(2), y(3).*c)    ...
                      - normcdf(-y(1), c-s-y(2), y(3).*c)))  ...
         -sum(nd.*log(1-normcdf(y(1),  c-s-y(2), y(3).*c)    ...
                      + normcdf(-y(1), c-s-y(2), y(3).*c)));
[xmin,value,flag,output]=fminsearch(f,[0.1*s,0,0.1]);

%% Fisher Information, Standard errors, Estimates and CIs
g=xmin(1);e=xmin(2);d=xmin(3);
h=0.0001;
I(1,1) =  -(f([g+h,e,d]) - 2*f([g,e,d]) + f([g-h,e,d])) / h^2;
I(2,2) =  -(f([g,e+h,d]) - 2*f([g,e,d]) + f([g,e-h,d])) / h^2;
I(3,3) =  -(f([g,e,d+h]) - 2*f([g,e,d]) + f([g,e,d-h])) / h^2;
I(1,2) =  -(f([g+h,e+h,d]) + f([g-h,e-h,d]) - f([g-h,e+h,d]) - f([g+h,e-h,d])) / (4*h^2);
I(1,3) =  -(f([g+h,e,d+h]) + f([g-h,e,d-h]) - f([g-h,e,d+h]) - f([g+h,e,d-h])) / (4*h^2);
I(2,3) =  -(f([g,e+h,d+h]) + f([g,e-h,d-h]) - f([g,e-h,d+h]) - f([g,e+h,d-h])) / (4*h^2);
I(2,1) =  I(1,2); I(3,1) = I(1,3); I(3,2) = I(2,3);  I = abs(inv(I));

r12 = I(1,2)/sqrt(I(1,1)*I(2,2));
r13 = I(1,3)/sqrt(I(1,1)*I(3,3));
r23 = I(2,3)/sqrt(I(2,2)*I(3,3));
r12_r13_r_23 = [r12 r13 r23] %Estimated correlations among estimates

SE_gamma=sqrt(I(1,1));
SE_eta=sqrt(I(2,2));
SE_w=sqrt(I(3,3));

gamma=g; eta=e; w=d; 
gamma_eta_w = [gamma eta w; SE_gamma SE_eta SE_w]
conf95_gamma = [gamma-1.96*SE_gamma, gamma+1.96*SE_gamma]
conf95_eta = [eta-1.96*SE_eta, eta+1.96*SE_eta]
conf95_w = [w-1.96*SE_w, w+1.96*SE_w]


%% True and Estimated function
psyest = @(x) normcdf( xmin(1), x-s-xmin(2), x.*xmin(3)) ...
            - normcdf(-xmin(1), x-s-xmin(2), x.*xmin(3));
x=min(c):max(c);
F=psyest(x); n=ns+nd;
plot(x,F,'black-',c,ns./n,'blacko')
xlabel(' Comparison Level c')
ylabel(' Relative Frequency of R_{same}')
ylim([-0.03 1.03])
legend('Estimated psychometric function',...
       'Observed relative frequency','Location','NorthEast')
