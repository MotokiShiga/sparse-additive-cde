function pdf = pdf_SACDE(X,Y, optP)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% probability density function using optimized parameters by SA-CDE
%
% [Input]
%  X:    test inputs
%  Y:    test outputs
%  optP: parameters optimized by SA-CDE algorithms
%
% [Output]
%  pdf:  estimated probability density p(Y|X)
%
%
% (c) Motoki Shiga, 
%     Department of Electrical, Electronic and Computer Engineering, 
%     Gifu University, Japan.
%     shiga_m@gifu-u.ac.jp
%
%
% Reference:
% [1] Motoki Shiga, Voot Tangkaratt, Masashi Sugiyama
%     "Direct Conditional Probability Density Estimation with Sparse Feature Selection",
%     Machine Learning, vol.100, no.2, pp.161-182, 2015.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[num,dim_X] = size(X);
dim_Y = size(Y,2);

C1 = (sqrt(2*pi) * optP.sigma)^(dim_Y);
C2 = 2*optP.sigma^2;
distY2nu = repmat(sum( (Y').^2, 1),[optP.B,1]) - 2*optP.nu*Y' ...
           + repmat(sum( optP.nu.^2, 2),[1,num]);

T1 = zeros(1,num);  T2 = zeros(1,num);
for d = 1:dim_X
  distX2mu = repmat(sum( (X(:,d)').^2, 1),[optP.B,1])...
    - 2*optP.mu(:,d)*X(:,d)' ...
    + repmat(sum(optP.mu(:,d).^2, 2),[1,num]);
  tmp1 = exp( -( distX2mu + distY2nu ) /C2 );
  tmp2 = exp( -( distX2mu ) /C2 );
  a = optP.alpha( (d-1)*optP.B+(1:optP.B) );
  T1 = T1 + a'*tmp1;
  T2 = T2 + a'*tmp2;
end
pdf = T1./max(eps,T2*C1);

pdf( pdf<eps ) = eps;
pdf = pdf(:);


end