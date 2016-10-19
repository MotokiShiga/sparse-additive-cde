function [optK] = SACDE(X, Y, param)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Sparse Additive Conditional Density Estimation
%
% [Input]
%  X: Training inputs
%  Y: Training outputs
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


lambda_list  = param.lambda;
sigma_list   = param.sigma;

if ~isfield(param,'flag_path')
  param.flag_path = false;
end
if ~isfield(param,'B')
  param.B = 100;
end

%length
[num, dim_X] = size(X);
[~, dim_Y] = size(Y);

% setting
maxIter  = 50;
num_fold = 5;

cv_index = floor( (0:num-1) * num_fold ./ num)+1;
cv_index = cv_index( randperm(num) );


% sampling for basis function
B = min( num, param.B);
rnum = randsample(num,B);
mu = X(rnum,:);
nu = Y(rnum,:);

% output
optK.mu    = mu;
optK.nu    = nu;
optK.B     = B;

% constant
one2B = (1:B);

%distance between Y and nu
dist_nu_Y = repmat(sum( (Y').^2, 1),[B,1]) - 2*nu*Y'...
             + repmat(sum( nu.^2, 2 ),[1,num]);
dist_nu_Y_dup = repmat(dist_nu_Y,dim_X,1);
% distance between X and mu 
dist_mu_X = zeros(dim_X*B,num);
for g = 1:dim_X
  X_tmp = X(:,g)';
  dist_mu_X((g-1)*B+one2B,:) = repmat( X_tmp.^2, [B,1]) - 2*mu(:,g)*X_tmp...
                               + repmat(sum( mu(:,g).^2 ,2),[1,num]);
end
% distance between nu and nu
tmp1 = nu*nu';
tmp2 = repmat(diag(tmp1),[1,B]);
dist_nu_nu = tmp2 - 2*tmp1 + tmp2';


% Choose parameters by Cross-Validation
nll = zeros(length(lambda_list),length(sigma_list));
for k_fold = 1:num_fold

  ind_test  = (cv_index==k_fold);
  ind_train = ~ind_test;
  num = sum(ind_train);
  
  %compute fonstant for h and H
  C_h = dist_nu_Y_dup(:,ind_train) + dist_mu_X(:,ind_train);
  dist_mu_X_for_H = dist_mu_X(:,ind_train);
  
  for c_sigma = 1:length(sigma_list)
%     disp(['CV: ', num2str(k_fold), ', c_sigma: ',num2str(c_sigma)])
    sigma = sigma_list(c_sigma);
        
    %initialization
    alpha = zeros(dim_X*B,1);
%     cost  = nan(maxIter,1);
    
    %coefficients
    h  = cal_h;
    H = cal_H_dependent;
    L       = eigs(H,1);
    
    % compute for pdf
    tmp1 = dist_mu_X(:,ind_test);
    tmp2 = dist_nu_Y_dup(:,ind_test);
    pdf1 = exp( -( tmp1 + tmp2 ) /(2*sigma^2) )/(sqrt(2*pi) * sigma)^(dim_Y);
    pdf2 = exp( -( tmp1 ) /(2*sigma^2) );
    
    %main loop
    for li = 1:length(param.lambda)
      lambda = lambda_list(li);
      alpha = zeros(dim_X*B,1);
      for iter = 1:maxIter
        alpha_old = alpha;
        update_alpha;
%         cost(iter) = cal_cost;
        epsilon = norm(alpha - alpha_old) / (dim_X*B);
        if epsilon <= 1e-5
          break;
        end
      end
      optK.sigma  = sigma;
      optK.lambda = lambda;
      optK.alpha  = alpha;
      ph = pdf_val;
      nll(li,c_sigma) = nll(li,c_sigma) - mean(log(ph));
    end
  end
end
nll = nll / num_fold;
[min_nll,c_l] = min(nll);
[~,c_s] = min(min_nll);
c_l = c_l( c_s );
lambda = lambda_list(c_l);
sigma  = sigma_list(c_s);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% optimize alpha with optimized parameters
num = size(X,1);

%initialization
alpha = zeros(dim_X*B,1);
% cost  = nan(maxIter,1);

%compute for h and H
C_h = dist_nu_Y_dup + dist_mu_X;
dist_mu_X_for_H = dist_mu_X;

%coefficients
h = cal_h;
H = cal_H_dependent;
L = max( eigs(H) );

%outputs
optK.sigma   = sigma;
optK.lambda  = lambda;

%main loop
alpha = zeros(dim_X*B,1);
for iter = 1:maxIter
  alpha_old = alpha;
  update_alpha;
%   cost(iter) = cal_cost;
  epsilon = norm(alpha - alpha_old) / (dim_X*B);
  if epsilon <= 1e-5
    break;
  end
end
optK.alpha = alpha;
optK.alpha_l2 = zeros(1,dim_X);
for g = 1:dim_X
  ind = (g-1)*B + (1:B);
  optK.alpha_l2(g) = norm( alpha(ind) );
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing reguralization path
%
if param.flag_path
  optK.alpha_l2 = zeros(dim_X,length(lambda_list));
%   cost  = nan(maxIter,1);    
  for li = 1:length(lambda_list)
    %main loop
    lambda = lambda_list(li);
    alpha = zeros(dim_X*B,1);
    for iter = 1:maxIter
      alpha_old = alpha;
      update_alpha;
%       cost(iter) = cal_cost;
      epsilon = norm(alpha - alpha_old);
      if epsilon <= 1e-10
        break;
      end
    end
    for g = 1:dim_X
      ind = (g-1)*B + (1:B);
      optK.alpha_l2(g,li) = norm( alpha(ind) );
    end
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = cal_h
  h = mean( exp( -C_h/(2*sigma^2) ), 2);
  h( isinf(h) ) = 1/eps;
  h( isnan(h) ) = 0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function H = cal_H_dependent
  H = zeros( dim_X*B, dim_X*B );
  C_nu = exp( -dist_nu_nu /(4*sigma^2) );
  C_nu = C_nu * ((sqrt(pi)*sigma)^dim_Y/num);  
  C1 = exp( -dist_mu_X_for_H / (2*sigma^2) );
  C1 = C1*C1';
  for g1 = 1:dim_X
    ind1 = (g1-1)*B+one2B;
    H( ind1, ind1 ) = C1(ind1,ind1) .* C_nu;
    for g2 = 1:(g1-1)
      ind2 = (g2-1)*B+one2B;
      tmp = C1(ind1,ind2) .* C_nu;
      H( ind1, ind2 ) = tmp;
      H( ind2, ind1 ) = tmp';
    end
  end  
  H( isinf(H) ) = 1/eps;
  H( isnan(H) ) = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_alpha
  u = alpha - (H*alpha - h) / L;
  u = 0.5*(u + abs(u));
  for g_sub = 1:dim_X
    id = (g_sub-1)*B + one2B;
    const = norm( u(id) );
    if const==0
      alpha(id) = 0;
    else
      tmp = 1 - lambda/(L*const);
      alpha(id) = 0.5*(tmp + abs(tmp)) .* u(id);
    end
  end
end

%   function cost = cal_cost
%     cost = 0.5*(alpha'*H*alpha) - h'*alpha;
%     cost2 = 0;
%     for g_sub = 1:dim_X
%       id = (g_sub-1)*B + (1:B);
%       cost2 = cost2 + norm( alpha(id) );
%     end
%     cost = cost + lambda*cost2;
%   end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function pdf = pdf_val

T1   = alpha'*pdf1;
T2   = alpha'*pdf2;
pdf  = T1./max(eps,T2);
pdf( pdf<eps ) = eps;

end

end