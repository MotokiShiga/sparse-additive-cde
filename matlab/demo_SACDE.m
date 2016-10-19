function demo_SACDE(dataset)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% demo of SA-CDE
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

if nargin==0
  dataset = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% setting of SA-CDE
param.lambda    = logspace(log10(2),-2,20);
param.sigma     = logspace(log10(2),-2,20);
param.flag_path = true;
% setting of dataset
dim_noise = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng('default');
rng('shuffle');


%%%%%%%%%%%%%%%%%%%%%%%%% Generating data
switch dataset
 case {1,2,3} % Artificial data
  ntrain=300;
  xtrain=rand(1,ntrain)*2-1;
  switch dataset
   case 1
    noise=randn(1,ntrain);
   case 2
    dummy=(rand(1,ntrain)>0.5);
    noise=randn(1,ntrain)*(2/3)+(dummy*2-1);
   case 3
    dummy = (rand(1,ntrain)>0.75);
    noise = randn(1,ntrain).*((dummy==0)*1+(dummy==1)/3) + dummy*(3/2);
  end
  ytrain=sinc(0.75*pi*xtrain)+exp(1-xtrain).*noise/8;
  xtest0=[-0.5 0 0.5];
  ytest0=linspace(-3,3,300);
  axis_limit=[-1 1 -3 3];
end

xtest1=repmat(xtest0,length(ytest0),1);
ytest1=repmat(ytest0',1,length(xtest0));
xtest=xtest1(:)';
ytest=ytest1(:)';
ntest=length(xtest);

xtrain = [xtrain' rand(ntrain,dim_noise)];
ytrain = ytrain';
xtest  = [xtest' zeros(ntest, dim_noise) ];
ytest  = ytest';

%normalization
xscale=std(xtrain,0);
yscale=std(ytrain,0);
xmean=mean(xtrain);
ymean=mean(ytrain);
xtrain_normalized=(xtrain - repmat(xmean,[ntrain 1]))./repmat(xscale,[ntrain 1]);
ytrain_normalized=(ytrain - repmat(ymean,[ntrain 1]))./repmat(yscale,[ntrain 1]);
xtest_normalized= (xtest  - repmat(xmean,[ntest  1]))./repmat(xscale,[ntest 1]);
ytest_normalized= (ytest  - repmat(ymean,[ntest  1]))./repmat(yscale,[ntest 1]);

% figure
% plot(xtrain_normalized(:,1),xtrain_normalized(:,2),'*')
% axis equal

%True conditional density for artificial data
switch dataset
 case 1
  ptest=pdf_Gaussian(ytest,sinc(0.75*pi*xtest(:,1)),exp(1-xtest(:,1))/8);
 case 2
  tmp=exp(1-xtest(:,1))/8;
  ptest=pdf_Gaussian(ytest,sinc(0.75*pi*xtest(:,1))-tmp,tmp*2/3)/2 ...
        +pdf_Gaussian(ytest,sinc(0.75*pi*xtest(:,1))+tmp,tmp*2/3)/2;
 case 3
  tmp=exp(1-xtest(:,1))/8;
  ptest=pdf_Gaussian(ytest,sinc(0.75*pi*xtest(:,1)),tmp)*3/4 ...
        +pdf_Gaussian(ytest,sinc(0.75*pi*xtest(:,1))+tmp*3/2,tmp/3)/4;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SA-CDE
tic
optP = SACDE(xtrain_normalized, ytrain_normalized, param);
toc
ph = pdf_SACDE(xtest_normalized,ytest_normalized, optP);
ph = ph(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


hf1 = figure;
clf
hold on
set(gca,'FontName','Helvetica')
set(gca,'FontSize',12)
plot(xtrain(:,1),ytrain,'ko','LineWidth',1,'MarkerSize',4)
xtest_unique=unique(xtest(:,1));
for xtest_index=1:length(xtest_unique)
  x=xtest_unique(xtest_index);
  cdf_scale=(xtest0(2)-xtest0(1))*0.8/max(max(ptest(xtest(:,1)==x)),max(ph(xtest(:,1)==x)/yscale));
  h1 = plot(xtest(xtest(:,1)==x,1)+ptest(xtest(:,1)==x)*cdf_scale,...
    ytest(xtest(:,1)==x),'b--','LineWidth',2);
  h2 = plot(xtest(xtest(:,1)==x,1)+ph(xtest(:,1)==x)*cdf_scale/yscale,...
            ytest(xtest(:,1)==x),'g--','LineWidth',2);
end

set(gca,'FontName','Helvetica', 'FontSize',14)
switch dataset
 case {1}
  h = legend([h1 h2], 'True','SA-CDE','Location','SouthEast');
 case {2,3}
  h = legend([h1 h2], 'True','SA-CDE');
end
set(h,'FontSize',10);
axis(axis_limit)
xlabel('x','FontName','Helvetica', 'FontSize',14)
ylabel('y','FontName','Helvetica', 'FontSize',14)
% axis square
% set(gcf,'PaperUnits','centimeters');
% set(gcf,'PaperPosition',[0 0 12 10]);
% print('-depsc',sprintf('Illust_Toy%g',dataset))
  
  
figure
semilogx(param.lambda, optP.alpha_l2','*-')
hold on
x = repmat(optP.lambda,[1,100]);
y_max = max(optP.alpha_l2(:))*1.1;
y = linspace(0,y_max);
plot(x, y, 'k--');
xlim([ min(param.lambda), max(param.lambda)])
ylim([0,y_max])
set(gca,'FontName','Helvetica', 'FontSize',14)
xlabel('\lambda','FontName','Helvetica', 'FontSize',14);
ylabel('|| \alpha_d ||_2','FontName','Helvetica', 'FontSize',14);
h = legend('Relevant feature');
set(h,'FontName','Helvetica', 'FontSize',10);
% axis square
% set(gcf,'PaperUnits','centimeters');
% set(gcf,'PaperPosition',[0 0 12 10]);
% print('-depsc',sprintf('RPath_Toy%g',dataset))

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function px=pdf_Gaussian(x,mu,sigma) 
px=(1./sqrt(2*pi*sigma.^2)).*exp(-((x-mu).^2)./(2*sigma.^2));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y=sinc(x)
y=ones(size(x));
i=(x~=0);
y(i)=sin(pi*x(i))./(pi*x(i));   
end
