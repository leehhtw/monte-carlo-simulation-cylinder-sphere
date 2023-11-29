% ********** Setup the directory on your computer **********
clear
restoredefaultpath
filePath = matlab.desktop.editor.getActiveFilename;
root0 = fileparts(filePath);
addpath(genpath(fullfile(root0,'lib')));
root = fullfile(root0,'data');

projname = 'sphere_1';

%% Read simulation result
files = dir(fullfile(root,projname,'sphere_*'));
sim = struct([]);
tic;
for i = 1:numel(files)
    sim(i).data = simul3Dcylinder_cuda(fullfile(files(i).folder,files(i).name));
end
toc;
save(fullfile(root,projname,'simResult_allsetup.mat'),'sim');

%% Analyze packing
files = dir(fullfile(root,projname,'sphere_*'));
pck = struct([]);
for i = 1:numel(files)
    FOV = load(fullfile(root,projname,files(i).name,'phantom_res.txt'));        % length size of the whole geometry, um
    rc  = load(fullfile(root,projname,files(i).name,'phantom_rCir.txt'));       % sphere radius, um
    pck(i).f_in  = sum(4/3*pi*rc.^3);               % intra-cellular volume fraction
    pck(i).r     = mean(FOV*rc);                    % sphere radius, um
    pck(i).sv_in = sum(4*rc.^2)/sum(4/3*rc.^3)/FOV; % intra-cellular surface-to-volume ratio
    pck(i).kappa = sim(i).data.kappa;               % permeability, um/ms
    pck(i).tex   = round((1-pck(i).f_in)/(pck(i).kappa * pck(i).sv_in));        % exchange time, ms
    pck(i).D_in  = sim(i).data.Din;                 % intra-cellular diffusivity, um2/ms
    pck(i).D_ex  = sim(i).data.Dex;                 % extra-cellular diffusivity, um2/ms
end
save(fullfile(root,projname,'packing.mat'),'pck');

%% Calculate diffusivity and kurtosis in narrow pulse limit
load(fullfile(root,projname,'simResult_allsetup.mat'));
load(fullfile(root,projname,'packing.mat'));

t = sim(1).data.TD;     % diffusion time, ms
f = [pck.f_in];         % intra-cellular volume fraction
r = [pck.r];            % cylinder radius, um
[C, ~, IC] = unique([r(:),f(:)],'row');
ru = C(:,1); 
fu = C(:,2); 
nr = numel(ru);

tex = [pck.tex];        % exchange time, ms
texu = unique(tex); nt = numel(texu);

Din = [pck.D_in];       % intra-cellular diffusivity, um2/ms
Dex = [pck.D_ex];       % extra-cellular diffusivity, um2/ms

% gradient direction in 100 random directions
gdir = randn(100,3);
gdir = gdir./sqrt(sum(gdir.^2,2));

figure('unit','inch','position',[-25 0 25 10]);
cmap = colormap('jet'); cmap = cmap(1:23:end,:);
for i = 1:numel(f)
    % Sphere: diffusivity and kurtosis averaged over random directions
    simi = sim(i).data;
    [Ki,Di] = simi.akc_mom(gdir);
    RDi = mean(Di,2);
    RKi = mean(Ki,2);
    
    It = find(tex(i)==texu);
    subplot(2,nr,IC(i))
    hold on;
    plot(t,RDi,'-','color',cmap(It,:),'linewidth',0.75);

    subplot(2,nr,IC(i)+nr)
    hold on;
    plot(t,RKi,'-','color',cmap(It,:),'linewidth',0.75);
end

subplot(2,nr,1)
hold on;
clear hd
for i = 1:nt
    hd(i) = plot(-1,0,'-','color',cmap(i,:),'linewidth',1);
    lgtxt{i} = sprintf('%u ms',tex(i));
end
legend(hd,lgtxt,'interpreter','latex','fontsize',10,...
    'location','southeast','box','off','NumColumns',1);

for i = 1:nr
   subplot(2,nr,i)
   xlim([0 200]); ylim([0 2]);
   box on; grid on;
   set(gca,'fontsize',12,'xtick',0:50:200);
   xlabel('$t$ (ms)','interpreter','latex','fontsize',20);
   ylabel('RD ($\mu$m$^2$/ms)','interpreter','latex','fontsize',20);
   title(sprintf('$d=%.0f$ $\\mu$m, $f=%.2f$',2*ru(i),fu(i)),...
       'interpreter','latex','fontsize',16);
   
   subplot(2,nr,i+nr);
   xlim([0 200]); ylim([0 3]);
   box on; grid on;
   set(gca,'fontsize',12,'xtick',0:50:200,'ytick',0:4);
   xlabel('$t$ (ms)','interpreter','latex','fontsize',20);
   ylabel('RK','interpreter','latex','fontsize',20);
   title(sprintf('$d=%.0f$ $\\mu$m, $f=%.2f$',2*ru(i),fu(i)),...
       'interpreter','latex','fontsize',16);
end
