% ********** Setup the directory on your computer **********
clear
restoredefaultpath
filePath = matlab.desktop.editor.getActiveFilename;
root0 = fileparts(filePath);
addpath(genpath(fullfile(root0,'lib')));
root = fullfile(root0,'data');

projname = 'cylinder_1';

%% Create b-table using mrtrix
ndir = 30;              % # gradient direction per b-shell
bvec = dirgen(ndir);    % gradient direction [gx, gy, gz]
bval = 0.25:0.25:1.5;   % b-value, ms/um2
btab = [];              % b-table: [b-value, gx, gy, gz]
for i = 1:numel(bval)
    bvali = bval(i);
    for j = 1:size(bvec,1)
        bvecj = bvec(j,:);
        btab = cat(1,btab,[bvali; bvecj(:)]);
    end
end

%% Set up simulation
% Geometry parameters
d = [1     1.1   1.2   1.3   1.4];  % cylinder diameter (um)
f = [0.4   0.45  0.5   0.55  0.6];  % cylinder volume fraction
Din = 1;                            % intra-cellular diffusivity, um^2/ms
Dex = 2;                            % extra-cellular diffusivity, µm^2/ms
tex = [10:10:50 75 100:50:200];     % exchange time, ms
nite = 1;                           % repetition of packing

% Simulation parameters
dt = 2e-4;                          % time of each step, ms
TN = 1e6;                           % # steps
NPar = 2e6;                         % # random walkers
Nbvec = size(btab,1);               % # gradient direction in total        
pinit = 3;                          % initial position, 1=ICS, 2=ECS, 3=ICS+ECS, 4=center
threadpb = 256;                     % Thread per block for cuda
seed = 0;                           % seed for random number generator

for i = 1:numel(d)
    di = d(i);
    for j = 1:numel(f)
        fj = f(j);
        for k = 1:numel(tex)
            texk = tex(k);
            kappak = (1-fj)/texk/(2/(di/2));    % permeability = (1-f)/tex/(S/V), um/ms
            for l = 1:nite
                % Assign the seed for random number generator of paking
                % generation for reproducibility
                seed = seed+1;
                
                % Create directory for simulation
                target = fullfile(root,projname,sprintf('cylinder_%04u',seed));
                mkdir(target);
                
                % Directory to the packing code
                root_code = fullfile(root0,'lib','cylinder','packing');
                
                % Save the b-table
                fileID = fopen(fullfile(target,'btable.txt'),'w');
                fprintf(fileID,sprintf('%g\n',btab));
                fclose(fileID);
                
                % Save the simulation parameters
                fileID = fopen(fullfile(target,'simParamInput.txt'),'w');
                fprintf(fileID,sprintf('%g\n%u\n%u\n%u\n%g\n%g\n%g\n%u\n%u\n',dt,TN,NPar,Nbvec,Din,Dex,kappak,pinit,threadpb));
                fclose(fileID);
                
                % Generate cylinder packing
                pc = packcyl();
                maxdensity = fj;            % targeted intra-cellular volume fraction
                hardwallBC = 0;             % boundary condition, 0=periodic, 1=hard wall
                N = 999;                    % # cylinders
                rinit = (di/2)*ones(N,1);   % cylinder radius, um
                % Complie the packing C code if you have not run it on your machine
                if seed==1
                    compileFlag = 1;        % 0=do not compile the C++ code, 1=compile the C++ code
                else
                    compileFlag = 0;        % 0=do not compile the C++ code, 1=compile the C++ code
                end 
                n = 500;                    % matrix size of the lookup table n x n
                gap = 0;                    % minimal distance between spheres (um), default=0
                pc.packing(maxdensity,hardwallBC,rinit,root_code,target,compileFlag,n,gap);
            end
        end
    end
end

%% Create a shell script to run the codes
% Decide the list of the simulations labeled with the seed
files = dir(fullfile(root,projname,'cylinder_*'));
list = 1:numel(files);

% Compile the CUDA C++ code if you have not run it on your machine
root_cuda = fullfile(root0,'lib','cylinder','rms');
% Please go to the directory and run sh compile_rms.sh in terminal to compile the code

% Create a shell script for simulations
fileID = fopen(fullfile(root,projname,'job.sh'),'w');   
fprintf(fileID,'#!/bin/bash\n');
for i = list
    target = fullfile(root,projname,sprintf('cylinder_%04u',i));
    fprintf(fileID,sprintf('cd %s\n',target));
    fprintf(fileID,sprintf('cp -a %s .\n',fullfile(root_cuda,'main_cuda')));
    fprintf(fileID,'./main_cuda\n');
end
fclose(fileID);
% Please go to the directory and run sh job.sh to run the simulations

%% Have a look for the microstructure
% The lookup table A saves two cylinder labels in one integer. If the first
% and the second cylinder labels are a1 and a2, a1 = mod(A,Nmax), and a2 =
% floor(A/Nmax).
% Other parameters:
%   Nmax: the smallest integer that is a power of 10 and larger than nCir
%   FOV: field of view of the entire gemoetry, um
%   nCir: # cylinders
%   rCir: cylinder radius, unitless, normalized by FOV
%   [xCir, yCir]: cylinder center position, unitless, normalized by FOV
root_packing = fullfile(root,'cylinder_1','cylinder_0001');
pc = packcyl();
[A, Nmax, FOV, nCir, xCir, yCir, rCir] = pc.readpacking(root_packing);

% Plot the microstructure and the lookup table
figure; set(gcf,'unit','inch','position',[0 0 12 5])

% Cylinder packing in continuous space
subplot(121);
pc.plotpack(xCir, yCir, rCir);

% Lookup table
subplot(122);
pc.plotlookup(A, nCir);

