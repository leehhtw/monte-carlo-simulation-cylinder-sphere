classdef packsph < handle
    % Packing spheres
    % 
    % Usage:
    % pc = packsph();
    % pc.packing(maxdensity,hardwallBC,rinit,root,target,compileFlag,n,gap)
    %
    % Input:
    % maxdensity: maximal sphere volume fraction
    % hardwallBC: boundary condition, 0 for periodic, 1 for hard wall
    % rinit: cylinder radius, [# axon x 1], micron
    % root: directory to Donev's C code
    % target: directory saving the packing
    % compileFlag: 0 for not compiling the C code, 1 for compiling the C code
    % n: size of the lookup table = n x n
    % gap: the shortest distance between cylinders, micron
    % 
    % Output:
    % packing_parameter.mat
    % xc, yc, rc: position of the sphere center and the sphere radius
    % within a normalized box
    % res: length of the box side in micron
    %
    % lookup_table.mat
    % A: lookup table
    % n: size of the lookup table
    % Nmax: the smallest integer larger than # sphere, in the base of 10
    %
    % phantom_*.txt
    % input files for cuda code
    %
    % (c) Hong-Hsi Lee, February, 2020
    
    properties (Constant = true, Access = protected)
        
    end
    
    properties (GetAccess = public, SetAccess = protected)
        
    end
    
    properties (GetAccess = private, SetAccess = protected)
        
    end
    
    methods (Access = public)
        function this = packsph()
            
        end
        
        function packing(this,maxdensity,hardwallBC,rinit,root,target,compileFlag,n,gap)
            % Initial position and rescaled outer radius
            rinit_adj = rinit+gap/2;
            maxdensity_adj = maxdensity*sum(rinit_adj.^3)/sum(rinit.^3);
            this.initposition(rinit_adj,root);
            N = length(rinit_adj);
            
            % Prepare input file for C++
            C = cell(1,15);
            fid = fopen(fullfile(root,'spheres_poly/input'),'r');
            for k = 1:15, C{k} = fgetl(fid); end
            fclose(fid);

            % Change N and maxpf variables in inputN file 
            C{2}=['int N = ' num2str(N) ';                        // number of spheres'];
            C{4}=['double maxpf = ' num2str(maxdensity_adj) ';                  // max packing fraction'];
            C{12}=['int hardwallBC = ' num2str(hardwallBC) ';                   // 0 for periodic, 1 for hard wall BC'];
            fid = fopen(fullfile(root,'spheres_poly/inputN'), 'w');
            fprintf(fid, '%s\n', C{:});
            fclose(fid);
            
            % compile and run Donev's C code
            root_tmp = pwd;
            cd(fullfile(root,'spheres_poly'));
            if compileFlag
                disp('compiling .............');
                system('g++ -O3 -ffast-math -o spheres neighbor.C spheres.C box.C sphere.C event.C heap.C read_input.C');
            end
            disp('running C ...........');
            tic;
            system('./spheres inputN'); % run
            toc;
            cd(root_tmp);
            
            % Save final packing
            outputfilename = fullfile(root,'spheres_poly/write.dat');
            M = dlmread(outputfilename,'',6,0);
            xc = M(:,1); yc = M(:,2); zc = M(:,3); rc = M(:,4)/2;
            
            % Adjust radius for targeted density
            density = sum(4/3*pi*rc.^3);
            if density>maxdensity
                rc = (maxdensity_adj/density)^(1/3)*rc;
            end
            
            % Calculate field-of-view in micron
            res = mean(rinit_adj)/mean(rc);
            rc = rc-gap/res/2;
            
            fprintf('Sphere volume fraction = %.4f\n',sum(4/3*pi*rc.^3));
            
            % Create lookup table
            [A,~,Nmax] = this.createlookuptable(n,xc,yc,zc,rc);
            
            % Save files for C++ simulation code
            fid = fopen(fullfile(target,'phantom_res.txt'),'w');
            fprintf(fid,'%f',res);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_NPix.txt'),'w');
            fprintf(fid,'%u',n);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_APix.txt'),'w');
            for i = 1:size(A,1)
                for j = 1:size(A,2)
                    fprintf(fid,sprintf('%u\n',A(i,j,:)));
                end
            end
            fclose(fid);
            
            fid = fopen(fullfile(target,'phantom_NAx.txt'),'w');
            fprintf(fid,'%u',length(rc));
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_xCir.txt'),'w');
            fprintf(fid,'%f\n',xc);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_yCir.txt'),'w');
            fprintf(fid,'%f\n',yc);
            fclose(fid);
            
            fid = fopen(fullfile(target,'phantom_zCir.txt'),'w');
            fprintf(fid,'%f\n',zc);
            fclose(fid);
            
            fid = fopen(fullfile(target,'phantom_rCir.txt'),'w');
            fprintf(fid,'%f\n',rc);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_Nmax.txt'),'w');
            fprintf(fid,'%u',Nmax);
            fclose(fid);
        end
        
        function arrange(this,target,xc,yc,zc,rc,res,n)
            [A,~,Nmax] = this.createlookuptable(n,xc,yc,zc,rc);
            
            fid = fopen(fullfile(target,'phantom_res.txt'),'w');
            fprintf(fid,'%f',res);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_NPix.txt'),'w');
            fprintf(fid,'%u',n);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_APix.txt'),'w');
            for i = 1:size(A,1)
                for j = 1:size(A,2)
                    fprintf(fid,sprintf('%u\n',A(i,j,:)));
                end
            end
            fclose(fid);
            
            fid = fopen(fullfile(target,'phantom_NAx.txt'),'w');
            fprintf(fid,'%u',length(rc));
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_xCir.txt'),'w');
            fprintf(fid,'%f\n',xc);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_yCir.txt'),'w');
            fprintf(fid,'%f\n',yc);
            fclose(fid);
            
            fid = fopen(fullfile(target,'phantom_zCir.txt'),'w');
            fprintf(fid,'%f\n',zc);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_rCir.txt'),'w');
            fprintf(fid,'%f\n',rc);
            fclose(fid);

            fid = fopen(fullfile(target,'phantom_Nmax.txt'),'w');
            fprintf(fid,'%u',Nmax);
            fclose(fid);
        end
        
        function [A,B,Nmax] = createlookuptable(this,n,xc,yc,zc,r)
        %CREATELOOKUPTABLE    create lookup table for packed spheres
        %   [A,B,Nmax] = createlookuptable(n,xc,yc,zc,r)
        %   
        %   Input:
        %   n: size of the lookup table = n x n x n
        %   xc, yc, zc: center of spheres, 0 <= xc,yc,zc <= 1
        %   r: outer radius of cylinders
        %
        %   Output:
        %   A: axon labels/lookup table
        %   B: # axon in each pixel
        %   Nmax: the smallest integer larger than # axon, in the base of 10
        %
        %   Author: Hong-Hsi Lee, 2016 (orcid.org/0000-0002-3663-6559)
        %           Lauren M Burcaw, 2014
            MSX = n;
            MSY = n;
            MSZ = n;
            A = zeros(MSX,MSY,MSZ);
            B = zeros(MSX,MSY,MSZ);
            N = length(r);
            Nmax = 10^(ceil(log10(N+1)));

            for i = 1:size(r,1)
                for ii = ceil((xc(i)-r(i))*n):(ceil((xc(i)+r(i))*n)+1)
                    if (ii>MSX)
                       ti = ii-MSX;
                    elseif (ii < 1)
                       ti = ii+MSX;
                    else
                       ti = ii;
                    end
                    for jj = ceil((yc(i)-r(i))*n):(ceil((yc(i)+r(i))*n)+1)
                        if jj > MSY
                            tj = jj-MSY;
                        elseif (jj < 1)
                            tj = jj+MSY;
                        else
                            tj = jj;
                        end
                        for kk = ceil((zc(i)-r(i))*n):(ceil((zc(i)+r(i))*n)+1)
                            if kk > MSZ
                                tk = kk-MSZ;
                            elseif (kk < 1)
                                tk = kk+MSZ;
                            else
                                tk = kk;
                            end
                            if this.inside_sphere(ii,jj,kk,xc(i),yc(i),zc(i),r(i),n)
                                if A(ti,tj,tk) == 0        % 1 sphere
                                    A(ti,tj,tk) = i;
                                    B(ti,tj,tk) = 1;
                                else
                                    if A(ti,tj,tk) < Nmax  % 2 spheres
                                        A(ti,tj,tk) = A(ti,tj,tk)*Nmax + i;
                                        B(ti,tj,tk) = 2;
                                    else                   % > 2 spheres
                                        fprintf('More than 2 spheres in one pixel.\n');
                                        fprintf('Use larger matrix size for the table.\n');
                                        pause;
                                    end
                                end                    
                            end
                        end
                    end
                end
            end
            A = uint32(A);
            B = uint16(B);
            fprintf(' * Matrix filling done ! *\n');
            fprintf(' ----------------------------\n');
        end
    end

    methods (Static)
        function inside = inside_sphere(ii,jj,kk,xc,yc,zc,r,n)
        %INSIDE_SPHERE    True for the pixel overlapping the sphere
        %   inside_sphere(ii,jj,kk,xc,yc,zc,r,n) returns 1 if the pixel (ii,jj,kk)
        %   overlaps with the sphere (xc,yc,zc,r), otherwise, return 0. The size of
        %   the lookup table is n x n x n.
        %
        %   Author: Hong-Hsi Lee, 2016 (orcid.org/0000-0002-3663-6559)
        
        x = max( ii-1, min(xc*n,ii) );
        y = max( jj-1, min(yc*n,jj) );
        z = max( kk-1, min(zc*n,kk) );
        
        d2 = (x-xc*n)^2 + (y-yc*n)^2 + (z-zc*n)^2;
        inside = d2 < (r*n)^2;
        
%             di = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1; 1 1 1];
%             inside = 0;
%             for i = 1:size(di,1)
%                 vi = ( (ii-di(i,1)-xc*n)^2 + (jj-di(i,2)-yc*n)^2 + (kk-di(i,3)-zc*n)^2 ) <= (r*n)^2;
%                 inside = inside + vi;
%             end
% 
%             di = [-1 0 0; 1 0 0; 0 -1 0; 0 1 0; 0 0 -1; 0 0 1];
%             for i = size(di,1)
%                vi = ( floor((xc+di(i,1)*r)*n)==(ii-1) ).*( floor((yc+di(i,2)*r)*n)==(jj-1) ).*( floor((zc+di(i,3)*r)*n)==(kk-1) );
%                inside = inside + vi;
%             end
%             inside = inside > 0;
        end
        
        function [x, y, z, rs] = initposition(rinit,root) 
        %INITPOSITION    Initialize positions and radii of densly packed spheres
        %   [x,y,z,rs] = initposition(rinit) returns initial positions x, y, z and
        %   rescaled radii rs of spheres for Donev's C++ input file, based on
        %   initial radii rinit (column vector), and saves them in the file
        %   specified in readfilename.
        %
        %   ---------------------------------------------------------
        %     in box.C, in the function
        %     void box::ReadPositions(const char* filename)
        %
        %   infile.ignore(256, '\n');  // ignore the dim line
        %   infile.ignore(256, '\n');  // ignore the #sphere 1 line
        %   infile.ignore(256, '\n');  // ignore the #sphere line
        %   infile.ignore(256, '\n');  // ignore the diameter line
        %   infile.ignore(1000, '\n'); // ignore the 100 010 001 line
        %   infile.ignore(256, '\n');  // ignore the T T T line
        % 
        %   for (int i=0; i<N; i++)
        %     {
        %       infile >> s[i].r;      // read in radius    
        %       infile >> s[i].gr;     // read in growth rate
        %       infile >> s[i].m;      // read in mass
        %       for (int k=0; k<DIM; k++)  
        %          infile >> s[i].x[k]; // read in position 
        %     }
        %    ..... 
            readfilename=fullfile(root,'spheres_poly/read.dat');
            dim=3;              % dimension
            N=length(rinit);    % # cylinders

            % rescale radii to make them small enough to not overlap, but not too small
            rmax=max(rinit); 
            dens0=0.01;         % dens0 ~ N(rmax/Rscale)^3
            % divide all radii by Rscale for less initial overlap
            Rscale=rmax*(N/dens0)^(1/3); 
            rinit=sort(rinit(:),'descend');
            rs=rinit/Rscale;

            % assign random positions and check for no overlap
            x=zeros(N,1); y=zeros(N,1); z=zeros(N,1); 
            % distance^2 function
            dist2 = @(x1,y1,z1, x2,y2,z2) min(abs(x1-x2),1-abs(x1-x2))^2 + min(abs(y1-y2),1-abs(y1-y2))^2 + min(abs(z1-z2),1-abs(z1-z2))^2;
            for ncurr=1:N
               overlap=1;
               while (overlap==1)
                  overlap=0;
                  x(ncurr)=rand; y(ncurr)=rand; z(ncurr)=rand;
                  for nprev=1:ncurr-1
                     if dist2(x(ncurr),y(ncurr),z(ncurr),x(nprev),y(nprev),z(nprev))<=(rs(ncurr)+rs(nprev))^2, overlap=1; ncurr; break, end
                  end
               end
            end

            % create read.dat file
            fid=fopen(readfilename,'w');
            fprintf(fid,'%d\n', dim);
            fprintf(fid,'%d\n', N);
            fprintf(fid,'%d\n', N);
            fprintf(fid,'%e\n', 2*max(rs));
            fprintf(fid,'10 01\n');
            fprintf(fid,'T T\n');
            for n=1:N
               fprintf(fid, '%e %e %f %f %f\n', rs(n), rs(n), 1.0, x(n), y(n), z(n));
            end
            fclose(fid);
        end
        
        function plothist(rinit,nbin)
        %PLOTHIST    Plot diameter histogram
        %   plothist(rinig,nbin) plots diameter histogram based on the
        %   radii rinit, and number of bins nbin.
            edges = linspace(0,rinit,nbin);
            Nc = histcounts(2*rinit,edges);
            Nc = Nc/sum(Nc)/mean(diff(edges));
            bar(edges(2:end),Nc,1);
            box on; pbaspect([2 1 1])
            xlabel('Inner Diameter ($\mu$m)','interpreter','latex','fontsize',20)
            ylabel('PDF ($\mu$m$^{-1}$)','interpreter','latex','fontsize',20)
        end
        
        function plotpack(xc,yc,zc,rc)
        %PLOTPACK    Plot cylinder packing
        %   plotpack(xc,yc,rc) plots cylinder packing based on the cylinder
        %   positions xc, yc and radii rc. Positions and radii are rescaled
        %   to fit within a 1 x 1 square.
            hold on;
            [xs, ys, zs] = sphere;
            for ii = 0%-1:1
                for jj = 0%-1:1
                    for kk = 0%-1:1
                        for i = 1:numel(rc)
                            surf(xs*rc(i)+xc(i)+ii, ys*rc(i)+yc(i)+jj, zs*rc(i)+zc(i)+kk,'edgealpha',0,'facecolor',0.7*[1 1 1]);
                        end
                    end
                end
            end
            xlim([0 1]); ylim([0 1]); zlim([0 1]);
            pbaspect([1 1 1]); box on
%             title('Sphere Packing','interpreter','latex','fontsize',20)
            set(gca,'xtick',[],'ytick',[],'ztick',[])
            view(3)
            material dull
            camlight
        end
        
        function plotlookup (A,Nsph,sl)
        %PLOTLOOKUP    Plot lookup table
        %   plotlookup(A,Nsph,sl) plots the lookup table A of Nsph packed 
        %   spheres on the sl-th slice. The background is black, and the 
        %   pixels with two spheres are white.
            cmap = colormap('parula');
            Ibg = A==0;                     % background region
            Iol = A>Nsph;                   % two-axon region
            A2 = ceil(single(A)/Nsph*64);   % rescale the colormap for # axons
            A2(Ibg) = 1;
            A2(Iol) = 1;
            [nx,ny,nz] = size(A2);
            imgc = cmap(uint16(A2(:)),:);
            imgc(Ibg,:) = 0;                % background region is black
            imgc(Iol,:) = 1;                % two-axon region is white
            imgc = reshape(imgc,[nx,ny,nz,3]);

            image(rot90(squeeze(imgc(:,:,sl,:)))); caxis([0 Nsph]);
            box on; axis off; pbaspect([1 1 1]);
            title('Lookup Table','interpreter','latex','fontsize',20)
        end
        
        function rinit = gaussdist(N,rmean,rstd,seed)
        %GAUSSDIST    Create Gaussian distributed radii
        %   rinit=gaussdist(N,rmean,rstd,seed) creates Gaussian distributed
        %   cylindrical radii rinit with mean rmean and standard deviation
        %   rstd. The output has N radii, and the randn is initilized by
        %   seed.
            rng(seed);
            rinit = [];
            while(numel(rinit)<N)
                ri = rmean + rstd*randn(N,1);
                ri = ri(ri>0);
                rinit = cat(1,rinit,ri(:));
            end
            rinit = rinit(1:N);
            rinit = rinit/mean(rinit)*rmean;
            rinit = sort(rinit);
        end
        
        function rinit = anydist(N,frequency,ri)
        %ANYDIST    Create radii based on a given distribution
        %   rinit=anydist(N,frequency,ri) creates cylindrical radii rinit 
        %   based on any given discretized distribution, frequency(ri). The
        %   output has N radii.
            pct = frequency/sum(frequency);         % frequency, count in percentage
            Nct = round(N*pct);                     % actual count
            if sum(Nct)<N
                I = randperm(length(Nct),N-sum(Nct));
                Nct(I) = Nct(I) + 1;
            elseif sum(Nct)>N
                Ip = find(Nct>0);
                I = randperm(numel(Ip),sum(Nct)-N);
                Nct(Ip(I)) = Nct(Ip(I)) - 1;
            end
            
            rinit = zeros(N,1);                     % output radius
            i = 0;
            for bin = 1:length(Nct)
              rinit(i+1:i+Nct(bin)) = ri(bin);
              i = i+Nct(bin);
            end
        end
        
        function ds = sphgap(xc,yc,zc,rc)
            xc = xc(:); yc = yc(:); zc = zc(:); rc = rc(:);
            ds = sqrt((xc-xc.').^2 + (yc-yc.').^2 + (zc-zc.').^2) - (rc+rc.');
            n = length(rc);
            ds(1:n+1:end) = 0;
        end
        
        function [A, Nmax, FOV, nCir, xCir, yCir, zCir, rCir] = readpacking(root_packing)
            A    = load(fullfile(root_packing,'phantom_APix.txt'));
            Nmax = load(fullfile(root_packing,'phantom_Nmax.txt'));
            NPix = load(fullfile(root_packing,'phantom_NPix.txt'));
            FOV  = load(fullfile(root_packing,'phantom_res.txt'));
            A    = permute(reshape(A,NPix,NPix,NPix),[3,2,1]);
            
            nCir = load(fullfile(root_packing,'phantom_NAx.txt'));
            xCir = load(fullfile(root_packing,'phantom_xCir.txt'));
            yCir = load(fullfile(root_packing,'phantom_yCir.txt'));
            zCir = load(fullfile(root_packing,'phantom_zCir.txt'));
            rCir = load(fullfile(root_packing,'phantom_rCir.txt'));
       end
        
    end
    
end