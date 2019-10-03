function SJget_install
% SJget_install - Install SJget toolbox.
%
% Copyright 2008, Leslie Foster
% Created Nikolay Botev March 2008
% 5/2/2008:  modified by L. Foster

    % Compute path to SJget toolbox
    pkgdir = which(mfilename);
    i = find(pkgdir == filesep);
    pkgdir = pkgdir(1:i(end)-1);

    version_str = version;
    idot = find(version_str == '.');
    version_num = str2num( version_str(1:idot(1)+1) );
    if version_num >= 7.2 
       % Check if already installed
       % textscan(path,...) fails in early versions of Matlab 7
       %   so skip the check for these versions
       pth = textscan(path, '%s', 'delimiter', pathsep);
       pth = pth{1};
       if ~isempty(find(strcmpi(pth, pkgdir), 1))
           disp('SJget toolbox is already installed.');
           return;
       end
    end

    % Add SJget package to MATLAB path
    addpath(pkgdir);    
 
    % Download SJ_Index.mat
    matdir = [pkgdir filesep 'mat'];
    if ~exist(matdir, 'dir')
        mkdir(matdir);
    end
    SJget;

    % Done
    disp(' ')
    disp(' ');
    disp('Your path has been modified by:');
    disp(['addpath ',pkgdir]);    

    disp(' ');
    disp('SJget toolbox successfully installed.');
    disp('Remember to save your path using savepath or pathtool!');
end
