function options = edgecut_options  %#ok
%EDGECUT_OPTIONS create a struct of default options for edge cuts.
%   options = edgecutoptions ; returns an options struct with defaults set.
%   If modifications to the default options are needed to modify how EDGECUT
%   functions, call EDGECUT_OPTIONS and modify the struct as needed.
%
%   Example:
%       options = edgecut_options ;
%       options.targetSplit = 0.3;
%       options.matchingStrategy = 0;   % Random matching
%       Prob = ssget('DNVS/troll'); A = Prob.A;
%       part = edgecut(A, O);
%       sum(part)/length(part)    % 0.3000
%
%   See also EDGECUT.

%   Copyright (c) 2018, N. Yeralan, S. Kolodziej, T. Davis, W. Hager

error ('edgecut_options mexFunction not found') ;
