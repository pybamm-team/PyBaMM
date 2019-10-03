function comparisonData = compare(trials, percent_to_keep, plot_outliers, use_weights)
    if (nargin < 1 || trials < 1)
        trials = 5;
    end
    if (nargin < 2 || percent_to_keep <= 0)
        percent_to_keep = 60;
    end
    if (nargin < 3)
        plot_outliers = 0;
    end
    if (nargin < 4)
        use_weights = 0;
    end
    
    index = ssget;
    j = 1;
    
    comparisonData = struct('avg_mongoose_times', [], ...
                            'avg_metis_times', [], ...
                            'rel_mongoose_times',  [], ...
                            'avg_mongoose_imbalance', [], ...
                            'avg_metis_imbalance', [], ...
                            'avg_mongoose_cut_weight', [], ...
                            'avg_mongoose_cut_size', [], ...
                            'avg_metis_cut_weight', [], ...
                            'avg_metis_cut_size', [], ...
                            'rel_mongoose_cut_size', [], ...
                            'problem_id', [], ...
                            'problem_name', [], ...
                            'problem_kind', [], ...
                            'problem_nnz', [], ...
                            'problem_n', []);
    for i = 1:length(index.nrows)
        if (index.isReal(i))
            
            % For comparing graph performance
            % if (~index.isGraph(i))
            %     continue;
            % end
            % use_weights = 1;
            
            Prob = ssget(i);
            A = Prob.A;
            
            % If matrix is unsymmetric, form the augmented system
            if (index.numerical_symmetry(i) < 1)
                [m_rows, n_cols] = size(A);
                A = [sparse(m_rows,m_rows) A; A' sparse(n_cols,n_cols)];
            end

            % Make matrix binary - matrix values are not necessarily edge weights
            %A = sign(abs(A));
            A = abs(A); % Edit for graph comparison

            % Sanitize the matrix (remove diagonal, make symmetric)
            A = sanitize(A, ~use_weights);
            
            % If the sanitization removed all vertices, skip this matrix
            if nnz(A) < 2
                continue
            end
            
            fprintf('Computing separator for %d: %s\n', i, Prob.name);
            
            [~, n_cols] = size(A);
            comparisonData(j).problem_id = Prob.id;
            comparisonData(j).problem_name = Prob.name;
            comparisonData(j).problem_kind = Prob.kind;
            comparisonData(j).problem_nnz = nnz(A);
            comparisonData(j).problem_n = n_cols;
            
            % Run Mongoose with default options to partition the graph.
            O = edgecut_options();
            O.randomSeed = 123456789;
            for k = 1:trials
                tic;
                partition = edgecut(A,O);
                t = toc;
                fprintf('Mongoose: %0.2f\n', t);
                mongoose_times(j,k) = t;
                part_A = find(partition);
                part_B = find(1-partition);
                perm = [part_A part_B];
                p = length(part_A);
                A_perm = A(perm, perm);
                mongoose_cut_weight(j,k) = sum(sum(A_perm((p+1):n_cols, 1:p)));
                mongoose_cut_size(j,k) = sum(sum(sign(A_perm((p+1):n_cols, 1:p))));
                mongoose_imbalance(j,k) = abs(0.5-(length(part_A)/(length(part_A) + length(part_B))));
            end
            
            % Run METIS to partition the graph.
            for k = 1:trials
                tic;
                [part_A,part_B] = metispart(A, 0, 123456789);
                t = toc;
                fprintf('METIS:    %0.2f\n', t);
                metis_times(j,k) = t;
                p = length(part_A);
                perm = [part_A part_B];
                A_perm = A(perm, perm);
                metis_cut_weight(j,k) = sum(sum(A_perm((p+1):n_cols, 1:p)));
                metis_cut_size(j,k) = sum(sum(sign(A_perm((p+1):n_cols, 1:p))));
                metis_imbalance(j,k) = abs(0.5-(length(part_A)/(length(part_A) + length(part_B))));
            end
            j = j + 1;
        end
    end
    
    n = length(mongoose_times);
    
    for i = 1:n
        % Compute trimmed means - trim lowest and highest 20%
        comparisonData(i).avg_mongoose_times = trimmean(mongoose_times(i,:), 100-percent_to_keep);
        comparisonData(i).avg_mongoose_cut_weight = trimmean(mongoose_cut_weight(i,:), 100-percent_to_keep);
        comparisonData(i).avg_mongoose_cut_size = trimmean(mongoose_cut_size(i,:), 100-percent_to_keep);
        comparisonData(i).avg_mongoose_imbalance = trimmean(mongoose_imbalance(i,:), 100-percent_to_keep);
        
        comparisonData(i).avg_metis_times = trimmean(metis_times(i,:), 100-percent_to_keep);
        comparisonData(i).avg_metis_cut_weight = trimmean(metis_cut_weight(i,:), 100-percent_to_keep);
        comparisonData(i).avg_metis_cut_size = trimmean(metis_cut_size(i,:), 100-percent_to_keep);
        comparisonData(i).avg_metis_imbalance = trimmean(metis_imbalance(i,:), 100-percent_to_keep);
        
        % Compute times relative to METIS
        comparisonData(i).rel_mongoose_times = (comparisonData(i).avg_mongoose_times / comparisonData(i).avg_metis_times);

        % Compute cut weight relative to METIS
        comparisonData(i).rel_mongoose_cut_weight = (comparisonData(i).avg_mongoose_cut_weight / comparisonData(i).avg_metis_cut_weight);

        % Compute cut size relative to METIS
        comparisonData(i).rel_mongoose_cut_size = (comparisonData(i).avg_mongoose_cut_size / comparisonData(i).avg_metis_cut_size);
        
        % Check for outliers
        prob_id = comparisonData(i).problem_id;
        outlier = 0;
        
        if (comparisonData(i).rel_mongoose_times > 2)
            disp(['Outlier! Mongoose time significantly worse. ID: ', num2str(prob_id)]);
            outlier = 1;
            comparisonData(i).outlier.time = 1;
        end
        if (comparisonData(i).rel_mongoose_times < 0.5)
            disp(['Outlier! METIS time significantly worse. ID: ', num2str(prob_id)]);
            outlier = 1;
            comparisonData(i).outlier.time = -1;
        end
        
        if (comparisonData(i).rel_mongoose_cut_size > 2)
            disp(['Outlier! Mongoose cut size significantly worse. ID: ', num2str(prob_id)]);
            outlier = 1;
            comparisonData(i).outlier.cut_size = 1;
        end
        if (comparisonData(i).rel_mongoose_cut_size < 0.5)
            disp(['Outlier! METIS cut size significantly worse. ID: ', num2str(prob_id)]);
            outlier = 1;
            comparisonData(i).outlier.cut_size = -1;
        end
        
        if (comparisonData(i).avg_mongoose_imbalance > 2*comparisonData(i).avg_metis_imbalance)
            disp(['Outlier! Mongoose imbalance significantly worse. ID: ', num2str(prob_id)]);
            comparisonData(i).outlier.imbalance = 1;
            outlier = 1;
        end
        if (comparisonData(i).avg_metis_imbalance > 2*comparisonData(i).avg_mongoose_imbalance)
            disp(['Outlier! METIS imbalance significantly worse. ID: ', num2str(prob_id)]);
            comparisonData(i).outlier.imbalance = -1;
            outlier = 1;
        end
        
        if (outlier && plot_outliers)
            plotGraphs(prob_id);
        end
    end
    
    % Sort metrics
    sorted_rel_mongoose_times = sort([comparisonData.rel_mongoose_times]);
    sorted_rel_mongoose_cut_weight = sort([comparisonData.rel_mongoose_cut_weight]);
    sorted_rel_mongoose_cut_size = sort([comparisonData.rel_mongoose_cut_size]);
    sorted_avg_mongoose_imbalance = sort([comparisonData.avg_mongoose_imbalance]);
    sorted_avg_metis_imbalance = sort([comparisonData.avg_metis_imbalance]);
    
    % Get the Git commit hash for labeling purposes
    [error, commit] = system('git rev-parse --short HEAD');
    git_found = ~error;
    commit = strtrim(commit);
    
    %%%%% Plot performance profiles %%%%%
    
    % Plot timing profiles
    figure;
    semilogy(1:n, sorted_rel_mongoose_times, 'Color', 'b');
    hold on;
    semilogy(1:n, ones(1,n), 'Color','r');
    axis([1 n min(sorted_rel_mongoose_times) max(sorted_rel_mongoose_times)]);
    xlabel('Matrix');
    ylabel('Wall Time Relative to METIS');
    hold off;
    
    plt = Plot();
    plt.LineStyle = {'-', '--'};
    plt.Legend = {'Mongoose', 'METIS'};
    plt.LegendLoc = 'SouthEast';
    plt.BoxDim = [6, 5];
    
    filename = ['Timing' date];
    if(git_found)
        title(['Timing Profile - Commit ' commit]);
        filename = ['Timing-' commit];
    end
    
    plt.export([filename '.png']);

    % Plot separator weight profiles
    figure;
    semilogy(1:n, sorted_rel_mongoose_cut_weight, 'Color', 'b');
    hold on;
    semilogy(1:n, ones(1,n), 'Color','r');
    axis([1 n min(sorted_rel_mongoose_cut_weight) max(sorted_rel_mongoose_cut_weight)]);
    xlabel('Matrix');
    ylabel('Cut Weight Relative to METIS');
    hold off;

    plt = Plot();
    plt.LineStyle = {'-', '--'};
    plt.Legend = {'Mongoose', 'METIS'};
    plt.LegendLoc = 'SouthEast';
    plt.BoxDim = [6, 5];

    filename = ['SeparatorWeight' date];
    if(git_found)
        title(['Separator Weight Profile - Commit ' commit]);
        filename = ['SeparatorWeight-' commit];
    end
    plt.export([filename '.png']);

    % Plot separator size profiles
    figure;
    semilogy(1:n, sorted_rel_mongoose_cut_size, 'Color', 'b');
    hold on;
    semilogy(1:n, ones(1,n), 'Color','r');
    axis([1 n min(sorted_rel_mongoose_cut_size) max(sorted_rel_mongoose_cut_size)]);
    xlabel('Matrix');
    ylabel('Cut Size Relative to METIS');
    hold off;

    plt = Plot();
    plt.LineStyle = {'-', '--'};
    plt.Legend = {'Mongoose', 'METIS'};
    plt.LegendLoc = 'SouthEast';
    plt.BoxDim = [6, 5];

    filename = ['SeparatorSize' date];
    if(git_found)
        title(['Separator Size Profile - Commit ' commit]);
        filename = ['SeparatorSize-' commit];
    end
    plt.export([filename '.png']);
    
    % Plot imbalance profiles
    figure;
    plot(1:n, sorted_avg_mongoose_imbalance, 'Color', 'b');
    hold on;
    plot(1:n, sorted_avg_metis_imbalance, 'Color','r');
    axis([1 n 0 max([sorted_avg_metis_imbalance sorted_avg_mongoose_imbalance])]);
    xlabel('Matrix');
    ylabel('Imbalance');
    hold off;
    
    plt = Plot();
    plt.LineStyle = {'-', '--'};
    plt.Legend = {'Mongoose', 'METIS'};
    plt.LegendLoc = 'NorthWest';
    plt.BoxDim = [6, 5];
    
    filename = ['Imbalance' date];
    if(git_found)
        title(['Imbalance Profile - Commit ' commit]);
        filename = ['Imbalance-' commit];
    end
    
    plt.export([filename '.png']);
    
    %% Plot only big matrices to compare
    
    
    
    %% Write data to file for future comparisons
    if(git_found)
        writetable(struct2table(comparisonData), [commit '.txt']);
    end
end

function plotGraphs(prob_id)
    index = ssget;
    Prob = ssget(prob_id);
    A = Prob.A;
    if (index.numerical_symmetry(prob_id) < 1)
        [m_rows, n_cols] = size(A);
        A = [sparse(m_rows,m_rows) A; A' sparse(n_cols,n_cols)];
    end
    A = mongoose_sanitizeMatrix(A);
    
    % Compute partitioning using Mongoose
    partition = mongoose_computeEdgeSeparator(A);
%     part_A = find(partition);
%     part_B = find(1-partition);
%     perm = [part_A part_B];
%     p = length(partition);
%     A_perm = A(perm, perm);
%     subplot(1,2,1);
%     hold on;
%     spy(A);
%     subplot(1,2,2);
%     spy(A_perm);
%     hold off;
    mongoose_separator_plot(A, partition, 1-partition, ['mongoose_' num2str(prob_id)]);
    
    % Compute partitioning using METIS
    [perm, ~] = metispart(A, 0, 123456789);
    [m, ~] = size(A);
    partition = zeros(m,1);
    for j = 1:m
        partition(j,1) = sum(sign(find(j == perm)));
    end
    mongoose_separator_plot(A, partition, 1-partition, ['metis_' num2str(prob_id)]);
end

function plotMatrix(prob_id)
    index = ssget;
    Prob = ssget(prob_id);
    A = Prob.A;
    if (index.numerical_symmetry(prob_id) < 1)
        [m_rows, n_cols] = size(A);
        A = [sparse(m_rows,m_rows) A; A' sparse(n_cols,n_cols)];
    end
    A = mongoose_sanitizeMatrix(A);
    
    % Compute partitioning using Mongoose
    partition = mongoose_computeEdgeSeparator(A);
    part_A = find(partition);
    part_B = find(1-partition);
    perm = [part_A part_B];
    A_perm = A(perm, perm);
    subplot(1,2,1);
    hold on;
    spy(A);
    subplot(1,2,2);
    spy(A_perm);
    hold off;
end
