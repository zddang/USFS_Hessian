function PlotExampleUSFSHessianDataset
    % PlotExampleUSFSHessianDataset
    % This script creates plots and figures related to the results obtained 
    % from running your example script (e.g., exampleUSFSHessianDatasetBayesOptTOX171.m).
    %
    % Prerequisite:
    %   - The main example script should have run first, so that 'obj_values', 
    %     'W', 'feature_idx', 'fea', and 'gnd' are in the workspace.
    %
    % Adjust paths and variable names if needed.

    % Check if required variables are in workspace
    vars_needed = {'obj_values', 'W', 'feature_idx', 'fea', 'gnd'};
    for v = vars_needed
        if ~evalin('base', sprintf('exist(''%s'', ''var'')', v{1}))
            error('Variable %s not found in the workspace. Please run the main example script first.', v{1});
        end
    end

    % Load required variables from base workspace
    obj_values = evalin('base', 'obj_values');
    W = evalin('base', 'W');
    feature_idx = evalin('base', 'feature_idx');
    fea = evalin('base', 'fea');
    gnd = evalin('base', 'gnd');
    
    %% 1. Plot Objective Values Over Iterations
    figure('Name','Objective Function Convergence','NumberTitle','off');
    plot(obj_values, 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Objective Value');
    title('Convergence of the Objective Function');
    grid on;

    %% 2. Plot Feature Importance
    feature_importance = sum(W.^2, 2);
    figure('Name','Feature Importance','NumberTitle','off');
    bar(feature_importance);
    xlabel('Feature Index');
    ylabel('Sum of Squared Weights');
    title('Feature Importance by W Norm');
    grid on;
    hold on;
    % Mark selected features with a red star
    plot(feature_idx, feature_importance(feature_idx), 'r*', 'MarkerSize', 5);
    hold off;

    %% 3. Clustering Performance Comparison
    % Check if litekmeans and evaluation functions are available
    if exist('litekmeans','file') && exist('NMI_sqrt_lei','file') && exist('ACC_Lei','file')
        nClass = length(unique(gnd));
        
        % Clustering with all features
        rand('twister',5748);
        label_all = litekmeans(fea, nClass, 'Replicates', 10);
        NMI_all = NMI_sqrt_lei(gnd,label_all);
        ACC_all = ACC_Lei(gnd,label_all);

        % Clustering with selected features
        X_selected = fea(:, feature_idx);
        rand('twister',5748);
        label_selected = litekmeans(X_selected, nClass, 'Replicates', 10);
        NMI_selected = NMI_sqrt_lei(gnd,label_selected);
        ACC_selected = ACC_Lei(gnd,label_selected);

        % Display the results
        disp('Clustering performance with all features:');
        disp(['NMI: ', num2str(NMI_all), '   ACC: ', num2str(ACC_all)]);
        disp('Clustering performance with selected features:');
        disp(['NMI: ', num2str(NMI_selected), '   ACC: ', num2str(ACC_selected)]);

        % Plot comparison of NMI
        figure('Name','Clustering NMI Comparison','NumberTitle','off');
        bar([NMI_all, NMI_selected]);
        set(gca, 'XTickLabel', {'All Features','Selected Features'});
        ylabel('NMI (sqrt)');
        title('Clustering Performance (NMI)');
        grid on;
        
        % Plot comparison of ACC
        figure('Name','Clustering ACC Comparison','NumberTitle','off');
        bar([ACC_all, ACC_selected]);
        set(gca, 'XTickLabel', {'All Features','Selected Features'});
        ylabel('Accuracy');
        title('Clustering Performance (ACC)');
        grid on;
        
    else
        disp('Clustering comparison skipped because required functions are not available.');
    end

    %% Display selected features
    fprintf('Selected features indices:\n');
    disp(feature_idx');
end