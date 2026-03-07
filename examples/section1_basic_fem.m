%% section1_basic_fem.m
% Section 1 - Question 1.2: Basic Finite Element Implementation
%
% Functionality:
% - Implements 1D rod element finite element method
% - Supports arbitrary number of elements
% - Three boundary condition methods
% - Comparison and verification with analytical solutions
% - Comparison on the same graph: elimination / penalty / lagrange + exact

clear; clc; close all;

addpath(genpath('../src'));

fprintf('\n');
fprintf('========================================\n');
fprintf('  Section 1 - Question 1.2\n');
fprintf('  Basic Finite Element Implementation\n');
fprintf('========================================\n\n');

params = config_params();

mesh = generate_mesh(params);

%% Assemble the overall system
[K, F] = assemble_global_system(mesh, params);

%% Three boundary condition methods for solving the problem, along with analytical solutions, are compared on the same graph.
bc_methods = {'elimination', 'penalty', 'lagrange'};

fprintf('========================================\n');
fprintf('The three boundary condition methods will be solved sequentially and compared graphically:\n');
fprintf('  - %s\n', bc_methods{:});
fprintf('========================================\n');

% ---- Get node coordinates ----
if isfield(mesh, 'x')
    x = mesh.x(:);
elseif isfield(mesh, 'nodes')
    x = mesh.nodes(:);
elseif isfield(mesh, 'X')
    x = mesh.X(:);
else
    error('The node coordinate field was not found in the mesh. Please check generate_mesh.m');
end

% ---- Analytical solution ----
u_exact = (params.Fd / (params.E * params.S)) * x;

% ---- Solve the three methods sequentially, and save the displacement and post-processing results. ----
U_all = struct();
R_all = struct();

for k = 1:numel(bc_methods)
    bc_method = bc_methods{k};

    fprintf('\n----------------------------------------\n');
    fprintf('Solution method: %s\n', bc_method);
    fprintf('----------------------------------------\n');

    u = apply_boundary_conditions(K, F, mesh, params, bc_method);

    % If the lagrange method returns an augmented unknown (u; lambda), only the first Nn are taken here.
    if numel(u) > params.Nn
        u = u(1:params.Nn);
    end

    results = postprocess(u, mesh, params, K, F);

    U_all.(bc_method) = u(:);
    R_all.(bc_method) = results;

    fprintf('  Maximum displacement: %.6e m\n', max(abs(results.u)));
    fprintf('  L2 relative error: %.6e\n', results.error.u_L2);
    fprintf('  Support reaction force: %.6f N (Theoretical: %.6f N)\n', results.reaction, -params.Fd);
end

fig_compare = figure('Name','Q1.2 - 3 methods vs exact','NumberTitle','off');
hold on; grid on;

plot(x, u_exact, 'LineWidth', 2);                 % Exact
plot(x, U_all.elimination, 'LineWidth', 1.5);     % Elimination
plot(x, U_all.penalty,     'LineWidth', 1.5);     % Penalty
plot(x, U_all.lagrange,    'LineWidth', 1.5);     % Lagrange

xlabel('x');
ylabel('u(x)');
title(sprintf('Displacement: Exact vs 3 BC methods (Ne=%d)', params.Ne));
legend({'Exact','Elimination','Penalty','Lagrange'}, 'Location', 'best');

%% Error plot
fig_error = figure('Name','Q1.2 - pointwise error','NumberTitle','off');
hold on; grid on;

plot(x, abs(U_all.elimination - u_exact), 'LineWidth', 1.5);
plot(x, abs(U_all.penalty     - u_exact), 'LineWidth', 1.5);
plot(x, abs(U_all.lagrange    - u_exact), 'LineWidth', 1.5);

xlabel('x');
ylabel('|u_{num}-u_{exact}|');
title(sprintf('Pointwise error (Ne=%d)', params.Ne));
legend({'Elimination','Penalty','Lagrange'}, 'Location', 'best');

%% Save image and results
if params.save_results
    fig_dir = '../results/section1/figures/';
    if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end

    fn_cmp = fullfile(fig_dir, sprintf('comparison_3methods_exact_Ne%d.png', params.Ne));
    fn_err = fullfile(fig_dir, sprintf('error_3methods_exact_Ne%d.png', params.Ne));
    saveas(fig_compare, fn_cmp);
    saveas(fig_error,  fn_err);
    fprintf('\nImage saved:\n  %s\n  %s\n', fn_cmp, fn_err);

    data_dir = '../results/section1/data/';
    if ~exist(data_dir, 'dir'); mkdir(data_dir); end

    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    mat_file = fullfile(data_dir, sprintf('results_3methods_Ne%d_%s.mat', params.Ne, timestamp));
    save(mat_file, 'U_all', 'R_all', 'u_exact', 'x', 'mesh', 'params', 'bc_methods');
    fprintf('Data saved: %s\n', mat_file);

    report_dir = '../results/section1/reports/';
    if ~exist(report_dir, 'dir'); mkdir(report_dir); end

    for k = 1:numel(bc_methods)
        bc_method = bc_methods{k};
        save_results_to_file(R_all.(bc_method), mesh, params, sprintf('Q1.2_%s', bc_method));
    end
end

fprintf('\n');
fprintf('========================================\n');
fprintf('  Question 1.2 Completed!\n');
fprintf('========================================\n');
fprintf('Number of units: %d\n', params.Ne);
fprintf('Number of nodes: %d\n', params.Nn);

for k = 1:numel(bc_methods)
    bc_method = bc_methods{k};
    res = R_all.(bc_method);
    fprintf('\n[%s]\n', bc_method);
    fprintf('  Maximum displacement: %.6e m\n', max(abs(res.u)));
    fprintf('  L2 relative error: %.6e\n', res.error.u_L2);
    fprintf('  Support reaction force: %.6f N (Theoretical: %.6f N)\n', res.reaction, -params.Fd);
end

fprintf('\n========================================\n\n');
fprintf(' can modify the following parameters and run it again.:\n ');
fprintf('  - params.Ne \n');
fprintf('  - params.plot_results \n');
fprintf('  - params.save_results \n\n');