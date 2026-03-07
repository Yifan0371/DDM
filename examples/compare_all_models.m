% compare_all_methods.m
% Global comparison figure: displacement u(x) curves of all methods in a single plot
%
% Covered methods:
% Section 1
%   Elimination (direct elimination method)
%   Penalty method
%   Lagrange multiplier method
% Section 2.1 Primal
%   Schur direct method (Q2.1)
%   Distributed CG (Q2.3)
%   BDD PCG (Q2.5)
% Section 2.2 Dual
%   FETI direct method (Q2.7)
%   FETI PCG (Q2.8/Q2.9)
% Section 2.3 Mixed
%   Monoscale LaTIn (Q2.10)
%   Multiscale LaTIn (Q2.13)
% Analytical
%   Analytical solution
%
% Usage: run this script inside the examples/ directory

clear; clc; close all;

addpath(genpath('../src'));

fprintf('\n');
fprintf('All methods displacement comparison\n');
fprintf('\n');

params = config_params();
mesh   = generate_mesh(params);
[K, F] = assemble_global_system(mesh, params);
subs   = generate_substructures(mesh, params);

x_mm = mesh.x * 1e3;
Nn   = params.Nn;
x_nodes = linspace(0, params.L, Nn)';

u_exact = (params.Fd / (params.E * params.S)) * mesh.x;

fprintf('Parameters: Ne=%d, N_sub=%d, h=%.3e m\n\n', params.Ne, params.N_sub, params.h);

fprintf('Section 1 solving boundary condition methods\n');

u_elim = apply_boundary_conditions(K, F, mesh, params, 'elimination');
if numel(u_elim) > Nn, u_elim = u_elim(1:Nn); end

u_pen = apply_boundary_conditions(K, F, mesh, params, 'penalty');
if numel(u_pen) > Nn, u_pen = u_pen(1:Nn); end

u_lag = apply_boundary_conditions(K, F, mesh, params, 'lagrange');
if numel(u_lag) > Nn, u_lag = u_lag(1:Nn); end

fprintf('\nSection 2.1 Primal methods\n');

[~, u_schur, ~] = solve_schur_direct(subs, F, params);
[~, u_pcg,   ~] = solve_schur_pcg(subs, F, params);
[~, u_bdd,   ~] = solve_bdd_pcg(subs, F, params);

fprintf('\nSection 2.2 Dual methods\n');

[~, u_dual_direct, ~]          = solve_dual_direct(subs, F, params);
[lambda_feti, ~, feti_info]    = solve_dual_feti(subs, F, params);
[u_feti, ~]                    = recover_rigid_modes(lambda_feti, subs, F, params, feti_info);

fprintf('\nSection 2.3 LaTIn methods\n');

k_opt = params.E * params.S / params.H;
[~, ~, info_mono]  = solve_latin_mono(subs, F, params, k_opt, k_opt);
u_latin_mono       = info_mono.u_global;
[~, ~, info_multi] = solve_latin_multi(subs, F, params);
u_latin_multi      = info_multi.u_global;

ref_norm = norm(u_exact);

err_elim        = abs(u_elim        - u_exact) / ref_norm;
err_pen         = abs(u_pen         - u_exact) / ref_norm;
err_lag         = abs(u_lag         - u_exact) / ref_norm;
err_schur       = abs(u_schur       - u_exact) / ref_norm;
err_pcg         = abs(u_pcg         - u_exact) / ref_norm;
err_bdd         = abs(u_bdd         - u_exact) / ref_norm;
err_dual_direct = abs(u_dual_direct - u_exact) / ref_norm;
err_feti        = abs(u_feti        - u_exact) / ref_norm;
err_mono        = abs(u_latin_mono  - u_exact) / ref_norm;
err_multi       = abs(u_latin_multi - u_exact) / ref_norm;

floor_val = 1e-17;
fix_zeros = @(e) max(e, floor_val);

err_elim        = fix_zeros(err_elim);
err_pen         = fix_zeros(err_pen);
err_lag         = fix_zeros(err_lag);
err_schur       = fix_zeros(err_schur);
err_pcg         = fix_zeros(err_pcg);
err_bdd         = fix_zeros(err_bdd);
err_dual_direct = fix_zeros(err_dual_direct);
err_feti        = fix_zeros(err_feti);
err_mono        = fix_zeros(err_mono);
err_multi       = fix_zeros(err_multi);

fprintf('\nGenerating relative error scatter plot\n');

methods = { ...
        'Elimination',      err_elim,        'm',  '+',  8; ...
        'Dual Direct',      err_dual_direct, 'c',  'x',  8; ...
        'Primal CG',        err_pcg,         'b',  '*',  7; ...
        'Primal BDD',       err_bdd,         [1 0.6 0], 's', 6; ...
        'Dual FETI',        err_feti,        [0.8 0.7 0], 's', 6; ...
        };

n_methods = size(methods, 1);

fig2 = figure('Name', 'Relative errors for different methods', ...
        'Position', [100 100 900 600], ...
        'Color', 'white');

hold on;
rng(42);

handles = zeros(n_methods, 1);
jitter_width = 0.25;

for m = 1:n_methods
        name   = methods{m, 1};
        errors = methods{m, 2};
        col    = methods{m, 3};
        mkr    = methods{m, 4};
        msz    = methods{m, 5};
        
        x_jitter = m + (rand(Nn, 1) - 0.5) * jitter_width;
        
        handles(m) = semilogy(x_jitter, errors, ...
                mkr, ...
                'Color', col, ...
                'MarkerSize', msz, ...
                'LineStyle', 'none', ...
                'DisplayName', name);
end

hold off;

set(gca, 'YScale', 'log');
set(gca, 'XTick', 1:n_methods);
set(gca, 'XTickLabel', methods(:, 1));
set(gca, 'FontSize', 12);
set(gca, 'XLim', [0.5, n_methods + 0.5]);

all_errs = [err_elim; err_dual_direct; err_pcg; err_bdd; err_feti];
y_min = 10^(floor(log10(min(all_errs))) - 0.5);
y_max = 10^(ceil(log10(max(all_errs)))  + 0.5);
set(gca, 'YLim', [y_min, y_max]);

grid on; box on;
xlabel('methods', 'FontSize', 14);
ylabel('relative errors', 'FontSize', 14);
title('relative errors for different methods', 'FontSize', 14);

legend(handles, methods(:,1), 'Location', 'northeast', 'FontSize', 11);

out_dir = '../results/comparison/';
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

fname2 = fullfile(out_dir, sprintf('relative_errors_scatter_Nsub%d_Ne%d.png', ...
        params.N_sub, params.Ne));
saveas(fig2, fname2);
fprintf('Scatter plot saved: %s\n', fname2);

fprintf('\nGenerating displacement comparison figure\n');

fig1 = figure('Name', 'All DDM Methods Displacement Comparison', ...
        'Position', [50 50 1200 650], ...
        'Color', 'white');
hold on;

h_exact    = plot(x_mm, u_exact*1e6,       'k-',   'LineWidth', 3.0, 'DisplayName', 'Analytical (exact)');
h_elim     = plot(x_mm, u_elim*1e6,        'r-o',  'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerIndices', 1:3:Nn, 'DisplayName', 'S1 Elimination');
h_pen      = plot(x_mm, u_pen*1e6,         'Color', [0.85 0.33 0.10], 'LineStyle', '-',  'Marker', 's', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerIndices', 2:3:Nn, 'DisplayName', 'S1 Penalty');
h_lag      = plot(x_mm, u_lag*1e6,         'Color', [0.93 0.69 0.13], 'LineStyle', '-',  'Marker', '^', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerIndices', 3:3:Nn, 'DisplayName', 'S1 Lagrange');
h_schur    = plot(x_mm, u_schur*1e6,       'b--d', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerIndices', 1:4:Nn, 'DisplayName', 'S2.1 Schur Direct Q2.1');
h_dcg      = plot(x_mm, u_pcg*1e6,         'Color', [0.30 0.75 0.93], 'LineStyle', '--', 'Marker', 'p', 'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerIndices', 2:4:Nn, 'DisplayName', 'S2.1 Distributed CG Q2.3');
h_bdd      = plot(x_mm, u_bdd*1e6,         'Color', [0.00 0.45 0.74], 'LineStyle', '--', 'Marker', 'h', 'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerIndices', 3:4:Nn, 'DisplayName', 'S2.1 BDD PCG Q2.5');
h_ddirect  = plot(x_mm, u_dual_direct*1e6, 'Color', [0.47 0.67 0.19], 'LineStyle', '-.', 'Marker', 'd', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerIndices', 1:4:Nn, 'DisplayName', 'S2.2 Dual Direct Q2.7');
h_feti     = plot(x_mm, u_feti*1e6,        'Color', [0.13 0.55 0.13], 'LineStyle', '-.', 'Marker', 'v', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerIndices', 3:4:Nn, 'DisplayName', 'S2.2 FETI PCG Q2.8 Q2.9');
h_mono     = plot(x_mm, u_latin_mono*1e6,  'Color', [0.49 0.18 0.56], 'LineStyle', ':',  'Marker', '<', 'LineWidth', 2.0, 'MarkerSize', 5, 'MarkerIndices', 2:4:Nn, 'DisplayName', sprintf('S2.3 LaTIn Mono Q2.10 %d iters', info_mono.n_iter));
h_multi    = plot(x_mm, u_latin_multi*1e6, 'Color', [0.75 0.00 0.75], 'LineStyle', ':',  'Marker', '>', 'LineWidth', 2.0, 'MarkerSize', 5, 'MarkerIndices', 4:4:Nn, 'DisplayName', sprintf('S2.3 LaTIn Multi Q2.13 %d iters', info_multi.n_iter));

hold off;

grid on; box on;
xlabel('Position x mm', 'FontSize', 14);
ylabel('Displacement u(x) um', 'FontSize', 14);
title(sprintf('All DDM Methods Displacement Comparison N_sub=%d N_e=%d h/H=1/%d', ...
        params.N_sub, params.Ne, params.Ne_sub), 'FontSize', 15, 'FontWeight', 'bold');
legend([h_exact,h_elim,h_pen,h_lag,h_schur,h_dcg,h_bdd,h_ddirect,h_feti,h_mono,h_multi], ...
        'Location', 'best', 'FontSize', 10, 'NumColumns', 2);
set(gca, 'FontSize', 12);

yl = ylim;
for s = 1:params.N_sub - 1
        x_b = (s * params.Ne_sub) * params.h * 1e3;
        line([x_b x_b], yl, 'Color', [0.7 0.7 0.7], 'LineStyle', '--', ...
                'LineWidth', 0.8, 'HandleVisibility', 'off');
end
ylim(yl);

fname1 = fullfile(out_dir, sprintf('all_methods_displacement_Nsub%d_Ne%d.png', ...
        params.N_sub, params.Ne));
saveas(fig1, fname1);
fprintf('Displacement figure saved %s\n', fname1);

fprintf('\nL2 relative error summary\n');
fprintf('%-35s  %12s\n', 'Method', 'L2 error');

methods_summary = { ...
        'Analytical reference',       0; ...
        'S1 Elimination',              norm(u_elim        -u_exact)/ref_norm; ...
        'S1 Penalty',                  norm(u_pen         -u_exact)/ref_norm; ...
        'S1 Lagrange',                 norm(u_lag         -u_exact)/ref_norm; ...
        'S2.1 Schur Direct Q2.1',    norm(u_schur       -u_exact)/ref_norm; ...
        'S2.1 Distributed CG Q2.3', norm(u_pcg         -u_exact)/ref_norm; ...
        'S2.1 BDD PCG Q2.5',         norm(u_bdd         -u_exact)/ref_norm; ...
        'S2.2 Dual Direct Q2.7',     norm(u_dual_direct -u_exact)/ref_norm; ...
        'S2.2 FETI Q2.8 Q2.9',      norm(u_feti        -u_exact)/ref_norm; ...
        'S2.3 LaTIn Mono Q2.10',    norm(u_latin_mono  -u_exact)/ref_norm; ...
        'S2.3 LaTIn Multi Q2.13',   norm(u_latin_multi -u_exact)/ref_norm; ...
        };

for k = 1:size(methods_summary, 1)
        fprintf('  %-35s  %12.4e\n', methods_summary{k,1}, methods_summary{k,2});
end

fprintf('\nTwo figures generated\n');
fprintf('1 %s\n', fname1);
fprintf('2 %s\n', fname2);
fprintf('Substructure boundaries N_sub=%d\n\n', params.N_sub);