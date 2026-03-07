%% section2_3_mixed.m
% Section 2.3: Mixed Approach - LaTIn
%
% Q2.10: Monoscale LaTIn, solve and verify
% Q2.11: Optimize search directions k+, k-
% Q2.12: Numerical scalability study
% Q2.13: Multiscale LaTIn (coarse problem) - degenerate 1D case
% Q2.14: Scalability of multiscale approach

%% Init
clear; clc; close all;
addpath(genpath('../src'));

fprintf('\n========================================\n');
fprintf('  Section 2.3: Mixed LaTIn Approach\n');
fprintf('========================================\n\n');

params = config_params();
mesh   = generate_mesh(params);
[K, F] = assemble_global_system(mesh, params);
subs   = generate_substructures(mesh, params);

u_ref = apply_boundary_conditions(K, F, mesh, params, 'elimination');


%  Q2.10: Monoscale LaTIn - direct solve and verify
fprintf('\n--- Q2.10: Monoscale LaTIn ---\n');

k_init = params.E * params.S / params.H;  % theoretical optimal
[u_sub, lam_sub, info_210] = solve_latin_mono(subs, F, params, k_init, k_init);

fprintf('  Converged: %d iter | L2 err = %.4e\n', info_210.n_iter, info_210.error_L2);

if params.plot_results
    figure('Name','Q2.10 - LaTIn solution','Position',[100 100 900 400]);

    subplot(1,2,1);
    x_plt = linspace(0, params.L, params.Nn)' * 1e3;
    plot(x_plt, info_210.u_global*1e6, 'b-o', 'LineWidth',2,'MarkerSize',5,'DisplayName','LaTIn');
    hold on;
    plot(x_plt, info_210.u_exact*1e6,  'r--','LineWidth',2,'DisplayName','Exact');
    plot(x_plt, u_ref*1e6, 'k:', 'LineWidth',1.5,'DisplayName','Global FEM');
    grid on; box on;
    xlabel('x [mm]'); ylabel('u [\mum]');
    title(sprintf('Q2.10: Displacement (N_{sub}=%d, err=%.1e)',params.N_sub,info_210.error_L2));
    legend('Location','best','FontSize',10);

    subplot(1,2,2);
    semilogy(info_210.eta_history, 'b-o', 'LineWidth',2,'MarkerSize',5);
    yline(params.tol,'r--','LineWidth',2);
    grid on; box on;
    xlabel('Iteration'); ylabel('\eta');
    title('Convergence history');

    if params.save_results
        out = '../results/section2_3/figures/';
        if ~exist(out,'dir'); mkdir(out); end
        saveas(gcf, fullfile(out,'Q2_10_latin_solution.png'));
    end
end

%  Q2.11: Optimize search directions
fprintf('\n--- Q2.11: Search direction optimization ---\n');
[k_opt, k_minus_opt, info_211] = optimize_search_directions(params);

%  Q2.12: Numerical scalability
fprintf('\n--- Q2.12: Scalability study ---\n');

N_sub_list  = [2, 4, 8, 16, 32];
Ne_sub_fixed = max(params.Ne_sub, 4);
n_iters_scale = zeros(length(N_sub_list), 1);

fprintf('  %6s  %10s  %10s\n', 'N_sub', 'n_iter', 'L2 err');

for idx = 1:length(N_sub_list)
    Ns = N_sub_list(idx);
    p_tmp        = params;
    p_tmp.N_sub  = Ns;
    p_tmp.Ne_sub = Ne_sub_fixed;
    p_tmp.Ne     = Ns * Ne_sub_fixed;
    p_tmp.h      = p_tmp.L / p_tmp.Ne;
    p_tmp.Nn     = p_tmp.Ne + 1;
    p_tmp.H      = Ne_sub_fixed * p_tmp.h;
    p_tmp.plot_results = false;

    k_s = p_tmp.E * p_tmp.S / p_tmp.H;  % optimal k for this config

    mesh_tmp  = generate_mesh(p_tmp);
    [~, F_tmp] = assemble_global_system(mesh_tmp, p_tmp);
    subs_tmp  = generate_substructures(mesh_tmp, p_tmp);

    [~, ~, inf_tmp] = solve_latin_mono(subs_tmp, F_tmp, p_tmp, k_s, k_s);
    n_iters_scale(idx) = inf_tmp.n_iter;

    fprintf('  %6d  %10d  %10.3e\n', Ns, inf_tmp.n_iter, inf_tmp.error_L2);
end

if params.plot_results
    figure('Name','Q2.12 - LaTIn scalability','Position',[200 200 700 400]);
    plot(N_sub_list, n_iters_scale, 'b-o', 'LineWidth',2,'MarkerSize',8,'DisplayName','LaTIn (k=ES/H)');
    hold on;
    plot(N_sub_list, ones(size(N_sub_list))*n_iters_scale(1), 'g--', 'LineWidth',2,'DisplayName','Ideal (const)');
    grid on; box on;
    xlabel('N_{sub} (fixed h/H = 1/N_{e,sub})','FontSize',12);
    ylabel('Iterations','FontSize',12);
    title(sprintf('Q2.12: Monoscale LaTIn scalability  (N_{e,sub}=%d)',Ne_sub_fixed),'FontSize',12);
    legend('Location','best','FontSize',11);
    xticks(N_sub_list);

    if params.save_results
        out = '../results/section2_3/figures/';
        saveas(gcf, fullfile(out,'Q2_12_scalability.png'));
    end

    fprintf('\n  Monoscale LaTIn: iterations ~constant = SCALABLE (with k=ES/H)\n');
end

%  Q2.13: Multiscale LaTIn - coarse/macro problem

fprintf('\n--- Q2.13: Multiscale LaTIn ---\n');
[u_sub_ms, lam_sub_ms, info_213] = solve_latin_multi(subs, F, params);

%  Q2.14: Multiscale scalability
fprintf('\n--- Q2.14: Multiscale scalability ---\n');

n_iters_ms = zeros(length(N_sub_list), 1);
fprintf('  %6s  %10s\n', 'N_sub', 'n_iter');

for idx = 1:length(N_sub_list)
    Ns = N_sub_list(idx);
    p_tmp        = params;
    p_tmp.N_sub  = Ns;
    p_tmp.Ne_sub = Ne_sub_fixed;
    p_tmp.Ne     = Ns * Ne_sub_fixed;
    p_tmp.h      = p_tmp.L / p_tmp.Ne;
    p_tmp.Nn     = p_tmp.Ne + 1;
    p_tmp.H      = Ne_sub_fixed * p_tmp.h;
    p_tmp.plot_results = false;

    mesh_tmp   = generate_mesh(p_tmp);
    [~, F_tmp] = assemble_global_system(mesh_tmp, p_tmp);
    subs_tmp   = generate_substructures(mesh_tmp, p_tmp);

    [~, ~, inf_tmp] = solve_latin_multi(subs_tmp, F_tmp, p_tmp);
    n_iters_ms(idx) = inf_tmp.n_iter;
    fprintf('  %6d  %10d\n', Ns, inf_tmp.n_iter);
end

if params.plot_results
    figure('Name','Q2.14 - Multiscale LaTIn scalability','Position',[300 200 900 400]);

    subplot(1,2,1);
    plot(N_sub_list, n_iters_scale, 'b-o','LineWidth',2,'MarkerSize',8,'DisplayName','Monoscale LaTIn');
    hold on;
    plot(N_sub_list, n_iters_ms, 'r-s','LineWidth',2,'MarkerSize',8,'DisplayName','Multiscale LaTIn');
    plot(N_sub_list, ones(size(N_sub_list)), 'g--','LineWidth',2,'DisplayName','Ideal (1 iter)');
    grid on; box on;
    xlabel('N_{sub}','FontSize',12); ylabel('Iterations','FontSize',12);
    title('Scalability comparison','FontSize',12);
    legend('Location','best','FontSize',10);
    xticks(N_sub_list);

    subplot(1,2,2);
    semilogy(info_213.eta_history, 'r-o','LineWidth',2,'MarkerSize',5);
    yline(params.tol,'k--','LineWidth',1.5);
    grid on; box on;
    xlabel('Iteration'); ylabel('\eta');
    title(sprintf('Multiscale LaTIn convergence  (N_{sub}=%d)',params.N_sub));

    sgtitle('Q2.14: Multiscale LaTIn scalability','FontSize',13);

    if params.save_results
        out = '../results/section2_3/figures/';
        saveas(gcf, fullfile(out,'Q2_14_multiscale_scalability.png'));
    end
end

fprintf('\n========================================\n');
fprintf('  Section 2.3 complete!\n');
fprintf('========================================\n');