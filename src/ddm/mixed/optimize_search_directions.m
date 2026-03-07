function [k_plus_opt, k_minus_opt, info] = optimize_search_directions(params)
% OPTIMIZE_SEARCH_DIRECTIONS - Question 2.11
%
% Numerically determine optimal k+, k- for fastest LaTIn convergence.
%
% Inputs:
%   params - parameter struct (from config_params)
%
% Outputs:
%   k_plus_opt  - optimal k+ [N/m]
%   k_minus_opt - optimal k- [N/m]
%   info        - struct with sweep results


    fprintf('------------------------------------\n');
    fprintf('Q2.11: Optimize LaTIn search directions\n');
    fprintf('------------------------------------\n');

    mesh   = generate_mesh(params);
    [~, F] = assemble_global_system(mesh, params);
    subs   = generate_substructures(mesh, params);

    %% Theoretical optimal: k = E*S/H = Schur complement value
    k_theory = params.E * params.S / params.H;
    fprintf('  Theoretical k = E*S/H = %.4e N/m\n', k_theory);
    fprintf('  (H = %.3e m, E*S = %.3e N)\n', params.H, params.E*params.S);

    %% 1D sweep: k+ = k- = k  (symmetric search directions)
    n_sweep  = 25;
    k_values = logspace(log10(k_theory/50), log10(k_theory*50), n_sweep);

    n_iters  = zeros(n_sweep, 1);
    p_tmp           = params;
    p_tmp.tol       = 1e-8;
    p_tmp.max_iter  = 800;
    p_tmp.plot_results = false;

    fprintf('\n  Sweep k+ = k- = k:\n');
    fprintf('  %12s  %8s\n', 'k [N/m]', 'n_iter');

    for i = 1:n_sweep
        k = k_values(i);
        [~, ~, inf_tmp] = solve_latin_mono(subs, F, p_tmp, k, k);
        n_iters(i) = inf_tmp.n_iter;
    end

    [n_opt, idx_opt] = min(n_iters);
    k_opt = k_values(idx_opt);
    fprintf('  k_opt (numerical) = %.4e  (n_iter=%d)\n', k_opt, n_opt);
    fprintf('  k_theory = E*S/H  = %.4e\n', k_theory);

    k_plus_opt  = k_opt;
    k_minus_opt = k_opt;

    %% Verify at theory value
    [~, ~, info_th] = solve_latin_mono(subs, F, params, k_theory, k_theory);
    fprintf('  At k=E*S/H: n_iter=%d, L2 err=%.2e\n', info_th.n_iter, info_th.error_L2);

    %% Plots
    if params.plot_results
        figure('Name','Q2.11 - Search direction optimization','Position',[100 100 1100 450]);

        subplot(1,3,1);
        semilogx(k_values/k_theory, n_iters, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
        hold on;
        xline(1, 'r--', 'LineWidth', 2);
        grid on; box on;
        xlabel('k / (E\cdotS/H)', 'FontSize', 12);
        ylabel('Iterations', 'FontSize', 12);
        title('1D sweep: k^+ = k^-', 'FontSize', 12);
        text(1.1, max(n_iters)*0.9, 'k=E\cdotS/H', 'Color','r', 'FontSize',10);

        subplot(1,3,2);
        % Convergence curves: k_opt vs k/10 vs k*10
        k_cases = [k_theory/10, k_theory, k_theory*10];
        lbls    = {'k=E\cdotS/(10H)', 'k=E\cdotS/H (opt)', 'k=10\cdotE\cdotS/H'};
        cols    = {'r-', 'b-', 'g-'};
        for ic = 1:3
            [~, ~, ic_info] = solve_latin_mono(subs, F, p_tmp, k_cases(ic), k_cases(ic));
            semilogy(1:ic_info.n_iter, ic_info.eta_history, cols{ic}, ...
                     'LineWidth', 2, 'DisplayName', lbls{ic});
            hold on;
        end
        yline(params.tol, 'k--', 'LineWidth', 1.5, 'DisplayName', sprintf('tol=%.0e', params.tol));
        grid on; box on;
        xlabel('Iteration', 'FontSize', 12);
        ylabel('\eta', 'FontSize', 12);
        title('Convergence curves', 'FontSize', 12);
        legend('Location','northeast','FontSize',9);

        subplot(1,3,3);
        % Displacement at converged optimal solution
        x_plt = linspace(0, params.L, params.Nn)' * 1e3;
        [~, ~, info_opt] = solve_latin_mono(subs, F, params, k_theory, k_theory);
        plot(x_plt, info_opt.u_global*1e6, 'b-o', 'LineWidth',2,'MarkerSize',5,'DisplayName','LaTIn');
        hold on;
        plot(x_plt, info_opt.u_exact*1e6, 'r--', 'LineWidth',2,'DisplayName','Exact');
        grid on; box on;
        xlabel('x [mm]','FontSize',12); ylabel('u [\mum]','FontSize',12);
        title(sprintf('Displacement (k=E\\cdotS/H, err=%.1e)',info_opt.error_L2),'FontSize',11);
        legend('Location','best','FontSize',10);

        sgtitle(sprintf('Q2.11: LaTIn search direction opt.  (N_{sub}=%d, N_{e,sub}=%d)', ...
                        params.N_sub, params.Ne_sub), 'FontSize',13);

        if params.save_results
            out = '../results/section2_3/figures/';
            if ~exist(out,'dir'); mkdir(out); end
            saveas(gcf, fullfile(out,'Q2_11_search_directions.png'));
        end
    end

    info.k_values  = k_values;
    info.n_iters   = n_iters;
    info.k_theory  = k_theory;
    info.k_opt     = k_opt;
    info.info_theory = info_th;

    fprintf('------------------------------------\n\n');
end