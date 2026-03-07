%% section2_1_primal.m
% Section 2.1: Primal Approach
%
% Implementation:
% - Question 2.1: Direct Solution of Schur Systems
% - Question 2.2: Condition Number Analysis
% - Question 2.3: Distributed Conjugate Gradient Method
% - Question 2.5: BDD Preconditioning
% - Question 2.6: Condition Number of Preconditioned Systems kappa(Sptilde^{-1} Sp) = O(1)

clear; clc; close all;

addpath(genpath('../src'));

fprintf('\n');
fprintf('========================================\n');
fprintf('  Section 2.1: Primal Approach\n');
fprintf('========================================\n\n');

%% Preparation: Generate mesh, assemble global system, generate substructures
params = config_params();
mesh   = generate_mesh(params);
[K, F] = assemble_global_system(mesh, params);
subs   = generate_substructures(mesh, params);

%% Reference Solution
u_ref = apply_boundary_conditions(K, F, mesh, params, 'elimination');
fprintf('Reference Solution u(L) = %.12e m\n\n', u_ref(end));


%  Question 2.1: Direct solution of the Schur  
fprintf('\n');
fprintf('Question 2.1: Direct solution of the Schur system\n');
[u_interface, u_global, info] = solve_schur_direct(subs, F, params);

if params.plot_results
    figure('Name', 'Q2.1 - Primal Schur direct solution', 'Position', [100 100 800 500]);
    x_nodes = linspace(0, params.L, params.Nn)';

    plot(x_nodes * 1e3, u_global * 1e6, 'b-o', ...
         'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'DDM (Schur direct method)');
    hold on;
    plot(x_nodes * 1e3, info.u_exact * 1e6, 'r--', ...
         'LineWidth', 2, 'DisplayName', 'Analytical solution');
    plot(x_nodes * 1e3, u_ref * 1e6, 'k:', ...
         'LineWidth', 1.5, 'DisplayName', 'Overall FEM Reference Solution');

    grid on;
    xlabel('Location x [mm]', 'FontSize', 12);
    ylabel('Displacement u(x) [μm]', 'FontSize', 12);
    title(sprintf('Q2.1: Primal Schur direct solution (N_{sub}=%d, N_e=%d)', ...
                  params.N_sub, params.Ne), 'FontSize', 13);
    legend('Location', 'best');

    if params.save_results
        output_dir = '../results/section2_1/figures/';
        if ~exist(output_dir, 'dir'); mkdir(output_dir); end
        saveas(gcf, fullfile(output_dir, 'Q2_1_schur_direct.png'));
    end
end

fprintf('\n========================================\n');
fprintf('  Q2.1 finished!\n');
fprintf('  κ(Sp) = %.4e  |  L2 error = %.4e\n', info.kappa_Sp, info.error_L2);
fprintf('========================================\n\n');

%  Question 2.2: Condition number analysis

fprintf('\n');
fprintf('------------------------------------\n');
fprintf('Question 2.2: Condition number analysis\n');
fprintf('------------------------------------\n');

cond_results = analyze_conditioning(params);

%% Save image
if params.save_results
    output_dir = '../results/section2_1/figures/';
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end

    figs = get(0, 'Children');
    for idx = 1:min(2, length(figs))
        fname = fullfile(output_dir, sprintf('Q2_2_conditioning_exp%c.png', 'A'+idx-1));
        saveas(figs(idx), fname);
        fprintf('Image saved: %s\n', fname);
    end
end

fprintf('\n========================================\n');
fprintf('  Q2.2 finished!\n');
fprintf('  See the power estimation and conclusions output above.\n');
fprintf('========================================\n\n');


%  Question 2.3: Distributed conjugate gradient method

fprintf('------------------------------------\n');
fprintf('Question 2.3: Distributed CG\n');
fprintf('------------------------------------\n');

%% --- Solve using the current params ---
[u_interface_pcg, u_global_pcg, info_pcg] = solve_schur_pcg(subs, F, params);

%% --- Figure 1: Displacement Comparison ---
if params.plot_results
    figure('Name', 'Q2.3 - Distributed CG', 'Position', [120 120 900 500]);
    x_nodes = linspace(0, params.L, params.Nn)';

    plot(x_nodes*1e3, u_global_pcg*1e6, 'b-o', ...
         'LineWidth', 1.5, 'MarkerSize', 5, 'DisplayName', 'PCG (distributed)');
    hold on;
    plot(x_nodes*1e3, info_pcg.u_exact*1e6, 'r--', ...
         'LineWidth', 2, 'DisplayName', 'Exact');
    plot(x_nodes*1e3, u_ref*1e6, 'k:', ...
         'LineWidth', 1.5, 'DisplayName', 'Global FEM ref');

    grid on; box on;
    xlabel('x [mm]', 'FontSize', 12);
    ylabel('u(x) [\mum]', 'FontSize', 12);
    title(sprintf('Q2.3: Distributed CG  (N_{sub}=%d,  %d iter,  L2=%.1e)', ...
                  params.N_sub, info_pcg.n_iter, info_pcg.error_L2), 'FontSize', 13);
    legend('Location', 'best');

    if params.save_results
        output_dir = '../results/section2_1/figures/';
        if ~exist(output_dir, 'dir'); mkdir(output_dir); end
        saveas(gcf, fullfile(output_dir, 'Q2_3_displacement.png'));
    end
end

%% Figure 2: Number of CG iterations vs. N_sub (non-scalability graph) ---
N_sub_list  = [2, 4, 8, 16, 32, 64];
Ne_sub_conv = max(params.Ne_sub, 4);
n_iters     = zeros(length(N_sub_list), 1);

fprintf('\n  Scalability test (fixed Ne_sub=%d):\n', Ne_sub_conv);
fprintf('  %8s  %10s  %10s\n', 'N_sub', 'n_iter', 'theory');

for idx = 1:length(N_sub_list)
    Ns = N_sub_list(idx);

    p_tmp        = params;
    p_tmp.N_sub  = Ns;
    p_tmp.Ne_sub = Ne_sub_conv;
    p_tmp.Ne     = Ns * Ne_sub_conv;
    p_tmp.h      = p_tmp.L / p_tmp.Ne;
    p_tmp.Nn     = p_tmp.Ne + 1;
    p_tmp.H      = Ne_sub_conv * p_tmp.h;

    mesh_tmp          = generate_mesh(p_tmp);
    [K_tmp, F_tmp]    = assemble_global_system(mesh_tmp, p_tmp);
    subs_tmp          = generate_substructures(mesh_tmp, p_tmp);
    [~, ~, info_tmp]  = solve_schur_pcg(subs_tmp, F_tmp, p_tmp);

    n_iters(idx) = info_tmp.n_iter;
    fprintf('  %8d  %10d  %10d\n', Ns, info_tmp.n_iter, Ns-1);
end

if params.plot_results
    figure('Name', 'Q2.3 - CG scalability', 'Position', [200 200 750 450]);

    plot(N_sub_list, n_iters, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, ...
         'DisplayName', 'CG iterations (measured)');
    hold on;
    plot(N_sub_list, N_sub_list - 1, 'r--', 'LineWidth', 1.8, ...
         'DisplayName', 'N_{sub} - 1  (theory)');
    yline(n_iters(1), 'g-', 'LineWidth', 1.8, ...
          'DisplayName', 'Ideal: const  (scalable)');

    grid on; box on;
    xlabel('N_{sub}', 'FontSize', 12);
    ylabel('CG iterations to convergence', 'FontSize', 12);
    title(sprintf('Q2.3: CG iterations ~ N_{sub}-1  \\Rightarrow  not scalable  (N_{e,sub}=%d)', ...
                  Ne_sub_conv), 'FontSize', 13);
    legend('Location', 'northwest', 'FontSize', 11);
    xticks(N_sub_list);

    if params.save_results
        output_dir = '../results/section2_1/figures/';
        if ~exist(output_dir, 'dir'); mkdir(output_dir); end
        saveas(gcf, fullfile(output_dir, 'Q2_3_scalability.png'));
    end
end

fprintf('\n========================================\n');
fprintf('  Q2.3 Done!\n');
fprintf('  N_sub=%d | Iterations: %d | L2 error: %.4e\n', ...
        params.N_sub, info_pcg.n_iter, info_pcg.error_L2);
fprintf('========================================\n\n');


%  Question 2.5: BDD preconditioning conjugate gradient (Neumann preconditioner)

fprintf('------------------------------------\n');
fprintf('Question 2.5: BDD PCG\n');
fprintf('------------------------------------\n');

%% Solve using the current params 
[u_interface_bdd, u_global_bdd, info_bdd] = solve_bdd_pcg(subs, F, params);

%%  Figure 1: Displacement Comparison 
if params.plot_results
    figure('Name', 'Q2.5 - BDD PCG', 'Position', [130 130 900 500]);
    x_nodes = linspace(0, params.L, params.Nn)';

    plot(x_nodes*1e3, u_global_bdd*1e6, 'b-o', ...
         'LineWidth', 1.5, 'MarkerSize', 5, 'DisplayName', 'BDD PCG');
    hold on;
    plot(x_nodes*1e3, info_bdd.u_exact*1e6, 'r--', ...
         'LineWidth', 2, 'DisplayName', 'Exact');
    plot(x_nodes*1e3, u_ref*1e6, 'k:', ...
         'LineWidth', 1.5, 'DisplayName', 'Global FEM ref');

    grid on; box on;
    xlabel('x [mm]', 'FontSize', 12);
    ylabel('u(x) [\mum]', 'FontSize', 12);
    title(sprintf('Q2.5: BDD PCG  (N_{sub}=%d,  %d iter,  L2=%.1e)', ...
                  params.N_sub, info_bdd.n_iter, info_bdd.error_L2), 'FontSize', 13);
    legend('Location', 'best');

    if params.save_results
        output_dir = '../results/section2_1/figures/';
        if ~exist(output_dir, 'dir'); mkdir(output_dir); end
        saveas(gcf, fullfile(output_dir, 'Q2_5_displacement.png'));
    end
end

%%  Scalable Scan 
N_sub_list   = [2, 4, 8, 16, 32, 64];
Ne_sub_fixed = max(params.Ne_sub, 4);

n_cg  = zeros(length(N_sub_list), 1);
n_bdd = zeros(length(N_sub_list), 1);

fprintf('\n  Scalability comparison (fixed Ne_sub=%d):\n', Ne_sub_fixed);
fprintf('  %6s  %10s  %10s\n', 'N_sub', 'CG iters', 'BDD iters');

for idx = 1:length(N_sub_list)
    Ns = N_sub_list(idx);

    p_tmp        = params;
    p_tmp.N_sub  = Ns;
    p_tmp.Ne_sub = Ne_sub_fixed;
    p_tmp.Ne     = Ns * Ne_sub_fixed;
    p_tmp.h      = p_tmp.L / p_tmp.Ne;
    p_tmp.Nn     = p_tmp.Ne + 1;
    p_tmp.H      = Ne_sub_fixed * p_tmp.h;

    mesh_tmp             = generate_mesh(p_tmp);
    [~, F_tmp]           = assemble_global_system(mesh_tmp, p_tmp);
    subs_tmp             = generate_substructures(mesh_tmp, p_tmp);
    [~, ~, info_cg]      = solve_schur_pcg(subs_tmp, F_tmp, p_tmp);
    [~, ~, info_bdd_tmp] = solve_bdd_pcg(subs_tmp, F_tmp, p_tmp);

    n_cg(idx)  = info_cg.n_iter;
    n_bdd(idx) = info_bdd_tmp.n_iter;

    fprintf('  %6d  %10d  %10d\n', Ns, n_cg(idx), n_bdd(idx));
end

%% Figure 2: Comparison of iteration counts 
if params.plot_results
    figure('Name', 'Q2.5 - Scalability', 'Position', [200 150 850 500]);

    plot(N_sub_list, n_cg, 'r-s', 'LineWidth', 2, 'MarkerSize', 9, ...
         'DisplayName', 'Primal CG  (no precond)');
    hold on;
    plot(N_sub_list, N_sub_list-1, 'r--', 'LineWidth', 1.4, ...
         'HandleVisibility', 'off');
    plot(N_sub_list, max(n_bdd, zeros(size(n_bdd))), 'b-o', ...
         'LineWidth', 2.5, 'MarkerSize', 10, ...
         'DisplayName', 'BDD PCG  (Neumann precond)');
    plot(N_sub_list, ones(size(N_sub_list)) * max(n_bdd(1), 0), 'g-', ...
         'LineWidth', 2, 'DisplayName', 'Ideal: const  (scalable)');

    grid on; box on;
    xlabel('N_{sub}  (fixed h/H = 1/N_{e,sub})', 'FontSize', 12);
    ylabel('Iterations to convergence', 'FontSize', 12);
    title(sprintf(['Q2.5: Scalability comparison  (N_{e,sub}=%d)\n' ...
                   'BDD: O(1) iters  \\Rightarrow  numerically scalable'], ...
                  Ne_sub_fixed), 'FontSize', 12);
    legend('Location', 'northwest', 'FontSize', 11);
    xticks(N_sub_list);

    if params.save_results
        output_dir = '../results/section2_1/figures/';
        if ~exist(output_dir, 'dir'); mkdir(output_dir); end
        saveas(gcf, fullfile(output_dir, 'Q2_5_scalability.png'));
    end
end

%% --- Figure 3 : kappa(Sp) vs kappa(P^T Sp P) ---
if params.plot_results
    kappa_Sp   = zeros(length(N_sub_list), 1);
    kappa_PSpP = zeros(length(N_sub_list), 1);

    for idx = 1:length(N_sub_list)
        Ns = N_sub_list(idx);
        p_tmp        = params;
        p_tmp.N_sub  = Ns;
        p_tmp.Ne_sub = Ne_sub_fixed;
        p_tmp.Ne     = Ns * Ne_sub_fixed;
        p_tmp.h      = p_tmp.L / p_tmp.Ne;
        p_tmp.Nn     = p_tmp.Ne + 1;
        p_tmp.H      = Ne_sub_fixed * p_tmp.h;

        kappa_Sp(idx)   = compute_kappa_Sp(p_tmp);
        kappa_PSpP(idx) = compute_kappa_PSpP(p_tmp);
    end

    figure('Name', 'Q2.5 - Condition numbers', 'Position', [250 200 850 480]);

    semilogy(N_sub_list, kappa_Sp,   'r-s', 'LineWidth', 2, 'MarkerSize', 9, ...
             'DisplayName', '\kappa(S_p)  ~  O(N_{sub}^2)');
    hold on;
    semilogy(N_sub_list, kappa_PSpP, 'b-o', 'LineWidth', 2.5, 'MarkerSize', 10, ...
             'DisplayName', '\kappa(P^T S_p P)  =  O(1)  [BDD]');
    semilogy(N_sub_list, ones(size(N_sub_list)), 'g--', 'LineWidth', 1.8, ...
             'DisplayName', 'Ideal: \kappa = 1');

    grid on; box on;
    xlabel('N_{sub}  (fixed h/H = 1/N_{e,sub})', 'FontSize', 12);
    ylabel('Condition number \kappa', 'FontSize', 12);
    title(sprintf(['Q2.5: Condition number after BDD projection  (N_{e,sub}=%d)\n' ...
                   'BDD reduces \\kappa(S_p) from O(N_{sub}^2) to O(1)'], ...
                  Ne_sub_fixed), 'FontSize', 12);
    legend('Location', 'northwest', 'FontSize', 11);
    xticks(N_sub_list);

    if params.save_results
        output_dir = '../results/section2_1/figures/';
        saveas(gcf, fullfile(output_dir, 'Q2_5_condition_numbers.png'));
    end
end

fprintf('\n========================================\n');
fprintf('  Q2.5 Done!\n');
fprintf('  N_sub=%d | BDD iters: %d | L2 err: %.4e\n', ...
        params.N_sub, info_bdd.n_iter, info_bdd.error_L2);
fprintf('  BDD is numerically scalable: kappa(P^T Sp P) = O(1)\n');
fprintf('========================================\n\n');

%% 
%  Question 2.6: Verify the condition number of the preconditioned system kappa(Sptilde^{-1} Sp) = O(1)

fprintf('------------------------------------\n');
fprintf('Question 2.6: kappa(Sptilde^{-1} Sp) = O(1)\n');
fprintf('------------------------------------\n');

N_sub_list_q26 = [2, 4, 8, 16, 32, 64];
Ne_sub_q26     = max(params.Ne_sub, 4);

kappa_Sp_q26      = zeros(length(N_sub_list_q26), 1);
kappa_precond_q26 = zeros(length(N_sub_list_q26), 1);

fprintf('\n  %6s  %14s  %22s\n', 'N_sub', 'kappa(Sp)', 'kappa(Sptilde^{-1}Sp)');
fprintf('  %s\n', repmat('-', 1, 48));

for idx = 1:length(N_sub_list_q26)
    Ns = N_sub_list_q26(idx);

    p_tmp        = params;
    p_tmp.N_sub  = Ns;
    p_tmp.Ne_sub = Ne_sub_q26;
    p_tmp.Ne     = Ns * Ne_sub_q26;
    p_tmp.h      = p_tmp.L / p_tmp.Ne;
    p_tmp.Nn     = p_tmp.Ne + 1;
    p_tmp.H      = Ne_sub_q26 * p_tmp.h;

    [Sp_q26, Sp_tilde_q26] = assemble_Sp_and_Sptilde(p_tmp);

    %% kappa(Sp): eigenvalues restricted to range (remove near-zero)
    eigs_Sp  = sort(real(eig(Sp_q26)));
    tol_eig  = max(abs(eigs_Sp)) * 1e-10;
    pos_Sp   = eigs_Sp(eigs_Sp > tol_eig);
    kappa_Sp_q26(idx) = pos_Sp(end) / pos_Sp(1);

    %% kappa(Sptilde^{-1} Sp) via generalized eigenvalue problem:
    %%   Sp v = lambda Sp_tilde v  <=>  Sptilde^{-1} Sp v = lambda v
    eigs_gen  = sort(real(eig(Sp_q26, Sp_tilde_q26)));
    pos_gen   = eigs_gen(eigs_gen > max(abs(eigs_gen)) * 1e-10);
    if length(pos_gen) >= 2
        kappa_precond_q26(idx) = pos_gen(end) / pos_gen(1);
    else
        kappa_precond_q26(idx) = 1.0;
    end

    fprintf('  %6d  %14.4e  %22.4e\n', Ns, kappa_Sp_q26(idx), kappa_precond_q26(idx));
end

%% Power-law fitting
log_N    = log(N_sub_list_q26(:));
p_fit_Sp = polyfit(log_N, log(kappa_Sp_q26), 1);

fprintf('\n  kappa(Sp)                ~ N_sub^%.2f  (theory: N_sub^2)\n', p_fit_Sp(1));
fprintf('  kappa(Sptilde^{-1} Sp)  ~ O(1)  (mean=%.4f, max=%.4f)\n', ...
        mean(kappa_precond_q26), max(kappa_precond_q26));

%%  Figure: Q2.6 Double logarithmic coordinates 
if params.plot_results
    figure('Name', 'Q2.6 - Preconditioned system conditioning', 'Position', [300 200 850 500]);

    loglog(N_sub_list_q26, kappa_Sp_q26, 'r-s', 'LineWidth', 2, 'MarkerSize', 9, ...
           'DisplayName', sprintf('\\kappa(S_p)  ~  N_{sub}^{%.1f}', p_fit_Sp(1)));
    hold on;
    loglog(N_sub_list_q26, kappa_precond_q26, 'b-o', 'LineWidth', 2.5, 'MarkerSize', 10, ...
           'DisplayName', '\kappa(\tilde{S}_p^{-1} S_p)  =  O(1)  [BDD]');

    % Reference line O(N_sub^2)
    ref_quad = N_sub_list_q26.^2 * (kappa_Sp_q26(1) / N_sub_list_q26(1)^2);
    loglog(N_sub_list_q26, ref_quad, 'r--', 'LineWidth', 1.4, ...
           'DisplayName', 'O(N_{sub}^2)  reference');

    % ideal O(1)
    yline(1, 'g--', 'LineWidth', 1.8, 'DisplayName', 'Ideal: \kappa = 1');

    grid on; box on;
    xlabel('N_{sub}  (fixed h/H = 1/N_{e,sub})', 'FontSize', 12);
    ylabel('Condition number \kappa', 'FontSize', 12);
    title(sprintf(['Q2.6: Conditioning of preconditioned system  (N_{e,sub}=%d)\n' ...
                   '\\kappa(\\tilde{S}_p^{-1} S_p) = O(1)  \\Rightarrow  BDD is numerically scalable'], ...
                  Ne_sub_q26), 'FontSize', 12);
    legend('Location', 'northwest', 'FontSize', 11);
    xticks(N_sub_list_q26);
    xticklabels(arrayfun(@num2str, N_sub_list_q26, 'UniformOutput', false));

    if params.save_results
        output_dir = '../results/section2_1/figures/';
        if ~exist(output_dir, 'dir'); mkdir(output_dir); end
        saveas(gcf, fullfile(output_dir, 'Q2_6_precond_conditioning.png'));
        fprintf('\n  Image saved: Q2_6_precond_conditioning.png\n');
    end
end

fprintf('\n========================================\n');
fprintf('  Q2.6 finished!\n');
fprintf('  kappa(Sp)               ~ O(N_sub^%.1f)\n', p_fit_Sp(1));
fprintf('  kappa(Sptilde^{-1} Sp)  = O(1)  (mean=%.4f)\n', mean(kappa_precond_q26));
fprintf('  Conclusion: Neumann preconditioning reduces the condition number from O(N_sub^2) to O(1).\n');
fprintf('========================================\n\n');



%  Auxiliary function: Calculate kappa(Sp) (for Q2.5, Figure 3)

function kappa = compute_kappa_Sp(params)
    Nn     = params.Nn;
    Ne_sub = params.Ne_sub;
    N_sub  = params.N_sub;
    E = params.E; S = params.S; h = params.h;
    ke = (E*S/h) * [1,-1;-1,1];

    if_nodes = ((1:N_sub-1)*Ne_sub+1)';
    n_if = length(if_nodes);
    if_map = zeros(Nn,1);
    for k=1:n_if; if_map(if_nodes(k))=k; end

    Sp = zeros(n_if, n_if);
    for s = 1:N_sub
        ns = (s-1)*Ne_sub+1; ne = s*Ne_sub+1; nloc = Ne_sub+1;
        K_sub = zeros(nloc);
        for e=1:Ne_sub; K_sub([e,e+1],[e,e+1])=K_sub([e,e+1],[e,e+1])+ke; end
        b_loc=[]; b_glo=[];
        if if_map(ns)>0; b_loc(end+1)=1;    b_glo(end+1)=ns;  end
        if if_map(ne)>0; b_loc(end+1)=nloc; b_glo(end+1)=ne;  end
        d_loc=[]; if s==1; d_loc=[1]; end
        all_b=unique([b_loc, d_loc]); i_loc=setdiff(1:nloc, all_b);
        if isempty(b_loc); continue; end
        Kii=K_sub(i_loc,i_loc); Kib=K_sub(i_loc,b_loc);
        Kbi=K_sub(b_loc,i_loc); Kbb=K_sub(b_loc,b_loc);
        Ss = Kbb - Kbi*(Kii\Kib);
        for ii=1:length(b_glo)
            ki=if_map(b_glo(ii)); if ki==0; continue; end
            for jj=1:length(b_glo)
                kj=if_map(b_glo(jj)); if kj==0; continue; end
                Sp(ki,kj)=Sp(ki,kj)+Ss(ii,jj);
            end
        end
    end
    eigs_Sp = eig(Sp); eigs_pos = sort(eigs_Sp(eigs_Sp>1e3));
    kappa = eigs_pos(end)/eigs_pos(1);
end



%  Auxiliary function: Calculate kappa(P^T Sp P) over range(P) (for Q2.5, Figure 3)

function kappa = compute_kappa_PSpP(params)
    Nn=params.Nn; Ne_sub=params.Ne_sub; N_sub=params.N_sub;
    E=params.E; S=params.S; h=params.h;
    ke=(E*S/h)*[1,-1;-1,1];

    if_nodes=((1:N_sub-1)*Ne_sub+1)';
    n_if=length(if_nodes);
    if_map=zeros(Nn,1);
    for k=1:n_if; if_map(if_nodes(k))=k; end

    Sp=zeros(n_if,n_if); G_cols={};
    for s=1:N_sub
        ns=(s-1)*Ne_sub+1; ne=s*Ne_sub+1; nloc=Ne_sub+1;
        K_sub=zeros(nloc);
        for e=1:Ne_sub; K_sub([e,e+1],[e,e+1])=K_sub([e,e+1],[e,e+1])+ke; end
        b_loc=[]; b_glo=[];
        if if_map(ns)>0; b_loc(end+1)=1;    b_glo(end+1)=ns; end
        if if_map(ne)>0; b_loc(end+1)=nloc; b_glo(end+1)=ne; end
        d_loc=[]; if s==1; d_loc=[1]; end
        all_b=unique([b_loc,d_loc]); i_loc=setdiff(1:nloc,all_b);
        if isempty(b_loc); continue; end
        Kii=K_sub(i_loc,i_loc); Kib=K_sub(i_loc,b_loc);
        Kbi=K_sub(b_loc,i_loc); Kbb=K_sub(b_loc,b_loc);
        Ss=Kbb-Kbi*(Kii\Kib);
        for ii=1:length(b_glo)
            ki=if_map(b_glo(ii)); if ki==0; continue; end
            for jj=1:length(b_glo)
                kj=if_map(b_glo(jj)); if kj==0; continue; end
                Sp(ki,kj)=Sp(ki,kj)+Ss(ii,jj);
            end
        end
        % G: Floating substructure
        if s==1; continue; end
        g=zeros(n_if,1);
        for gj=[ns, (s<N_sub)*ne]
            if gj==0; continue; end
            ki=if_map(gj); if ki>0; g(ki)=1; end
        end
        G_cols{end+1}=g;
    end
    G=cell2mat(G_cols);
    GtSpG=G'*Sp*G;
    P=eye(n_if)-G*(GtSpG\(G'*Sp));
    PtSpP=P'*Sp*P;
    eigs_proj=sort(real(eig(PtSpP)));
    eigs_pos=eigs_proj(eigs_proj>1e3);
    if length(eigs_pos)<2; kappa=1.0; return; end
    kappa=eigs_pos(end)/eigs_pos(1);
end


% Auxiliary functions: Assemble Sp and Sp_tilde (for Q2.6)

function [Sp, Sp_tilde] = assemble_Sp_and_Sptilde(params)
    N_sub  = params.N_sub;
    Ne_sub = params.Ne_sub;
    Nn     = params.Nn;
    E = params.E; S = params.S; h = params.h;
    ke = (E*S/h) * [1,-1;-1,1];

    if_nodes = ((1:N_sub-1)*Ne_sub+1)';
    n_if     = length(if_nodes);
    if_map   = zeros(Nn, 1);
    for k = 1:n_if; if_map(if_nodes(k)) = k; end

    Sp       = zeros(n_if, n_if);
    Sp_tilde = zeros(n_if, n_if);

    for s = 1:N_sub
        ns = (s-1)*Ne_sub+1; ne = s*Ne_sub+1; nloc = Ne_sub+1;

        K_sub = zeros(nloc);
        for e = 1:Ne_sub
            K_sub([e,e+1],[e,e+1]) = K_sub([e,e+1],[e,e+1]) + ke;
        end

        b_loc = []; b_glo = [];
        if if_map(ns) > 0; b_loc(end+1) = 1;    b_glo(end+1) = ns; end
        if if_map(ne) > 0; b_loc(end+1) = nloc;  b_glo(end+1) = ne; end
        b_loc = b_loc(:); b_glo = b_glo(:);
        if isempty(b_loc); continue; end

        %% Sp: avec Dirichlet BC sur le premier sous-domaine
        d_loc  = []; if s==1; d_loc = [1]; end
        all_b  = unique([b_loc; d_loc(:)]);
        i_loc  = setdiff((1:nloc)', all_b);

        if ~isempty(i_loc)
            Kii = K_sub(i_loc, i_loc);
            Kib = K_sub(i_loc, b_loc);
            Kbi = K_sub(b_loc, i_loc);
            Kbb = K_sub(b_loc, b_loc);
            Ss  = Kbb - Kbi * (Kii \ Kib);
        else
            Ss  = K_sub(b_loc, b_loc);
        end

        for ii = 1:length(b_glo)
            ki = if_map(b_glo(ii)); if ki==0; continue; end
            for jj = 1:length(b_glo)
                kj = if_map(b_glo(jj)); if kj==0; continue; end
                Sp(ki,kj) = Sp(ki,kj) + Ss(ii,jj);
            end
        end

        %% Sp_tilde: Neumann BC (pas de Dirichlet, pinv pour la singularite)
        i_loc_N = setdiff((1:nloc)', b_loc);

        if ~isempty(i_loc_N)
            Kii_N = K_sub(i_loc_N, i_loc_N);
            Kib_N = K_sub(i_loc_N, b_loc);
            Kbi_N = K_sub(b_loc, i_loc_N);
            Kbb_N = K_sub(b_loc, b_loc);
            Ss_N  = Kbb_N - Kbi_N * pinv(Kii_N) * Kib_N;
        else
            Ss_N  = K_sub(b_loc, b_loc);
        end

        for ii = 1:length(b_glo)
            ki = if_map(b_glo(ii)); if ki==0; continue; end
            for jj = 1:length(b_glo)
                kj = if_map(b_glo(jj)); if kj==0; continue; end
                Sp_tilde(ki,kj) = Sp_tilde(ki,kj) + Ss_N(ii,jj);
            end
        end
    end
end