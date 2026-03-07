%% section2_2_dual.m
% Section 2.2: Dual Approach
%
% Implementation Content:
% - Question 2.7: Direct Solution of Dual Interface Problem
% - Question 2.8: FETI Method (Projected PCG + Dirichlet Preconditions)
% - Question 2.9: Rigid Body Modal Recovery and Displacement Plotting

clear; clc; close all;

addpath(genpath('../src'));

fprintf('\n');
fprintf('========================================\n');
fprintf('  Section 2.2: Dual Approach\n');
fprintf('========================================\n\n');

fprintf('Preparation: Generating domain decomposition data...\n');
params = config_params();
mesh = generate_mesh(params);
[K, F] = assemble_global_system(mesh, params);
subs = generate_substructures(mesh, params);

%  Question 2.7: Directly solving the dual interface problem

fprintf('\n');
fprintf('============================================\n');
fprintf('  Question 2.7: Directly solving the dual interface problem\n');
fprintf('============================================\n');

[lambda_direct, u_direct, info_direct] = solve_dual_direct(subs, F, params);

% Plot the results of Q2.7
figure('Name', 'Q2.7: Dual direct method displacement', 'Position', [50 50 800 500]);
u_exact = info_direct.u_exact;
plot(mesh.x*1e3, u_direct*1e6, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 5, ...
     'DisplayName', 'Dual direct');
hold on;
plot(mesh.x*1e3, u_exact*1e6, 'r--', 'LineWidth', 2, 'DisplayName', 'Analytical');
grid on;
xlabel('Position x [mm]', 'FontSize', 12);
ylabel('u(x) [\mum]', 'FontSize', 12);
title('Q2.7: Dual Schur Direct Method – Displacement Field', 'FontSize', 13);
legend('Location', 'best');
set(gca, 'FontSize', 11);

%  Question 2.8: FETI method (projection PCG + Dirichlet preconditioning)
fprintf('\n');
fprintf('============================================\n');
fprintf('  Question 2.8: FETI method\n');
fprintf('============================================\n');

[lambda_feti, u_feti, feti_info] = solve_dual_feti(subs, F, params);

% Plot the convergence history of Q2.8 (only if there are iterations)
if ~isempty(feti_info.residuals)
    figure('Name', 'Q2.8: FETI Convergence', 'Position', [100 100 800 500]);
    semilogy(1:feti_info.n_iter, feti_info.residuals, 'b-o', ...
             'LineWidth', 1.5, 'MarkerSize', 6);
    hold on;
    yline(params.tol, '--r', 'LineWidth', 1.5);
    grid on;
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Relative residual', 'FontSize', 12);
    title(sprintf('Q2.8: FETI PCG convergence (N_{sub}=%d, N_e=%d)', ...
          params.N_sub, params.Ne), 'FontSize', 14);
    legend('residual', sprintf('tol = %.0e', params.tol), 'Location', 'best');
    set(gca, 'FontSize', 11);
end

%  Question 2.9: Rigid body mode recovery
fprintf('\n');
fprintf('============================================\n');
fprintf('  Question 2.9: Rigid body mode recovery\n');
fprintf('============================================\n');

[u_recovered, rigid_info] = recover_rigid_modes(lambda_feti, subs, F, params, feti_info);

%  Scalability Analysis: FETI Performance at Different N_sub Values
fprintf('\n');
fprintf('============================================\n');
fprintf('  FETI Scalability Analysis\n');
fprintf('============================================\n');


Ne_list   = [6, 12, 20, 30, 40];
Nsub_list = [2,  4,  5, 10, 20];

fprintf('\n--- h/H is constant, increase N_sub ---\n');
fprintf('%6s %6s %6s %8s %10s %10s\n', 'Ne', 'N_sub', 'Ne_sub', 'h/H', 'PCG_iter', 'L2_err');
fprintf('-------------------------------------------------------\n');

%  Added: Data collection 
iter_hH_const   = [];
Nsub_hH_const   = [];
err_hH_const    = [];

for k = 1:length(Ne_list)
    p = config_params();
    p.Ne = Ne_list(k);
    p.N_sub = Nsub_list(k);
    if mod(p.Ne, p.N_sub) ~= 0, continue; end
    p.Ne_sub = p.Ne / p.N_sub;
    p.h = p.L / p.Ne;
    p.Nn = p.Ne + 1;
    p.H = p.Ne_sub * p.h;

    m = generate_mesh(p);
    [~, Fk] = assemble_global_system(m, p);
    sk = generate_substructures(m, p);

    try
        [~, uk, ik] = solve_dual_feti(sk, Fk, p);
        u_ex = (p.Fd / (p.E * p.S)) * m.x;
        err_k = norm(uk - u_ex) / norm(u_ex);
        fprintf('%6d %6d %6d %8.4f %10d %10.2e\n', ...
                p.Ne, p.N_sub, p.Ne_sub, p.h/p.H, ik.n_iter, err_k);
        %  Data collection 
        iter_hH_const(end+1) = ik.n_iter;
        Nsub_hH_const(end+1) = p.N_sub;
        err_hH_const(end+1)  = err_k;
    catch ME
        fprintf('%6d %6d %6d %8.4f %10s %10s\n', ...
                p.Ne, p.N_sub, p.Ne_sub, p.h/p.H, 'ERR', ME.message(1:min(10,end)));
    end
end

%%  Scenario 2: Keep h constant (Ne=20), increase N_sub 
fprintf('\n--- h is constant (Ne=20), N_sub is increased. ---\n');
fprintf('%6s %6s %6s %8s %10s %10s %10s\n', ...
        'Ne', 'N_sub', 'Ne_sub', 'h/H', 'PCG_iter', 'L2_err', 'kappa_Sd');
fprintf('-------------------------------------------------------------------\n');

%  Added: Data collection 
iter_h_const  = [];
Nsub_h_const  = [];
kappa_h_const = [];

Nsub_test = [2, 4, 5, 10, 20];
for k = 1:length(Nsub_test)
    p = config_params();
    p.Ne = 20;
    p.N_sub = Nsub_test(k);
    if mod(p.Ne, p.N_sub) ~= 0, continue; end
    p.Ne_sub = p.Ne / p.N_sub;
    p.h = p.L / p.Ne;
    p.Nn = p.Ne + 1;
    p.H = p.Ne_sub * p.h;

    m = generate_mesh(p);
    [~, Fk] = assemble_global_system(m, p);
    sk = generate_substructures(m, p);

    try
        [~, uk, ik] = solve_dual_feti(sk, Fk, p);
        u_ex = (p.Fd / (p.E * p.S)) * m.x;
        err_k = norm(uk - u_ex) / norm(u_ex);
        kappa = cond(ik.Sd);
        fprintf('%6d %6d %6d %8.4f %10d %10.2e %10.2e\n', ...
                p.Ne, p.N_sub, p.Ne_sub, p.h/p.H, ik.n_iter, err_k, kappa);
        %  Data collection 
        iter_h_const(end+1)  = ik.n_iter;
        Nsub_h_const(end+1)  = p.N_sub;
        kappa_h_const(end+1) = kappa;
    catch ME
        fprintf('%6d %6d %6d %8.4f %10s %10s\n', ...
                p.Ne, p.N_sub, p.Ne_sub, p.h/p.H, 'ERR', ME.message(1:min(10,end)));
    end
end

%  Q2.8 Scalable Drawing

%  Figure 1: Fixed h/H, FETI iteration count vs N_sub 
figure('Name', 'Q2.8: FETI Scalability', 'Position', [200 200 700 450]);

plot(Nsub_hH_const, iter_hH_const, 'rs-', ...
     'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'FETI (Dirichlet precond.)');
hold on;

Nsub_ref = Nsub_hH_const;
plot(Nsub_ref, Nsub_ref - 1, 'b--o', ...
     'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Unpreconditioned CG (reference, ~ S-1)');

grid on;
xlabel('Number of subdomains', 'FontSize', 13);
ylabel('PCG iterations', 'FontSize', 13);
title('Q2.8: FETI Scalability (fixed h/H)', 'FontSize', 13);
legend('Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 11);
ylim([-0.5, max(Nsub_ref)]);

%  Figure 2: Fixed h, Sd condition number vs N_sub 
if ~isempty(kappa_h_const)
    figure('Name', 'Q2.8: Sd condition number (fixed h)', 'Position', [250 250 700 450]);
    loglog(Nsub_h_const, kappa_h_const, 'ms-', ...
           'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '\kappa(S_d)');
    hold on;
    S_ref = Nsub_h_const;
    kappa_ref = kappa_h_const(1) * (S_ref / S_ref(1)).^2;
    loglog(S_ref, kappa_ref, 'k--', 'LineWidth', 1.5, ...
           'DisplayName', 'O(S^2) reference');
    grid on;
    xlabel('Number of subdomains', 'FontSize', 13);
    ylabel('\kappa(S_d)', 'FontSize', 13);
    title('Q2.8: Condition number of S_d (fixed h)', 'FontSize', 13);
    legend('Location', 'best', 'FontSize', 11);
    set(gca, 'FontSize', 11);
end