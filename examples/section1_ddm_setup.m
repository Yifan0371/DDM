%% section1_ddm_setup.m
% Section 1 - Question 1.3: Domain Decomposition Extension
%
% Function:
% - Define substructures
% - Compute substructure operators (K_sub, S_sub, R_sub)
% - Define the DOF interface
% - Prepare for the DDM method in Section 2

clear; clc; close all;

addpath(genpath('../src'));

fprintf('\n');
fprintf('========================================\n');
fprintf('  Section 1 - Question 1.3\n');
fprintf('  Domain decomposition extension\n');
fprintf('========================================\n\n');

params = config_params();

mesh = generate_mesh(params);

[K, F] = assemble_global_system(mesh, params);

fprintf('------------------------------------\n');
fprintf('Solve the overall problem (reference solution)\n');
fprintf('------------------------------------\n');
u_ref = apply_boundary_conditions(K, F, mesh, params, 'elimination');
results_ref = postprocess(u_ref, mesh, params, K, F);

fprintf('------------------------------------\n');
fprintf('Generate domain decomposition substructure\n');
fprintf('------------------------------------\n');
subs = generate_substructures(mesh, params);

fprintf('\n');
fprintf('========================================\n');
fprintf('Substructure details (Example: Substructure #1)\n');
fprintf('========================================\n');
s = 1;

fprintf('\n1. Basic Information:\n');
fprintf('   Includes unit: [%d, %d]\n', subs(s).elem_ids(1), subs(s).elem_ids(end));
fprintf('   Includes nodes: [%d, %d]\n', subs(s).node_ids(1), subs(s).node_ids(end));
fprintf('   Number of local nodes: %d\n', subs(s).nloc);

fprintf('\n2. Classification of Degrees of Freedom:\n');
fprintf('   Interface DOF (Global Number): '); disp(subs(s).interface_dofs.');
fprintf('   Internal DOF (Global Number): '); 
if isempty(subs(s).internal_dofs)
    fprintf('None (the substructure has only 1 unit)\n');
else
    disp(subs(s).internal_dofs.');
end

fprintf('\n3. Substructure stiffness matrix K_sub (%dx%d):\n', ...
        size(subs(s).K_sub,1), size(subs(s).K_sub,2));
disp(subs(s).K_sub);

fprintf('4. Schur complement S_sub (condensation to boundary, 2x2):\n');
disp(subs(s).S_sub);
fprintf('   condition number: %.3e\n', cond(subs(s).S_sub));

fprintf('\n5. Rigid body modes R_sub (normalized):\n');
disp(subs(s).R_sub.');

fprintf('========================================\n\n');

fprintf('------------------------------------\n');
fprintf('Assemble the global Schur complement matrix\n');
fprintf('------------------------------------\n');

% Collect all interfaces DOF
all_interface_dofs = [];
for s = 1:params.N_sub
    all_interface_dofs = [all_interface_dofs; subs(s).interface_dofs];
end
all_interface_dofs = unique(all_interface_dofs);
n_interface = length(all_interface_dofs);

fprintf('Total number of DOF (Domain of Frames): %d\n', n_interface);
fprintf('Interface DOF number: '); disp(all_interface_dofs.');

% Assemble global Schur patch
Sp_global = zeros(n_interface, n_interface);
for s = 1:params.N_sub
    local_interface = subs(s).interface_dofs;
    [~, idx] = ismember(local_interface, all_interface_dofs);
    Sp_global(idx, idx) = Sp_global(idx, idx) + subs(s).S_sub;
end

fprintf('\nGlobal Schur complement matrix:\n');
fprintf('  size: %d x %d\n', size(Sp_global, 1), size(Sp_global, 2));
fprintf('  condition number κ(Sp): %.3e\n', cond(Sp_global));
fprintf('  Overall stiffness condition number κ(K): %.3e\n', cond(K));
fprintf('  ratio κ(Sp)/κ(K): %.3f\n\n', cond(Sp_global)/cond(K));

%% Visual Reference Solution
if params.plot_results
    figs = plot_results(results_ref, mesh, params, false);
    
    figure(figs(1)); hold on;
    for s = 1:params.N_sub
        x_boundary = mesh.x(subs(s).node_ids(end));
        if s < params.N_sub  
            yline_pos = ylim;
            plot([x_boundary, x_boundary]*1e3, yline_pos, '--k', ...
                 'LineWidth', 1, 'DisplayName', '');
        end
    end
    title(sprintf('Nodal displacement distribution (N_{sub}=%d)', params.N_sub));
    legend('FEM', 'Location', 'best');
    
    % Save image
    if params.save_results
        output_dir = '../results/section1/figures/';
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        
        filename = fullfile(output_dir, ...
            sprintf('displacement_with_subdomains_Nsub%d.png', params.N_sub));
        saveas(figs(1), filename);
        fprintf('Image saved: %s\n', filename);
    end
end

%% Save substructure data
if params.save_results
    data_dir = '../results/section1/data/';
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end
    
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    mat_file = fullfile(data_dir, ...
        sprintf('substructures_Nsub%d_%s.mat', params.N_sub, timestamp));
    save(mat_file, 'subs', 'params', 'mesh', 'Sp_global', ...
         'all_interface_dofs', 'results_ref');
    fprintf('Substructure data saved: %s\n', mat_file);
end

%% Generate detailed report
if params.save_results
    report_dir = '../results/section1/reports/';
    if ~exist(report_dir, 'dir')
        mkdir(report_dir);
    end
    
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    report_file = fullfile(report_dir, ...
        sprintf('Q1.3_substructures_Nsub%d_%s.txt', params.N_sub, timestamp));
    
    fid = fopen(report_file, 'w');
    fprintf(fid, '====================================\n');
    fprintf(fid, 'Question 1.3: Domain Decomposition Setup Report\n');
    fprintf(fid, '====================================\n');
    fprintf(fid, 'Date: %s\n\n', datestr(now));
    
    fprintf(fid, 'Mesh parameters:\n');
    fprintf(fid, '  Total number of units Ne = %d\n', params.Ne);
    fprintf(fid, '  Total number of nodes Nn = %d\n', params.Nn);
    fprintf(fid, '  Unit size h  = %.6e m\n\n', params.h);
    
    fprintf(fid, 'Domain decomposition parameters:\n');
    fprintf(fid, '  Number of substructures N_sub  = %d\n', params.N_sub);
    fprintf(fid, '  Number of substructure units  = %d\n', params.Ne_sub);
    fprintf(fid, '  Substructure dimensions H    = %.6e m\n', params.H);
    fprintf(fid, '  h/H ratio        = %.6f\n\n', params.h/params.H);
    
    fprintf(fid, 'Interface Information:\n');
    fprintf(fid, '  Total number of DOF (Domain of Frames) = %d\n', n_interface);
    fprintf(fid, '  Interface DOF number: '); 
    fprintf(fid, '%d ', all_interface_dofs);
    fprintf(fid, '\n\n');
    
    fprintf(fid, 'Condition number analysis:\n');
    fprintf(fid, '  κ(K)  = %.6e\n', cond(K));
    fprintf(fid, '  κ(Sp) = %.6e\n', cond(Sp_global));
    fprintf(fid, '  ratio  = %.6f\n\n', cond(Sp_global)/cond(K));
    
    fprintf(fid, '====================================\n');
    fprintf(fid, 'Detailed information on each substructure\n');
    fprintf(fid, '====================================\n\n');
    
    for s = 1:params.N_sub
        fprintf(fid, '--- Substructure #%d ---\n', s);
        fprintf(fid, 'Unit range: [%d, %d]\n', ...
                subs(s).elem_ids(1), subs(s).elem_ids(end));
        fprintf(fid, 'Node range: [%d, %d]\n', ...
                subs(s).node_ids(1), subs(s).node_ids(end));
        fprintf(fid, 'Interface DOF: [%d, %d]\n', ...
                subs(s).interface_dofs(1), subs(s).interface_dofs(2));
        fprintf(fid, 'Internal DOF number: %d\n', length(subs(s).internal_dofs));
        fprintf(fid, 'κ(K_sub): %.6e\n', cond(subs(s).K_sub));
        fprintf(fid, 'κ(S_sub): %.6e\n\n', cond(subs(s).S_sub));
    end
    
    fclose(fid);
    fprintf('The report saved.: %s\n', report_file);
end

