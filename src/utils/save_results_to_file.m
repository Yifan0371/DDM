function save_results_to_file(results, mesh, params, method_name)
% SAVE_RESULTS_TO_FILE - Save results to a file

% Input:
% results - Result structure
% mesh - Mesh structure
% params - Parameter structure
% method_name - Method name string (used for file naming)
%

    if ~exist(params.output_dir, 'dir')
        mkdir(params.output_dir);
    end
    
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    base_name = sprintf('%s_Ne%d_%s', method_name, params.Ne, timestamp);
    
    mat_file = fullfile(params.output_dir, [base_name, '.mat']);
    save(mat_file, 'results', 'mesh', 'params');
    
    txt_file = fullfile(params.output_dir, [base_name, '.txt']);
    fid = fopen(txt_file, 'w');
    
    fprintf(fid, '====================================\n');
    fprintf(fid, 'FEM simulation results\n');
    fprintf(fid, '====================================\n');
    fprintf(fid, 'method: %s\n', method_name);
    fprintf(fid, 'date: %s\n\n', datestr(now));
    
    fprintf(fid, 'Parameter settings:\n');
    fprintf(fid, '  L  = %.6e m\n', params.L);
    fprintf(fid, '  S  = %.6e m^2\n', params.S);
    fprintf(fid, '  E  = %.6e Pa\n', params.E);
    fprintf(fid, '  Fd = %.6e N\n', params.Fd);
    fprintf(fid, '  Ne = %d\n', params.Ne);
    fprintf(fid, '  h  = %.6e m\n\n', params.h);
    
    fprintf(fid, 'Calculation results:\n');
    fprintf(fid, '  u(L) FEM     = %.12e m\n', results.u(end));
    fprintf(fid, '  u(L) Analytical solution  = %.12e m\n', results.u_exact(end));
    fprintf(fid, '  absolute error   = %.12e m\n', abs(results.u(end) - results.u_exact(end)));
    fprintf(fid, '  Support reaction force     = %.12e N\n', results.reaction);
    fprintf(fid, '  σ FEM mean    = %.12e Pa\n', mean(results.stress));
    fprintf(fid, '  σ Analytical solution     = %.12e Pa\n\n', results.sigma_exact);
    
    fprintf(fid, 'relative error:\n');
    fprintf(fid, '  L1 norm      = %.6e\n', results.error.u_L1);
    fprintf(fid, '  L2 norm      = %.6e\n', results.error.u_L2);
    fprintf(fid, '  L∞ norm      = %.6e\n\n', results.error.u_Linf);
    
    fprintf(fid, '====================================\n');
    fprintf(fid, 'Detailed data (nodal displacements):\n');
    fprintf(fid, '====================================\n');
    fprintf(fid, '%10s %20s %20s %20s\n', 'Node', 'x [m]', 'u_FEM [m]', 'u_exact [m]');
    fprintf(fid, '------------------------------------------------------------\n');
    for i = 1:mesh.Nn
        fprintf(fid, '%10d %20.12e %20.12e %20.12e\n', ...
                i, mesh.x(i), results.u(i), results.u_exact(i));
    end
    
    fclose(fid);
    
end
