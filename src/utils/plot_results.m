function figs = plot_results(results, mesh, params, save_figs)
% PLOT_RESULTS - Plots displacement, stress, and other results
%
% Input:
% results - Results structure (from postprocess)
% mesh - Mesh structure
% params - Parameter structure
% save_figs - Whether to save the image (default: false)

%
% Output:
% figs - Array of graphic handles



    if nargin < 4
        save_figs = false;
    end
    
    x = mesh.x;
    
    %% Figure 1: Nodal displacement distribution
    figs(1) = figure('Name', 'Nodal displacement distribution', 'Position', [100 100 800 500]);
    plot(x*1e3, results.u*1e6, '-o', 'LineWidth', 1.5, 'MarkerSize', 6); 
    grid on;
    xlabel('Location x [mm]', 'FontSize', 12);
    ylabel('Displacement u(x) [μm]', 'FontSize', 12);
    title('Nodal displacement distribution', 'FontSize', 14);
    set(gca, 'FontSize', 11);
    
    %% Figure 2: Element stress distribution
    figs(2) = figure('Name', 'Element stress distribution', 'Position', [150 150 800 500]);
    plot(results.xmid*1e3, results.stress*1e-6, '-s', 'LineWidth', 1.5, 'MarkerSize', 6);
    hold on;
    yline(results.sigma_exact*1e-6, '--r', 'LineWidth', 2, 'DisplayName', '解析解');
    grid on;
    xlabel('Location x (Midpoint of unit) [mm]', 'FontSize', 12);
    ylabel('Normal stress σ_{xx} [MPa]', 'FontSize', 12);
    title('Element stress distribution', 'FontSize', 14);
    legend('FEM', 'Analytical solution', 'Location', 'best');
    set(gca, 'FontSize', 11);
    
    %% Figure 3: Comparison of FEM and analytical solution
    figs(3) = figure('Name', 'Comparison of FEM and analytical solution', 'Position', [200 200 800 500]);
    plot(x*1e3, results.u*1e6, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6, ...
         'DisplayName', 'FEM numerical solution');
    hold on;
    plot(x*1e3, results.u_exact*1e6, '--', 'LineWidth', 2, ...
         'DisplayName', 'Analytical solution');
    grid on;
    xlabel('Location x [mm]', 'FontSize', 12);
    ylabel('Displacement u(x) [μm]', 'FontSize', 12);
    title('Comparison of FEM and analytical solution', 'FontSize', 14);
    legend('Location', 'best');
    set(gca, 'FontSize', 11);
    
    %% Figure 4: Error Distribution
    figs(4) = figure('Name', 'Error Distribution', 'Position', [250 250 800 500]);
    error_u = abs(results.u - results.u_exact);
    semilogy(x*1e3, error_u, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    grid on;
    xlabel('Location x [mm]', 'FontSize', 12);
    ylabel('|u_{FEM} - u_{exact}| [m]', 'FontSize', 12);
    title('Displacement absolute error distribution', 'FontSize', 14);
    set(gca, 'FontSize', 11);
    
    if save_figs
        if ~exist(params.output_dir, 'dir')
            mkdir(params.output_dir);
        end
        
        fig_names = {'displacement', 'stress', 'comparison', 'error'};
        for i = 1:length(figs)
            filename = fullfile(params.output_dir, ...
                sprintf('%s_Ne%d.png', fig_names{i}, params.Ne));
            saveas(figs(i), filename);
        end
    end
    
end
