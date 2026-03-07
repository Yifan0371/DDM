function params = config_params()
% CONFIG_PARAMS - Configures all simulation parameters
%
% Output:
% params - A structure containing all simulation parameters
%


    %% physical parameters
    params.L  = 100e-3;      % rod length [m] (100 mm)
    params.S  = 10e-6;       % cross-sectional area [m^2] (10 mm^2)
    params.E  = 2e5 * 1e6;   % Young modulus [Pa] (2e5 MPa = 2e11 Pa)
    params.Fd = 10;          % pull [N]
    
    %% Mesh parameters
    params.Ne = 20;          % Total number of units (can be modified)
    params.h  = params.L / params.Ne;  % Cell size (uniform grid)
    params.Nn = params.Ne + 1;         % Total number of nodes
    
    %% Domain decomposition parameters
    params.N_sub = 2;        % Number of substructures (must be divisible by Ne)
    if mod(params.Ne, params.N_sub) ~= 0
        error('The total number of units, Ne, must be divisible by the number of substructures, N_sub.');
    end
    params.Ne_sub = params.Ne / params.N_sub;  % Number of units in each substructure
    params.H = params.Ne_sub * params.h;        % Substructure feature size
    
    %% Solver parameters
    params.tol = 1e-8;       % Iterative solver convergence tolerance
    params.max_iter = 1000;  % Maximum number of iterations
    
    %% Output settings
    params.save_results = true;   % Save results?
    params.plot_results = true;   % Whether to draw
    params.output_dir = './results';  % Results Output Directory
    
end
