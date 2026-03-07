function mesh = generate_mesh(params)
% GENERATE_MESH - Generates the finite element mesh for 1D bars
%
% Input:
% params - Parameter structure (from config_params)
%
% Output:
% mesh - Mesh structure, containing:
% .x - Node coordinates [Nn x 1]
% .conn - Element connections [Ne x 2]
% .Ne - Number of elements
% .Nn - Number of nodes
% .h - Element size
%


    %% Extract parameters
    L  = params.L;
    Ne = params.Ne;
    Nn = params.Nn;
    h  = params.h;
    
    %% Generate node coordinates (column vector)
    mesh.x = linspace(0, L, Nn).';
    
    %% Generate unit connection relationship
    % The e-th unit connection node [e, e+1]
    mesh.conn = [(1:Ne).', (2:Nn).'];
    
    %% Save grid information
    mesh.Ne = Ne;
    mesh.Nn = Nn;
    mesh.h  = h;
    
    %% Display grid information
    fprintf('Mesh generation complete:\n');
    fprintf('  Number of nodes: %d\n', mesh.Nn);
    fprintf('  Number of units: %d\n', mesh.Ne);
    fprintf('  Unit size: %.6e m\n', mesh.h);
    fprintf('  coordinate range: [%.6e, %.6e] m\n\n', min(mesh.x), max(mesh.x));
    
end
