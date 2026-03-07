function [K, F] = assemble_global_system(mesh, params)
% ASSEMBLE_GLOBAL_SYSTEM - Assemble the overall stiffness matrix and load vector
%
% Input:
% mesh - Mesh structure (from generate_mesh)
% params - Parameter structure (from config_params)

% Output:
% K - Global stiffness matrix [Nn x Nn]
% F - Global load vector [Nn x 1]


    E  = params.E;
    S  = params.S;
    Fd = params.Fd;
    h  = mesh.h;
    Ne = mesh.Ne;
    Nn = mesh.Nn;
    conn = mesh.conn;
    
    K = zeros(Nn, Nn);
    F = zeros(Nn, 1);
    
    %% Calculate the element stiffness matrix
    ke = (E*S/h) * [ 1 -1;
                    -1  1];
    
    %% Assemble the overall stiffness matrix
    for e = 1:Ne
        nodes = conn(e,:);
        K(nodes, nodes) = K(nodes, nodes) + ke;
    end
    
    %% Apply an external load (at the last node).
    F(end) = Fd;
    
    fprintf('The overall system assembly is complete.:\n');
    fprintf('  Stiffness matrix size: %d x %d\n', size(K,1), size(K,2));
    fprintf('  condition number: %.3e\n', cond(K));
    fprintf('  External load node: %d, size: %.3e N\n\n', Nn, Fd);
    
end
