function subs = generate_substructures(mesh, params)
% GENERATE_SUBSTRUCTURES - Generates substructures for domain decomposition
% % Input:
% mesh - Mesh structure
% params - Parametric structure

% % Output:
% subs - Array of substructures (array of structures), each substructure contains:
% .elem_ids - List of global element numbers
% .node_ids - List of global node numbers
% .interface_dofs - Interface degrees of freedom (global numbers)
% .internal_dofs - Internal degrees of freedom (global numbers)
% .K_sub - Substructure stiffness matrix
% .S_sub - Schur complement (original)
% .R_sub - Rigid body modes
% .nloc - Number of local nodes

%


    N_sub  = params.N_sub;
    Ne_sub = params.Ne_sub;
    E = params.E;
    S = params.S;
    h = mesh.h;
    
    %% Element stiffness matrix
    ke = (E*S/h) * [ 1 -1;
                    -1  1];
    
    %% Initialize substructure array
    subs = struct('elem_ids', {}, 'node_ids', {}, ...
                  'interface_dofs', {}, 'internal_dofs', {}, ...
                  'K_sub', {}, 'S_sub', {}, 'R_sub', {}, 'nloc', {});
    
    %% Generate each substructure in a loop
    for s = 1:N_sub
        start_elem = (s-1)*Ne_sub + 1;
        end_elem   = s*Ne_sub;
        subs(s).elem_ids = start_elem : end_elem;
        
        subs(s).node_ids = (start_elem : end_elem+1).';
        subs(s).nloc = length(subs(s).node_ids);
        
        nloc = subs(s).nloc;
        idx_b = [1; nloc];              
        idx_i = (2:nloc-1).';           
        
        subs(s).interface_dofs = subs(s).node_ids(idx_b);
        subs(s).internal_dofs  = subs(s).node_ids(idx_i);
        
        K_sub = zeros(nloc, nloc);
        for k = 1:Ne_sub
            loc = [k, k+1];
            K_sub(loc, loc) = K_sub(loc, loc) + ke;
        end
        subs(s).K_sub = K_sub;
        
        if isempty(idx_i)
            S_sub = K_sub(idx_b, idx_b);
        else
            Kii = K_sub(idx_i, idx_i);
            Kbb = K_sub(idx_b, idx_b);
            Kib = K_sub(idx_i, idx_b);
            Kbi = K_sub(idx_b, idx_i);
            S_sub = Kbb - Kbi * (Kii \ Kib);
        end
        subs(s).S_sub = S_sub;
        
        R_sub = ones(nloc, 1);
        R_sub = R_sub / norm(R_sub);  
        subs(s).R_sub = R_sub;
        
    end
    
    fprintf('All substructures generated.!\n');
    fprintf('====================================\n\n');
    
end
