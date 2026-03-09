# DDM Project - Domain Decomposition Methods

A MATLAB implementation of various Domain Decomposition Methods (DDM) for 1D finite element problems.

## Code Structure

```
DDM/
├── examples/               # Run scripts (entry points)
│   ├── section1_basic_fem.m        # Section 1 – Q1.2: basic FEM, 3 BC methods
│   ├── section1_ddm_setup.m        # Section 1 – Q1.3: substructure setup
│   ├── section2_1_primal.m         # Section 2.1 – Primal methods (Schur, CG, BDD)
│   ├── section2_2_dual.m           # Section 2.2 – Dual methods (FETI)
│   ├── section2_3_mixed.m          # Section 2.3 – Mixed methods (LaTIn)
│   └── compare_all_models.m        # Global comparison of all methods
│
├── src/
│   ├── core/               # Section 1 – FEM core routines
│   │   ├── config_params.m             # Simulation parameters
│   │   ├── generate_mesh.m             # 1D mesh generation
│   │   ├── assemble_global_system.m    # Global K and F assembly
│   │   ├── apply_boundary_conditions.m # BC: elimination / penalty / Lagrange
│   │   ├── analytical_solution.m       # Exact solution u(x) = F/(ES) * x
│   │   └── postprocess.m               # Stress, strain, error computation
│   │
│   ├── ddm/                # Section 2 – Domain decomposition methods
│   │   ├── generate_substructures.m    # Substructure setup (K_sub, S_sub, R_sub)
│   │   ├── primal/                     # Section 2.1 – Primal approach
│   │   │   ├── solve_schur_direct.m    # Q2.1: direct Schur solve
│   │   │   ├── solve_schur_pcg.m       # Q2.3: distributed CG
│   │   │   ├── solve_bdd_pcg.m         # Q2.5: BDD preconditioned CG
│   │   │   ├── analyze_conditioning.m  # Q2.2: condition number study
│   │   │   └── analyze_conditioning_bdd.m  # Q2.6: preconditioned system κ
│   │   ├── dual/                       # Section 2.2 – Dual approach
│   │   │   ├── solve_dual_direct.m     # Q2.7: direct dual Schur solve
│   │   │   ├── solve_dual_feti.m       # Q2.8: FETI projected PCG
│   │   │   └── recover_rigid_modes.m   # Q2.9: rigid body mode recovery
│   │   └── mixed/                      # Section 2.3 – Mixed approach (LaTIn)
│   │       ├── solve_latin_mono.m      # Q2.10: monoscale LaTIn
│   │       ├── solve_latin_multi.m     # Q2.13: multiscale LaTIn
│   │       └── optimize_search_directions.m  # Q2.11: optimal k+, k-
│   │
│   └── utils/
│       ├── plot_results.m              # Displacement / stress plots
│       └── save_results_to_file.m      # Save results to .mat / .txt
│
└── results/                # All output figures and data (auto-generated)
    ├── section1/
    │   ├── figures/        # Displacement and error plots
    │   ├── data/           # .mat data files
    │   └── reports/        # .txt summary reports
    ├── section2_1/figures/ # Primal method plots
    ├── section2_3/figures/ # LaTIn method plots
    └── comparison/         # All-methods comparison figures
```

## Quick Start

All scripts are run from the `examples/` directory:

```matlab
cd examples
run section1_basic_fem.m       % Section 1: basic FEM
run section2_1_primal.m        % Section 2.1: primal DDM
run section2_2_dual.m          % Section 2.2: dual FETI
run section2_3_mixed.m         % Section 2.3: LaTIn
run compare_all_models.m       % Full comparison
```

## Output

All figures and results are saved automatically under `results/`. The comparison script generates two summary figures:
- `all_methods_displacement_*.png` — displacement curves for all methods
- `relative_errors_scatter_*.png` — relative error scatter plot
