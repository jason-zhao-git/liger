# iNMF Algorithm Analysis: GPU and Parallelization Support

## Summary

The **LIGER (rliger)** package implements integrative Non-negative Matrix Factorization (iNMF) and its variants with **CPU-based parallelization support via OpenMP**, but **does not have GPU acceleration** in its current implementation.

## iNMF Variants Implemented

The codebase contains multiple iNMF algorithm variants:

1. **Standard iNMF** (`runINMF()`) - R/integration.R:218
2. **Online iNMF** (`runOnlineINMF()`) - R/integration.R:660
3. **UINMF** (Mosaic integration with unshared features) (`runUINMF()`) - R/integration.R:1155
4. **Consensus iNMF** (`runCINMF()`) - R/cINMF.R:90

All methods can be invoked through the unified `runIntegration()` wrapper function.

## Parallelization Support

### ✅ CPU Parallelization: YES

All iNMF variants support **OpenMP-based CPU parallelization** via the `nCores` parameter:

**Key Parameters:**
- `nCores`: Number of parallel tasks for computation (default: `2L`)
- Requires platform with OpenMP support
- Specified in function signatures:
  - R/integration.R:179-180
  - R/integration.R:660-661
  - R/integration.R:1155-1156
  - R/cINMF.R:45-46

**Example Usage:**
```r
# Standard iNMF with 8 cores
pbmc <- runINMF(pbmc, k = 20, lambda = 5, nCores = 8L)

# Consensus iNMF with 8 cores
pbmc <- runCINMF(pbmc, k = 20, lambda = 5, nCores = 8L, nRandomStarts = 10)

# Online iNMF with 8 cores
pbmc <- runOnlineINMF(pbmc, k = 20, lambda = 5, nCores = 8L)
```

**Implementation Details:**
- The actual parallelization is handled by the **RcppPlanc** package (version >= 2.0.0)
- RcppPlanc wraps the PLANC (Parallel Low-rank Approximation with Nonnegativity Constraints) C++ library
- Uses block coordinate descent with alternating non-negative least squares (ANLS)
- Parallelized BPPNNLS (Block Principal Pivoting Non-negative Least Squares) solver

### ❌ GPU Acceleration: NO

**Current Status:**
- The rliger package **does not implement GPU acceleration**
- The Makevars file explicitly disables OpenMP in Armadillo: `PKG_CXXFLAGS = -DARMA_DONT_USE_OPENMP` (src/Makevars:16)
- No CUDA source files (.cu, .cuh) found in the codebase

**Underlying PLANC Library:**
- The PLANC library has **limited GPU support** as of version 0.8
- GPU support is restricted to "offloading dense operations to GPU through NVBLAS"
- This means only certain dense matrix operations can use NVIDIA's NVBLAS library
- Not comprehensive GPU acceleration of the entire algorithm

**Key Findings:**
1. No CUDA code in rliger package
2. RcppPlanc package focuses on CPU optimization
3. PLANC library's GPU support is minimal and not exposed through RcppPlanc/rliger

## Performance Optimization

The package has undergone significant performance improvements:

**Version 2.0.0 Changes (NEWS.md:79):**
> "Moved iNMF (previously optimizeALS()), UINMF (previously optimizeALS(unshared = TRUE)) and online iNMF (previously online_iNMF()) implementation to new package RcppPlanc with **vastly improved performance**."

**Optimization Strategies:**
1. **RcppPlanc Integration**: High-performance C++ implementation
2. **OpenMP Parallelization**: Multi-core CPU utilization
3. **Memory Efficiency**: Supports HDF5-backed out-of-core computation
4. **Matrix Operations**: Optimized BLAS/LAPACK operations
5. **Sparse Matrix Support**: Efficient handling of sparse data

## Algorithm Details

### Objective Function

The iNMF optimization problem (R/integration.R:125-127):

```
argmin(H≥0, W≥0, V≥0) Σᵢᵈ ||Eᵢ - (W + Vᵢ)Hᵢ||²F + λΣᵢᵈ ||VᵢHᵢ||²F
```

Where:
- **E**: Input non-negative data matrices
- **H**: Cell factor loadings (cells × k)
- **W**: Shared gene loadings (genes × k)
- **V**: Dataset-specific gene loadings (genes × k)
- **λ**: Regularization parameter
- **k**: Number of factors

### Solver Implementation

The computation uses RcppPlanc functions:
- `RcppPlanc::inmf()` - Main iNMF solver (R/integration.R:438)
- `RcppPlanc::onlineINMF()` - Online learning variant (R/integration.R:902)
- `RcppPlanc::uinmf()` - Unshared features variant (R/integration.R:1285)
- `RcppPlanc::bppnnls_prod()` - Parallel NNLS solver (R/cINMF.R:481, 489, 501)

## Scalability Features

1. **HDF5 Support**: Out-of-core computation for large datasets
2. **Online Learning**: Minibatch-based processing (online iNMF)
3. **Distributed Memory**: PLANC library supports MPI (not exposed in rliger)
4. **Multi-core Processing**: OpenMP parallelization

## Platform Requirements

**Dependencies:**
- RcppPlanc (>= 2.0.0) - Required for all iNMF methods
- OpenMP support (for parallelization)
- BLAS/LAPACK libraries
- Armadillo C++ library (via RcppArmadillo)

**Tested Platforms:**
- Linux variants with Intel processors (mentioned in PLANC documentation)
- Tested on NERSC, OLCF, and PACE clusters (for PLANC)

## Recommendations for GPU Acceleration

If GPU acceleration is needed, potential approaches include:

1. **Extend RcppPlanc**: Add CUDA kernels for matrix operations
2. **Use NVBLAS**: Configure PLANC to use NVBLAS for dense operations (requires PLANC library modifications)
3. **Integrate GPU libraries**:
   - cuBLAS for dense linear algebra
   - cuSPARSE for sparse matrix operations
   - Custom CUDA kernels for ANLS iterations
4. **Alternative**: Use GPU-accelerated NMF libraries and adapt for iNMF
   - PyTorch/TensorFlow implementations
   - NVIDIA's cuML library

## Code References

### Main Implementation Files:
- `R/integration.R` - All iNMF variants (runINMF, runOnlineINMF, runUINMF)
- `R/cINMF.R` - Consensus iNMF implementation
- `src/Makevars` - Build configuration (OpenMP settings)

### Key Functions with nCores Parameter:
- `runINMF.liger()` - Line 240
- `runOnlineINMF.liger()` - Line 740
- `runUINMF.liger()` - Line 1213
- `runCINMF.liger()` - Line 114
- `.runINMF.list()` - Line 380 (internal)
- `inmf_solveH()` - Line 477 (consensus iNMF helper)
- `inmf_solveV()` - Line 485 (consensus iNMF helper)
- `inmf_solveW()` - Line 493 (consensus iNMF helper)

## Conclusion

**Current Capabilities:**
- ✅ Multi-core CPU parallelization via OpenMP
- ✅ Highly optimized C++ implementation
- ✅ Memory-efficient HDF5 support
- ✅ Multiple iNMF algorithm variants

**Limitations:**
- ❌ No GPU/CUDA acceleration
- ❌ Limited to single-node processing (no MPI in rliger)

**Bottom Line:** The LIGER package provides excellent CPU-based parallel iNMF implementations but does not currently support GPU acceleration. For very large-scale datasets, the online iNMF method with HDF5 backing and multi-core processing is the recommended approach.
