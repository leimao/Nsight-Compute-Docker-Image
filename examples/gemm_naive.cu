#include <iostream>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t err, const char* const func, const char* const file,
                const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_cuda_last(__FILE__, __LINE__)
void check_cuda_last(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Non-coalesced read and write from global memory.
template <typename T>
__global__ void gemm_non_coalesced(size_t m, size_t n, size_t k, T alpha,
                                   T const* A, size_t lda, T const* B,
                                   size_t ldb, T beta, T* C, size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx].
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_non_coalesced(size_t m, size_t n, size_t k,
                                      T const* alpha, T const* A, size_t lda,
                                      T const* B, size_t ldb, T const* beta,
                                      T* C, size_t ldc, cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_non_coalesced<T><<<grid_dim, block_dim, 0U, stream>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Coalesced read and write from global memory.
template <typename T>
__global__ void gemm_coalesced(size_t m, size_t n, size_t k, T alpha,
                               T const* A, size_t lda, T const* B, size_t ldb,
                               T beta, T* C, size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx].
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_coalesced(size_t m, size_t n, size_t k, T const* alpha,
                                  T const* A, size_t lda, T const* B,
                                  size_t ldb, T const* beta, T* C, size_t ldc,
                                  cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_coalesced<T><<<grid_dim, block_dim, 0U, stream>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T>
void gemm_cpu(size_t m, size_t n, size_t k, T alpha, T const* A, size_t lda,
              T const* B, size_t ldb, T beta, T* C, size_t ldc)
{
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            T sum{static_cast<T>(0)};
            for (size_t k_idx{0U}; k_idx < k; ++k_idx)
            {
                sum += A[i * lda + k_idx] * B[k_idx * ldb + j];
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

template <typename T>
void verify_outputs(size_t m, size_t n, size_t ldc, T const* C, T const* C_ref,
                    T abs_error_tol)
{
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            T const abs_error{std::abs(C[i * ldc + j] - C_ref[i * ldc + j])};
            if (abs_error > abs_error_tol)
            {
                std::cerr << "Error: i = " << i << ", j = " << j
                          << ", abs_error = " << abs_error << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

int main()
{
    size_t const m{1024U};
    size_t const n{1024U};
    size_t const k{1024U};
    float const alpha{1.0f};
    float const beta{0.0f};
    float const abs_error_tol{1e-5f};

    size_t const lda{k};
    size_t const ldb{n};
    size_t const ldc{n};

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // Allocate memory on the host.
    float* A_host{nullptr};
    float* B_host{nullptr};
    float* C_host{nullptr};
    float* C_host_from_device{nullptr};
    CHECK_CUDA_ERROR(cudaMallocHost(&A_host, m * lda * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&B_host, k * ldb * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host, m * ldc * sizeof(float)));
    CHECK_CUDA_ERROR(
        cudaMallocHost(&C_host_from_device, m * ldc * sizeof(float)));

    // Initialize A and B.
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < k; ++j)
        {
            A_host[i * lda + j] = static_cast<float>(i + j);
        }
    }
    for (size_t i{0U}; i < k; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            B_host[i * ldb + j] = static_cast<float>(i + j);
        }
    }

    // Allocate memory on the device.
    float* A_device{nullptr};
    float* B_device{nullptr};
    float* C_device{nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&A_device, m * lda * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&B_device, k * ldb * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_device, m * ldc * sizeof(float)));

    // Copy A and B to the device.
    CHECK_CUDA_ERROR(cudaMemcpy(A_device, A_host, m * lda * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_device, B_host, k * ldb * sizeof(float),
                                cudaMemcpyHostToDevice));

    // Run the CPU version.
    gemm_cpu(m, n, k, alpha, A_host, lda, B_host, ldb, beta, C_host, ldc);

    // Launch the kernel.
    launch_gemm_kernel_non_coalesced(m, n, k, &alpha, A_device, lda, B_device,
                                     ldb, &beta, C_device, ldc, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy C from the device.
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_from_device, C_device,
                                m * ldc * sizeof(float),
                                cudaMemcpyDeviceToHost));

    // Compare the results.
    verify_outputs(m, n, ldc, C_host_from_device, C_host, abs_error_tol);

    // Launch the kernel.
    launch_gemm_kernel_coalesced(m, n, k, &alpha, A_device, lda, B_device, ldb,
                                 &beta, C_device, ldc, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy C from the device.
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_from_device, C_device,
                                m * ldc * sizeof(float),
                                cudaMemcpyDeviceToHost));

    // Compare the results.
    verify_outputs(m, n, ldc, C_host_from_device, C_host, abs_error_tol);

    // Free the memory.
    CHECK_CUDA_ERROR(cudaFree(A_device));
    CHECK_CUDA_ERROR(cudaFree(B_device));
    CHECK_CUDA_ERROR(cudaFree(C_device));
    CHECK_CUDA_ERROR(cudaFreeHost(A_host));
    CHECK_CUDA_ERROR(cudaFreeHost(B_host));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host_from_device));
}