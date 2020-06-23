#include "cuda_ex.hpp"

#define N 10000000

__global__ void add_gpu(int n, float *x, float *y, float *z){
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n; i += stride)
        z[i] = x[i] + y[i];
}

__host__ void add_cpu(int n, float *x, float *y, float *z){
    for(int i = 0; i < n; i++){
        z[i] = x[i] + y[i];
    }
}

__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

void Cuda_Computing::test(){
    cout << "GPU COUNT : ";
    cout << m_gpu_count << endl;

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    float *x, *y, *z;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&z, N * sizeof(float));

    float *xc, *yc, *zc;
    xc = new float[N];
    yc = new float[N];
    zc = new float[N];

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;

        xc[i] = 1.0f;
        yc[i] = 2.0f;
    }

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord( start, 0 );

    add_gpu<<<numBlocks, blockSize>>>(N, x, y, z);
    
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );

    cudaDeviceSynchronize();

    cout << "GPU : " << time << endl;
    cout << z[0] << " - " << z[N - 1] << endl;

    cudaEventRecord( start, 0 );

    add_cpu(N, xc, yc, zc);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );

    cout << "CPU : " << time << endl;
    cout << zc[0] << " - " << zc[N - 1] << endl;

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    int mn = 16;
    int SIZE = mn * mn;
    vector<float> hA(SIZE);
    vector<float> hB(SIZE);

    for (int i = 0; i < mn; i++){
        for (int j = 0; j < mn; j++){
            hA[i * mn + j] = 1.0f;
            hB[i * mn + j] = 2.0f;
        }
    }

    CudaArray<float> dA(SIZE);
    CudaArray<float> dB(SIZE);

    dA.set(&hA[0], SIZE);
    dB.set(&hB[0], SIZE);

    dim3 threadsPerBlock(mn, mn);
    dim3 numBlocksMat(N / threadsPerBlock.x, N / threadsPerBlock.y);
    // MatAdd<<<numBlocksMat, threadsPerBlock>>>(A, B, C);

    float *hC;
    hC = new float[SIZE];
    for(int r = 0; r < mn; r++){
        for(int c = 0; c < mn; c++){
            hC[r * mn + c] = hA[r * mn + c] + hB[r * mn + c];
        }
    }

    cout << hC[0] << endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    delete[] xc;
    delete[] yc;
    delete[] zc;

    delete[] hC;
}