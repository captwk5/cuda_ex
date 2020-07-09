#include "cuda_ex.hpp"

#define N 10000000
#define MN 32

#define RCSIZE 224
#define SIZE RCSIZE * RCSIZE

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

__global__ void MatAdd(float *A, float *B, float *C){
    // blockDim : Block Size
    // threadIdx : Block Size 에 대한 Thread Index
    // blockIdx : Block 갯수에 대한 Index
    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("BD : %d , BI : %d , TH : %d = %d\n", blockDim.x, blockIdx.x, threadIdx.x, COL);
    // printf("RBD : %d , RBI : %d , RTH : %d\n", blockDim.y, blockIdx.y, threadIdx.y);

    // float tmpSum = 0;
    // if (ROW < MN && COL < MN){
    //     // printf("%f\n", A[ROW * MN + COL]);

    //     // for (int i = 0; i < MN; i++) {
    //         // tmpSum += A[ROW * MN + i] * B[i * MN + COL];
    //     // }
    //     C[ROW * MN + COL] = A[ROW * MN + COL] + B[ROW * MN + COL];
    // }else{
    //     // printf("??");
    // }
    
    C[ROW * RCSIZE + COL] = A[ROW * RCSIZE + COL] + B[ROW * RCSIZE + COL];
}

void Cuda_Computing::test(){
    cout << "GPU COUNT : ";
    cout << m_gpu_count << endl;

    for(int i = 0; i < m_gpu_count; i++){
        cudaGetDeviceProperties( &prop, i );
        printf( "Name:  %s\n", prop.name );
        // printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        // printf( "Clock rate:  %d\n", prop.clockRate );

        printf( "Multiprocessor count:  %d\n", prop.multiProcessorCount );
        printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per mp:  %d\n", prop.regsPerBlock );

        printf( "Threads in warp:  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n", prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );

        break;
    }

    cudaEvent_t start, stop;
    float time;

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

    vector<float> hA(SIZE);
    vector<float> hB(SIZE);
    vector<float> hC_(SIZE);

    for (int i = 0; i < RCSIZE; i++){
        for (int j = 0; j < RCSIZE; j++){
            hA[i * RCSIZE + j] = 1.0f;
            hB[i * RCSIZE + j] = 2.0f;
        }
    }

    CudaArray<float> dA(SIZE);
    CudaArray<float> dB(SIZE);
    CudaArray<float> dC(SIZE);

    dA.set(&hA[0], SIZE);
    dB.set(&hB[0], SIZE);

    dim3 threadsPerBlock(MN, MN); // Block Size
    // dim3 numBlocks_(SIZE / threadsPerBlock.x, SIZE / threadsPerBlock.y);
    dim3 numBlocks_(7, 7);

    cudaEventRecord( start, 0 );
    MatAdd<<<numBlocks_, threadsPerBlock>>>(dA.getData(), dB.getData(), dC.getData());

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );

    cudaDeviceSynchronize();

    cout << "GPU Mat : " << time << endl;
    dC.get(&hC_[0], SIZE);

    cout << hC_[SIZE - 1] << endl;

    float *hC;
    hC = new float[SIZE];

    cudaEventRecord( start, 0 );
    for(int r = 0; r < RCSIZE; r++){
        for(int c = 0; c < RCSIZE; c++){
            hC[r * RCSIZE + c] = hA[r * RCSIZE + c] + hB[r * RCSIZE + c];
        }
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );

    cout << "CPU Mat : " << time << endl;

    cout << hC[SIZE - 1] << endl;

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    delete[] xc;
    delete[] yc;
    delete[] zc;

    delete[] hC;
}