#include "cuda_ex.hpp"

#define N 10000000
#define MN 32

#define RCSIZE 480
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

__global__ void cudaConvolution(float *A, float *B, float *C, float *D, float *E, float *F, float* kernel){
    // blockDim : Block Size // 32
    // threadIdx : Block Size 에 대한 Thread Index (0 ~ 32)
    // blockIdx : Block 갯수에 대한 Index // numberof block
    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;

    if ((ROW * RCSIZE + COL) % (RCSIZE) != 0 && (ROW * RCSIZE + COL) % (RCSIZE - 1) != 0){
        D[ROW * RCSIZE + COL] = (
            A[((ROW * RCSIZE + COL) - RCSIZE) - 1] * kernel[0] + A[((ROW * RCSIZE + COL) - RCSIZE)] * kernel[1] + A[((ROW * RCSIZE + COL) - RCSIZE) + 1] * kernel[2] + 
            A[(ROW * RCSIZE + COL) - 1] * kernel[3] + A[(ROW * RCSIZE + COL)] * kernel[4] + A[(ROW * RCSIZE + COL) + 1] * kernel[5] + 
            A[((ROW * RCSIZE + COL) + RCSIZE) - 1] * kernel[6] + A[((ROW * RCSIZE + COL) + RCSIZE)] * kernel[7] + A[((ROW * RCSIZE + COL) + RCSIZE) + 1] * kernel[8]
        );

        E[ROW * RCSIZE + COL] = (
            B[((ROW * RCSIZE + COL) - RCSIZE) - 1] * kernel[0] + B[((ROW * RCSIZE + COL) - RCSIZE)] * kernel[1] + B[((ROW * RCSIZE + COL) - RCSIZE) + 1] * kernel[2] + 
            B[(ROW * RCSIZE + COL) - 1] * kernel[3] + B[(ROW * RCSIZE + COL)] * kernel[4] + B[(ROW * RCSIZE + COL) + 1] * kernel[5] + 
            B[((ROW * RCSIZE + COL) + RCSIZE) - 1] * kernel[6] + B[((ROW * RCSIZE + COL) + RCSIZE)] * kernel[7] + B[((ROW * RCSIZE + COL) + RCSIZE) + 1] * kernel[8]
        );

        F[ROW * RCSIZE + COL] = (
            C[((ROW * RCSIZE + COL) - RCSIZE) - 1] * kernel[0] + C[((ROW * RCSIZE + COL) - RCSIZE)] * kernel[1] + C[((ROW * RCSIZE + COL) - RCSIZE) + 1] * kernel[2] + 
            C[(ROW * RCSIZE + COL) - 1] * kernel[3] + C[(ROW * RCSIZE + COL)] * kernel[4] + C[(ROW * RCSIZE + COL) + 1] * kernel[5] + 
            C[((ROW * RCSIZE + COL) + RCSIZE) - 1] * kernel[6] + C[((ROW * RCSIZE + COL) + RCSIZE)] * kernel[7] + C[((ROW * RCSIZE + COL) + RCSIZE) + 1] * kernel[8]
        );
    }
    // C[ROW * RCSIZE + COL] = A[ROW * RCSIZE + COL] + B[ROW * RCSIZE + COL];
}

void Cuda_Computing::cpuConvolution(int*** input, int*** output, int rows, int cols, int channels, vector<float> kernel){
    int stride = 1;
    // int filter = 3;
    int start_idx = int(3 / 2) * -1;
    int end_idx = start_idx * -1;
    float filter[3][3];
    filter[0][0] = kernel[0];
    filter[0][1] = kernel[1];
    filter[0][2] = kernel[2];
    filter[1][0] = kernel[3];
    filter[1][1] = kernel[4];
    filter[1][2] = kernel[5];
    filter[2][0] = kernel[6];
    filter[2][1] = kernel[7];
    filter[2][2] = kernel[8];

    for(int ch = 0; ch < channels; ch++){
        for(int r = 0; r < rows; r = r + stride){
            for(int c = 0; c < cols; c = c + stride){
                int value = 0;
                
                for (int fr = start_idx; fr <= end_idx; fr++) {
                    for (int fc = start_idx; fc <= end_idx; fc++) {
                        if ((r + fr >= 0 && c + fc >= 0) && (r + fr < rows && c + fc < cols)) {
                            value += (int)filter[fr + end_idx][fc + end_idx] * input[r + fr][c + fc][ch];
                        }
                    }
                }
                output[r][c][ch] = value;
            }
        }
    }
}

void Cuda_Computing::test(){
    cout << "GPU COUNT : ";
    cout << m_gpu_count << endl;

    /* GPU Device Information */
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

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*
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
    */

    vector<float> sharpen_kernel(9);
    sharpen_kernel[0] = 0.0f;
    sharpen_kernel[1] = -1.0f;
    sharpen_kernel[2] = 0.0f;
    sharpen_kernel[3] = -1.0f;
    sharpen_kernel[4] = 5.0f;
    sharpen_kernel[5] = -1.0f;
    sharpen_kernel[6] = 0.0f;
    sharpen_kernel[7] = -1.0f;
    sharpen_kernel[8] = 0.0f;


    Mat img = imread("test_img.jpg");

    vector<float> ch_1(SIZE);
    vector<float> ch_2(SIZE);
    vector<float> ch_3(SIZE);

    vector<float> ch_4(SIZE);
    vector<float> ch_5(SIZE);
    vector<float> ch_6(SIZE);

    for (int r = 0; r < RCSIZE; r++){
        Vec3b* ptr = img.ptr<Vec3b>(r);
        for (int c = 0; c < RCSIZE; c++){
            ch_1[r * RCSIZE + c] = ptr[c][0];
            ch_2[r * RCSIZE + c] = ptr[c][1];
            ch_3[r * RCSIZE + c] = ptr[c][2];
        }
    }

    CudaArray<float> d_kernel(9);

    CudaArray<float> d_ch_1(SIZE);
    CudaArray<float> d_ch_2(SIZE);
    CudaArray<float> d_ch_3(SIZE);

    CudaArray<float> d_ch_4(SIZE);
    CudaArray<float> d_ch_5(SIZE);
    CudaArray<float> d_ch_6(SIZE);

    d_kernel.set(&sharpen_kernel[0], SIZE);

    d_ch_1.set(&ch_1[0], SIZE);
    d_ch_2.set(&ch_2[0], SIZE);
    d_ch_3.set(&ch_3[0], SIZE);

    dim3 threadsPerBlock(MN, MN); // Block Size
    dim3 numBlocks_(RCSIZE / threadsPerBlock.x, RCSIZE / threadsPerBlock.y);
    // dim3 numBlocks_(16, 16);

    cudaEventRecord( start, 0 );

    cudaConvolution<<<numBlocks_, threadsPerBlock>>>(d_ch_1.getData(), d_ch_2.getData(), d_ch_3.getData(),
                                                    d_ch_4.getData(), d_ch_5.getData(), d_ch_6.getData(), d_kernel.getData());


    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    cudaDeviceSynchronize();

    d_ch_4.get(&ch_4[0], SIZE);
    d_ch_5.get(&ch_5[0], SIZE);
    d_ch_6.get(&ch_6[0], SIZE);

    for(int r = 0; r < RCSIZE; r++){
        for(int c = 0; c < RCSIZE; c++){
            img.at<Vec3b>(r, c)[0] = ch_4[r * RCSIZE + c];
            img.at<Vec3b>(r, c)[1] = ch_5[r * RCSIZE + c];
            img.at<Vec3b>(r, c)[2] = ch_6[r * RCSIZE + c];
        }
    }

    imwrite("result_gpu_conv.jpg", img);

    cout << "GPU Time : " << time << endl;


    Mat img_ = imread("test_img.jpg");

    int*** input = new int**[RCSIZE];

    for(int r = 0; r < RCSIZE; r++){
        Vec3b* ptr = img_.ptr<Vec3b>(r);
        *(input + r) = new int*[RCSIZE];
        for(int c = 0; c < RCSIZE; c++){
            *(*(input + r) + c) = new int[3];
            input[r][c][0] = ptr[c][0];
            input[r][c][1] = ptr[c][1];
            input[r][c][2] = ptr[c][2];
        }
    }

    int*** output = new int**[RCSIZE];
    for(int r = 0; r < RCSIZE; r++){
        *(output + r) = new int*[RCSIZE];
        for(int c = 0; c < RCSIZE; c++){
            *(*(output + r) + c) = new int[3];
        }
    }

    cudaEventRecord( start, 0 );

    cpuConvolution(input, output, RCSIZE, RCSIZE, 3, sharpen_kernel);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );

    cout << "CPU Time : " << time << endl;
    
    for(int r = 0; r < RCSIZE; r++){
        for(int c = 0; c < RCSIZE; c++){
            img_.at<Vec3b>(r, c)[0] = output[r][c][0];
            img_.at<Vec3b>(r, c)[1] = output[r][c][1];
            img_.at<Vec3b>(r, c)[2] = output[r][c][2];
        }
    }

    imwrite("result_cpu_conv.jpg", img_);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    // cudaFree(x);
    // cudaFree(y);
    // cudaFree(z);

    // delete[] xc;
    // delete[] yc;
    // delete[] zc;

    for(int i = 0; i < RCSIZE; i++){
        for(int j = 0; j < RCSIZE; j++){
            delete[] *(*(input + i) + j);
        }
    } 
    for(int i = 0; i < RCSIZE; i++){
        delete[] *(input + i);
    }
    delete[] input;

    for(int i = 0; i < RCSIZE; i++){
        for(int j = 0; j < RCSIZE; j++){
            delete[] *(*(output + i) + j);
        }
    } 
    for(int i = 0; i < RCSIZE; i++){
        delete[] *(output + i);
    }
    delete[] output;
}