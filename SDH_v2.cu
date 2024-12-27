/*
    Programmer: Ahmad Sibai
    Project 2: Spacial Distance Histogram Computation with Efficiency (CUDA)
    To compile: nvcc SDH_v2.c -o SDH_v2
*/

/* Testing
Atom Count: 10000, Bucket Range: 50.000000, Block Size: 128
Time to generate: 19.16787 ms

Atom Count: 10000, Bucket Range: 100.000000, Block Size: 128
Time to generate: 16.15504 ms

Atom Count: 10000, Bucket Range: 100.000000, Block Size: 256
Time to generate: 20.36294 ms

Atom Count: 100000, Bucket Range: 100.000000, Block Size: 128
Time to generate: 1118.93420 ms
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>               //header for cudaMalloc, cudaFree, cudaMemcpy
#include <device_launch_parameters.h>   //header for blockIdx, blockDim, threadIdx
// cuda_runtime.h and device_launch_parameters.h are not needed because of nvcc, i'm keeping them for practice

#define BOX_SIZE 23000 // size of the 3d data box

typedef struct atomdesc // struct for single atom
{
    double x_pos;
    double y_pos;
    double z_pos;
} atom;

typedef struct hist_entry // struct for histogram bucket
{
    unsigned long long d_cnt; // need a long long type as the count might be huge
} bucket;

// Host(CPU) variables
bucket *histogram; // pointer to array of all buckets in the histogram
long long PDH_acnt; // total number of atoms (data points)
int num_buckets; // total number of buckets in the histogram
double PDH_res; // range of each bucket in histogram
atom *atom_list; // list of all atoms

// Device(GPU) variables
bucket *GPUhistogram; // device memory for histogram buckets (GPU)
bucket *GPUhistogramOnHost; // host memory for histogram buckets (GPU)
double *GPUpositionX, *GPUpositionY, *GPUpositionZ; // pointer to 3 array of all atoms each of which contains one of the values (x, y, z)

int PDH_baseline() // brute-force SDH solution in a single CPU thread on Host(CPU)
{
    int i, j, h_pos;
    for (i = 0; i < PDH_acnt; i++)
    {
        double x1 = atom_list[i].x_pos;
        double y1 = atom_list[i].y_pos;
        double z1 = atom_list[i].z_pos;
        for (j = i + 1; j < PDH_acnt; j++)
        {
            double x2 = atom_list[j].x_pos;
            double y2 = atom_list[j].y_pos;
            double z2 = atom_list[j].z_pos;
            // calculate distance
            double dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
            h_pos = (int)(dist / PDH_res);
            histogram[h_pos].d_cnt++;
        }
    }
    return 0;
}

// CUDA kernel with advanced output privatization
__global__ void p2p_distancekernel(double *GPUpositionX, double *GPUpositionY, double *GPUpositionZ, bucket *histogram, long long PDH_acnt, double PDH_res, int num_buckets, int numCopies) {
    extern __shared__ unsigned long long sharedHist[];  // shared memory for private histograms
    int tid = threadIdx.x;                             // thread ID within the block
    int blockSize = blockDim.x;                        // block size
    int blockId = blockIdx.x;                          // block ID
    int laneID = tid & 0x1f;                           // lane ID within a warp (0 to 31)
    int privateCopy = laneID % numCopies;              // assigns each thread to a specific private copy

    // initialize private histogram copy to zero
    for (int i = tid; i < num_buckets * numCopies; i += blockSize) {
        sharedHist[i] = 0;
    }
    __syncthreads();
    // load own position (in registers)
    double ownX = GPUpositionX[blockId * blockSize + tid];
    double ownY = GPUpositionY[blockId * blockSize + tid];
    double ownZ = GPUpositionZ[blockId * blockSize + tid];

    // inter-block calculations
    for (int i = blockId + 1; i < gridDim.x; i++) {
        double *R_X = &GPUpositionX[i * blockSize];
        double *R_Y = &GPUpositionY[i * blockSize];
        double *R_Z = &GPUpositionZ[i * blockSize];
        for (int j = 0; j < blockSize; j++) {
            if ((i * blockSize + j) >= PDH_acnt) break; // boundary check
            double distance = sqrt((ownX - R_X[j]) * (ownX - R_X[j]) +
                                   (ownY - R_Y[j]) * (ownY - R_Y[j]) +
                                   (ownZ - R_Z[j]) * (ownZ - R_Z[j]));
            int h_pos = (int)(distance / PDH_res);
            if (h_pos < num_buckets) {
                atomicAdd(&sharedHist[privateCopy * num_buckets + h_pos], 1); // atomic write in shared memory
            }
        }
        __syncthreads();
    }

    // intra-block calculations
    for (int i = tid + 1; i < blockSize; i++) {
        if ((blockId * blockSize + i) >= PDH_acnt) break;
        double distance = sqrt((ownX - GPUpositionX[blockId * blockSize + i]) * (ownX - GPUpositionX[blockId * blockSize + i]) +
                               (ownY - GPUpositionY[blockId * blockSize + i]) * (ownY - GPUpositionY[blockId * blockSize + i]) +
                               (ownZ - GPUpositionZ[blockId * blockSize + i]) * (ownZ - GPUpositionZ[blockId * blockSize + i]));
        int h_pos = (int)(distance / PDH_res);
        if (h_pos < num_buckets) {
            atomicAdd(&sharedHist[privateCopy * num_buckets + h_pos], 1); // atomic write in shared memory
        }
    }
    __syncthreads();
    // reduction of private copies to global histogram
    for (int i = tid; i < num_buckets; i += blockSize) {
        unsigned long long sum = 0;
        for (int j = 0; j < numCopies; j++) {
            sum += sharedHist[j * num_buckets + i];
        }
        atomicAdd(&histogram[i].d_cnt, sum);
    }
}

// GPU calculation function
void PDH_baselineGPU(int blockSize) {
    int gridSize = (PDH_acnt + blockSize - 1) / blockSize;
    // calculate the number of copies based on block size with a warp size of 32
    int numCopies = (blockSize + 31) / 32; // round up to ensure full warps

    // allocate GPU memory
    cudaMalloc(&GPUhistogram, sizeof(bucket) * num_buckets);
    cudaMalloc(&GPUpositionX, sizeof(double) * PDH_acnt);
    cudaMalloc(&GPUpositionY, sizeof(double) * PDH_acnt);
    cudaMalloc(&GPUpositionZ, sizeof(double) * PDH_acnt);
    // allocate host memory
    double *hostPositionX = (double *)malloc(sizeof(double) * PDH_acnt);
    double *hostPositionY = (double *)malloc(sizeof(double) * PDH_acnt);
    double *hostPositionZ = (double *)malloc(sizeof(double) * PDH_acnt);
    // initalize host position arrays with atom positions
    for (int i = 0; i < PDH_acnt; i++) {
        hostPositionX[i] = atom_list[i].x_pos;
        hostPositionY[i] = atom_list[i].y_pos;
        hostPositionZ[i] = atom_list[i].z_pos;
    }
    //transfer atom positions from host to device memory
    cudaMemcpy(GPUpositionX, hostPositionX, sizeof(double) * PDH_acnt, cudaMemcpyHostToDevice);
    cudaMemcpy(GPUpositionY, hostPositionY, sizeof(double) * PDH_acnt, cudaMemcpyHostToDevice);
    cudaMemcpy(GPUpositionZ, hostPositionZ, sizeof(double) * PDH_acnt, cudaMemcpyHostToDevice);
    // initialize GPU histogram buckets to zero
    cudaMemset(GPUhistogram, 0, sizeof(bucket) * num_buckets);

    // to accurately record running time of kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // launch kernel with shared mem allocation
    size_t sharedMemSize = num_buckets * numCopies * sizeof(unsigned long long);
    p2p_distancekernel<<<gridSize, blockSize, sharedMemSize>>>(GPUpositionX, GPUpositionY, GPUpositionZ, GPUhistogram, PDH_acnt, PDH_res, num_buckets, numCopies);
    
    // to accurately record running time of kernel
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time to generate: %0.5f ms\n", elapsedTime );
    // transfer histogram from device to host
    cudaMemcpy(GPUhistogramOnHost, GPUhistogram, sizeof(bucket) * num_buckets, cudaMemcpyDeviceToHost);
    // free memory
    cudaFree(GPUhistogram);
    cudaFree(GPUpositionX);
    cudaFree(GPUpositionY);
    cudaFree(GPUpositionZ);
    free(hostPositionX);
    free(hostPositionY);
    free(hostPositionZ);
}

// print the counts in all buckets of the histogram
void output_histogram(bucket *histogram, const char *label)
{
    int i;
    long long total_cnt = 0;
    for (i = 0; i < num_buckets; i++)
    {
        if (i % 5 == 0) // we print 5 buckets in a row 
            printf("\n%02d: ", i);
        printf("%15lld ", histogram[i].d_cnt);
        total_cnt += histogram[i].d_cnt;
        // we also want to make sure the total distance count is correct
        if (i < num_buckets - 1)
            printf("| ");
    }
    if (strcmp(label, "Compute Difference") != 0)
    {
        printf("\nTotal number of distances calculated: %lld for %s\n", total_cnt, label);
    }
}

// compute differences between histogram (CPU) and GPUhistogramOnHost
int computeDifference(bucket *histogram, bucket *GPUhistogramOnHost, bucket *differenceHistogram, int num_buckets) {
    int hasDifference = 0; // used to track if there's any significant difference
    for (int i = 0; i < num_buckets; i++) {
        differenceHistogram[i].d_cnt = llabs(histogram[i].d_cnt - GPUhistogramOnHost[i].d_cnt); // calculate the absolute difference
        if (differenceHistogram[i].d_cnt > 1) { // check if the difference is greater than 1
            hasDifference = 1;
        }
    }
    return hasDifference; // return 0 if no differences, or 1 if any difference exists
}

int main(int argc, char **argv)
{
    if (argc < 4) // check command line arguments are provided
    {
        printf("Usage: %s <atomCount> <bucketRange> <blockSize>\n", argv[0]);
        return 1;
    }
    PDH_acnt = atoi(argv[1]); // number of atoms
    PDH_res = atof(argv[2]);  // range of each bucket
    int blockSize = atoi(argv[3]); // blocksize
    printf("Atom Count: %lld, Bucket Range: %f, Block Size: %d\n", PDH_acnt, PDH_res, blockSize);
    num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
    histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
    atom_list = (atom *)malloc(sizeof(atom) * PDH_acnt);
    GPUhistogramOnHost = (bucket *)malloc(sizeof(bucket) * num_buckets); // host memory to store GPU histogram
    bucket *differenceHistogram = (bucket *)malloc(sizeof(bucket) * num_buckets); // host memory to store GPU histogram
    // allocate memory on Host(CPU)
    srand(1);
    // generate data following a uniform distribution
    for (int i = 0; i < PDH_acnt; i++)
    {
        atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
        atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
        atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
    }
    // call Host(CPU) single thread version to compute the histogram
    PDH_baseline();
    // call Device(GPU) multi thread version to compute histogram
    PDH_baselineGPU(blockSize);
    // print out the Device(GPU) histogram
    output_histogram(GPUhistogramOnHost, "GPU");
    // compute difference
    int isdifferent = computeDifference(histogram, GPUhistogramOnHost, differenceHistogram, num_buckets);
    if(isdifferent == 0) // if there are no differences
    {
        printf("The differenceHistogram was empty, meaning the CPU histogram and GPU histogram were identical!\n");
    }
    else // if there are differences then we print out the computeDifference histogram to show the buckets where their are differences
    {
        printf("The CPU and GPU histograms had differences in their bucket values. The following printed differenceHistogram shows these differences:\n");
        output_histogram(differenceHistogram, "Compute Difference");
    }
    free(atom_list); //free all dynamically allocated memory
    free(histogram);
    free(GPUhistogramOnHost);
    free(differenceHistogram);
    return 0;
}
