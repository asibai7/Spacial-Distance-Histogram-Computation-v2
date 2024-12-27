# Spacial-Distance-Histogram-Computation-v2

This project is the 2nd version of the Spacial Distance Histogram Computation program, representing an optimized implementation of the CUDA-accelerated computation of a histogram categorizing distances between atoms in a 3D space. This version incorporates advanced optimization techniques, including shared memory usage, memory coalescing, and improved workload management, to achieve enhanced performance compared to Version 1.

The program calculates pairwise distances between atoms, updates the corresponding histogram buckets in parallel on the GPU, and compares the execution times with the CPU-only version. It ensures no discrepancies between CPU and GPU-generated histograms while delivering superior computational efficiency.

Requirements:<br>
To compile and run the program, you need access to a CUDA-enabled machine. Ensure you have nvcc installed to compile CUDA code.

To install and use:
1. Clone the repository:
```
git clone https://github.com/asibai7/Spacial-Distance-Histogram-Computation-v2.git
cd Spacial-Distance-Histogram-Computation-v2
```
2. Compile the code using nvcc on a CUDA-enabled machine:
```
nvcc SDH_v2.cu -o SDH_v2
```
3. Run program:
```
./SDH_v2 <atomCount> <bucketRange>
```
Example: 
```
./SDH_v2 200000 2000
```
With 200,000 atoms and a bucket range of 2,000:
- **CPU time**: 540.8 seconds
- **GPU time (Version 1)**: 19.3 seconds
- **GPU time (Version 2)**: 12.7 seconds
<br>(atomCount) The number of atoms to generate in 3D space.  
(bucketRange) The range for each bucket in the histogram.
