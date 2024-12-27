# Spacial-Distance-Histogram-Computation-v2

This project represents an optimized version of the CUDA-accelerated computation of a histogram categorizing distances between atoms in a 3D space. Enhancements include shared memory usage, memory coalescing, and improved workload management. These optimizations significantly improve performance over Version 1 while maintaining correctness and consistency between CPU and GPU-generated histograms.
The program calculates distances between pairs of atoms, updates the appropriate histogram buckets in parallel on the GPU, and compares execution times with the CPU-only version. Additionally, it ensures no discrepancies between the results of the CPU and GPU computations.

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
