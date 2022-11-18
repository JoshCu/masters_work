/* ACADEMIC INTEGRITY PLEDGE                                              */
/*                                                                        */
/* - I have not used source code obtained from another student nor        */
/*   any other unauthorized source, either modified or unmodified.        */
/*                                                                        */
/* - All source code and documentation used in my program is either       */
/*   my original work or was derived by me from the source code           */
/*   published in the textbook for this course or presented in            */
/*   class.                                                               */
/*                                                                        */
/* - I have not discussed coding details about this project with          */
/*   anyone other than my instructor. I understand that I may discuss     */
/*   the concepts of this program with other students and that another    */
/*   student may help me debug my program so long as neither of us        */
/*   writes anything during the discussion or modifies any computer       */
/*   file during the discussion.                                          */
/*                                                                        */
/* - I have violated neither the spirit nor letter of these restrictions. */
/*                                                                        */
/*                                                                        */
/*                                                                        */
/* Signed:_____________________________________ Date:_____________        */
/*                                                                        */
/*                                                                        */
/* 3460:677 CUDA Prefix Sum lab, V. 1.01, Fall 2016.                      */

#include <stdio.h>
#include <stdlib.h>
#include <helper_timer.h>

// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// + lst[n-1]}

#define BLOCK_SIZE 32 //@@ You can change this

// template <unsigned int blockSize>
// __device__ void warpReduce(volatile int *sdata, unsigned int tid)
// {
//   if (blockSize >= 64)
//     sdata[tid] += sdata[tid + 32];
//   if (blockSize >= 32)
//     sdata[tid] += sdata[tid + 16];
//   if (blockSize >= 16)
//     sdata[tid] += sdata[tid + 8];
//   if (blockSize >= 8)
//     sdata[tid] += sdata[tid + 4];
//   if (blockSize >= 4)
//     sdata[tid] += sdata[tid + 2];
//   if (blockSize >= 2)
//     sdata[tid] += sdata[tid + 1];
// }

__device__ void warpReduce(volatile int *sdata, unsigned int tid)
{
  if (BLOCK_SIZE >= 64)
    sdata[tid] += sdata[tid + 32];
  if (BLOCK_SIZE >= 32)
    sdata[tid] += sdata[tid + 16];
  if (BLOCK_SIZE >= 16)
    sdata[tid] += sdata[tid + 8];
  if (BLOCK_SIZE >= 8)
    sdata[tid] += sdata[tid + 4];
  if (BLOCK_SIZE >= 4)
    sdata[tid] += sdata[tid + 2];
  if (BLOCK_SIZE >= 2)
    sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n)
{
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + tid;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  // print g_idata
  // if (i < n)
  //   printf("bid:%d tid:%d g_idata[%d] = %d\n", blockIdx.x, tid, i, g_idata[i]);
  // if (i + blockSize < n)
  //   printf("bid:%d tid:%d g_idata[%d] = %d\n", blockIdx.x, tid, i, g_idata[i + blockSize]);
  // __syncthreads();
  sdata[tid] = 0;
  while (i < n)
  {
    sdata[tid] += g_idata[i] + g_idata[i + blockSize];
    i += gridSize;
  }
  // print shared data
  if (blockIdx.x * (blockSize * 2) + tid < n)
  {
    printf("sdata[%d] = %d\n", blockIdx.x * (blockSize * 2) + tid, sdata[blockIdx.x * (blockSize * 2) + tid]);
  }

  __syncthreads();
  if (blockSize >= 512)
  {
    if (tid < 256)
      sdata[tid] += sdata[tid + 256];
    __syncthreads();
  }
  if (blockSize >= 256)
  {
    if (tid < 128)
      sdata[tid] += sdata[tid + 128];
    __syncthreads();
  }
  if (blockSize >= 128)
  {
    if (tid < 64)
      sdata[tid] += sdata[tid + 64];
    __syncthreads();
  }
  if (tid < 32)
    warpReduce(sdata, tid);

  // copy all shared data to global memory
  g_odata[i] = sdata[tid];
}

int main(int argc, char **argv)
{
  int *hostInput;  // The input 1D list
  int *hostOutput; // The output list
  int *expectedOutput;
  int *deviceInput;
  int *deviceOutput;
  int *deviceAuxArray, *deviceAuxScannedArray;
  int numElements; // number of elements in the list

  FILE *infile, *outfile;
  int inputLength, outputLength;
  StopWatchLinux stw;
  unsigned int blog = 1;

  // Import host input data
  stw.start();
  if ((infile = fopen("input.raw", "r")) == NULL)
  {
    printf("Cannot open input.raw.\n");
    exit(EXIT_FAILURE);
  }
  fscanf(infile, "%i", &inputLength);
  hostInput = (int *)malloc(sizeof(int) * inputLength);
  for (int i = 0; i < inputLength; i++)
    fscanf(infile, "%i", &hostInput[i]);
  fclose(infile);
  numElements = inputLength;
  hostOutput = (int *)malloc(numElements * sizeof(int));
  stw.stop();
  printf("Importing data and creating memory on host: %f ms\n", stw.getTime());

  if (blog)
    printf("*** The number of input elements in the input is %i\n", numElements);

  stw.reset();
  stw.start();

  cudaMalloc((void **)&deviceInput, numElements * sizeof(int));
  cudaMalloc((void **)&deviceOutput, numElements * sizeof(int));

  cudaMalloc(&deviceAuxArray, (BLOCK_SIZE << 1) * sizeof(int));
  cudaMalloc(&deviceAuxScannedArray, (BLOCK_SIZE << 1) * sizeof(int));

  stw.stop();
  printf("Allocating GPU memory: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();

  cudaMemset(deviceOutput, 0, numElements * sizeof(int));

  stw.stop();
  printf("Clearing output memory: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();

  cudaMemcpy(deviceInput, hostInput, numElements * sizeof(int),
             cudaMemcpyHostToDevice);

  stw.stop();
  printf("Copying input memory to the GPU: %f ms\n", stw.getTime());

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements / (float)(BLOCK_SIZE << 1)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  stw.reset();
  stw.start();

  //@@ Modify this to complete the functionality of the scan
  //@@ on the device

  // print grid and block dimensions
  if (blog)
  {
    printf("Grid dimensions: %i %i %i, Block dimensions: %i %i %i \n", dimGrid.x,
           dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  }

  // reduce6
  reduce6<BLOCK_SIZE><<<dimGrid, dimBlock, BLOCK_SIZE * sizeof(int)>>>(deviceInput, deviceAuxArray, numElements);

  cudaDeviceSynchronize();

  stw.stop();
  printf("Performing CUDA computation: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();

  cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(int),
             cudaMemcpyDeviceToHost);

  stw.stop();
  printf("Copying output memory to the CPU: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();

  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxArray);
  cudaFree(deviceAuxScannedArray);

  stw.stop();
  printf("Freeing GPU Memory: %f ms\n", stw.getTime());

  if ((outfile = fopen("output.raw", "r")) == NULL)
  {
    printf("Cannot open output.raw.\n");
    exit(EXIT_FAILURE);
  }
  fscanf(outfile, "%i", &outputLength);
  expectedOutput = (int *)malloc(sizeof(int) * outputLength);
  for (int i = 0; i < outputLength; i++)
    fscanf(outfile, "%i", &expectedOutput[i]);
  fclose(outfile);

  int test = 1;
  for (int i = 0; i < outputLength; i++)
  {
    if (expectedOutput[i] != hostOutput[i])
      printf("%i %i %i\n", i, expectedOutput[i], hostOutput[i]);
    test = test && (expectedOutput[i] == hostOutput[i]);
  }

  if (test)
    printf("Results correct.\n");
  else
    printf("Results incorrect.\n");

  free(hostInput);
  cudaFreeHost(hostOutput);
  free(expectedOutput);

  return 0;
}
