/*
   Query CUDA devices and report properties of each.
   From "CUDA by Example" by Sanders and Kandrot.
*/

#include <stdio.h>

void printClockRate(int c) {
   double d = 1.0 * c;
   int k = 0;
   while (d > 1000) {
      d = d / 1000.0;
      k++;
   }
   printf("%.3f ", d);
   switch (k) {
      case 0: printf("KHz\n"); break;
      case 1: printf("MHz\n"); break;
      case 2: printf("GHz\n"); break;
      case 3: printf("THz\n"); break;
      case 4: printf("PHz\n"); break;
      case 5: printf("EHz\n"); break;
      case 6: printf("ZHz\n"); break;
      case 7: printf("YHz\n"); break;
      default: printf("?Hz\n"); break;
   }
}

void printMemory(size_t m) {
   unsigned long long d = (unsigned long long) m;
   printf("%llu bytes (~", d);
   int k = 0;
   while (d > 1000) {
      d = d / 1000;
      k++;
   }
   printf("%d ", d);
   switch (k) {
      case 0: printf(")\n"); break;
      case 1: printf("KB)\n"); break;
      case 2: printf("MB)\n"); break;
      case 3: printf("GB)\n"); break;
      case 4: printf("TB)\n"); break;
      case 5: printf("PB)\n"); break;
      case 6: printf("EB)\n"); break;
      case 7: printf("ZB)\n"); break;
      case 8: printf("YB)\n"); break;
      default: printf("?B)\n"); break;
   }
}

int main(void)
{
   cudaDeviceProp p;
   int c;

   cudaGetDeviceCount(&c);
   for (int i = 0; i < c; i++) {
      cudaGetDeviceProperties(&p,i);
      printf("GENERAL INFO FOR DEVICE %d ---\n",i);
      printf("   Name: %s\n", p.name);
      printf("   Compute capability: %d.%d\n", p.major, p.minor);
      printf("   Clock rate: ");
      printClockRate(p.clockRate);
      printf("   Device copy overlap: ");
      if (p.deviceOverlap)
         printf("enabled\n");
      else printf("disabled\n");
      printf("   Kernel execution timeout: ");
      if (p.kernelExecTimeoutEnabled)
         printf("enabled\n");
      else printf("disabled\n");
      printf("MEMORY INFO FOR DEVICE %d ---\n",i);
      printf("   Total global memory: ");
      printMemory(p.totalGlobalMem);
      printf("   Total constant memory: ");
      printMemory(p.totalConstMem);
      printf("   Maximum memory pitch: ");
      printMemory(p.memPitch);
      printf("   Texture alignment: %ld\n", p.textureAlignment);
      printf("MULTIPROCESSOR INFO FOR DEVICE %d ---\n", i);
      printf("   MP count: %d\n", p.multiProcessorCount);
      printf("   Shared memory per MP: ");
      printMemory(p.sharedMemPerBlock);
      printf("   Registers per MP: %d\n", p.regsPerBlock);
      printf("   Threads per warp: %d\n", p.warpSize);
      printf("   Maximum threads per block: %d\n", p.maxThreadsPerBlock);
      printf("   Maximum thread dimensions: (%d,%d,%d)\n",
         p.maxThreadsDim[0],p.maxThreadsDim[1],p.maxThreadsDim[2]);
      printf("   Maximum grid dimensions: (%d,%d,%d)\n",
         p.maxGridSize[0],p.maxGridSize[1],p.maxGridSize[2]);
      printf("\n");
   }
}
