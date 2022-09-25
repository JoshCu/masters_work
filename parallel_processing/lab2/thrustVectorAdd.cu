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
/* 3460:677 CUDA Thrust Vector Add lab, Version 1.01, Fall 2016.          */

#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cutil.h>

int main(int argc, char *argv[])
{
  float *hostInput1;
  float *hostInput2;
  float *expectedOutput;
  int inputLength1, inputLength2, outputLength;

  FILE *infile1, *infile2, *outfile;
  unsigned int generic, gpu1, gpu2, copy, compute;

  // Import host input data
  CUT_SAFE_CALL(cutCreateTimer(&generic));
  cutStartTimer(generic);
  if ((infile1 = fopen("input0.raw", "r")) == NULL)
  {
    printf("Cannot open input0.raw.\n");
    exit(EXIT_FAILURE);
  }
  if ((infile2 = fopen("input1.raw", "r")) == NULL)
  {
    printf("Cannot open input1.raw.\n");
    exit(EXIT_FAILURE);
  }
  fscanf(infile1, "%i", &inputLength1);
  hostInput1 = (float *)malloc(sizeof(float) * inputLength1);
  for (int i = 0; i < inputLength1; i++)
    fscanf(infile1, "%f", &hostInput1[i]);
  fscanf(infile2, "%i", &inputLength2);
  hostInput2 = (float *)malloc(sizeof(float) * inputLength2);
  for (int i = 0; i < inputLength2; i++)
    fscanf(infile2, "%f", &hostInput2[i]);
  fclose(infile1);
  fclose(infile2);
  cutStopTimer(generic);
  printf("Importing data to host: %f ms\n", cutGetTimerValue(generic));

  // Allocate host output vector
  //@@ Insert code here
  outputLength = inputLength1 + inputLength2;
  float *hostOutput = (float *)malloc(sizeof(float) * outputLength);

  CUT_SAFE_CALL(cutCreateTimer(&gpu1));
  cutStartTimer(gpu1);

  CUT_SAFE_CALL(cutCreateTimer(&gpu2));
  cutStartTimer(gpu2);

  // Declare and allocate thrust device input and output vectors
  //@@ Insert code here
  thrust::device_vector<float> deviceInput1(hostInput1, hostInput1 + inputLength1);
  thrust::device_vector<float> deviceInput2(hostInput2, hostInput2 + inputLength2);
  thrust::device_vector<float> deviceOutput(outputLength);

  cutStopTimer(gpu2);
  printf("Doing GPU memory allocation: %f ms\n", cutGetTimerValue(gpu2));

  CUT_SAFE_CALL(cutCreateTimer(&copy));
  cutStartTimer(copy);

  // Copy to device
  //@@ Insert code here
  thrust::copy(deviceInput1.begin(), deviceInput1.end(), deviceOutput.begin());
  thrust::copy(deviceInput2.begin(), deviceInput2.end(), deviceOutput.begin() + inputLength1);

  cutStopTimer(copy);
  printf("Copying data to the GPU: %f ms\n", cutGetTimerValue(copy));

  CUT_SAFE_CALL(cutCreateTimer(&compute));
  cutStartTimer(compute);

  // Execute vector addition
  //@@ Insert Code here
  thrust::transform(deviceInput1.begin(), deviceInput1.end(), deviceInput2.begin(), deviceOutput.begin(), thrust::plus<float>());

  cutStopTimer(compute);
  printf("Doing the computation on the GPU: %f ms\n", cutGetTimerValue(compute));
  /////////////////////////////////////////////////////////

  cutDeleteTimer(copy);
  CUT_SAFE_CALL(cutCreateTimer(&copy));
  cutStartTimer(copy);

  // Copy data back to host
  //@@ Insert code here
  thrust::copy(deviceOutput.begin(), deviceOutput.end(), hostOutput);

  cutStopTimer(copy);
  printf("Copying data from the GPU: %f ms\n", cutGetTimerValue(copy));

  cutStopTimer(gpu1);
  printf("Doing GPU computation (memory + compute): %f ms\n", cutGetTimerValue(gpu1));

  if ((outfile = fopen("output.raw", "r")) == NULL)
  {
    printf("Cannot open output.raw.\n");
    exit(EXIT_FAILURE);
  }
  fscanf(outfile, "%i", &outputLength);
  expectedOutput = (float *)malloc(sizeof(float) * outputLength);
  for (int i = 0; i < outputLength; i++)
    fscanf(outfile, "%f", &expectedOutput[i]);
  fclose(outfile);
  int test = 1;
  for (int i = 0; i < outputLength; i++)
    test = test && (abs(expectedOutput[i] - hostOutput[i]) < 0.005);
  if (test)
    printf("Results correct.\n");
  else
    printf("Results incorrect.\n");

  cutDeleteTimer(generic);
  cutDeleteTimer(gpu1);
  cutDeleteTimer(gpu2);
  cutDeleteTimer(copy);
  cutDeleteTimer(compute);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(expectedOutput);
  return 0;
}