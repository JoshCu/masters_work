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
/* 3460:677 CUDA Image Processing lab, Version 1.02, Fall 2016.           */

// Application, read and display bitmap file.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "cpu_bitmap.h"
#include "bitmap_help.h"

#define CHANNELS 4 // we have 4 channels corresponding to ARGB
#define BLUR_SIZE 1
__global__ void blurKernel(unsigned char *in, unsigned char *out)
{
   int x = blockIdx.x;
   int y = blockIdx.y;
   if (x < gridDim.x && y < gridDim.x)
   {
      int pixValR = 0;
      int pixValG = 0;
      int pixValB = 0;
      int pixels = 0;
      // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
      for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
      {
         for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol)
         {
            int curRow = y + blurRow;
            int curCol = x + blurCol;
            // Verify we have a valid image pixel
            if (curRow > -1 && curRow < gridDim.y && curCol > -1 && curCol < gridDim.x)
            {
               // get 1D coordinate for the grayscale image
               int grayOffset = (curRow * gridDim.x) + curCol;
               // get 1D coordinate for the color image
               int rgbOffset = grayOffset * CHANNELS;
               pixValR += in[rgbOffset];     // red value for pixel
               pixValG += in[rgbOffset + 1]; // green value for pixel
               pixValB += in[rgbOffset + 2]; // blue value for pixel
               pixels++;                     // Keep track of number of pixels in the accumulated total
            }
         }
      }
      int grayOffset = (y * gridDim.x) + x;
      int rgbOffset = grayOffset * CHANNELS;
      out[rgbOffset] = pixValR / pixels;     // red value for pixel
      out[rgbOffset + 1] = pixValG / pixels; // green value for pixel
      out[rgbOffset + 2] = pixValB / pixels; // blue value for pixel
   }
}

__host__ void imgProc(unsigned char *map, int size, int width, int height)
{
   // Allocate device memory.
   unsigned char *d_sourceImage, *d_blurImage;
   cudaMalloc((void **)&d_sourceImage, size * sizeof(unsigned char));
   cudaMalloc((void **)&d_blurImage, size * sizeof(unsigned char));

   // Copy host memory (the bitmap pixel data) to device.
   cudaMemcpy(d_sourceImage, map, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

   // Create a width-by-height grid of 1-by-1 blocks. Each block corresponds to an individual pixel, whose
   // coordinates are given as blockIdx.x + blockIdx.y * gridDim.x.
   dim3 dimGrid(width, height);
   dim3 dimBlock(1, 1);

   // Invoke a CUDA kernel
   blurKernel<<<dimGrid, dimBlock>>>(d_sourceImage, d_blurImage);

   // Copy results from device to host.
   cudaMemcpy(map, d_blurImage, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

   // Deallocate device memory.
   cudaFree(d_sourceImage);
   cudaFree(d_blurImage);
}

int main(void)
{
   char fname[50];
   FILE *infile;
   unsigned short ftype;
   tagBMFH bitHead;
   tagBMIH bitInfoHead;
   tagRGBQ *pRgb;

   printf("Please enter the .bmp file name: ");
   scanf("%s", fname);
   strcat(fname, ".bmp");
   infile = fopen(fname, "rb");

   if (infile != NULL)
   {
      printf("File open successful.\n");
      fread(&ftype, 1, sizeof(unsigned short), infile);
      if (ftype != 0x4d42)
      {
         printf("File not .bmp format.\n");
         return 1;
      }
      fread(&bitHead, 1, sizeof(tagBMFH), infile);
      fread(&bitInfoHead, 1, sizeof(tagBMIH), infile);
   }
   else
   {
      printf("File open fail.\n");
      return 1;
   }

   if (bitInfoHead.biBitCount < 24)
   {
      long nPlateNum = long(pow(2, double(bitInfoHead.biBitCount)));
      pRgb = (tagRGBQ *)malloc(nPlateNum * sizeof(tagRGBQ));
      memset(pRgb, 0, nPlateNum * sizeof(tagRGBQ));
      int num = fread(pRgb, 4, nPlateNum, infile);
   }

   int width = bitInfoHead.biWidth;
   int height = bitInfoHead.biHeight;
   int l_width = 4 * ((width * bitInfoHead.biBitCount + 31) / 32);
   long nData = height * l_width;
   unsigned char *pColorData = (unsigned char *)malloc(nData);
   memset(pColorData, 0, nData);
   fread(pColorData, 1, nData, infile);

   fclose(infile);

   CPUBitmap dataOfBmp(width, height);
   unsigned char *map = dataOfBmp.get_ptr();

   if (bitInfoHead.biBitCount < 24)
   {
      int k, index = 0;
      if (bitInfoHead.biBitCount == 1)
      {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
            {
               unsigned char mixIndex = 0;
               k = i * l_width + j / 8;
               mixIndex = pColorData[k];
               if (j % 8 < 7)
                  mixIndex = mixIndex << (7 - (j % 8));
               mixIndex = mixIndex >> 7;
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
      }
      else if (bitInfoHead.biBitCount == 2)
      {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
            {
               unsigned char mixIndex = 0;
               k = i * l_width + j / 4;
               mixIndex = pColorData[k];
               if (j % 4 < 3)
                  mixIndex = mixIndex << (6 - 2 * (j % 4));
               mixIndex = mixIndex >> 6;
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
      }
      else if (bitInfoHead.biBitCount == 4)
      {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
            {
               unsigned char mixIndex = 0;
               k = i * l_width + j / 2;
               mixIndex = pColorData[k];
               if (j % 2 == 0)
                  mixIndex = mixIndex << 4;
               mixIndex = mixIndex >> 4;
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
      }
      else if (bitInfoHead.biBitCount == 8)
      {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
            {
               unsigned char mixIndex = 0;
               k = i * l_width + j;
               mixIndex = pColorData[k];
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
      }
      else if (bitInfoHead.biBitCount == 16)
      {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
            {
               unsigned char mixIndex = 0;
               k = i * l_width + j * 2;
               unsigned char shortTemp = pColorData[k + 1] << 8;
               mixIndex = pColorData[k] + shortTemp;
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
      }
   }
   else
   {
      int k, index = 0;
      for (int i = 0; i < height; i++)
         for (int j = 0; j < width; j++)
         {
            k = i * l_width + j * 3;
            map[index * 4 + 0] = pColorData[k + 2];
            map[index * 4 + 1] = pColorData[k + 1];
            map[index * 4 + 2] = pColorData[k];
            index++;
         }
   }

   imgProc(map, dataOfBmp.image_size(), width, height);
   dataOfBmp.display_and_exit();
   return 0;
}