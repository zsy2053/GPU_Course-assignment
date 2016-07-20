/*$Id: imageFilter.cu 2016-03-04 18:27:54 (author: Reza Mokhtari)$*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <ctype.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <sys/time.h>


#include "imageFilter_kernel.cu"

#define IMG_DATA_OFFSET_POS 10
#define BITS_PER_PIXEL_POS 28

int swap;
void test_endianess();
void swap_bytes(char *bytes, int num_bytes);

int main(int argc, char *argv[]) 
{
	int i;
	int fd;
	char *fdata;
	struct stat finfo;
	char * inputfname;
	char * outputfname;

	if (argc < 4)
	{
		printf("USAGE: %s <bitmap input filename> <bitmap output file name> <part specifier>\n", argv[0]);
		exit(1);
	}

	inputfname = argv[1];
	outputfname = argv[2];
	char partId = argv[3][0];
	if(partId != 'a' && partId != 'b' && partId != 'c')
	{
		printf("Please provide a part specifier: a, b, or c\n");
		exit(1);
	}

	printf("Image filter: Running...\n");

	fd = open(inputfname, O_RDONLY);
	fstat(fd, &finfo);

	fdata = (char*) malloc(finfo.st_size);
	
	read (fd, fdata, finfo.st_size);

	if ((fdata[0] != 'B') || (fdata[1] != 'M')) 
	{
		printf("File is not a valid bitmap file. Terminating the program\n");
		exit(1);
	}

	test_endianess();     // will set the variable "swap"

	unsigned short *bitsperpixel = (unsigned short *)(&(fdata[BITS_PER_PIXEL_POS]));
	if (swap) 
	{
		printf("swapping\n");
		swap_bytes((char *)(bitsperpixel), sizeof(*bitsperpixel));
	}

 	// ensure its 3 bytes per pixel
	if (*bitsperpixel != 24) 
	{
		printf("Error: Invalid bitmap format - ");
		printf("This application only accepts 24-bit pictures. Exiting\n");
		exit(1);
	}

	unsigned short *data_pos = (unsigned short *)(&(fdata[IMG_DATA_OFFSET_POS]));
	if (swap) 
	{
		swap_bytes((char *)(data_pos), sizeof(*data_pos));
	}

	int imgdata_bytes = (int)finfo.st_size - (int)(*(data_pos));
	printf("This file has %d bytes of image data, %d pixels\n", imgdata_bytes, imgdata_bytes / 3);

	int width = *((int*)&fdata[18]);
	printf("Width: %d\n", width);
	int height = *((int*)&fdata[22]);
	printf("Height: %d\n", height);

	int fileSize = (int) finfo.st_size;	

	//p will point to the first pixel
	char* p = &(fdata[*data_pos]);

	//Set the number of blocks and threads
	dim3 grid(12, 1, 1);
	dim3 block(1024, 1, 1);

	char* d_inputPixels;
	cudaMalloc((void**) &d_inputPixels, width * height * 3);
	cudaMemcpy(d_inputPixels, p, width * height * 3, cudaMemcpyHostToDevice);
	
	char* d_outputPixels;
	cudaMalloc((void**) &d_outputPixels, width * height * 3);
	cudaMemset(d_outputPixels, 0, width * height * 3);

	struct timeval start_tv, end_tv;
	time_t sec;
	time_t ms;
	time_t diff;
	gettimeofday(&start_tv, NULL);

    int numberOfthread = grid.x * block.x;
    int pixelPerthread = (width * height) / numberOfthread;
	if((width * height) % numberOfthread != 0)
        pixelPerthread += 1;
	

    
	int blocksRows = (width - 8) / 120;
	int blocksCols = (height - 8) / 120;
	if((width - 8) % 120 != 0)
		blocksRows += 1;

	if((height - 8) % 120 != 0) 
		blocksCols += 1;
    
    int times = (blocksRows * blocksCols) / 12;
    if((blocksRows * blocksCols) % 12 != 0)
        times += 1;

	if(partId == 'a')
	{
		imageFilterKernelPartA<<<grid, block>>>((char3*) d_inputPixels, (char3*) d_outputPixels, width, height , pixelPerthread);
	} 
	else if(partId == 'b')
	{
		imageFilterKernelPartB<<<grid, block>>>((char3*) d_inputPixels, (char3*) d_outputPixels, width, height , pixelPerthread, numberOfthread);
	}
	else if(partId == 'c')
	{
		imageFilterKernelPartC<<<grid, block>>>((char3*) d_inputPixels, (char3*) d_outputPixels, width, height, blocksRows, blocksCols, times);
	}

	cudaThreadSynchronize();

	gettimeofday(&end_tv, NULL);
	sec = end_tv.tv_sec - start_tv.tv_sec;
	ms = end_tv.tv_usec - start_tv.tv_usec;

	diff = sec * 1000000 + ms;

	printf("%10s:\t\t%fms\n", "Time elapsed", (double)((double)diff/1000.0));


	char* outputPixels = (char*) malloc(height * width * 3);
	cudaMemcpy(outputPixels, d_outputPixels, height * width * 3, cudaMemcpyDeviceToHost);

	memcpy(&(fdata[*data_pos]), outputPixels, height * width * 3);

	FILE *writeFile; 
	writeFile = fopen(outputfname,"w+");
	for(i = 0; i < fileSize; i++)
		fprintf(writeFile,"%c", fdata[i]);
	fclose(writeFile);

	return 0;
} 

void test_endianess() {
    unsigned int num = 0x12345678;
    char *low = (char *)(&(num));
    if (*low ==  0x78) {
        //dprintf("No need to swap\n");
        swap = 0;
    }
    else if (*low == 0x12) {
        //dprintf("Need to swap\n");
        swap = 1;
    }
    else {
        printf("Error: Invalid value found in memory\n");
        exit(1);
    } 
}

void swap_bytes(char *bytes, int num_bytes) 
{
    int i;
    char tmp;
    
    for (i = 0; i < num_bytes/2; i++) {
        //dprintf("Swapping %d and %d\n", bytes[i], bytes[num_bytes - i - 1]);
        tmp = bytes[i];
        bytes[i] = bytes[num_bytes - i - 1];
        bytes[num_bytes - i - 1] = tmp;    
    }
}