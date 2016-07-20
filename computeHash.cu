
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <sys/time.h>

__global__ void hashKernel(char* input, int size, int* indices, int* hashOutput)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; /* assuming 1D grid and block */

	//Each thread hashes the data from indices[index] to indices[index + 1]
	int start = indices[index];
	int end = indices[index + 1];
	
	unsigned hash = 2166136261;
	unsigned FNVMultiple = 16777619;

	for(int i = start; i < end; i += sizeof(int))
	{
		int arrayValue = *((int*) (input + i));
		hash += arrayValue;
                hash = hash ^ (arrayValue);     /* xor  the entire 32 bits */
                hash -= arrayValue;
                hash = hash * FNVMultiple;  /* multiply by the magic number */
                hash *= (arrayValue == 0)? 1 : arrayValue;
	}

	hashOutput[index] = hash;
}

int main(int argc, char** argv)
{

	int fd;
        char *fdata;
        struct stat finfo;
        char *fname;

        if (argc < 2)
        {
                printf("USAGE: %s <inputfilename>\n", argv[0]);
                exit(1);
        }

        fname = argv[1];
        fd = open(fname, O_RDONLY);
        fstat(fd, &finfo);

        printf("Allocating %lluMB for the input file.\n", ((long long unsigned int)finfo.st_size) / (1 << 20));
        fdata = (char *) malloc(finfo.st_size);
        size_t successRead = read (fd, fdata, finfo.st_size);
        size_t fileSize = (size_t) finfo.st_size;

        if(successRead != fileSize)
	{
                printf("Not all of the file is read, terminating...\n"); /* happens when input data is too large. Not going to handle this for now */
		exit(1);
	}

	//setting fixed number of threads, do not modify.
	dim3 grid(8, 1, 1);
	dim3 block(512, 1, 1);
	int numThreads = grid.x * block.x;

	int* indices = (int*) malloc((numThreads + 1) * sizeof(int));
	
	//calculating indices. Each index shows the point from which a thread starts hashing the input data
	int inputChunkSize = (fileSize + (numThreads - 1)) / numThreads;
        
        inputChunkSize = inputChunkSize - (inputChunkSize % 4);
        
	for(int i = 0; i < numThreads ; i ++)
		indices[i] = i * inputChunkSize; /* last thread(s) might go out of boundary, but gonna be handled in the kernel */
	//Setting the (last + 1) index
	indices[numThreads] = (int) fileSize;

	int* d_indices;
	cudaMalloc((void**) &d_indices, numThreads * sizeof(int));
	cudaMemcpy(d_indices, indices, numThreads * sizeof(int), cudaMemcpyHostToDevice);

	char* d_input;
	cudaMalloc((void**) &d_input, fileSize);
	cudaMemcpy(d_input, fdata, fileSize, cudaMemcpyHostToDevice);

	//Each thread will store its hash value in an element of this array.
	int* d_hashOutput;
	cudaMalloc((void**) &d_hashOutput, numThreads * sizeof(int));
	cudaMemset(d_hashOutput, 0, numThreads * sizeof(int));

	struct timeval partial_start, partial_end;
        time_t sec, ms, diff;
        gettimeofday(&partial_start, NULL);

	hashKernel<<<grid, block>>>(d_input, fileSize, d_indices, d_hashOutput);
	cudaThreadSynchronize();
	
        cudaError_t errR = cudaGetLastError();
        if(errR != cudaSuccess)
	{
		printf("Kernel returned an error, terminating...\n");
                exit(1);
	}

	gettimeofday(&partial_end, NULL);
        sec = partial_end.tv_sec - partial_start.tv_sec;
        ms = partial_end.tv_usec - partial_start.tv_usec;
        diff = sec * 1000000 + ms;

        printf("\n%10s:\t\t%0.1fms\n", "Kernel elapsed time", (double)((double)diff/1000.0));
	int* hashOutput = (int*) malloc(numThreads * sizeof(int));
	cudaMemcpy(hashOutput, d_hashOutput, numThreads * sizeof(int), cudaMemcpyDeviceToHost);

	//Summing up the threads' hash values to form one final hash value.
	int finalHashValue = 0;
	for(int i = 0; i < numThreads; i ++)
		finalHashValue += hashOutput[i];
	
	printf("Final hash value: %d\n", finalHashValue);
	
	return 0;


}
