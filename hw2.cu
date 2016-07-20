#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
	
#define THREADS_PER_BLOCK	512
#define BLOCKS_PER_GRID_ROW 128
float cpu1;
float cpu2;
float gpu1;
float gpu2;
float max1;
float min1;
float m1;
float m2;
float cc1;
float cc2;
float cc3;

__global__ void arradd( float *A)
{

__shared__ float max[512];

int arrayIndex = 128*512*blockIdx.y + 512*blockIdx.x + threadIdx.x;
max[threadIdx.x] = A[arrayIndex];

__syncthreads();
int nTotalThreads = blockDim.x;

while(nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	
		if (threadIdx.x < halfPoint)
		{
                        float temp = max[threadIdx.x + halfPoint];
			if (temp > max[threadIdx.x]) max[threadIdx.x] = temp;
		}
		__syncthreads();

		nTotalThreads = (nTotalThreads >> 1);	// divide by two.
	}
if (threadIdx.x == 0)
	{
		A[128*blockIdx.y + blockIdx.x] = max[0];
	}
}



__global__ void erredd( float *A)
{
__shared__ float min[512];

int arrayIndex = 128*512*blockIdx.y + 512*blockIdx.x + threadIdx.x;
min[threadIdx.x] = A[arrayIndex];


__syncthreads();
int nTotalThreads = blockDim.x;

while(nTotalThreads > 1)
{
  int halfPoint = (nTotalThreads >> 1);
  if(threadIdx.x < halfPoint)
 {
 float temp = min[threadIdx.x + halfPoint];
 if (temp < min[threadIdx.x]) min[threadIdx.x] = temp;
}
__syncthreads();

nTotalThreads = (nTotalThreads >> 1);
}
if (threadIdx.x == 0)
{
A[128*blockIdx.y + blockIdx.x] = min[0];
}

}

void helper(float *A, int N){
cudaEvent_t start2, stop2;

float time1;


 
if (N <=0) return;
float max;
max = A[0];
for (int i=0; i<10; i++)
{
cudaEventCreate(&start2);
cudaEventRecord(start2,0);
for (int i=1; i<N; i++)
{
float temp = A[i];
if (temp > max) max = temp;
}
cudaEventCreate(&stop2);
cudaEventRecord(stop2,0);
cudaEventSynchronize(stop2);
cudaEventElapsedTime(&time1, start2, stop2);
time1 = time1 + time1;
}
            cpu1 = time1 / 10;
            cpu1 = cpu1 / 1000;
                           
		max1=max;
}

void helper2(float *B, int N){
cudaEvent_t start3, stop3;
float time2;



if (N <=0) return;
float min;
min = B[0];
for (int i=0; i<10; i++)
{
cudaEventCreate(&start3);
cudaEventRecord(start3,0);
for (int i=1; i<N; i++)
{
float temp = B[i];
if (temp < min) min = temp;
}
cudaEventCreate(&stop3);
cudaEventRecord(stop3,0);
cudaEventSynchronize(stop3);
cudaEventElapsedTime(&time2, start3, stop3);
time2 = time2 + time2;
}
		cpu2 = time2 / 10;
                cpu2 = cpu2 / 1000;
		min1=min;
}





void step1Max(int N){
cudaEvent_t start2, stop2;
cudaEvent_t start21, stop21;
cudaEvent_t start22, stop22;
float time22;
float time2;
float time29;
float time21;
N = N * 1048576;
        
   	float *d_A;  
     size_t size = N *sizeof(float);
     float *h_A = (float *)malloc(size);


	cudaMalloc((void **)&d_A, sizeof(float) * N);
	
	for(int i = 0; i < N; i++)
	{
		h_A[i] = (float)rand();
	}

		
	float tempMax;
                               
              
		

		         int blockGridWidth = BLOCKS_PER_GRID_ROW;
		          int blockGridHeight = (N / THREADS_PER_BLOCK) / blockGridWidth;

		         dim3 blockGridRows(blockGridWidth, blockGridHeight);
		         dim3 threadBlockRows(THREADS_PER_BLOCK, 1);

                        int k=0;
			while (k!=10)
			{		
                        cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);
                        cudaEventCreate(&start2);
                        cudaEventRecord(start2,0);
			arradd<<<blockGridRows, threadBlockRows>>>(d_A);
			cudaEventCreate(&stop2);
                        cudaEventRecord(stop2,0);
                        cudaEventSynchronize(stop2);
                        cudaEventElapsedTime(&time2, start2, stop2);
			cudaThreadSynchronize();
			cudaMemcpy(h_A, d_A, sizeof(float) * N / THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);
			tempMax = h_A[0];
			for (int i = N / THREADS_PER_BLOCK; i > 0; i = i / 2)
			{
			cudaMemcpy(d_A, h_A, sizeof(float) * i, cudaMemcpyHostToDevice);
                        cudaEventCreate(&start21);
                        cudaEventRecord(start21,0);
			arradd<<<blockGridRows, threadBlockRows>>>(d_A);
			cudaEventCreate(&stop21);
                        cudaEventRecord(stop21,0);
                        cudaEventSynchronize(stop21);
                        cudaEventElapsedTime(&time21, start21, stop21);
			time21 = time21 + time21;
			cudaThreadSynchronize();
			cudaMemcpy(h_A, d_A, sizeof(float) * i, cudaMemcpyDeviceToHost);
			tempMax = h_A[0];
			if (i==1)
			{
                        cudaMemcpy(d_A, h_A, sizeof(int) * THREADS_PER_BLOCK, cudaMemcpyHostToDevice);
                        cudaEventCreate(&start22);
                        cudaEventRecord(start22,0);
			arradd<<<blockGridRows, threadBlockRows>>>(d_A);
			cudaEventCreate(&stop22);
                        cudaEventRecord(stop22,0);
                        cudaEventSynchronize(stop22);
                        cudaEventElapsedTime(&time22, start22, stop22);
			time22 = time22 + time22;
			cudaThreadSynchronize();
			cudaMemcpy(h_A, d_A, sizeof(int) * 1, cudaMemcpyDeviceToHost);
			tempMax = h_A[0];
			}
			}
			k++;
			time2 = time2 + time2;
			}	
			time29 = (time2 + time22 + time21) / 10;
                       // time29 = time29/10;
                        time29 = time29/1000; 
                        m1 = tempMax;	  
		        gpu1 = time29;
        helper(h_A, N);
	cudaFree(d_A);
	free(h_A);
cc1 = cpu1 / gpu1;
}

void step1Min (int N){
cudaEvent_t start3, stop3;
cudaEvent_t start31, stop31;
cudaEvent_t start32, stop32;
float time3;
float time32;
float time31;
N = N * 1048576;
        

    

	float *d_B;
    int i;
    size_t size = N *sizeof(float);
    float *h_B = (float *)malloc(size);


	cudaMalloc( (void **)&d_B, sizeof(float) * N);
	

	for(i = 0; i < N; i++)
	{
		h_B[i] = (float)rand();
	}

		
	float tempMin;
                
		

		        int blockGridWidth = BLOCKS_PER_GRID_ROW;
		        int blockGridHeight = (N / THREADS_PER_BLOCK) / blockGridWidth;

		        dim3 blockGridRows(blockGridWidth, blockGridHeight);
		        dim3 threadBlockRows(THREADS_PER_BLOCK, 1);
                        int k=0;
			while (k!=10)
			{		
                        cudaMemcpy(d_B, h_B, sizeof(float) * N, cudaMemcpyHostToDevice);
                        cudaEventCreate(&start3);
                        cudaEventRecord(start3,0);
			erredd<<<blockGridRows, threadBlockRows>>>(d_B);
			cudaEventCreate(&stop3);
                        cudaEventRecord(stop3,0);
                        cudaEventSynchronize(stop3);
                        cudaEventElapsedTime(&time3, start3, stop3);
			cudaThreadSynchronize();
			cudaMemcpy(h_B, d_B, sizeof(float) * N / THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);
			tempMin = h_B[0];
			k++;
			time3 = time3 + time3;
			for (int i = N / THREADS_PER_BLOCK; i > 0; i = i / 2)
			{
			cudaMemcpy(d_B, h_B, sizeof(float) * i, cudaMemcpyHostToDevice);
                        cudaEventCreate(&start31);
                        cudaEventRecord(start31,0);
			erredd<<<blockGridRows, threadBlockRows>>>(d_B);
			cudaEventCreate(&stop31);
                        cudaEventRecord(stop31,0);
                        cudaEventSynchronize(stop31);
                        cudaEventElapsedTime(&time31, start31, stop31);
			cudaThreadSynchronize();
			time31 = time31 + time31;
			cudaMemcpy(h_B, d_B, sizeof(float) * i, cudaMemcpyDeviceToHost);
			tempMin = h_B[0];
			if (i==1)
			{
                        cudaMemcpy(d_B, h_B, sizeof(int) * THREADS_PER_BLOCK, cudaMemcpyHostToDevice);
                        cudaEventCreate(&start32);
                        cudaEventRecord(start32,0);
			erredd<<<blockGridRows, threadBlockRows>>>(d_B);
			cudaEventCreate(&stop32);
                        cudaEventRecord(stop32,0);
                        cudaEventSynchronize(stop32);
                        cudaEventElapsedTime(&time32, start32, stop32);
			cudaThreadSynchronize();
			time32 = time32 + time32;
			cudaMemcpy(h_B, d_B, sizeof(int) * 1, cudaMemcpyDeviceToHost);
			tempMin = h_B[0];
			}
			}
			}	        
			tempMin = h_B[0];			
		gpu2 = (time31+time3 + time32) / 10;	
   //               gpu2 = gpu2 / 100;
                gpu2 = gpu2 / 1000;
		m2 = tempMin;
		helper2(h_B, N);	
	cudaFree(d_B);
	free(h_B);

cc2 = cpu2 / gpu2;
}

int main(int argc, char **argv){
int a[3] = {2, 8, 32};
float element1;
printf("Step 1\n");
printf("Shuyang\n");
printf("Zang\n");
//printf("N   2M   GPUmax   %f   CPUmax  %f   GPUtime  %f   CPUtime  %f  GPUSpeedup \n");
for (int i=0; i<3;i++){
step1Max(a[i]);
element1 = a[i];
printf("N   %f   GPUmax   %f   CPUmax   %f   GPUtime   %f   CPUtime   %f   GPUSpeedup   %f \n", element1, m1, max1, gpu1, cpu1, cc1);
//printf("%6f   ", element1);
//printf("%12f   ", m1);
//printf("%12f   ", max1);
//printf("%12f   ", gpu1);
//printf("%16f   ", cpu1);
//printf("%25f   \n", cc1);
}
printf("\n");
//printf("N            GPUmin                    CPUmin           GPUtime           CPUtime                    GPUSpeedup \n");
for (int i=0; i<3;i++){
step1Min(a[i]);
element1 = a[i];
printf("N   %f   GPUmax   %f   CPUmax   %f   GPUtime   %f   CPUtime   %f   GPUSpeedup   %f \n", element1, m2, min1, gpu2, cpu2, cc2);
//printf("%6f   ", element1);
//printf("%16f   ", m2);
//printf("%16f   ", min1);
//printf("%16f   ", gpu2);
//printf("%20f   ", cpu2);
//printf("%24f   \n", cc2);
}
}
