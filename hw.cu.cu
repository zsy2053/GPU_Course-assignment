#include <stdio.h>
#include <cuda_runtime.h>
float element1;
float cputogpu1;
float kernel1;
float gputocpu1;
float element2;
float cputogpu2;
float kernel2;
float gputocpu2;
float element3;
float cputogpu3;
float kernel3;
float gputocpu3;
float element4;
float cputogpu4 = 0;
float kernel4 = 0;
float gputocpu4 = 0;
__global__ void arradd( float *A, int N)
{
int  B = 2000;
int  i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < N)
{
A[i] = A[i] + B;
}

}

__global__ void darradd( double *A2, int N2)
{
int B2 = 2000;
int i2 = blockDim.x * blockIdx.x + threadIdx.x;
if (i2 < N2)
{
A2[i2] = A2[i2] + B2;
}
}

__global__ void iarradd( int32_t *A3, int N3)
{
int B3 = 2000;
int i3 = blockDim.x * blockIdx.x + threadIdx.x;
if (i3 < N3)
{
A3[i3] = A3[i3] + B3;
}
}

__global__ void xarradd( float *A4, int N4, int B4, int num)
{
int i4 = blockDim.x * blockIdx.x + threadIdx.x;
if (i4 < N4)
{
for (int i=0; i<num; i++)
{
A4[i4] = A4[i4] + B4;
}
}
}

int helper4(int num){
cudaError_t err4 = cudaSuccess;
cudaEvent_t start41, stop41;
cudaEvent_t start42, stop42;
cudaEvent_t start43, stop43;
float time41;
float time42;
float time43;
int N4 = 128000000;
int B4 = 2000;
size_t size4 = N4 *sizeof(float);

float *h_A4 = (float *)malloc(size4);


//float *h_C4 = (float *)malloc(size4);


for (int i4 = 0; i4 < N4; i4++)
{
h_A4[i4] = i4/3.0f;
}

float *d_A4 = NULL;
err4 = cudaMalloc((void **)&d_A4, size4);

//float *d_C4 = NULL;
//err4 = cudaMalloc((void **)&d_C4, size4);


cudaEventCreate(&start41);
cudaEventRecord(start41,0);

//printf("COPY input data from the host to CUDA device\n");
err4 = cudaMemcpy(d_A4, h_A4, size4, cudaMemcpyHostToDevice);

cudaEventCreate(&stop41);
cudaEventRecord(stop41,0);
cudaEventSynchronize(stop41);
cudaEventElapsedTime(&time41, start41, stop41);
//printf("The time for CPU to GPU is %fms\n",time41);
cputogpu4 = time41;

cudaEventCreate(&start42);
cudaEventRecord(start42,0);

int threadsPerBlock = 256;
int blocksPerGrid = (N4 + threadsPerBlock - 1) / threadsPerBlock;
//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
xarradd<<<blocksPerGrid, threadsPerBlock>>>(d_A4, N4, B4, num);
err4 = cudaGetLastError();

/*if (err != cudaSuccess)
{
printf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
exit(EXIT_FAILURE);
*/
cudaEventCreate(&stop42);
cudaEventRecord(stop42,0);
cudaEventSynchronize(stop42);
cudaEventElapsedTime(&time42,start42,stop42);
//printf("The time for kernal is %fms\n",time42);
kernel4 = time42;
cudaEventCreate(&start43);
cudaEventRecord(start43,0);

//printf("Copy output data from the CUDA device to the host memory\n");

err4 = cudaMemcpy(h_A4, d_A4, size4, cudaMemcpyDeviceToHost);

cudaEventCreate(&stop43);
cudaEventRecord(stop43,0);
cudaEventSynchronize(stop43);
cudaEventElapsedTime(&time43,start43,stop43);
//printf("The time for GPU to CPU is %fms\n",time43);
gputocpu4 = time43;


err4 = cudaFree(d_A4);
//err4 = cudaFree(d_C4);


free(h_A4);
//free(h_C4);

err4 = cudaDeviceReset();


//printf("test done\n");
//printf("Done\n");
return 0;

}

int helper3(int N3){
cudaError_t err3 = cudaSuccess;
cudaEvent_t start31, stop31;
cudaEvent_t start32, stop32;
cudaEvent_t start33, stop33;
float time31;
float time32;
float time33;
N3=N3*1000000;
size_t size3 = N3 *sizeof(int32_t);

int32_t *h_A3 = (int32_t *)malloc(size3);


//float *h_C3 = (float *)malloc(size3);


for (int i = 0; i < N3; i++)
{
h_A3[i] = i/3.0f;
}

int32_t *d_A3 = NULL;
err3 = cudaMalloc((void **)&d_A3, size3);

//float *d_C3 = NULL;
//err3 = cudaMalloc((void **)&d_C3, size3);


cudaEventCreate(&start31);
cudaEventRecord(start31,0);

//printf("COPY input data from the host to CUDA device\n");
err3 = cudaMemcpy(d_A3, h_A3, size3, cudaMemcpyHostToDevice);

cudaEventCreate(&stop31);
cudaEventRecord(stop31,0);
cudaEventSynchronize(stop31);
cudaEventElapsedTime(&time31, start31, stop31);
//printf("The time for CPU to GPU is %fms\n",time31);
cputogpu3 = time31;

cudaEventCreate(&start32);
cudaEventRecord(start32,0);

int threadsPerBlock = 256;
int blocksPerGrid = (N3 + threadsPerBlock - 1)/threadsPerBlock;
//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
iarradd<<<blocksPerGrid, threadsPerBlock>>>(d_A3, N3);
err3 = cudaGetLastError();

/*if (err != cudaSuccess)
{
printf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
exit(EXIT_FAILURE);
}*/
cudaEventCreate(&stop32);
cudaEventRecord(stop32,0);
cudaEventSynchronize(stop32);
cudaEventElapsedTime(&time32,start32,stop32);
//printf("The time for kernal is %fms\n",time32);
kernel3 = time32;
cudaEventCreate(&start33);
cudaEventRecord(start33,0);

//printf("Copy output data from the CUDA device to the host memory\n");


err3 = cudaMemcpy(h_A3, d_A3, size3, cudaMemcpyDeviceToHost);

if (err3 != cudaSuccess)
{
fprintf(stderr, "Failed to copy vector c from device to host (error code %s)!\n", cudaGetErrorString(err3));
exit(EXIT_FAILURE);
}

cudaEventCreate(&stop33);
cudaEventRecord(stop33,0);
cudaEventSynchronize(stop33);
cudaEventElapsedTime(&time33,start33,stop33);
//printf("The time for GPU to CPU is %fms\n",time33);
gputocpu3 = time33;


err3 = cudaFree(d_A3);
//err3 = cudaFree(d_C3);


free(h_A3);
//free(h_C3);

err3 = cudaDeviceReset();


//printf("test done\n");
//printf("Done\n");
return 0;

}


int helper2(int N2) {
cudaError_t err2 = cudaSuccess;
cudaEvent_t start21, stop21;
cudaEvent_t start22, stop22;
cudaEvent_t start23, stop23;
float time21;
float time22;
float time23;
N2 = N2*1000000;

size_t size2 = N2 *sizeof(double);

double *h_A2 = (double *)malloc(size2);


//float *h_C2 = (float *)malloc(size2);


for (int i2 = 0; i2 < N2; i2++)
{
h_A2[i2] = i2/3.0f;
}

double *d_A2 = NULL;
err2 = cudaMalloc((void **)&d_A2, size2);

//float *d_C2 = NULL;
//err2 = cudaMalloc((void **)&d_C2, size2);


cudaEventCreate(&start21);
cudaEventRecord(start21,0);

//printf("COPY input data from the host to CUDA device\n");
err2 = cudaMemcpy(d_A2, h_A2, size2, cudaMemcpyHostToDevice);

cudaEventCreate(&stop21);
cudaEventRecord(stop21,0);
cudaEventSynchronize(stop21);
cudaEventElapsedTime(&time21, start21, stop21);
//printf("The time for CPU to GPU is %fms\n",time21);
cputogpu2 = time21;
cudaEventCreate(&start22);
cudaEventRecord(start22,0);

int threadsPerBlock = 256;
int blocksPerGrid = (N2 + threadsPerBlock - 1) / threadsPerBlock;
//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
darradd<<<blocksPerGrid, threadsPerBlock>>>(d_A2, N2);
err2 = cudaGetLastError();

/*if (err != cudaSuccess)
{
printf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
exit(EXIT_FAILURE);
}*/
cudaEventCreate(&stop22);
cudaEventRecord(stop22,0);
cudaEventSynchronize(stop22);
cudaEventElapsedTime(&time22,start22,stop22);
//printf("The time for kernal is %fms\n",time22);
kernel2 = time22;


cudaEventCreate(&start23);
cudaEventRecord(start23,0);

//printf("Copy output data from the CUDA device to the host memory\n");
err2 = cudaMemcpy(h_A2, d_A2, size2, cudaMemcpyDeviceToHost);

if (err2 != cudaSuccess)
{
fprintf(stderr, "Failed to copy vector c from device to host (error code %s)!\n", cudaGetErrorString(err2));
exit(EXIT_FAILURE);
}

cudaEventCreate(&stop23);
cudaEventRecord(stop23,0);
cudaEventSynchronize(stop23);
cudaEventElapsedTime(&time23,start23,stop23);
//printf("The time for GPU to CPU is %fms\n",time23);
gputocpu2=time23;


err2 = cudaFree(d_A2);
//err2 = cudaFree(d_C2);


free(h_A2);
//free(h_C2);

err2 = cudaDeviceReset();


//printf("test done\n");
//printf("Done\n");
return 0;

}




int helper(int N){
cudaError_t err = cudaSuccess;
cudaEvent_t start1, stop1;
cudaEvent_t start2, stop2;
cudaEvent_t start3, stop3;
float time1;
float time2;
float time3;
N = N * 1000000;


size_t size = N *sizeof(float);

float *h_A = (float *)malloc(size);


//float *h_C = (float *)malloc(size);


for (int i = 0; i < N; i++)
{
h_A[i] = i/3.0f;
}

float *d_A = NULL;
err = cudaMalloc((void **)&d_A, size);

//float *d_C = NULL;
//err = cudaMalloc((void **)&d_C, size);


cudaEventCreate(&start1);
cudaEventRecord(start1,0);

//printf("COPY input data from the host to CUDA device\n");
err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

cudaEventCreate(&stop1);
cudaEventRecord(stop1,0);
cudaEventSynchronize(stop1);
cudaEventElapsedTime(&time1, start1, stop1);
//printf("The time for CPU to GPU is %fms\n",time1);
cputogpu1 = time1;

cudaEventCreate(&start2);
cudaEventRecord(start2,0);

int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
arradd<<<blocksPerGrid, threadsPerBlock>>>(d_A, N);
err = cudaGetLastError();

/*if (err != cudaSuccess)
{
printf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
exit(EXIT_FAILURE);
}*/
cudaEventCreate(&stop2);
cudaEventRecord(stop2,0);
cudaEventSynchronize(stop2);
cudaEventElapsedTime(&time2,start2,stop2);
//printf("The time for kernal is %fms\n",time2);
kernel1 = time2;

cudaEventCreate(&start3);
cudaEventRecord(start3,0);

//printf("Copy output data from the CUDA device to the host memory\n");
err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

/*if (err != cudaSuccess)
{
fprintf(stderr, "Failed to copy vector c from device to host (error code %s)!\n", cudaGetErrorString(err));
exit(EXIT_FAILURE);
}*/

cudaEventCreate(&stop3);
cudaEventRecord(stop3,0);
cudaEventSynchronize(stop3);
cudaEventElapsedTime(&time3,start3,stop3);
//printf("The time for GPU to CPU is %fms\n",time3);
gputocpu1 = time3;


err = cudaFree(d_A);
//err = cudaFree(d_C);


free(h_A);
//free(h_C);

err = cudaDeviceReset();


//printf("test done\n");
//printf("Done\n");
return 0;

}

int main(void){

int a[9] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
printf("part A\n");
printf("Elements    CPUtoGPU(ms)    Kernel (ms)    GPUtoCPU (ms)\n");
for (int i=0; i<=8;i++){
helper(a[i]);
element1 = a[i];
printf("%6f   ", element1);
printf("%11f   ", cputogpu1);
printf("%15f   ", kernel1);
printf("%12f  \n ", gputocpu1);
}
int b[9] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
printf("part B\n");
printf("Elements    CPUtoGPU(ms)    Kernel (ms)    GPUtoCPU (ms)\n");
for (int i2=0; i2<=8;i2++){
helper2(b[i2]);
element2 = b[i2];
printf("%6f   ", element2);
printf("%11f   ", cputogpu2);
printf("%15f   ", kernel2);
printf("%12f  \n ", gputocpu2);

}
printf("part C\n");
printf("Elements    CPUtoGPU(ms)    Kernel (ms)    GPUtoCPU (ms)\n");

int c[9] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
for (int i3=0; i3<=8;i3++){
helper3(c[i3]);
element3 = b[i3];
printf("%6f   ", element3);
printf("%11f   ", cputogpu3);
printf("%15f   ", kernel3);
printf("%12f  \n ", gputocpu3);


}

printf("part D\n");
printf("XaddedTimes    CPUtoGPU(ms)    Kernel (ms)    GPUtoCPU (ms)     Elements (m)\n");
int x4 = 128;
int d[9] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
for (int i4=0; i4<=8;i4++){
helper4(d[i4]);
element4 = d[i4];
printf("%6f   ", element4);
printf("%12f   ", cputogpu4);
printf("%16f   ", kernel4);
printf("%13f   ", gputocpu4);
printf("%13d  \n", x4);


}
}
    