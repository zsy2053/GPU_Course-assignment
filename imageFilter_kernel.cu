#ifndef _IMAGEFILTER_KERNEL_H_
#define _IMAGEFILTER_KERNEL_H_



__global__ void imageFilterKernelPartA(char3* inputPixels, char3* outputPixels, uint width, uint height, int pixelPerthread)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

               
	for(int h = index * pixelPerthread; h < (index+1)*pixelPerthread; h ++){
		
                
		int3 sum = {0, 0, 0};
		int count = 0;
		
	    
		for(int i = -4; i <= 4; i++)
		{
			for(int j = -4; j <= 4; j++)
			{
			         int px = h + j * width + i;
				if(px >= 0 && px < (width * height))
				{
					
					sum.x += (int)inputPixels[px].x;
					sum.y += (int)inputPixels[px].y;
					sum.z += (int)inputPixels[px].z;
					count++;
				}
			}
	    }	
		outputPixels[h].x = sum.x/count;
		outputPixels[h].y = sum.y/count;
		outputPixels[h].z = sum.z/count;				
	}
}


__global__ void imageFilterKernelPartB(char3* inputPixels, char3* outputPixels, uint width, uint height, int pixelPerthread, int numberOfthread)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i = 0; i < pixelPerthread; i++)
	{
		int location = i * numberOfthread + index;
		int3 sum = {0, 0, 0};
		int count = 0;
		
		for(int i = -4; i <= 4; i++)
		{
			for(int j = -4; j <= 4; j++)
			{
			        int px = location + j * width + i;
				if(px >= 0 && px < (width * height))
				{
					sum.x += (int)inputPixels[px].x;
					sum.y += (int)inputPixels[px].y;
					sum.z += (int)inputPixels[px].z;
					count++;
				}
			}
	    }
		outputPixels[location].x = sum.x/count;
		outputPixels[location].y = sum.y/count;
		outputPixels[location].z = sum.z/count;
	}
}

__global__ void imageFilterKernelPartC(char3* inputPixels, char3* outputPixels, uint width, uint height, int blocksRows, int blocksCols, int times)
{
	__shared__ char3 tile[128 * 128];
	
	int shared_x = threadIdx.x % 32;
	int shared_y = threadIdx.x / 32;

	for(int i = 0; i < times; i++)
	{
		int global_x = (blockIdx.x + i * 12) % blocksRows;
		int global_y = (blockIdx.x + i * 12) / blocksRows;

		for(int k = 0; k < 4; k++)
		{
			for(int j = 0; j < 4; j++)
			{
				int idx = (global_y * 120 + shared_y + k * 32) * width + global_x * 120 + shared_x + j * 32;
				int shared_idx = (shared_y + k * 32) * 128 + shared_x + j * 32;

				if(    ((global_y * 120 + shared_y + k * 32) >=0) 
                    && ((global_y * 120 + shared_y + k * 32) < height) 
					&& ((global_x * 120+ shared_x + j * 32) >= 0) 
                    && ((global_x * 120 + shared_x + j * 32) < width))
				{
					tile[shared_idx] = inputPixels[idx];
				}
			}	
		}
		__syncthreads();

		for(int k = 0; k < 4; k++)
		{
			for(int j = 0; j < 4; j++)
			{
                if((shared_x + j * 32 >= 4) && (shared_x + j * 32 <= 123) && (shared_y + k * 32 >= 4) && (shared_y + k * 32 <= 123))   
                {
                    int3 sum = {0, 0, 0};
			        int count = 0;
				    for(int dx = -4; dx <= 4; dx++)
				    {
					    for(int dy = -4; dy <= 4; dy++)
					    {
							sum.x += (int)tile[(shared_y + k * 32 + dy) * 128 + (shared_x + dx) + j * 32].x;
							sum.y += (int)tile[(shared_y + k * 32 + dy) * 128 + (shared_x + dx) + j * 32].y;
							sum.z += (int)tile[(shared_y + k * 32 + dy) * 128 + (shared_x + dx) + j * 32].z;
							count++;
					    }
				    }
	
				    int out_idx = (global_y * 120 + shared_y + k * 32) * width + global_x * 120 + shared_x + j * 32;
				    if(    ((global_y * 120 + shared_y + k * 32) >=0) 
                        && ((global_y * 120 + shared_y + k * 32) < height) 
					    && ((global_x * 120 + shared_x + j * 32) >= 0) 
                        && ((global_x * 120 + shared_x + j * 32) < width))
				    {
					    outputPixels[out_idx].x = sum.x / count;
					    outputPixels[out_idx].y = sum.y / count;
					    outputPixels[out_idx].z = sum.z / count;
				    }
			    }
                
                if(    (global_y * 120 + shared_y + k * 32) <= 3
                    || ((global_y * 120 + shared_y + k * 32) >= height-4 && (global_y * 120 + shared_y + k * 32) < height)
                    || (global_x * 120 + shared_x + j * 32) <= 3
                    || ((global_x * 120 + shared_x + j * 32) >= width-4 && (global_x * 120 + shared_x + j * 32) < width))
                {
                    int3 sum = {0, 0, 0};
			        int count = 0;
				    for(int dx = -4; dx <= 4; dx++)
				    {
					    for(int dy = -4; dy <= 4; dy++)
					    {
                            if(    ((global_y * 120 + shared_y + k * 32 + dy) >=0) 
                                && ((global_y * 120 + shared_y + k * 32 + dy) < height) 
				                && ((global_x * 120 + shared_x + j * 32 + dx) >= 0) 
                                && ((global_x * 120 + shared_x + j * 32 + dx) < width))
						    {
							    sum.x += (int)inputPixels[(global_y * 120 + shared_y + k * 32 + dy) * width + (global_x * 120 + shared_x + dx)  + j * 32].x;
							    sum.y += (int)inputPixels[(global_y * 120 + shared_y + k * 32 + dy) * width + (global_x * 120 + shared_x + dx)  + j * 32].y;
							    sum.z += (int)inputPixels[(global_y * 120 + shared_y + k * 32 + dy) * width + (global_x * 120 + shared_x + dx)  + j * 32].z;
							    count++;
						    }
					    }
				    }
	
				    int out_boundary = (global_y * 120 + shared_y + k * 32) * width + global_x * 120 + shared_x + j * 32;
				    if(    ((global_y * 120 + shared_y + k * 32) >=0) 
                        && ((global_y * 120 + shared_y + k * 32) < height) 
				        && ((global_x * 120 + shared_x + j * 32) >= 0) 
                        && ((global_x * 120 + shared_x + j * 32) < width))
				    {
					    outputPixels[out_boundary].x = sum.x / count;
					    outputPixels[out_boundary].y = sum.y / count;
					    outputPixels[out_boundary].z = sum.z / count;
				    }
                }
			}
		}
        __syncthreads();
	}
}


#endif // _IMAGEFILTER_KERNEL_H_