#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>  

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL

static vector3*dPos=NULL;
static vector3*dVel=NULL;
static double*dMass=NULL;
static vector3*dAccels=NULL;
__global__
void compute_accels_kernel(vector3 *pos,double *mass,vector3 *accels,int n){
	int i=blockIdx.y*blockDim.y+threadIdx.y;
	int j=blockIdx.x*blockDim.x+threadIdx.x;
	if (i>=n||j>=n) return;

	if (i == j) {
		for (int k = 0; k < 3; ++k) {
			accels[i * n + j][k] = 0.0;
		}
		return;
	}
	vector3 distance;
	for (int k = 0; k < 3; ++k) {
		distance[k] = pos[i][k] - pos[j][k];
	}
	double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
	double magnitude=sqrt(magnitude_sq);
	double accelmag=-1.0*GRAV_CONSTANT*mass[j]/magnitude_sq;

	accels[i * n + j][0] = accelmag * distance[0] / magnitude;
	accels[i * n + j][1] = accelmag * distance[1] / magnitude;
	accels[i * n + j][2] = accelmag * distance[2] / magnitude;
}
__global__
void sum_and_update_kernel(vector3 *pos,vector3 *vel,vector3 *accels,int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	vector3 accel_sum = {0.0, 0.0, 0.0};
	for (int j = 0; j < n; ++j) {
		for (int k = 0; k < 3; ++k) {
			accel_sum[k] += accels[i * n + j][k];
		}
	}
	for (int k = 0; k < 3; ++k) {
		vel[i][k] += accel_sum[k] * INTERVAL;
		pos[i][k] += vel[i][k] * INTERVAL;
	}
}

void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int N=NUMENTITIES;
	if (dPos == NULL) {
		cudaMalloc((void**)&dPos,N*sizeof(vector3));
		cudaMalloc((void**)&dVel,N*sizeof(vector3));
		cudaMalloc((void**)&dMass,N*sizeof(double));
		cudaMalloc((void**)&dAccels,N*N*sizeof(vector3));
		cudaMemcpy(dMass,mass,N*sizeof(double),cudaMemcpyHostToDevice);
	}
	//copy the latest hPos/hVel values to the GPU
	cudaMemcpy(dPos, hPos, N * sizeof(vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(dVel, hVel, N * sizeof(vector3), cudaMemcpyHostToDevice);
	//Kernel 1: Calculate the accels matrix 
	dim3 block(16, 16);
	dim3 grid((N+block.x-1)/block.x, (N+block.y-1)/block.y);
	compute_accels_kernel<<<grid, block>>>(dPos, dMass, dAccels, N);
	cudaDeviceSynchronize();
	//Kernel 2: Accumulation and Update
	int threads = 256;
	int blocks=(N+threads-1)/threads;
	sum_and_update_kernel<<<blocks, threads>>>(dPos,dVel,dAccels,N);
	cudaDeviceSynchronize();
	// Copy the result back to host
	cudaMemcpy(hPos, dPos, N*sizeof(vector3), cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, dVel, N*sizeof(vector3), cudaMemcpyDeviceToHost);
}