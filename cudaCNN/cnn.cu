/* Kernel */

#include "cnn.h"

// Free up memory
cnnLayer::~cnnLayer()
{

    cudaFree(bias);
    cudaFree(weight);
    
	cudaFree(output);
	cudaFree(prev_value);

	cudaFree(backward_output);
	cudaFree(backward_prev_value);
	cudaFree(backward_weight);
}

cnnLayer::cnnLayer(int Filter, int FilterNum, int ImageSize)
{
	this->Filter = Filter;
	this->FilterNum = FilterNum;
	this->ImageSize = ImageSize;

	output = NULL;
	prev_value = NULL;
	bias   = NULL;
	weight = NULL;

    // allocate memory
    cudaMalloc(&output, sizeof(float) * ImageSize);
    cudaMalloc(&output, sizeof(float) * FilterNum * Filter);

}

