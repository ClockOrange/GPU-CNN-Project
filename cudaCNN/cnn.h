#define _LAYER_H_
#define _LAYER_H_

#include <string>

#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>


/* Layer constrcute */
class cnnLayer{
    public:
        int ImageSize;
        int Filter;
        int FilterNum;

        int M, N, O;

        float *bias;
        float *weight;
        float *preact;

        float *output;
        float *prev_value;

        float *backward_output;
	    float *backward_prev_value;
	    float *backward_weight;
        float *d_output;
	    float *d_preact;
	    float *d_weight;

        cnnLayer(int Filter, int FilterNum, int ImageSize);
        ~cnnLayer();

        /* support functions in the constructer */
        void calculateOutput(float *data);
	    void cleanMemory();
	    void backwardCleanMemory();
};

