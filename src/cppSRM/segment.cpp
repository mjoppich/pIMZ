// Define C functions for the C++ class - as ctypes can only talk to C...



#include "./src/SRM.h"


extern "C" 
{
    SRM* StatisticalRegionMerging_New( uint8_t const colorDims, float* pQValues, uint8_t iQCount)
    {
        return new SRM(colorDims, pQValues, iQCount);
    }

    uint32_t* SRM_processFloat(SRM* pSRM, uint32_t xcount, uint32_t ycount, float* pImage)
    {
        return pSRM->getSegmentedImageFloat(xcount, ycount, pImage);
    }

};