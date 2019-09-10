// Define C functions for the C++ class - as ctypes can only talk to C...



#include "./src/srm.h"


extern "C" 
{
    SRM* StatisticalRegionMerging_New( uint8_t const colorDims, float* pQValues, uint8_t iQCount)
    {
        return new SRM(colorDims, pQValues, iQCount);
    }

    void* StatisticalRegionMerging_mode_dot( SRM* pSRM)
    {
        pSRM->setDotMode(true);
    }

    void* StatisticalRegionMerging_mode_eucl( SRM* pSRM)
    {
        pSRM->setDotMode(false);
    }

    float* SRM_calc_similarity( SRM* pSRM, uint32_t xcount, uint32_t ycount, float* pImage)
    {
        
        return pSRM->calculateSimilarity(xcount, ycount, pImage);
    }

    uint32_t* SRM_processFloat(SRM* pSRM, uint32_t xcount, uint32_t ycount, float* pImage)
    {
        return pSRM->getSegmentedImageFloat(xcount, ycount, pImage);
    }

};