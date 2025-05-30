#ifndef SRM_REGION_H
#define SRM_REGION_H

#include <vector>
#include <time.h>
#include <stdlib.h>
#include <array>

#ifdef _MSC_VER

typedef __int8 int8_t;
typedef unsigned __int8 uint8_t;

typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;

#else
#include <stdint.h>
#endif

#ifndef  SAFEDEL
#define SAFEDEL(x) {if (x != NULL) {delete x; x=NULL;}}
#endif
        
class PixelRegion;


class ImageRegion
{
public:
    ImageRegion(uint8_t iDims);
    ImageRegion(ImageRegion *pRegion1, ImageRegion *pRegion2);

    ~ImageRegion();

    uint8_t getDims()
    {
        return m_iDims;
    }

    bool addPixel(PixelRegion *pRegion, float *puintColor);
    ImageRegion *addRegion(ImageRegion *pRegion);

    int size();
    float* getAvgColor();

    void setPixelsColor(float *pImage);
    void setPixelsCluster(uint32_t *pImage, uint32_t iRegion);


    int regionId; //test
private:

    float* a_fAvgColor;
    uint8_t m_iDims;
protected:
    std::vector< PixelRegion* > *m_pPixels;
    std::vector< std::vector<float> > *m_pPixelColors;
};

#endif // IGEM_SRM_REGION_H
