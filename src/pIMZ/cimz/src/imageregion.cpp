#include "imageregion.h"
#include "srm.h"

ImageRegion::ImageRegion(uint8_t iDims)
: m_iDims(iDims)
{
    m_pPixels = new std::vector< PixelRegion* >();
    m_pPixelColors = new std::vector< std::vector<float> >();

    this->a_fAvgColor = new float[m_iDims];
}


ImageRegion::ImageRegion(ImageRegion *pRegion1, ImageRegion *pRegion2)
: ImageRegion(pRegion1->m_iDims)
{

    m_pPixels->insert(m_pPixels->end(), pRegion1->m_pPixels->begin(), pRegion1->m_pPixels->end());
    m_pPixels->insert(m_pPixels->end(), pRegion2->m_pPixels->begin(), pRegion2->m_pPixels->end());

    m_pPixelColors->insert(m_pPixelColors->end(), pRegion1->m_pPixelColors->begin(), pRegion1->m_pPixelColors->end());
    m_pPixelColors->insert(m_pPixelColors->end(), pRegion2->m_pPixelColors->begin(), pRegion2->m_pPixelColors->end());

    std::vector< PixelRegion* >::iterator oIt;

    for (oIt = m_pPixels->begin(); oIt != m_pPixels->end(); ++oIt)
    {
        (*(oIt))->updateRegion(this);
    }

    //qDebug() << m_pIndices->size();
}


ImageRegion::~ImageRegion()
{
    m_pPixels->clear();
    m_pPixelColors->clear();

    delete(m_pPixels);
    delete(m_pPixelColors);
}


bool ImageRegion::addPixel(PixelRegion *pRegion, float *puintColor)
{
    // new color vector
    float* newAvgCol = new float[m_iDims];

    // factors for re-averaging
	size_t iTotal = m_pPixels->size() + 1;
	float fFac1 = (float)m_pPixels->size() / (float)iTotal;
	float fFac2 = (float)1 / (float)iTotal;

    std::vector<float> vColors(this->m_iDims);

    for (uint8_t d=0; d < m_iDims; ++d)
    {
        newAvgCol[d] = this->a_fAvgColor[d] * fFac1 + puintColor[d] * fFac2;
        vColors[d] = puintColor[d];
    }

    SAFEDEL(this->a_fAvgColor)
    this->a_fAvgColor = newAvgCol;

	m_pPixels->insert(m_pPixels->end(), pRegion);
	m_pPixelColors->insert(m_pPixelColors->end(), vColors);

    return true;
}


ImageRegion* ImageRegion::addRegion(ImageRegion *pRegion)
{
    // new color vector
    float* newAvgCol = new float[m_iDims];

    // factors for re-averaging
	size_t iTotal = m_pPixelColors->size() + pRegion->m_pPixelColors->size();
	float fFac1 = (float)m_pPixelColors->size() / (float)iTotal;
	float fFac2 = (float)pRegion->m_pPixelColors->size() / (float)iTotal;

    for (uint8_t d=0; d < m_iDims; ++d)
    {
        newAvgCol[d] = this->a_fAvgColor[d] * fFac1 + pRegion->a_fAvgColor[d] * fFac2;
    }

    SAFEDEL(this->a_fAvgColor)
    this->a_fAvgColor = newAvgCol;

	m_pPixels->insert(m_pPixels->end(), pRegion->m_pPixels->begin(), pRegion->m_pPixels->end());
	m_pPixelColors->insert(m_pPixelColors->end(), pRegion->m_pPixelColors->begin(), pRegion->m_pPixelColors->end());
	
    std::vector< PixelRegion* >::iterator oIt;
    for (oIt = pRegion->m_pPixels->begin(); oIt != pRegion->m_pPixels->end(); ++oIt)
    {
        (*(oIt))->updateRegion(this);
    }

    return this;
}


int ImageRegion::size()
{
    return m_pPixels->size();
}


float* ImageRegion::getAvgColor()
{
	return a_fAvgColor;
}


void ImageRegion::setPixelsColor(float* pImage)
{

    unsigned int iVecLen = m_pPixels->size();

    if (iVecLen != m_pPixelColors->size())
        exit(1);

    int iIndex = 0;

    //srand(time(NULL) + clock());

    //QRgb oRandomColor = QColor::fromRgbF((float)(rand() % 255) / 255.0f,(float)(rand() % 255) / 255.0f,(float)(rand() % 255) / 255.0f, 1.0f).rgb();
    float* pAvgColor = this->getAvgColor();
    for (int i = 0; i < iVecLen; ++i)
    {
        PixelRegion* pPixel = m_pPixels->at(i);      
        uint32_t iBaseIndex = pPixel->getIndex();

        for (uint8_t d=0; d < m_iDims; ++d)
        {
            pImage[iBaseIndex+d] = pAvgColor[d];
        }
    }

}

void ImageRegion::setPixelsCluster(uint32_t* pImage, uint32_t iRegion)
{

    unsigned int iVecLen = m_pPixels->size();

    if (iVecLen != m_pPixelColors->size())
        exit(1);

    int iIndex = 0;

    //srand(time(NULL) + clock());

    //QRgb oRandomColor = QColor::fromRgbF((float)(rand() % 255) / 255.0f,(float)(rand() % 255) / 255.0f,(float)(rand() % 255) / 255.0f, 1.0f).rgb();
    float* pAvgColor = this->getAvgColor();
    for (int i = 0; i < iVecLen; ++i)
    {
        PixelRegion* pPixel = m_pPixels->at(i);      

        //std::cerr << pPixel->getIndex() << " " << pPixel->getXIndex() << " " << pPixel->getYIndex() << std::endl;

        pImage[pPixel->getIndex()] = iRegion;
    }

}

