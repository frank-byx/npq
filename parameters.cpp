#include "parameters.h"


namespace npq
{

Parameters::Parameters(id_t trueNumVectors, double targetDistortion, double targetDistortionMargin, dim_t mstMaxDegree, bool useCorrection)
{
	this->trueNumVectors = trueNumVectors;
	this->targetDistortion = targetDistortion;
	this->targetDistortionMargin = targetDistortionMargin;
	this->mstMaxDegree = mstMaxDegree;
	this->useCorrection = useCorrection;
}

} // namespace npq