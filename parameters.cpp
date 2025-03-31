#include "parameters.h"


namespace npq
{

Parameters::Parameters(id_t trueNumVectors, double targetDistortion, double targetDistortionMargin, dim_t mstMaxDegree)
{
	this->trueNumVectors = trueNumVectors;
	this->targetDistortion = targetDistortion;
	this->targetDistortionMargin = targetDistortionMargin;
	this->mstMaxDegree = mstMaxDegree;
}

} // namespace npq