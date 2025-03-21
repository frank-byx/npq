#include "parameters.h"


namespace npq
{

Parameters::Parameters(double targetDistortion, double targetDistortionMargin, dim_t mstMaxDegree)
{
	this->targetDistortion = targetDistortion;
	this->targetDistortionMargin = targetDistortionMargin;
	this->mstMaxDegree = mstMaxDegree;
}

} // namespace npq