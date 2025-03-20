#include "parameters.h"


namespace npq
{

Parameters::Parameters(double targetDistortion, double targetDistortionMargin)
{
	this->targetDistortion = targetDistortion;
	this->targetDistortionMargin = targetDistortionMargin;
}

} // namespace npq