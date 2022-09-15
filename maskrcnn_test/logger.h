#pragma once
#include "logging.h"

constexpr long long operator"" _MiB(long long unsigned val)
{
	return val * (1 << 20);
}

namespace sample
{
	extern Logger gLogger;
	extern LogStreamConsumer gLogVerbose;
	extern LogStreamConsumer gLogInfo;
	extern LogStreamConsumer gLogWarning;
	extern LogStreamConsumer gLogError;
	extern LogStreamConsumer gLogFatal;

	void setReportableSeverity(Logger::Severity severity);
} // namespace sample
