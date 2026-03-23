#include "temporal_filter.hpp"

#include <algorithm>
#include <cmath>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

void LowPassFilter::reset() {
  initialized_ = false;
  state_ = 0.0;
}

double LowPassFilter::filter(double value, double alpha) {
  alpha = std::clamp(alpha, 0.0, 1.0);
  if (!initialized_) {
    initialized_ = true;
    state_ = value;
    return state_;
  }

  state_ = alpha * value + (1.0 - alpha) * state_;
  return state_;
}

bool LowPassFilter::initialized() const { return initialized_; }

OneEuroLandmarkFilter::OneEuroLandmarkFilter(const OneEuroFilterConfig &config)
    : config_(config) {}

void OneEuroLandmarkFilter::reset() {
  for (auto &filter : valueFilters_) {
    filter.reset();
  }
  for (auto &filter : derivativeFilters_) {
    filter.reset();
  }
  previousRawValues_.fill(0.0);
  initialized_ = false;
}

OneEuroLandmarkFilter::LandmarkArray
OneEuroLandmarkFilter::filterAbsolute(const LandmarkArray &absoluteLandmarks,
                                      double dtSeconds) {
  LandmarkArray filteredLandmarks{};
  const double fallbackDt =
      config_.nominalFrequency > 0.0 ? 1.0 / config_.nominalFrequency
                                     : 1.0 / 30.0;
  const double safeDt = dtSeconds > 0.0 ? dtSeconds : fallbackDt;

  for (std::size_t i = 0; i < kCoordinateCount; ++i) {
    const double rawValue = absoluteLandmarks[i];
    const double derivative =
        initialized_ ? (rawValue - previousRawValues_[i]) / safeDt : 0.0;
    const double filteredDerivative =
        derivativeFilters_[i].filter(derivative,
                                     alpha(config_.derivativeCutoff, safeDt));
    const double cutoff =
        config_.minCutoff + config_.beta * std::abs(filteredDerivative);
    const double filteredValue =
        valueFilters_[i].filter(rawValue, alpha(cutoff, safeDt));

    filteredLandmarks[i] = static_cast<float>(filteredValue);
    previousRawValues_[i] = rawValue;
  }

  initialized_ = true;
  return filteredLandmarks;
}

double OneEuroLandmarkFilter::alpha(double cutoff, double dtSeconds) const {
  const double safeCutoff = std::max(cutoff, 1e-6);
  const double safeDt = dtSeconds > 0.0
                            ? dtSeconds
                            : (config_.nominalFrequency > 0.0
                                   ? 1.0 / config_.nominalFrequency
                                   : 1.0 / 30.0);
  const double tau = 1.0 / (2.0 * kPi * safeCutoff);
  return 1.0 / (1.0 + tau / safeDt);
}
