#pragma once

#include <array>
#include <cstddef>

struct OneEuroFilterConfig {
  double nominalFrequency{30.0};
  double minCutoff{1.2};
  double beta{0.02};
  double derivativeCutoff{1.0};
  double resetGapSeconds{0.5};
};

class LowPassFilter {
public:
  void reset();
  double filter(double value, double alpha);
  bool initialized() const;

private:
  bool initialized_{false};
  double state_{0.0};
};

class OneEuroLandmarkFilter {
public:
  static constexpr std::size_t kLandmarkCount = 98;
  static constexpr std::size_t kCoordinateCount = kLandmarkCount * 2;
  using LandmarkArray = std::array<float, kCoordinateCount>;

  explicit OneEuroLandmarkFilter(
      const OneEuroFilterConfig &config = OneEuroFilterConfig{});

  void reset();

  LandmarkArray filterAbsolute(const LandmarkArray &absoluteLandmarks,
                               double dtSeconds);

private:
  double alpha(double cutoff, double dtSeconds) const;

  OneEuroFilterConfig config_;
  std::array<LowPassFilter, kCoordinateCount> valueFilters_;
  std::array<LowPassFilter, kCoordinateCount> derivativeFilters_;
  std::array<double, kCoordinateCount> previousRawValues_{};
  bool initialized_{false};
};
