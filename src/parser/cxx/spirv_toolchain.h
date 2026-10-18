#pragma once

#include <cxx/toolchain.h>

namespace cxx {

class SPIRVToolchain final : public Toolchain {
 public:
  SPIRVToolchain(Preprocessor* preprocessor, Triple triple);

  [[nodiscard]] auto hasThreads() const -> bool override { return false; }

  void addSystemIncludePaths() override;
  void addSystemCppIncludePaths() override;
  void addPredefinedMacros() override;
};

}  // namespace cxx
