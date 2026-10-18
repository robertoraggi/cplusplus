#include <cxx/memory_layout.h>
#include <cxx/spirv_toolchain.h>

#include <format>

namespace cxx {

SPIRVToolchain::SPIRVToolchain(Preprocessor* preprocessor, Triple triple)
    : Toolchain(preprocessor, std::move(triple)) {
  setMemoryLayout(std::make_unique<MemoryLayout>(this->triple()));
}

void SPIRVToolchain::addSystemIncludePaths() { addBuiltinIncludePath(); }

void SPIRVToolchain::addSystemCppIncludePaths() {}

void SPIRVToolchain::addPredefinedMacros() {
  addCommonMacros();
  if (language() == LanguageKind::kCXX)
    addCommonCxx26Macros();
  else
    addCommonC23Macros();
  addSPIRVMacros();
  if (memoryLayout()->deviceProfile() == DeviceProfile::kVulkan11F32)
    defineMacro("__CXX_VULKAN11_F32__", "1");
}

void Toolchain::addSPIRVMacros() {
  auto bits = memoryLayout()->bits();
  auto bytes = std::to_string(memoryLayout()->sizeOfPointer());
  defineMacro("__SPIRV", "1");
  defineMacro("__SPIRV__", "1");
  defineMacro("SPIRV", "1");
  auto arch = std::format("SPIRV{}", bits);
  defineMacro(arch, "1");
  defineMacro("__" + arch, "1");
  defineMacro("__" + arch + "__", "1");
  defineMacro("__BIGGEST_ALIGNMENT__", "8");
  defineMacro("__LONG_WIDTH__", "64");
  defineMacro("__LONG_MAX__", "9223372036854775807L");
  defineMacro("__SIZEOF_LONG__", "8");
  defineMacro("__SIZEOF_LONG_DOUBLE__", "8");
  defineMacro("__SIZEOF_POINTER__", bytes);
  defineMacro("__POINTER_WIDTH__", std::to_string(bits));
  defineMacro("__SIZEOF_SIZE_T__", bytes);
  defineMacro("__SIZEOF_PTRDIFF_T__", bytes);
  defineMacro("__SIZEOF_WCHAR_T__", "4");
  defineMacro("__SIZEOF_WINT_T__", "4");
  defineMacro("__WCHAR_TYPE__", "int");
  defineMacro("__WCHAR_MAX__", "2147483647");
  defineMacro("__WCHAR_MIN__", "(-__WCHAR_MAX__ - 1)");
  defineMacro("__WCHAR_WIDTH__", "32");
  defineMacro("__WINT_TYPE__", "int");
  defineMacro("__WINT_MAX__", "2147483647");
  defineMacro("__WINT_WIDTH__", "32");
  defineMacro("__clang_wide_literal_encoding__", "\"UTF-32\"");
  defineMacro("__INT64_TYPE__", "long long int");
  defineMacro("__INT64_MAX__", "9223372036854775807LL");
  defineMacro("__INT64_C(c)", "c##LL");
  defineMacro("__INT64_C_SUFFIX__", "LL");
  defineMacro("__UINT64_TYPE__", "long long unsigned int");
  defineMacro("__UINT64_MAX__", "18446744073709551615ULL");
  defineMacro("__UINT64_C(c)", "c##ULL");
  defineMacro("__UINT64_C_SUFFIX__", "ULL");
  defineMacro("__INTMAX_TYPE__", "long long int");
  defineMacro("__INTMAX_MAX__", "9223372036854775807LL");
  defineMacro("__INTMAX_C(c)", "c##LL");
  defineMacro("__INTMAX_C_SUFFIX__", "LL");
  defineMacro("__UINTMAX_TYPE__", "long long unsigned int");
  defineMacro("__UINTMAX_MAX__", "18446744073709551615ULL");
  defineMacro("__UINTMAX_C(c)", "c##ULL");
  defineMacro("__UINTMAX_C_SUFFIX__", "ULL");
  for (const auto* prefix : {"__INT64", "__INTMAX"}) {
    defineMacro(std::format("{}_FMTd__", prefix), "\"lld\"");
    defineMacro(std::format("{}_FMTi__", prefix), "\"lli\"");
  }
  for (const auto* prefix : {"__UINT64", "__UINTMAX"}) {
    for (auto suffix : {"X", "o", "u", "x"})
      defineMacro(std::format("{}_FMT{}__", prefix, suffix),
                  std::format("\"ll{}\"", suffix));
  }
  for (const auto* prefix : {"__INT_FAST64", "__INT_LEAST64"}) {
    defineMacro(std::format("{}_TYPE__", prefix), "long int");
    defineMacro(std::format("{}_MAX__", prefix), "9223372036854775807L");
    defineMacro(std::format("{}_FMTd__", prefix), "\"ld\"");
    defineMacro(std::format("{}_FMTi__", prefix), "\"li\"");
  }
  for (const auto* prefix : {"__UINT_FAST64", "__UINT_LEAST64"}) {
    defineMacro(std::format("{}_TYPE__", prefix), "long unsigned int");
    defineMacro(std::format("{}_MAX__", prefix), "18446744073709551615UL");
    for (auto suffix : {"X", "o", "u", "x"})
      defineMacro(std::format("{}_FMT{}__", prefix, suffix),
                  std::format("\"l{}\"", suffix));
  }
  defineMacro("__LDBL_MANT_DIG__", "53");
  defineMacro("__LDBL_DECIMAL_DIG__", "17");
  defineMacro("__LDBL_DIG__", "15");
  defineMacro("__LDBL_EPSILON__", "2.2204460492503131e-16L");
  defineMacro("__LDBL_DENORM_MIN__", "4.9406564584124654e-324L");
  defineMacro("__LDBL_MAX__", "1.7976931348623157e+308L");
  defineMacro("__LDBL_MIN__", "2.2250738585072014e-308L");
  defineMacro("__LDBL_NORM_MAX__", "1.7976931348623157e+308L");
  defineMacro("__LDBL_MAX_EXP__", "1024");
  defineMacro("__LDBL_MAX_10_EXP__", "308");
  defineMacro("__LDBL_MIN_EXP__", "(-1021)");
  defineMacro("__LDBL_MIN_10_EXP__", "(-307)");
  defineMacro("__USER_LABEL_PREFIX__", "");
  if (definesStrictAnsi()) defineMacro("__STRICT_ANSI__", "1");
  if (language() == LanguageKind::kCXX)
    defineMacro("__STDCPP_DEFAULT_NEW_ALIGNMENT__",
                std::format("{}U", memoryLayout()->defaultNewAlignment()));

  std::string signedType = "int";
  std::string unsignedType = "unsigned int";
  std::string signedMax = "2147483647";
  std::string unsignedMax = "4294967295U";
  std::string formatPrefix;
  if (bits == 64) {
    signedType = "long int";
    unsignedType = "long unsigned int";
    signedMax = "9223372036854775807L";
    unsignedMax = "18446744073709551615UL";
    formatPrefix = "l";
    defineMacro("__LP64__", "1");
    defineMacro("_LP64", "1");
  }
  defineMacro("__SIZE_TYPE__", unsignedType);
  defineMacro("__SIZE_MAX__", unsignedMax);
  defineMacro("__SIZE_WIDTH__", std::to_string(bits));
  for (auto suffix : {"B", "X", "b", "o", "u", "x"})
    defineMacro(std::format("__SIZE_FMT{}__", suffix),
                std::format("\"{}{}\"", formatPrefix, suffix));
  for (const auto* prefix : {"__PTRDIFF", "__INTPTR"}) {
    defineMacro(std::format("{}_TYPE__", prefix), signedType);
    defineMacro(std::format("{}_MAX__", prefix), signedMax);
    defineMacro(std::format("{}_WIDTH__", prefix), std::to_string(bits));
    defineMacro(std::format("{}_FMTd__", prefix),
                std::format("\"{}d\"", formatPrefix));
    defineMacro(std::format("{}_FMTi__", prefix),
                std::format("\"{}i\"", formatPrefix));
  }
  defineMacro("__UINTPTR_TYPE__", unsignedType);
  defineMacro("__UINTPTR_MAX__", unsignedMax);
  defineMacro("__UINTPTR_WIDTH__", std::to_string(bits));
  for (auto suffix : {"B", "X", "b", "o", "u", "x"})
    defineMacro(std::format("__UINTPTR_FMT{}__", suffix),
                std::format("\"{}{}\"", formatPrefix, suffix));
}

}  // namespace cxx
