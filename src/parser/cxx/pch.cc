// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cxx/freeze_audit.h>
#include <cxx/pch.h>
#include <cxx/preprocessor.h>
#include <cxx/private/semantic_codec.h>
#include <cxx/time_trace.h>
#include <cxx/translation_unit.h>

#include <format>
#include <limits>

namespace cxx {

namespace {

void encodePreprocessorState(ByteWriter& out,
                             const PreprocessorSnapshot& snapshot) {
  out.str(snapshot.date);
  out.str(snapshot.time);
  out.i32(snapshot.counter);
  out.i32(snapshot.currentPack);

  out.u32(static_cast<std::uint32_t>(snapshot.packStack.size()));
  for (auto value : snapshot.packStack) out.i32(value);

  out.u32(static_cast<std::uint32_t>(snapshot.macros.size()));
  for (const auto& macro : snapshot.macros) {
    out.str(macro.name);
    out.boolean(macro.isFunctionLike);
    out.boolean(macro.isVariadic);

    out.u32(static_cast<std::uint32_t>(macro.formals.size()));
    for (const auto& formal : macro.formals) out.str(formal);

    out.u32(static_cast<std::uint32_t>(macro.body.size()));
    for (const auto& token : macro.body) {
      out.u32(static_cast<std::uint32_t>(token.kind));
      out.str(token.spelling);
      std::uint8_t flags = 0;
      if (token.startOfLine) flags |= 1;
      if (token.leadingSpace) flags |= 2;
      if (token.isFromMacroBody) flags |= 4;
      if (token.noexpand) flags |= 8;
      out.u8(flags);
    }
  }

  out.u32(static_cast<std::uint32_t>(snapshot.undefinedBuiltins.size()));
  for (const auto& name : snapshot.undefinedBuiltins) out.str(name);

  out.u32(static_cast<std::uint32_t>(snapshot.protectedFiles.size()));
  for (const auto& file : snapshot.protectedFiles) {
    out.str(file.fileName);
    out.str(file.headerGuardName);
    out.i32(file.headerProtectionLevel);
    out.boolean(file.pragmaOnceProtected);
    out.boolean(file.isSystemHeader);
  }

  out.u32(static_cast<std::uint32_t>(snapshot.includedFiles.size()));
  for (const auto& [fileName, isSystemHeader] : snapshot.includedFiles) {
    out.str(fileName);
    out.boolean(isSystemHeader);
  }
}

void decodePreprocessorState(ByteReader& in, PreprocessorSnapshot& snapshot) {
  snapshot.date = in.str();
  snapshot.time = in.str();
  snapshot.counter = in.i32();
  snapshot.currentPack = in.i32();

  const auto packStackSize = in.count(4);
  for (std::uint32_t i = 0; in.ok() && i < packStackSize; ++i)
    snapshot.packStack.push_back(in.i32());

  const auto macroCount = in.count(4);
  for (std::uint32_t i = 0; in.ok() && i < macroCount; ++i) {
    MacroRecord macro;
    macro.name = in.str();
    macro.isFunctionLike = in.boolean();
    macro.isVariadic = in.boolean();

    const auto formalCount = in.count(4);
    for (std::uint32_t j = 0; in.ok() && j < formalCount; ++j)
      macro.formals.push_back(in.str());

    const auto bodySize = in.count(4);
    for (std::uint32_t j = 0; in.ok() && j < bodySize; ++j) {
      PreprocessingTokenRecord token;

      const auto kind = in.u32();

      if (kind > std::numeric_limits<std::uint8_t>::max()) {
        in.fail();
        break;
      }

      token.kind = static_cast<TokenKind>(kind);
      token.spelling = in.str();
      const auto flags = in.u8();
      token.startOfLine = (flags & 1) != 0;
      token.leadingSpace = (flags & 2) != 0;
      token.isFromMacroBody = (flags & 4) != 0;
      token.noexpand = (flags & 8) != 0;
      macro.body.push_back(std::move(token));
    }

    snapshot.macros.push_back(std::move(macro));
  }

  const auto undefinedCount = in.count(4);
  for (std::uint32_t i = 0; in.ok() && i < undefinedCount; ++i)
    snapshot.undefinedBuiltins.push_back(in.str());

  const auto protectedCount = in.count(4);
  for (std::uint32_t i = 0; in.ok() && i < protectedCount; ++i) {
    ProtectedFileRecord file;
    file.fileName = in.str();
    file.headerGuardName = in.str();
    file.headerProtectionLevel = in.i32();
    file.pragmaOnceProtected = in.boolean();
    file.isSystemHeader = in.boolean();
    snapshot.protectedFiles.push_back(std::move(file));
  }

  const auto includedCount = in.count(4);
  for (std::uint32_t i = 0; in.ok() && i < includedCount; ++i) {
    auto fileName = in.str();
    const auto isSystemHeader = in.boolean();
    snapshot.includedFiles.emplace_back(std::move(fileName), isSystemHeader);
  }
}

}  // namespace

auto precompiledHeaderSerializationAbi() -> std::string {
  return std::format("cxx/{}.{}", ArchiveWriter::kSchemaMajor,
                     ArchiveWriter::kSchemaMinor);
}

auto PrecompiledHeaderWriter::operator()() -> std::vector<std::uint8_t> {
  TimeTrace::Scope trace{unit_->timeTrace(), "Write precompiled header"};

  FreezeAudit audit{unit_};

  if (!audit.checkSemanticBoundary()) {
    errors_ = audit.errors();
    return {};
  }

  ArchiveWriter archive;

  ArchiveEnvelope envelope;
  envelope.kind = ArchiveKind::kPrecompiledHeader;
  envelope.serializationAbi = precompiledHeaderSerializationAbi();
  envelope.targetKey = keys_.targetKey;
  envelope.languageKey = keys_.languageKey;
  envelope.optionDigest = keys_.optionDigest;
  envelope.artifactIdentity = unit_->fileName();
  envelope.dependencies = dependencies_;
  archive.setEnvelope(std::move(envelope));

  ByteWriter preprocessor;
  encodePreprocessorState(preprocessor, preprocessorState_);
  archive.addSection(ArchiveSection::kPreprocessor, preprocessor.take());

  SemanticEncoder encoder{unit_};

  TimeTrace::Scope encodeTrace{unit_->timeTrace(), "Encode semantic graph"};

  if (!encoder(unit_->semanticArchiveRoots(), archive)) {
    errors_ = encoder.errors();
    return {};
  }

  return archive();
}

auto PrecompiledHeaderReader::operator()(std::span<const std::uint8_t> data)
    -> bool {
  TimeTrace::Scope trace{unit_->timeTrace(), "Read precompiled header",
                         std::format("{} bytes", data.size())};

  ArchiveReader archive;
  archive.setVerifyChecksums(verifyChecksums_);

  if (!archive(data)) {
    error_ = archive.error();
    return false;
  }

  const auto& envelope = archive.envelope();

  if (envelope.kind != ArchiveKind::kPrecompiledHeader) {
    error_ = "not a precompiled header";
    return false;
  }

  if (envelope.serializationAbi != precompiledHeaderSerializationAbi()) {
    error_ = "precompiled header was built by a different compiler version";
    return false;
  }

  if (envelope.targetKey != keys_.targetKey) {
    error_ = "precompiled header was built for a different target";
    return false;
  }

  if (envelope.languageKey != keys_.languageKey) {
    error_ = "precompiled header was built for a different language mode";
    return false;
  }

  if (envelope.optionDigest != keys_.optionDigest) {
    error_ = "precompiled header was built with different options";
    return false;
  }

  PreprocessorSnapshot preprocessorState;

  {
    auto section = archive.section(ArchiveSection::kPreprocessor);
    decodePreprocessorState(section, preprocessorState);
    if (!section.ok()) {
      error_ = "precompiled header preprocessor state is truncated";
      return false;
    }
  }

  // Decode into a shard first: on failure the active unit must not contain a
  // partially loaded prefix (7.6).
  SemanticDecoder decoder{unit_};
  SemanticArchiveRoots roots;

  TimeTrace::Scope decodeTrace{unit_->timeTrace(), "Decode semantic graph"};

  if (!decoder(archive, roots)) {
    error_ = decoder.error();
    return false;
  }

  if (!roots.globalScope) {
    error_ = "precompiled header has no global scope";
    return false;
  }

  dependencies_ = envelope.dependencies;

  unit_->preprocessor()->restore(preprocessorState);
  unit_->adoptPrefix(std::move(roots), decoder.takeSourceMap());

  return true;
}

}  // namespace cxx
