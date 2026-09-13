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

#pragma once

#include <cxx/ast_fwd.h>
#include <cxx/diagnostic.h>
#include <cxx/diagnostics_client.h>
#include <cxx/literals_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/parser_fwd.h>
#include <cxx/preprocessor_fwd.h>
#include <cxx/semantic_archive.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/time_trace.h>
#include <cxx/token.h>
#include <cxx/type_traits.h>

#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cxx {
class TranslationUnit {
 public:
  explicit TranslationUnit(DiagnosticsClient* diagosticsClient = nullptr);
  ~TranslationUnit();

  void enableTimeTrace() { timeTrace_ = std::make_unique<TimeTrace>(); }
  [[nodiscard]] auto timeTrace() const -> TimeTrace* {
    return timeTrace_.get();
  }

  [[nodiscard]] auto control() const -> Control* { return control_.get(); }

  [[nodiscard]] auto arena() const -> Arena* { return arena_.get(); }

  [[nodiscard]] auto typeTraits() -> TypeTraits;

  [[nodiscard]] auto diagnosticsClient() const -> DiagnosticsClient*;

  [[nodiscard]] auto changeDiagnosticsClient(
      DiagnosticsClient* diagnosticsClient) -> DiagnosticsClient*;

  [[nodiscard]] auto ast() const -> UnitAST* { return ast_; }

  void setAST(UnitAST* ast) { ast_ = ast; }

  [[nodiscard]] auto globalScope() const -> ScopeSymbol*;

  void addPendingMemberInstantiation(ClassSymbol* instance);
  void reopenMemberInstantiation(ClassSymbol* instance);
  [[nodiscard]] auto takePendingMemberInstantiations()
      -> std::vector<ClassSymbol*>;
  [[nodiscard]] auto beginMemberInstantiation(ClassSymbol* instance) -> bool;

  [[nodiscard]] auto cachedConstraintSatisfaction(
      Symbol* symbol, const std::vector<ExpressionAST*>& constraints,
      const std::vector<TemplateArgument>& arguments) -> std::optional<bool>;

  void cacheConstraintSatisfaction(Symbol* symbol,
                                   std::vector<ExpressionAST*> constraints,
                                   std::vector<TemplateArgument> arguments,
                                   bool value);

  void addPendingBodyCompletion(FunctionSymbol* function);
  [[nodiscard]] auto takePendingBodyCompletions()
      -> std::vector<FunctionSymbol*>;

  void markFunctionBodyUnparsed(FunctionDefinitionAST* definition);
  void markFunctionBodyParsed(FunctionDefinitionAST* definition);
  [[nodiscard]] auto isFunctionBodyUnparsed(
      FunctionDefinitionAST* definition) const -> bool;

  [[nodiscard]] auto isPotentiallyEvaluated() const -> bool {
    return potentiallyEvaluated_;
  }

  class PotentiallyEvaluatedScope {
   public:
    PotentiallyEvaluatedScope(const PotentiallyEvaluatedScope&) = delete;
    auto operator=(const PotentiallyEvaluatedScope&)
        -> PotentiallyEvaluatedScope& = delete;

    PotentiallyEvaluatedScope(TranslationUnit* unit, bool potentiallyEvaluated)
        : unit_(unit), saved_(unit->potentiallyEvaluated_) {
      unit_->potentiallyEvaluated_ = potentiallyEvaluated;
    }

    ~PotentiallyEvaluatedScope() { unit_->potentiallyEvaluated_ = saved_; }

   private:
    TranslationUnit* unit_;
    bool saved_;
  };

  [[nodiscard]] auto isImmediateFunctionContext() const -> bool {
    return immediateFunctionContext_;
  }

  class ImmediateFunctionContextScope {
   public:
    ImmediateFunctionContextScope(const ImmediateFunctionContextScope&) =
        delete;
    auto operator=(const ImmediateFunctionContextScope&)
        -> ImmediateFunctionContextScope& = delete;

    ImmediateFunctionContextScope(TranslationUnit* unit, bool active)
        : unit_(unit), saved_(unit->immediateFunctionContext_) {
      unit_->immediateFunctionContext_ = saved_ || active;
    }

    ~ImmediateFunctionContextScope() {
      unit_->immediateFunctionContext_ = saved_;
    }

   private:
    TranslationUnit* unit_;
    bool saved_;
  };

  [[nodiscard]] auto isDeferredInitializer() const -> bool {
    return deferredInitializer_;
  }

  class DeferredInitializerScope {
   public:
    DeferredInitializerScope(const DeferredInitializerScope&) = delete;
    auto operator=(const DeferredInitializerScope&)
        -> DeferredInitializerScope& = delete;

    DeferredInitializerScope(TranslationUnit* unit, bool active)
        : unit_(unit), saved_(unit->deferredInitializer_) {
      unit_->deferredInitializer_ = saved_ || active;
    }

    ~DeferredInitializerScope() { unit_->deferredInitializer_ = saved_; }

   private:
    TranslationUnit* unit_;
    bool saved_;
  };

  [[nodiscard]] auto reportingDiagnosticsClient() const -> DiagnosticsClient* {
    return reportingDiagnosticsClient_;
  }

  void setReportingDiagnosticsClient(DiagnosticsClient* client) {
    reportingDiagnosticsClient_ = client;
  }

  [[nodiscard]] auto templateInstantiationDepth() const -> int {
    return templateInstantiationDepth_;
  }

  [[nodiscard]] auto isInstantiatingTemplate() const -> bool {
    return templateInstantiationDepth_ > 0;
  }

  class TemplateInstantiationScope {
   public:
    explicit TemplateInstantiationScope(TranslationUnit* unit) : unit_(unit) {
      ++unit_->templateInstantiationDepth_;
    }

    ~TemplateInstantiationScope() { --unit_->templateInstantiationDepth_; }

   private:
    TranslationUnit* unit_;
  };

  static constexpr int kMaxTemplateInstantiationDepth = 1024;

  [[nodiscard]] auto fileName() const -> const std::string&;

  [[nodiscard]] auto preprocessor() const -> Preprocessor* {
    return preprocessor_.get();
  }

  [[nodiscard]] auto language() const -> LanguageKind;

  void parse(ParserConfiguration config = {});

  void beginParsing(ParserConfiguration config = {});

  /**
   * The resume entry point of section 9.5. It accepts a validated prefix that
   * has already been adopted and starts the parser at the base of the
   * consumer's token segment; `beginParsing`'s "there is no AST yet"
   * precondition stays exactly as it is for an ordinary compilation.
   */
  void resumeParsing(ParserConfiguration config = {});

  void resume(ParserConfiguration config = {});

  /**
   * Adopts a decoded prefix as the committed prefix of this unit: the global
   * scope, the open AST declaration list, the identity counters and the
   * end-of-translation-unit queues.
   */
  void adoptPrefix(SemanticArchiveRoots roots,
                   std::unique_ptr<PrefixSourceMap> sourceMap);

  [[nodiscard]] auto hasAdoptedPrefix() const -> bool {
    return prefixSourceMap_ != nullptr;
  }

  [[nodiscard]] auto prefixSourceLocationInfo(SourceLocation loc) const
      -> std::optional<PrefixSourceLocationInfo>;

  /**
   * Whether a location belongs to the source file being compiled. This is a
   * location question, not a token question: a location in an adopted prefix
   * has no token in this unit, and a prefix is never the main file.
   */
  [[nodiscard]] auto isMainFileLocation(SourceLocation loc) const -> bool;

  /** Whether a location belongs to the builtin declarations. */
  [[nodiscard]] auto isBuiltinsLocation(SourceLocation loc) const -> bool;

  [[nodiscard]] auto continueParsing() -> ParsingState;

  void endParsing();

  [[nodiscard]] auto config() const -> const ParserConfiguration&;

  void setSource(std::string source, std::string fileName);

  void beginPreprocessing(std::string source, std::string fileName);

  [[nodiscard]] auto continuePreprocessing() -> PreprocessingState;

  void endPreprocessing();

  [[nodiscard]] auto fatalErrors() const -> bool;
  void setFatalErrors(bool fatalErrors);

  auto blockErrors(bool blockErrors = true) -> bool;

  void error(SourceLocation loc, std::string message) const;
  void warning(SourceLocation loc, std::string message) const;
  void note(SourceLocation loc, std::string message) const;

  [[nodiscard]] inline auto tokenCount() const -> unsigned {
    return static_cast<unsigned>(tokens_.size());
  }

  [[nodiscard]] inline auto tokens() const -> const std::vector<Token>& {
    return tokens_;
  }

  [[nodiscard]] inline auto tokenSegmentBase() const -> unsigned {
    return tokenSegmentBase_;
  }

  void setTokenSegmentBase(unsigned base) {
    const auto available = std::numeric_limits<unsigned>::max() - base;
    if (tokens_.size() > available) {
      cxx_runtime_error("source location range overflows");
    }
    tokenSegmentBase_ = base;
  }

  [[nodiscard]] inline auto ownsLocation(SourceLocation loc) const -> bool {
    if (!loc) return false;
    if (loc.index() < tokenSegmentBase_) return false;
    return loc.index() - tokenSegmentBase_ < tokens_.size();
  }

  [[nodiscard]] inline auto locationOfIndex(unsigned index) const
      -> SourceLocation {
    return SourceLocation(tokenSegmentBase_ + index);
  }

  [[nodiscard]] inline auto indexOfLocation(SourceLocation loc) const
      -> unsigned {
    if (!ownsLocation(loc)) {
      cxx_runtime_error(std::format(
          "source location {} is outside the current token segment [{}, {})",
          loc.index(), tokenSegmentBase_, tokenSegmentBase_ + tokens_.size()));
    }
    return loc.index() - tokenSegmentBase_;
  }

  [[nodiscard]] inline auto tokenForDiagnostic(SourceLocation loc) const
      -> const Token& {
    static const Token nullToken{};
    if (!loc || !ownsLocation(loc)) return nullToken;
    return tokenAt(loc);
  }

  [[nodiscard]] inline auto tokenAt(SourceLocation loc) const -> const Token& {
    if (tokens_.empty()) cxx_runtime_error("translation unit has no tokens");
    if (!loc) return tokens_.front();
    return tokens_[indexOfLocation(loc)];
  }

  [[nodiscard]] inline auto tokenAtIndex(unsigned index) const -> const Token& {
    if (index >= tokens_.size()) return tokens_.back();
    return tokens_[index];
  }

  void setTokenKind(SourceLocation loc, TokenKind kind) {
    tokens_[indexOfLocation(loc)].setKind(kind);
  }

  [[nodiscard]] inline auto tokenKind(SourceLocation loc) const -> TokenKind {
    return tokenAt(loc).kind();
  }

  void setTokenValue(SourceLocation loc, TokenValue value) {
    tokens_[indexOfLocation(loc)].setValue(value);
  }

  [[nodiscard]] auto tokenLength(SourceLocation loc) const -> int;

  [[nodiscard]] auto tokenText(SourceLocation loc) const -> const std::string&;

  [[nodiscard]] auto presumedTokenStartPosition(SourceLocation loc) const
      -> SourcePosition;

  [[nodiscard]] auto tokenStartPosition(SourceLocation loc) const
      -> SourcePosition;

  [[nodiscard]] auto tokenEndPosition(SourceLocation loc) const
      -> SourcePosition;

  [[nodiscard]] auto identifier(SourceLocation loc) const -> const Identifier*;

  [[nodiscard]] auto packAlignmentAt(SourceLocation loc) const -> int;

  void extractPackAlignments();

  [[nodiscard]] auto packAlignmentChanges() const
      -> const std::vector<std::pair<unsigned, int>>& {
    return packAlignments_;
  }

  void captureSnippet(SourceLocationRange range);

  [[nodiscard]] auto snippetText(SourceLocationRange range) const
      -> std::string_view;

  [[nodiscard]] auto literal(SourceLocation loc) const -> const Literal*;

  [[nodiscard]] auto semanticArchiveRoots() -> SemanticArchiveRoots;

 private:
  struct ConstraintSatisfaction {
    std::vector<ExpressionAST*> constraints;
    std::vector<TemplateArgument> arguments;
    bool value = false;
  };

  struct ConstraintSatisfactionCache {
    std::vector<ConstraintSatisfaction> entries;
    std::optional<std::size_t> lastIndex;
  };

  std::unique_ptr<Control> control_;
  std::unique_ptr<Arena> arena_;
  std::unique_ptr<Preprocessor> preprocessor_;
  std::unique_ptr<Parser> parser_;
  std::vector<Token> tokens_;
  std::string fileName_;
  UnitAST* ast_ = nullptr;
  unsigned tokenSegmentBase_ = 0;
  const char* yyptr = nullptr;
  DiagnosticsClient* diagnosticsClient_ = nullptr;
  DiagnosticsClient* reportingDiagnosticsClient_ = nullptr;
  NamespaceSymbol* globalNamespace_ = nullptr;
  ParserConfiguration config_;
  std::vector<ClassSymbol*> pendingMemberInstantiations_;
  std::unordered_set<ClassSymbol*> instantiatedMemberClasses_;
  std::vector<FunctionSymbol*> pendingBodyCompletions_;
  std::unordered_set<FunctionDefinitionAST*> unparsedFunctionBodies_;
  std::unordered_map<Symbol*, ConstraintSatisfactionCache>
      constraintSatisfactionCaches_;
  std::unordered_map<std::uint64_t, const Identifier*> snippets_;
  std::vector<std::pair<unsigned, int>> packAlignments_;
  std::unique_ptr<PrefixSourceMap> prefixSourceMap_;
  std::unique_ptr<TimeTrace> timeTrace_;
  int templateInstantiationDepth_ = 0;
  bool potentiallyEvaluated_ = true;
  bool immediateFunctionContext_ = false;
  bool deferredInitializer_ = false;
};
}  // namespace cxx
