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

#include <cxx/arena.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/lexer.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/parser.h>
#include <cxx/preprocessor.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>

#ifndef CXX_NO_FLATBUFFERS
#endif

#include <algorithm>
#include <limits>
#include <ostream>

namespace cxx {
TranslationUnit::TranslationUnit(DiagnosticsClient* diagnosticsClient)
    : control_(std::make_unique<Control>()) {
  diagnosticsClient_ = diagnosticsClient;
  reportingDiagnosticsClient_ = diagnosticsClient;
  arena_ = std::make_unique<Arena>();
  globalNamespace_ = control_->newNamespaceSymbol(nullptr, {});

  preprocessor_ =
      std::make_unique<Preprocessor>(control_.get(), diagnosticsClient_);

  if (diagnosticsClient_) {
    diagnosticsClient_->setSourceResolver(preprocessor_.get());
  }
}

TranslationUnit::~TranslationUnit() {}

auto TranslationUnit::typeTraits() -> TypeTraits { return TypeTraits{this}; }

auto TranslationUnit::diagnosticsClient() const -> DiagnosticsClient* {
  return diagnosticsClient_;
}

auto TranslationUnit::changeDiagnosticsClient(
    DiagnosticsClient* diagnosticsClient) -> DiagnosticsClient* {
  std::swap(diagnosticsClient_, diagnosticsClient);

  if (diagnosticsClient_) {
    diagnosticsClient_->setSourceResolver(preprocessor_.get());
    if (!diagnosticsClient_->isSfinae()) {
      reportingDiagnosticsClient_ = diagnosticsClient_;
    }
  }

  return diagnosticsClient;
}

void TranslationUnit::setSource(std::string source, std::string fileName) {
  beginPreprocessing(std::move(source), std::move(fileName));
  DefaultPreprocessorState state{*preprocessor_};
  while (state) {
    std::visit(state, continuePreprocessing());
  }
  endPreprocessing();
}

void TranslationUnit::beginPreprocessing(std::string source,
                                         std::string fileName) {
  fileName_ = std::move(fileName);
  preprocessor_->beginPreprocessing(std::move(source), fileName_, tokens_);
}

auto TranslationUnit::continuePreprocessing() -> PreprocessingState {
  return preprocessor_->continuePreprocessing(tokens_);
}

void TranslationUnit::endPreprocessing() {
  preprocessor_->endPreprocessing(tokens_);
  extractPackAlignments();
}

void TranslationUnit::extractPackAlignments() {
  const auto available =
      std::numeric_limits<unsigned>::max() - tokenSegmentBase_;

  if (tokens_.size() > available) {
    cxx_runtime_error("source location range overflows");
  }

  std::size_t out = 0;

  for (std::size_t index = 0; index < tokens_.size(); ++index) {
    const auto& token = tokens_[index];

    if (token.is(TokenKind::T_PRAGMA_PACK)) {
      packAlignments_.emplace_back(
          tokenSegmentBase_ + static_cast<unsigned>(out),
          token.value().intValue);
      continue;
    }

    tokens_[out] = token;
    ++out;
  }

  tokens_.resize(out);
}

auto TranslationUnit::packAlignmentAt(SourceLocation loc) const -> int {
  if (packAlignments_.empty()) return 0;

  auto index = tokenSegmentBase_;
  if (loc) index = loc.index();

  const auto it = std::upper_bound(
      packAlignments_.begin(), packAlignments_.end(), index,
      [](unsigned index, const std::pair<unsigned, int>& change) {
        return index < change.first;
      });

  if (it == packAlignments_.begin()) return 0;

  return std::prev(it)->second;
}

auto TranslationUnit::fatalErrors() const -> bool {
  return diagnosticsClient_->fatalErrors();
}

void TranslationUnit::setFatalErrors(bool fatalErrors) {
  diagnosticsClient_->setFatalErrors(fatalErrors);
}

auto TranslationUnit::blockErrors(bool blockErrors) -> bool {
  return diagnosticsClient_->blockErrors(blockErrors);
}

void TranslationUnit::error(SourceLocation loc, std::string message) const {
  diagnosticsClient_->report(tokenForDiagnostic(loc), Severity::Error,
                             std::move(message), loc);
}

void TranslationUnit::warning(SourceLocation loc, std::string message) const {
  TranslationUnit::diagnosticsClient_->report(
      tokenForDiagnostic(loc), Severity::Warning, std::move(message), loc);
}

void TranslationUnit::note(SourceLocation loc, std::string message) const {
  diagnosticsClient_->report(tokenForDiagnostic(loc), Severity::Note,
                             std::move(message), loc);
}

auto TranslationUnit::tokenLength(SourceLocation loc) const -> int {
  const auto& tk = tokenAt(loc);
  if (tk.kind() == TokenKind::T_IDENTIFIER) {
    const std::string* id = tk.value().stringValue;
    return static_cast<int>(id->size());
  }
  return static_cast<int>(Token::spell(tk.kind()).size());
}

auto TranslationUnit::identifier(SourceLocation loc) const
    -> const Identifier* {
  const auto& tk = tokenAt(loc);
  return tk.value().idValue;
}

auto TranslationUnit::literal(SourceLocation loc) const -> const Literal* {
  const auto& tk = tokenAt(loc);
  return tk.value().literalValue;
}

auto TranslationUnit::tokenText(SourceLocation loc) const
    -> const std::string& {
  // A prefix location has no token to spell: the spellings the semantic graph
  // still needs travel as snippets instead (6.8).
  if (prefixSourceLocationInfo(loc)) {
    static const std::string empty;
    return empty;
  }
  const auto& tk = tokenAt(loc);
  switch (tk.kind()) {
    case TokenKind::T_IDENTIFIER:
      return tk.value().idValue->name();

    case TokenKind::T_STRING_LITERAL:
    case TokenKind::T_WIDE_STRING_LITERAL:
    case TokenKind::T_UTF8_STRING_LITERAL:
    case TokenKind::T_UTF16_STRING_LITERAL:
    case TokenKind::T_UTF32_STRING_LITERAL:
    case TokenKind::T_USER_DEFINED_STRING_LITERAL:
    case TokenKind::T_CHARACTER_LITERAL:
    case TokenKind::T_INTEGER_LITERAL:
      return tk.value().literalValue->value();

    default:
      return Token::spell(tk.kind());
  }
}

auto TranslationUnit::presumedTokenStartPosition(SourceLocation loc) const
    -> SourcePosition {
  if (auto info = prefixSourceLocationInfo(loc))
    return {info->presumedFileName, info->presumedLine, info->startColumn};
  return preprocessor_->presumedTokenStartPosition(tokenAt(loc));
}

auto TranslationUnit::tokenStartPosition(SourceLocation loc) const
    -> SourcePosition {
  if (auto info = prefixSourceLocationInfo(loc))
    return {info->fileName, info->startLine, info->startColumn};
  return preprocessor_->tokenStartPosition(tokenAt(loc));
}

auto TranslationUnit::tokenEndPosition(SourceLocation loc) const
    -> SourcePosition {
  if (auto info = prefixSourceLocationInfo(loc))
    return {info->fileName, info->endLine, info->endColumn};
  return preprocessor_->tokenEndPosition(tokenAt(loc));
}

void TranslationUnit::parse(ParserConfiguration config) {
  beginParsing(std::move(config));

  while (!std::holds_alternative<ParsingComplete>(continueParsing())) {
  }

  endParsing();
}

void TranslationUnit::beginParsing(ParserConfiguration config) {
  if (tokens_.empty()) cxx_runtime_error("translation unit has no tokens");

  if (ast_) {
    cxx_runtime_error("translation unit already parsed");
  }

  config_ = std::move(config);

  parser_ = std::make_unique<Parser>(this);
  parser_->beginParsing(ast_);
}

void TranslationUnit::resumeParsing(ParserConfiguration config) {
  if (tokens_.empty()) cxx_runtime_error("translation unit has no tokens");

  if (!prefixSourceMap_) {
    cxx_runtime_error("resumeParsing without an adopted prefix");
  }

  if (!ast_) {
    cxx_runtime_error("the adopted prefix contributed no AST");
  }

  config_ = std::move(config);

  parser_ = std::make_unique<Parser>(this);
  parser_->resumeParsing(ast_);
}

void TranslationUnit::resume(ParserConfiguration config) {
  resumeParsing(std::move(config));

  while (!std::holds_alternative<ParsingComplete>(continueParsing())) {
  }

  endParsing();
}

void TranslationUnit::adoptPrefix(SemanticArchiveRoots roots,
                                  std::unique_ptr<PrefixSourceMap> sourceMap) {
  globalNamespace_ = symbol_cast<NamespaceSymbol>(roots.globalScope);
  ast_ = roots.ast;
  prefixSourceMap_ = std::move(sourceMap);

  control_->setAnonymousIdCount(roots.anonymousIdCount);
  control_->setClosureNameCount(roots.closureNameCount);

  setTokenSegmentBase(roots.prefixTokenCount);

  // The point of instantiation of a prefix request stays at the end of the
  // consumer's translation unit (9.2), so the queues are restored rather than
  // drained at the freeze boundary.
  pendingBodyCompletions_ = std::move(roots.pendingBodyCompletions);
  pendingMemberInstantiations_ = std::move(roots.pendingMemberInstantiations);

  for (auto instance : roots.instantiatedMemberClasses)
    instantiatedMemberClasses_.insert(instance);

  for (const auto& [key, text] : roots.snippets)
    snippets_.emplace(key, control_->getIdentifier(text));
}

auto TranslationUnit::prefixSourceLocationInfo(SourceLocation loc) const
    -> std::optional<PrefixSourceLocationInfo> {
  if (!prefixSourceMap_) return std::nullopt;
  if (!loc) return std::nullopt;
  if (loc.index() >= tokenSegmentBase_) return std::nullopt;
  return prefixSourceMap_->positionOf(loc.index());
}

auto TranslationUnit::isMainFileLocation(SourceLocation loc) const -> bool {
  if (!loc) return false;
  if (!ownsLocation(loc)) return false;
  return tokenAt(loc).fileId() == preprocessor_->mainSourceFileId();
}

auto TranslationUnit::isBuiltinsLocation(SourceLocation loc) const -> bool {
  if (!loc) return false;

  if (auto info = prefixSourceLocationInfo(loc))
    return info->fileName == Preprocessor::kBuiltinsFileName;

  if (!ownsLocation(loc)) return false;

  return tokenAt(loc).fileId() == preprocessor_->builtinsFileId();
}

auto TranslationUnit::semanticArchiveRoots() -> SemanticArchiveRoots {
  SemanticArchiveRoots roots;

  roots.globalScope = globalScope();
  roots.ast = ast_;
  roots.anonymousIdCount = control_->anonymousIdCount();
  roots.closureNameCount = control_->closureNameCount();
  roots.prefixTokenCount = locationOfIndex(tokenCount()).index();
  roots.pendingBodyCompletions = pendingBodyCompletions_;
  roots.pendingMemberInstantiations = pendingMemberInstantiations_;
  roots.instantiatedMemberClasses.assign(instantiatedMemberClasses_.begin(),
                                         instantiatedMemberClasses_.end());

  for (const auto& [key, text] : snippets_)
    roots.snippets.emplace_back(key, text->name());

  std::ranges::sort(roots.snippets);

  // The archive bytes must not depend on hash iteration order (7.5).
  std::ranges::sort(
      roots.instantiatedMemberClasses, {},
      [](ClassSymbol* symbol) { return symbol->location().index(); });

  return roots;
}

auto TranslationUnit::continueParsing() -> ParsingState {
  if (!parser_) return ParsingComplete{};
  return parser_->continueParsing();
}

void TranslationUnit::endParsing() {
  if (!parser_) return;
  parser_->endParsing();
  parser_.reset();
}

auto TranslationUnit::language() const -> LanguageKind {
  return preprocessor_->language();
}

auto TranslationUnit::config() const -> const ParserConfiguration& {
  return config_;
}

auto TranslationUnit::globalScope() const -> ScopeSymbol* {
  if (!globalNamespace_) return nullptr;
  return globalNamespace_;
}

void TranslationUnit::addPendingMemberInstantiation(ClassSymbol* instance) {
  if (!instance) return;
  if (std::ranges::contains(pendingMemberInstantiations_, instance)) return;
  pendingMemberInstantiations_.push_back(instance);
}

void TranslationUnit::reopenMemberInstantiation(ClassSymbol* instance) {
  if (!instance) return;
  instantiatedMemberClasses_.erase(instance);
  addPendingMemberInstantiation(instance);
}

auto TranslationUnit::beginMemberInstantiation(ClassSymbol* instance) -> bool {
  if (!instance) return false;
  return instantiatedMemberClasses_.insert(instance).second;
}

auto TranslationUnit::cachedConstraintSatisfaction(
    Symbol* symbol, const std::vector<ExpressionAST*>& constraints,
    const std::vector<TemplateArgument>& arguments) -> std::optional<bool> {
  if (!symbol) return std::nullopt;
  auto cacheIt = constraintSatisfactionCaches_.find(symbol);
  if (cacheIt == constraintSatisfactionCaches_.end()) return std::nullopt;

  auto& cache = cacheIt->second;
  auto matches = [&](const ConstraintSatisfaction& entry) {
    if (entry.constraints != constraints) return false;
    return compare_args(this, entry.arguments, arguments);
  };

  if (cache.lastIndex) {
    auto index = *cache.lastIndex;
    if (index < cache.entries.size()) {
      if (matches(cache.entries[index])) return cache.entries[index].value;
    }
  }

  for (std::size_t i = 0; i < cache.entries.size(); ++i) {
    if (cache.lastIndex) {
      if (i == *cache.lastIndex) continue;
    }
    if (!matches(cache.entries[i])) continue;
    cache.lastIndex = i;
    return cache.entries[i].value;
  }

  return std::nullopt;
}

void TranslationUnit::cacheConstraintSatisfaction(
    Symbol* symbol, std::vector<ExpressionAST*> constraints,
    std::vector<TemplateArgument> arguments, bool value) {
  if (!symbol) return;
  auto& cache = constraintSatisfactionCaches_[symbol];
  cache.entries.push_back(
      {std::move(constraints), std::move(arguments), value});
  cache.lastIndex = cache.entries.size() - 1;
}

auto TranslationUnit::takePendingMemberInstantiations()
    -> std::vector<ClassSymbol*> {
  auto pending = std::move(pendingMemberInstantiations_);
  pendingMemberInstantiations_.clear();
  return pending;
}

void TranslationUnit::addExplicitInstantiationDefinition(
    FunctionSymbol* function) {
  if (!function) return;
  if (std::ranges::contains(explicitInstantiationDefinitions_, function))
    return;
  explicitInstantiationDefinitions_.push_back(function);
}

auto TranslationUnit::explicitInstantiationDefinitions() const
    -> const std::vector<FunctionSymbol*>& {
  return explicitInstantiationDefinitions_;
}

auto TranslationUnit::isExplicitInstantiationDefinition(
    FunctionSymbol* function) const -> bool {
  if (!function) return false;
  return std::ranges::contains(explicitInstantiationDefinitions_, function);
}

void TranslationUnit::addPendingBodyCompletion(FunctionSymbol* function) {
  if (!function) return;
  if (!function->hasUninstantiatedBody()) return;
  if (!function->isDefinitionRequired()) return;
  if (isEnclosedInDependentTemplate(this, function, true)) return;
  if (std::ranges::contains(pendingBodyCompletions_, function)) return;
  pendingBodyCompletions_.push_back(function);
}

auto TranslationUnit::takePendingBodyCompletions()
    -> std::vector<FunctionSymbol*> {
  auto pending = std::move(pendingBodyCompletions_);
  pendingBodyCompletions_.clear();
  return pending;
}

void TranslationUnit::markFunctionBodyUnparsed(
    FunctionDefinitionAST* definition) {
  if (definition) unparsedFunctionBodies_.insert(definition);
}

void TranslationUnit::markFunctionBodyParsed(
    FunctionDefinitionAST* definition) {
  if (definition) unparsedFunctionBodies_.erase(definition);
}

auto TranslationUnit::isFunctionBodyUnparsed(
    FunctionDefinitionAST* definition) const -> bool {
  return definition && unparsedFunctionBodies_.contains(definition);
}

namespace {

auto snippetKey(SourceLocationRange range) -> std::uint64_t {
  auto [first, last] = range;
  return (std::uint64_t(first.index()) << 32) | last.index();
}

}  // namespace

void TranslationUnit::captureSnippet(SourceLocationRange range) {
  const auto key = snippetKey(range);
  if (snippets_.contains(key)) return;

  auto [first, last] = range;

  // A range outside this unit's token segment belongs to an adopted prefix,
  // whose snippets are restored from the archive rather than re-read (6.8).
  if (first && !ownsLocation(first)) return;

  std::string text;

  for (auto loc = first; loc && loc != last; loc = loc.next()) {
    const auto& token = tokenAt(loc);
    if (loc != first && (token.leadingSpace() || token.startOfLine())) {
      text += ' ';
    }
    text += token.spell();
  }

  snippets_.emplace(key, control_->getIdentifier(text));
}

auto TranslationUnit::snippetText(SourceLocationRange range) const
    -> std::string_view {
  auto it = snippets_.find(snippetKey(range));
  if (it == snippets_.end()) return {};
  return it->second->name();
}

auto TranslationUnit::fileName() const -> const std::string& {
  return fileName_;
}

}  // namespace cxx
