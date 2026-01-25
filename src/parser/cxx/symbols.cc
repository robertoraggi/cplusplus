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

#include <cxx/symbols.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/types.h>
#include <cxx/util.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {

auto compare_args(const std::vector<TemplateArgument>& args1,
                  const std::vector<TemplateArgument>& args2) -> bool {
  if (args1.size() != args2.size()) return false;
  for (size_t i = 0; i < args1.size(); ++i) {
    const auto& arg = args1[i];
    const auto& otherArg = args2[i];
    auto sym = std::get_if<Symbol*>(&arg);
    auto otherSym = std::get_if<Symbol*>(&otherArg);
    if (!sym || !otherSym) {
      // If either is not a symbol, we cannot compare them
      return false;
    }
    auto var = symbol_cast<VariableSymbol>(*sym);
    auto otherVar = symbol_cast<VariableSymbol>(*otherSym);
    if (var && otherVar) {
      auto cst = std::get<std::intmax_t>(var->constValue().value());
      auto otherCst = std::get<std::intmax_t>(otherVar->constValue().value());
      if (cst != otherCst) return false;
      continue;
    }
    auto symType = (*sym)->type();
    auto otherSymType = (*otherSym)->type();
    if (symType != otherSymType) {
      // If the types are different, we cannot compare them
      return false;
    }
  }
  return true;
};

auto Symbol::EnclosingSymbolIterator::operator++() -> EnclosingSymbolIterator& {
  symbol_ = symbol_->parent();
  return *this;
}

auto Symbol::EnclosingSymbolIterator::operator++(int)
    -> EnclosingSymbolIterator {
  auto it = *this;
  symbol_ = symbol_->parent();
  return it;
}

auto Symbol::templateParameters() -> TemplateParametersSymbol* {
  return symbol_cast<TemplateParametersSymbol>(parent());
}

auto Symbol::hasEnclosingSymbol(Symbol* symbol) const -> bool {
  for (auto enclosingSymbol : enclosingSymbols()) {
    if (enclosingSymbol == symbol) return true;
  }
  return false;
}

auto Symbol::kind() const -> SymbolKind { return kind_; }

auto Symbol::name() const -> const Name* { return name_; }

void Symbol::setName(const Name* name) { name_ = name; }

auto Symbol::type() const -> const Type* { return type_; }

void Symbol::setType(const Type* type) { type_ = type; }

auto Symbol::location() const -> SourceLocation { return location_; }

void Symbol::setLocation(SourceLocation location) { location_ = location; }

auto Symbol::parent() const -> ScopeSymbol* { return parent_; }

void Symbol::setParent(ScopeSymbol* enclosingScope) {
  parent_ = enclosingScope;
}

auto Symbol::next() const -> Symbol* {
  for (auto sym = link_; sym; sym = sym->link_) {
    if (sym->name_ == name_) return sym;
  }
  return nullptr;
}

auto Symbol::enclosingNamespace() const -> NamespaceSymbol* {
  for (auto scope = parent(); scope; scope = scope->parent()) {
    if (auto ns = symbol_cast<NamespaceSymbol>(scope)) {
      return ns;
    }
  }
  return nullptr;
}

auto Symbol::enclosingNonTemplateParametersScope() const -> ScopeSymbol* {
  auto scope = parent();

  while (scope && scope->isTemplateParameters()) {
    scope = scope->parent();
  }

  return scope;
}

ScopeSymbol::ScopeSymbol(SymbolKind kind, ScopeSymbol* enclosingScope)
    : Symbol(kind, enclosingScope) {}

ScopeSymbol::~ScopeSymbol() {}

void ScopeSymbol::addMember(Symbol* symbol) { addSymbol(symbol); }

auto ScopeSymbol::members() const -> const std::vector<Symbol*>& {
  return members_;
}

void ScopeSymbol::reset() {
  for (auto symbol : members_) {
    symbol->link_ = nullptr;
    symbol->setParent(nullptr);
  }
  members_.clear();
  buckets_.clear();
  usingDirectives_.clear();
}

auto ScopeSymbol::isTransparent() const -> bool {
  if (isTemplateParameters()) return true;
  if (isFunctionParameters()) return true;
  return false;
}

void ScopeSymbol::addSymbol(Symbol* symbol) {
  if (symbol->isTemplateParameters()) {
    cxx_runtime_error("trying to add a template parameters symbol to a scope");
    return;
  }

  if (isTemplateParameters()) {
    if (!(symbol->isTypeParameter() || symbol->isTemplateTypeParameter() ||
          symbol->isNonTypeParameter() ||
          symbol->isConstraintTypeParameter())) {
      cxx_runtime_error("invalid symbol in template parameters scope");
    }
  }

  if (!symbol->parent_ || symbol->isFunctionParameters()) {
    symbol->setParent(this);
  }

  members_.push_back(symbol);

  if (name_cast<ConversionFunctionId>(symbol->name())) {
    if (auto functionSymbol = symbol_cast<FunctionSymbol>(symbol)) {
      if (auto classSymbol = symbol_cast<ClassSymbol>(this)) {
        classSymbol->addConversionFunction(functionSymbol);
      }
    }
  }

  if (3 * members_.size() >= 2 * buckets_.size()) {
    rehash();
  } else {
    auto h = symbol->name() ? symbol->name()->hashValue() : 0;
    h = h % buckets_.size();
    symbol->link_ = buckets_[h];
    buckets_[h] = symbol;
  }
}

void ScopeSymbol::rehash() {
  const auto newSize = std::max(std::size_t(8), buckets_.size() * 2);

  buckets_ = std::vector<Symbol*>(newSize);

  for (auto symbol : members_) {
    auto h = symbol->name() ? symbol->name()->hashValue() : 0;
    auto index = h % newSize;
    symbol->link_ = buckets_[index];
    buckets_[index] = symbol;
  }
}

void ScopeSymbol::replaceSymbol(Symbol* symbol, Symbol* newSymbol) {
  if (symbol == newSymbol) return;

  auto it = std::find(members_.begin(), members_.end(), symbol);

  if (it == members_.end()) return;

  *it = newSymbol;

  newSymbol->link_ = symbol->link_;

  auto h = newSymbol->name() ? newSymbol->name()->hashValue() : 0;
  h = h % buckets_.size();

  if (buckets_[h] == symbol) {
    buckets_[h] = newSymbol;
  } else {
    for (auto p = buckets_[h]; p; p = p->link_) {
      if (p->link_ == symbol) {
        p->link_ = newSymbol;
        break;
      }
    }
  }

  symbol->link_ = nullptr;
}

void ScopeSymbol::addUsingDirective(ScopeSymbol* scope) {
  usingDirectives_.push_back(scope);
}

auto ScopeSymbol::find(const Name* name) const -> SymbolChainView {
  if (!members_.empty()) {
    auto h = name ? name->hashValue() : 0;
    h = h % buckets_.size();
    for (auto symbol = buckets_[h]; symbol; symbol = symbol->link_) {
      if (symbol->name() == name) {
        return SymbolChainView{symbol};
      }
    }
  }
  return SymbolChainView{nullptr};
}

auto ScopeSymbol::find(TokenKind op) const -> SymbolChainView {
  if (!members_.empty()) {
    const auto h = OperatorId::hash(op) % buckets_.size();
    for (auto symbol = buckets_[h]; symbol; symbol = symbol->link_) {
      auto id = name_cast<OperatorId>(symbol->name());
      if (id && id->op() == op) return SymbolChainView{symbol};
    }
  }
  return SymbolChainView{nullptr};
}

auto ScopeSymbol::find(const std::string_view& name) const -> SymbolChainView {
  if (!members_.empty()) {
    const auto h = Identifier::hash(name) % buckets_.size();
    for (auto symbol = buckets_[h]; symbol; symbol = symbol->link_) {
      auto id = name_cast<Identifier>(symbol->name());
      if (id && id->name() == name) return SymbolChainView{symbol};
    }
  }
  return SymbolChainView{nullptr};
}

NamespaceSymbol::NamespaceSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

NamespaceSymbol::~NamespaceSymbol() {}

auto NamespaceSymbol::isInline() const -> bool { return isInline_; }

void NamespaceSymbol::setInline(bool isInline) { isInline_ = isInline; }

auto NamespaceSymbol::unnamedNamespace() const -> NamespaceSymbol* {
  return unnamedNamespace_;
}

void NamespaceSymbol::setUnnamedNamespace(NamespaceSymbol* unnamedNamespace) {
  unnamedNamespace_ = unnamedNamespace;
}

auto NamespaceSymbol::anonNamespaceIndex() const -> std::optional<int> {
  if (anonNamespaceIndex_ < 0) return std::nullopt;
  return anonNamespaceIndex_;
}

void NamespaceSymbol::setAnonNamespaceIndex(int index) {
  anonNamespaceIndex_ = index;
}

ConceptSymbol::ConceptSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

ConceptSymbol::~ConceptSymbol() {}

BaseClassSymbol::BaseClassSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

BaseClassSymbol::~BaseClassSymbol() {}

auto BaseClassSymbol::isVirtual() const -> bool { return isVirtual_; }

void BaseClassSymbol::setVirtual(bool isVirtual) { isVirtual_ = isVirtual; }

auto BaseClassSymbol::accessSpecifier() const -> AccessSpecifier {
  return accessSpecifier_;
}

void BaseClassSymbol::setAccessSpecifier(AccessSpecifier accessSpecifier) {
  accessSpecifier_ = accessSpecifier;
}

auto BaseClassSymbol::symbol() const -> Symbol* { return symbol_; }

void BaseClassSymbol::setSymbol(Symbol* symbol) { symbol_ = symbol; }

ClassSymbol::ClassSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

ClassSymbol::~ClassSymbol() {}

auto ClassSymbol::flags() const -> std::uint32_t { return flags_; }

void ClassSymbol::setFlags(std::uint32_t flags) { flags_ = flags; }

auto ClassSymbol::isUnion() const -> bool { return isUnion_; }

void ClassSymbol::setIsUnion(bool isUnion) { isUnion_ = isUnion; }

auto ClassSymbol::isFinal() const -> bool { return isFinal_; }

void ClassSymbol::setFinal(bool isFinal) { isFinal_ = isFinal; }

auto ClassSymbol::baseClasses() const -> const std::vector<BaseClassSymbol*>& {
  return baseClasses_;
}

void ClassSymbol::addBaseClass(BaseClassSymbol* baseClass) {
  baseClasses_.push_back(baseClass);
}

auto ClassSymbol::constructors() const -> const std::vector<FunctionSymbol*>& {
  return constructors_;
}

void ClassSymbol::addConstructor(FunctionSymbol* constructor) {
  constructors_.push_back(constructor);
}

auto ClassSymbol::isComplete() const -> bool { return isComplete_; }

void ClassSymbol::setComplete(bool isComplete) { isComplete_ = isComplete; }

auto ClassSymbol::sizeInBytes() const -> int { return sizeInBytes_; }

void ClassSymbol::setSizeInBytes(int sizeInBytes) {
  sizeInBytes_ = sizeInBytes;
}

auto ClassSymbol::alignment() const -> int { return std::max(alignment_, 1); }

void ClassSymbol::setAlignment(int alignment) { alignment_ = alignment; }

auto ClassSymbol::hasBaseClass(Symbol* symbol) const -> bool {
  std::unordered_set<const ClassSymbol*> processed;
  return hasBaseClass(symbol, processed);
}

auto ClassSymbol::hasBaseClass(
    Symbol* symbol, std::unordered_set<const ClassSymbol*>& processed) const
    -> bool {
  if (!processed.insert(this).second) {
    return false;
  }

  for (auto baseClass : baseClasses_) {
    auto baseClassSymbol = baseClass->symbol();
    if (baseClassSymbol == symbol) return true;
    if (auto baseClassType = type_cast<ClassType>(baseClassSymbol->type())) {
      if (baseClassType->symbol()->hasBaseClass(symbol, processed)) return true;
    }
  }
  return false;
}

auto ClassSymbol::conversionFunctions() const
    -> const std::vector<FunctionSymbol*>& {
  return conversionFunctions_;
}

void ClassSymbol::addConversionFunction(FunctionSymbol* conversionFunction) {
  conversionFunctions_.push_back(conversionFunction);
}

auto ClassSymbol::buildClassLayout(Control* control)
    -> std::expected<bool, std::string> {
  int offset = 0;
  int alignment = 1;

  auto memoryLayout = control->memoryLayout();

  for (auto base : baseClasses()) {
    auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());

    if (!baseClassSymbol) {
      return std::unexpected(
          std::format("base class '{}' not found", to_string(base->name())));
    }

    offset = align_to(offset, baseClassSymbol->alignment());
    offset += baseClassSymbol->sizeInBytes();
    alignment = std::max(alignment, baseClassSymbol->alignment());
  }

  FieldSymbol* lastField = nullptr;

  for (auto member : members()) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field) continue;
    if (field->isStatic()) continue;

    if (lastField && control->is_unbounded_array(lastField->type())) {
      // If the last field is an unbounded array, we cannot compute the offset
      // of the next field, so we report an error.
      return std::unexpected(
          std::format("size of incomplete type '{}'",
                      to_string(lastField->type(), lastField->name())));
    }

    if (!field->alignment()) {
      return std::unexpected(
          std::format("alignment of incomplete type '{}'",
                      to_string(field->type(), field->name())));
    }

    std::optional<std::size_t> size;

    if (control->is_unbounded_array(field->type())) {
      size = 0;
    } else {
      size = memoryLayout->sizeOf(field->type());
    }

    if (!size.has_value()) {
      return std::unexpected(
          std::format("size of incomplete type '{}'",
                      to_string(field->type(), field->name())));
    }

    if (isUnion()) {
      offset = std::max(offset, int(size.value()));
    } else {
      offset = align_to(offset, field->alignment());
      field->setOffset(offset);
      offset += size.value();
    }

    alignment = std::max(alignment, field->alignment());

    lastField = field;
  }

  offset = align_to(offset, alignment);

  setAlignment(alignment);
  setSizeInBytes(offset);

  return true;
}

EnumSymbol::EnumSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

EnumSymbol::~EnumSymbol() {}

auto EnumSymbol::hasFixedUnderlyingType() const -> bool {
  return hasFixedUnderlyingType_;
}

void EnumSymbol::setHasFixedUnderlyingType(bool hasFixedUnderlyingType) {
  hasFixedUnderlyingType_ = hasFixedUnderlyingType;
}

auto EnumSymbol::underlyingType() const -> const Type* {
  return underlyingType_;
}

void EnumSymbol::setUnderlyingType(const Type* underlyingType) {
  underlyingType_ = underlyingType;
}

ScopedEnumSymbol::ScopedEnumSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

ScopedEnumSymbol::~ScopedEnumSymbol() {}

auto ScopedEnumSymbol::underlyingType() const -> const Type* {
  return underlyingType_;
}

void ScopedEnumSymbol::setUnderlyingType(const Type* underlyingType) {
  underlyingType_ = underlyingType;
}

FunctionSymbol::FunctionSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

FunctionSymbol::~FunctionSymbol() {}

auto FunctionSymbol::isDefined() const -> bool { return isDefined_; }

void FunctionSymbol::setDefined(bool isDefined) { isDefined_ = isDefined; }

auto FunctionSymbol::isStatic() const -> bool { return isStatic_; }

void FunctionSymbol::setStatic(bool isStatic) { isStatic_ = isStatic; }

auto FunctionSymbol::isExtern() const -> bool { return isExtern_; }

void FunctionSymbol::setExtern(bool isExtern) { isExtern_ = isExtern; }

auto FunctionSymbol::isFriend() const -> bool { return isFriend_; }

void FunctionSymbol::setFriend(bool isFriend) { isFriend_ = isFriend; }

auto FunctionSymbol::isConstexpr() const -> bool { return isConstexpr_; }

void FunctionSymbol::setConstexpr(bool isConstexpr) {
  isConstexpr_ = isConstexpr;
}

auto FunctionSymbol::isConsteval() const -> bool { return isConsteval_; }

void FunctionSymbol::setConsteval(bool isConsteval) {
  isConsteval_ = isConsteval;
}

auto FunctionSymbol::isInline() const -> bool { return isInline_; }

void FunctionSymbol::setInline(bool isInline) { isInline_ = isInline; }

auto FunctionSymbol::isVirtual() const -> bool { return isVirtual_; }

void FunctionSymbol::setVirtual(bool isVirtual) { isVirtual_ = isVirtual; }

auto FunctionSymbol::isExplicit() const -> bool { return isExplicit_; }

void FunctionSymbol::setExplicit(bool isExplicit) { isExplicit_ = isExplicit; }

auto FunctionSymbol::isDeleted() const -> bool { return isDeleted_; }

void FunctionSymbol::setDeleted(bool isDeleted) { isDeleted_ = isDeleted; }

auto FunctionSymbol::isDefaulted() const -> bool { return isDefaulted_; }

void FunctionSymbol::setDefaulted(bool isDefaulted) {
  isDefaulted_ = isDefaulted;
}

auto FunctionSymbol::isConstructor() const -> bool {
  auto p = symbol_cast<ClassSymbol>(parent());
  if (!p) return false;

  auto functionType = type_cast<FunctionType>(type());
  if (!functionType) return false;
  if (!functionType->returnType()) return false;
  if (functionType->returnType()->kind() != TypeKind::kVoid) {
    return false;
  }

  auto id = name_cast<Identifier>(name());
  if (!id) return false;

  if (p->name() != id) return false;

  return true;
}

auto FunctionSymbol::languageLinkage() const -> LanguageKind {
  return hasCLinkage_ ? LanguageKind::kC : LanguageKind::kCXX;
}

void FunctionSymbol::setLanguageLinkage(LanguageKind linkage) {
  hasCLinkage_ = (linkage == LanguageKind::kC);
}

auto FunctionSymbol::hasCLinkage() const -> bool { return hasCLinkage_; }

OverloadSetSymbol::OverloadSetSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

OverloadSetSymbol::~OverloadSetSymbol() {}

auto OverloadSetSymbol::functions() const
    -> const std::vector<FunctionSymbol*>& {
  return functions_;
}

void OverloadSetSymbol::setFunctions(std::vector<FunctionSymbol*> functions) {
  functions_ = std::move(functions);
}

void OverloadSetSymbol::addFunction(FunctionSymbol* function) {
  functions_.push_back(function);
}

LambdaSymbol::LambdaSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

LambdaSymbol::~LambdaSymbol() {}

auto LambdaSymbol::isConstexpr() const -> bool { return isConstexpr_; }

void LambdaSymbol::setConstexpr(bool isConstexpr) {
  isConstexpr_ = isConstexpr;
}

auto LambdaSymbol::isConsteval() const -> bool { return isConsteval_; }

void LambdaSymbol::setConsteval(bool isConsteval) {
  isConsteval_ = isConsteval;
}

auto LambdaSymbol::isMutable() const -> bool { return isMutable_; }

void LambdaSymbol::setMutable(bool isMutable) { isMutable_ = isMutable; }

auto LambdaSymbol::isStatic() const -> bool { return isStatic_; }

void LambdaSymbol::setStatic(bool isStatic) { isStatic_ = isStatic; }

FunctionParametersSymbol::FunctionParametersSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

FunctionParametersSymbol::~FunctionParametersSymbol() {}

TemplateParametersSymbol::TemplateParametersSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

TemplateParametersSymbol::~TemplateParametersSymbol() {}

auto TemplateParametersSymbol::isExplicitTemplateSpecialization() const
    -> bool {
  return isExplicitTemplateSpecialization_;
}

void TemplateParametersSymbol::setExplicitTemplateSpecialization(
    bool isExplicit) {
  isExplicitTemplateSpecialization_ = isExplicit;
}

BlockSymbol::BlockSymbol(ScopeSymbol* enclosingScope)
    : ScopeSymbol(Kind, enclosingScope) {}

BlockSymbol::~BlockSymbol() {}

TypeAliasSymbol::TypeAliasSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

TypeAliasSymbol::~TypeAliasSymbol() {}

VariableSymbol::VariableSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

VariableSymbol::~VariableSymbol() {}

auto VariableSymbol::isStatic() const -> bool { return isStatic_; }

void VariableSymbol::setStatic(bool isStatic) { isStatic_ = isStatic; }

auto VariableSymbol::isThreadLocal() const -> bool { return isThreadLocal_; }

void VariableSymbol::setThreadLocal(bool isThreadLocal) {
  isThreadLocal_ = isThreadLocal;
}

auto VariableSymbol::isExtern() const -> bool { return isExtern_; }

void VariableSymbol::setExtern(bool isExtern) { isExtern_ = isExtern; }

auto VariableSymbol::isConstexpr() const -> bool { return isConstexpr_; }

void VariableSymbol::setConstexpr(bool isConstexpr) {
  isConstexpr_ = isConstexpr;
}

auto VariableSymbol::isConstinit() const -> bool { return isConstinit_; }

void VariableSymbol::setConstinit(bool isConstinit) {
  isConstinit_ = isConstinit;
}

auto VariableSymbol::isInline() const -> bool { return isInline_; }

void VariableSymbol::setInline(bool isInline) { isInline_ = isInline; }

auto VariableSymbol::initializer() const -> ExpressionAST* {
  return initializer_;
}

void VariableSymbol::setInitializer(ExpressionAST* initializer) {
  initializer_ = initializer;
}

auto VariableSymbol::constValue() const -> const std::optional<ConstValue>& {
  return constValue_;
}

void VariableSymbol::setConstValue(std::optional<ConstValue> value) {
  constValue_ = std::move(value);
}

FieldSymbol::FieldSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

FieldSymbol::~FieldSymbol() {}

auto FieldSymbol::isBitField() const -> bool { return isBitField_; }

void FieldSymbol::setBitField(bool isBitField) { isBitField_ = isBitField; }

auto FieldSymbol::bitFieldOffset() const -> int { return bitFieldOffset_; }

void FieldSymbol::setBitFieldOffset(int bitFieldOffset) {
  bitFieldOffset_ = bitFieldOffset;
}

auto FieldSymbol::bitFieldWidth() const -> const std::optional<ConstValue>& {
  return bitFieldWidth_;
}

void FieldSymbol::setBitFieldWidth(std::optional<ConstValue> bitFieldWidth) {
  bitFieldWidth_ = std::move(bitFieldWidth);
}

auto FieldSymbol::isStatic() const -> bool { return isStatic_; }

void FieldSymbol::setStatic(bool isStatic) { isStatic_ = isStatic; }

auto FieldSymbol::isThreadLocal() const -> bool { return isThreadLocal_; }

void FieldSymbol::setThreadLocal(bool isThreadLocal) {
  isThreadLocal_ = isThreadLocal;
}

auto FieldSymbol::isConstexpr() const -> bool { return isConstexpr_; }

void FieldSymbol::setConstexpr(bool isConstexpr) { isConstexpr_ = isConstexpr; }

auto FieldSymbol::isConstinit() const -> bool { return isConstinit_; }

void FieldSymbol::setConstinit(bool isConstinit) { isConstinit_ = isConstinit; }

auto FieldSymbol::isInline() const -> bool { return isInline_; }

void FieldSymbol::setInline(bool isInline) { isInline_ = isInline; }

auto FieldSymbol::isMutable() const -> bool { return isMutable_; }

void FieldSymbol::setMutable(bool isMutable) { isMutable_ = isMutable; }

auto FieldSymbol::offset() const -> int { return offset_; }

void FieldSymbol::setOffset(int offset) { offset_ = offset; }

auto FieldSymbol::alignment() const -> int { return alignment_; }

void FieldSymbol::setAlignment(int alignment) { alignment_ = alignment; }

ParameterSymbol::ParameterSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

ParameterSymbol::~ParameterSymbol() {}

ParameterPackSymbol::ParameterPackSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

ParameterPackSymbol::~ParameterPackSymbol() {}

auto ParameterPackSymbol::elements() const -> const std::vector<Symbol*>& {
  return elements_;
}

void ParameterPackSymbol::addElement(Symbol* element) {
  elements_.push_back(element);
}

TypeParameterSymbol::TypeParameterSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

TypeParameterSymbol::~TypeParameterSymbol() {}

NonTypeParameterSymbol::NonTypeParameterSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

NonTypeParameterSymbol::~NonTypeParameterSymbol() {}

auto NonTypeParameterSymbol::index() const -> int { return index_; }

void NonTypeParameterSymbol::setIndex(int index) { index_ = index; }

auto NonTypeParameterSymbol::depth() const -> int { return depth_; }

void NonTypeParameterSymbol::setDepth(int depth) { depth_ = depth; }

auto NonTypeParameterSymbol::objectType() const -> const Type* {
  return objectType_;
}

void NonTypeParameterSymbol::setObjectType(const Type* objectType) {
  objectType_ = objectType;
}

auto NonTypeParameterSymbol::isParameterPack() const -> bool {
  return isParameterPack_;
}

void NonTypeParameterSymbol::setParameterPack(bool isParameterPack) {
  isParameterPack_ = isParameterPack;
}

TemplateTypeParameterSymbol::TemplateTypeParameterSymbol(
    ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

TemplateTypeParameterSymbol::~TemplateTypeParameterSymbol() {}

ConstraintTypeParameterSymbol::ConstraintTypeParameterSymbol(
    ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

ConstraintTypeParameterSymbol::~ConstraintTypeParameterSymbol() {}

auto ConstraintTypeParameterSymbol::index() const -> int { return index_; }

void ConstraintTypeParameterSymbol::setIndex(int index) { index_ = index; }

auto ConstraintTypeParameterSymbol::depth() const -> int { return depth_; }

void ConstraintTypeParameterSymbol::setDepth(int depth) { depth_ = depth; }

auto ConstraintTypeParameterSymbol::isParameterPack() const -> bool {
  return isParameterPack_;
}

void ConstraintTypeParameterSymbol::setParameterPack(bool isParameterPack) {
  isParameterPack_ = isParameterPack;
}

EnumeratorSymbol::EnumeratorSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

EnumeratorSymbol::~EnumeratorSymbol() {}

auto EnumeratorSymbol::value() const -> const std::optional<ConstValue>& {
  return value_;
}

void EnumeratorSymbol::setValue(const std::optional<ConstValue>& value) {
  value_ = value;
}

UsingDeclarationSymbol::UsingDeclarationSymbol(ScopeSymbol* enclosingScope)
    : Symbol(Kind, enclosingScope) {}

UsingDeclarationSymbol::~UsingDeclarationSymbol() {}

auto UsingDeclarationSymbol::target() const -> Symbol* { return target_; }

void UsingDeclarationSymbol::setTarget(Symbol* symbol) { target_ = symbol; }

auto UsingDeclarationSymbol::declarator() const -> UsingDeclaratorAST* {
  return declarator_;
}

void UsingDeclarationSymbol::setDeclarator(UsingDeclaratorAST* declarator) {
  declarator_ = declarator;
}

bool is_type(Symbol* symbol) {
  if (!symbol) return false;
  switch (symbol->kind()) {
    case SymbolKind::kTypeParameter:
    case SymbolKind::kConstraintTypeParameter:
    case SymbolKind::kTypeAlias:
    case SymbolKind::kClass:
    case SymbolKind::kEnum:
    case SymbolKind::kScopedEnum:
      return true;
    case SymbolKind::kUsingDeclaration: {
      auto usingDeclaration = symbol_cast<UsingDeclarationSymbol>(symbol);
      return is_type(usingDeclaration->target());
    }
    default:
      return false;
  }  // switch
}

}  // namespace cxx
