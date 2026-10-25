#include <cxx/ast.h>
#include <cxx/codegen/codegen.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <format>

namespace cxx {

struct Codegen::InlineAsmBuilder {
  enum class OutputStorage { Register, Memory, RegisterOrMemory };

  struct Output {
    AsmOperandAST* ast;
    ir::ValueRef address;
    ir::TypeRef type;
    OutputStorage storage;
    bool isReadWrite;
    std::optional<BitFieldAccess> bitField;
  };

  Codegen& gen;
  AsmDeclarationAST* ast;
  std::vector<AsmOperandAST*> operands;
  std::vector<Output> outputs;
  std::vector<ir::ValueRef> arguments;
  std::vector<ir::TypeRef> results;
  std::vector<ir::TypeRef> elementTypes;
  std::string constraints;
  std::string assembly;
  bool hasSideEffects = false;
  bool isGoto = false;

  [[nodiscard]] auto gotoOperandIndex(std::string_view name)
      -> std::optional<std::size_t> {
    std::size_t index = operands.size();
    for (const auto& output : outputs)
      if (output.isReadWrite) ++index;
    for (auto label : ListView{ast->gotoLabelList}) {
      if (label->identifier->name() == name) return index;
      ++index;
    }
    return std::nullopt;
  }

  [[nodiscard]] auto error(SourceLocation loc, std::string_view message)
      -> bool {
    gen.unit_->error(loc, std::string(message));
    return false;
  }

  [[nodiscard]] static auto text(const Literal* literal) -> std::string {
    if (!literal) return {};
    return StringLiteral::Components::from(literal->value(),
                                           StringLiteralEncoding::kNone)
        .value;
  }

  [[nodiscard]] static auto isBitFieldOperand(ExpressionAST* expression)
      -> bool {
    while (auto nested = ast_cast<NestedExpressionAST>(expression))
      expression = nested->expression;
    Symbol* symbol = nullptr;
    if (auto id = ast_cast<IdExpressionAST>(expression)) symbol = id->symbol;
    if (auto member = ast_cast<MemberExpressionAST>(expression))
      symbol = member->symbol;
    auto field = symbol_cast<FieldSymbol>(symbol);
    return field && field->isBitField();
  }

  void addConstraint(std::string_view constraint) {
    if (!constraints.empty()) constraints += ',';
    constraints += constraint;
  }

  void addArgument(ir::ValueRef value, ir::TypeRef elementType = {}) {
    arguments.push_back(value);
    elementTypes.push_back(elementType);
  }

  [[nodiscard]] auto operandIndex(std::string_view name)
      -> std::optional<std::size_t> {
    for (std::size_t index = 0; index < operands.size(); ++index) {
      auto id = operands[index]->symbolicName;
      if (id && id->name() == name) return index;
    }
    return std::nullopt;
  }

  [[nodiscard]] auto collectOperands() -> bool {
    for (auto operand : ListView{ast->outputOperandList})
      operands.push_back(operand);
    for (auto operand : ListView{ast->inputOperandList})
      operands.push_back(operand);
    for (std::size_t index = 0; index < operands.size(); ++index) {
      auto operand = operands[index];
      if (!operand->symbolicName) continue;
      if (operandIndex(operand->symbolicName->name()) != index)
        return error(operand->symbolicNameLoc, "duplicate asm operand name");
    }
    return true;
  }

  [[nodiscard]] auto registerConstraint(std::string_view constraint) const
      -> std::optional<std::string_view> {
    if (constraint == "r") return "r";
    if (gen.control()->memoryLayout()->isAArch64() && constraint == "w")
      return "w";
    if (!gen.control()->memoryLayout()->isX86_64()) return {};
    if (constraint.size() != 1) return {};
    switch (constraint.front()) {
      case 'a':
        return "{ax}";
      case 'b':
        return "{bx}";
      case 'c':
        return "{cx}";
      case 'd':
        return "{dx}";
      default:
        return {};
    }
  }

  [[nodiscard]] auto convertConstraint(std::string& constraint,
                                       SourceLocation loc) -> bool {
    if (constraint.starts_with('[') && constraint.ends_with(']')) {
      auto index = operandIndex(
          std::string_view(constraint).substr(1, constraint.size() - 2));
      if (!index || *index >= outputs.size())
        return error(loc,
                     "asm matching constraint must name an output operand");
      constraint = std::to_string(*index);
      return true;
    }
    if (!constraint.empty() &&
        std::isdigit(static_cast<unsigned char>(constraint.front()))) {
      std::size_t index = 0;
      auto [end, ec] = std::from_chars(
          constraint.data(), constraint.data() + constraint.size(), index);
      if (ec != std::errc{} || end != constraint.data() + constraint.size() ||
          index >= outputs.size())
        return error(loc, "invalid asm matching constraint");
      return true;
    }
    std::string converted;
    for (auto ch : constraint) {
      if (auto code = registerConstraint(std::string_view(&ch, 1))) {
        converted += *code;
        continue;
      }
      if (ch == ',') {
        converted += '|';
        continue;
      }
      if (ch == 'm' || ch == 'i' || ch == 'n') {
        converted += ch;
        continue;
      }
      return error(loc,
                   std::format("unsupported asm constraint '{}'", constraint));
    }
    if (converted.empty()) return error(loc, "empty asm constraint");
    constraint = std::move(converted);
    return true;
  }

  [[nodiscard]] auto addOutput(AsmOperandAST* operand) -> bool {
    auto constraint = text(operand->constraintLiteral);
    if (constraint.empty() ||
        (constraint.front() != '=' && constraint.front() != '+'))
      return error(operand->constraintLiteralLoc,
                   "asm output constraint must begin with '=' or '+'");
    const auto isReadWrite = constraint.front() == '+';
    constraint.erase(0, 1);
    bool earlyClobber = constraint.starts_with('&');
    if (earlyClobber) constraint.erase(0, 1);
    bool allowsRegister = false;
    for (auto ch : constraint) {
      if (ch == 'm' || ch == ',') continue;
      if (!registerConstraint(std::string_view(&ch, 1)))
        return error(operand->constraintLiteralLoc,
                     "invalid asm output constraint");
      allowsRegister = true;
    }
    if (!convertConstraint(constraint, operand->constraintLiteralLoc))
      return false;
    auto expression = operand->expression;
    const auto isMemory = constraint.find('m') != std::string::npos;
    auto storage = OutputStorage::Register;
    if (isMemory) {
      storage = OutputStorage::Memory;
      if (allowsRegister) storage = OutputStorage::RegisterOrMemory;
    }
    if (!expression || !is_lvalue(expression))
      return error(operand->firstSourceLocation(),
                   "asm output operand must be an lvalue");
    if (gen.traits.is_const(expression->type))
      return error(operand->firstSourceLocation(),
                   "asm output operand must be modifiable");
    if (!isMemory && !gen.traits.is_scalar(expression->type))
      return error(operand->firstSourceLocation(),
                   "unsupported asm output operand type");
    auto type = gen.convertType(expression->type);
    auto bitField = gen.bitFieldAccess(expression);
    ir::ValueRef address;
    if (bitField) {
      if (isMemory)
        return error(operand->firstSourceLocation(),
                     "asm memory operand cannot be a bit-field");
      address = gen.newTemp(expression->type, operand->firstSourceLocation());
      if (isReadWrite) {
        auto initial = gen.emitter_.loadBitfield(
            operand->firstSourceLocation(), type, bitField->address,
            Codegen::bitfieldInfo(
                symbol_cast<ClassSymbol>(bitField->field->parent()),
                bitField->info),
            gen.traits.is_signed(expression->type), bitField->isVolatile);
        gen.emitter_.store(operand->firstSourceLocation(), initial, address,
                           gen.getAlignment(expression->type));
      }
    } else {
      address = gen.expression(expression).value;
    }
    if (!address) return false;
    outputs.push_back({operand, address, type, storage, isReadWrite, bitField});
    std::string converted = "=";
    if (earlyClobber) converted += '&';
    if (isMemory) converted += '*';
    converted += constraint;
    addConstraint(converted);
    if (isMemory) {
      addArgument(address, type);
    } else {
      results.push_back(type);
    }
    return true;
  }

  [[nodiscard]] auto addInput(AsmOperandAST* operand) -> bool {
    auto constraint = text(operand->constraintLiteral);
    if (!convertConstraint(constraint, operand->constraintLiteralLoc))
      return false;
    auto expression = operand->expression;
    if (!expression)
      return error(operand->firstSourceLocation(), "missing asm input operand");
    if (isBitFieldOperand(expression) &&
        constraint.find('m') != std::string::npos)
      return error(operand->firstSourceLocation(),
                   "asm memory operand cannot be a bit-field");
    auto value = gen.expression(expression);
    if (!value.value) return false;
    if (constraint.find('m') != std::string::npos) {
      if (!is_lvalue(expression))
        return error(expression->firstSourceLocation(),
                     "asm memory operand must be an lvalue");
      addConstraint("*" + constraint);
      addArgument(value.value, gen.convertType(expression->type));
      return true;
    }
    if (!gen.traits.is_scalar(expression->type))
      return error(expression->firstSourceLocation(),
                   "unsupported asm input operand type");
    if (is_glvalue(expression) && !isBitFieldOperand(expression))
      value.value = gen.emitter_.load(
          expression->firstSourceLocation(), gen.convertType(expression->type),
          value.value, gen.lvalueAccess(expression));
    addConstraint(constraint);
    addArgument(value.value);
    return true;
  }

  void addReadWriteInputs() {
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      auto& output = outputs[index];
      if (!output.isReadWrite) continue;
      if (output.storage == OutputStorage::Memory) {
        addConstraint("*m");
        addArgument(output.address, output.type);
        continue;
      }
      auto expression = output.ast->expression;
      auto value =
          gen.emitter_.load(expression->firstSourceLocation(), output.type,
                            output.address, gen.lvalueAccess(expression));
      addConstraint(std::to_string(index));
      addArgument(value);
    }
  }

  void addClobbers() {
    const auto isX86 = gen.control()->memoryLayout()->isX86_64();
    const auto isAArch64 = gen.control()->memoryLayout()->isAArch64();
    for (auto clobber : ListView{ast->clobberList}) {
      auto name = text(clobber->literal);
      if (name.starts_with('%')) name.erase(0, 1);
      if (name == "cc") {
        if (isAArch64)
          name = "cc";
        else if (isX86)
          name = "flags";
        else
          continue;
      }
      addConstraint(std::format("~{{{}}}", name));
    }
    if (isX86) {
      addConstraint("~{dirflag}");
      addConstraint("~{fpsr}");
      if (constraints.find("~{flags}") == std::string::npos)
        addConstraint("~{flags}");
    }
  }

  [[nodiscard]] auto convertAssembly() -> bool {
    const auto source = text(ast->literal);
    for (std::size_t pos = 0; pos < source.size(); ++pos) {
      auto ch = source[pos];
      if (ch == '$') {
        assembly += "$$";
        continue;
      }
      if (ast->colonLoc && gen.control()->memoryLayout()->isX86_64()) {
        if (ch == '{') {
          assembly += "$(";
          continue;
        }
        if (ch == '|') {
          assembly += "$|";
          continue;
        }
        if (ch == '}') {
          assembly += "$)";
          continue;
        }
      }
      if (!ast->colonLoc || ch != '%') {
        assembly += ch;
        continue;
      }
      if (++pos == source.size())
        return error(ast->literalLoc, "incomplete asm operand reference");
      ch = source[pos];
      if (ch == '%' || ch == '{' || ch == '|' || ch == '}') {
        assembly += ch;
        continue;
      }
      if (ch == '=') {
        assembly += "${:uid}";
        continue;
      }
      char modifier = 0;
      if (std::isalpha(static_cast<unsigned char>(ch))) {
        modifier = ch;
        if (++pos == source.size())
          return error(ast->literalLoc, "incomplete asm operand reference");
        ch = source[pos];
      }
      std::optional<std::size_t> index;
      if (ch == '[') {
        auto end = source.find(']', pos + 1);
        if (end == std::string::npos)
          return error(ast->literalLoc, "unterminated asm operand name");
        auto name = std::string_view(source).substr(pos + 1, end - pos - 1);
        if (modifier == 'l')
          index = gotoOperandIndex(name);
        else
          index = operandIndex(name);
        pos = end;
      } else if (std::isdigit(static_cast<unsigned char>(ch))) {
        std::size_t value = 0;
        const auto begin = source.data() + pos;
        auto [end, ec] =
            std::from_chars(begin, source.data() + source.size(), value);
        if (ec == std::errc{}) index = value;
        pos += end - begin - 1;
      }
      auto count = operands.size();
      if (modifier == 'l') {
        for (const auto& output : outputs)
          if (output.isReadWrite) ++count;
        for (auto label : ListView{ast->gotoLabelList}) ++count;
      }
      if (!index || *index >= count)
        return error(ast->literalLoc, "invalid asm operand reference");
      if (modifier) {
        assembly += std::format("${{{}:{}}}", *index, modifier);
      } else {
        assembly += std::format("${}", *index);
      }
    }
    return true;
  }

  void storeOutputs(std::span<const ir::ValueRef> values,
                    ir::ValueRef packed = {}) {
    std::size_t index = 0;
    for (auto& output : outputs) {
      if (output.storage != OutputStorage::Register) continue;
      auto expression = output.ast->expression;
      auto loc = expression->firstSourceLocation();
      ir::ValueRef value;
      if (packed) {
        value = packed;
        if (results.size() > 1)
          value = gen.emitter_.extractValue(loc, output.type, packed, index);
      } else {
        value = values[index];
      }
      ++index;
      if (output.bitField) {
        gen.emitter_.storeBitfield(
            loc, value, output.bitField->address,
            Codegen::bitfieldInfo(
                symbol_cast<ClassSymbol>(output.bitField->field->parent()),
                output.bitField->info),
            output.bitField->isVolatile);
        continue;
      }
      gen.emitter_.store(loc, value, output.address,
                         gen.lvalueAccess(expression));
    }
  }

  void emit() {
    if (!gen.function_) {
      gen.emitter_.moduleAsm(ast->asmLoc, text(ast->literal));
      return;
    }
    hasSideEffects = !ast->outputOperandList;
    for (auto qualifier : ListView{ast->asmQualifierList}) {
      if (qualifier->qualifier == TokenKind::T_VOLATILE) hasSideEffects = true;
      if (qualifier->qualifier == TokenKind::T_GOTO) {
        isGoto = true;
        hasSideEffects = true;
      }
    }
    if (!collectOperands()) return;
    for (auto operand : ListView{ast->outputOperandList})
      if (!addOutput(operand)) return;
    for (auto operand : ListView{ast->inputOperandList})
      if (!addInput(operand)) return;
    addReadWriteInputs();
    for (auto label : ListView{ast->gotoLabelList}) addConstraint("!i");
    addClobbers();
    if (!convertAssembly()) return;
    ir::InlineAsmInfo info{.asmString = assembly,
                           .constraints = constraints,
                           .arguments = arguments,
                           .results = results,
                           .argumentElementTypes = elementTypes,
                           .hasSideEffects = hasSideEffects};
    if (!isGoto) {
      storeOutputs(gen.emitter_.inlineAsm(ast->asmLoc, info));
      return;
    }
    auto fallthrough = gen.newBlock();
    std::vector<ir::BlockRef> destinations;
    for (auto label : ListView{ast->gotoLabelList})
      destinations.push_back(gen.newBlock());
    auto packed = gen.emitter_.inlineAsmGoto(ast->asmLoc, info, fallthrough,
                                             destinations);
    std::size_t index = 0;
    for (auto label : ListView{ast->gotoLabelList}) {
      gen.emitter_.setInsertionBlock(destinations[index++]);
      storeOutputs({}, packed);
      gen.emitGotoLabel(ast->asmLoc, label->identifier->name());
      gen.branch(ast->asmLoc, fallthrough);
    }
    gen.emitter_.setInsertionBlock(fallthrough);
    storeOutputs({}, packed);
  }
};

void Codegen::inlineAsm(AsmDeclarationAST* ast) {
  InlineAsmBuilder{*this, ast}.emit();
}

}  // namespace cxx
