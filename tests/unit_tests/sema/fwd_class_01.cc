// RUN: %cxx -verify -fsyntax-only -dump-symbols %s | %filecheck %s

namespace cxx {
class BasicBlock;
class ExpressionAST;
}  // namespace cxx

namespace cxx::ir {
class BasicBlock;
}

namespace cxx {
struct Codegen {
  void cg_condition(ExpressionAST* ast, ir::BasicBlock* iftrue,
                    ir::BasicBlock* iffalse);
};
}  // namespace cxx

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  namespace cxx
// CHECK-NEXT:    class BasicBlock
// CHECK-NEXT:    class ExpressionAST
// CHECK-NEXT:    namespace ir
// CHECK-NEXT:      class BasicBlock
// CHECK-NEXT:    class Codegen
// CHECK-NEXT:      constructor defaulted void Codegen()
// CHECK-NEXT:      constructor defaulted void Codegen(const ::cxx::Codegen&)
// CHECK-NEXT:        parameters
// CHECK-NEXT:          parameter const ::cxx::Codegen&
// CHECK-NEXT:      constructor defaulted void Codegen(::cxx::Codegen&&)
// CHECK-NEXT:        parameters
// CHECK-NEXT:          parameter ::cxx::Codegen&&
// CHECK-NEXT:      injected class name Codegen
// CHECK-NEXT:      function void cg_condition(::cxx::ExpressionAST*, ::cxx::ir::BasicBlock*, ::cxx::ir::BasicBlock*)
// CHECK-NEXT:        parameters
// CHECK-NEXT:          parameter ::cxx::ExpressionAST* ast
// CHECK-NEXT:          parameter ::cxx::ir::BasicBlock* iftrue
// CHECK-NEXT:          parameter ::cxx::ir::BasicBlock* iffalse
// CHECK-NEXT:      function defaulted ::cxx::Codegen& operator =(const ::cxx::Codegen&)
// CHECK-NEXT:        parameters
// CHECK-NEXT:          parameter const ::cxx::Codegen&
// CHECK-NEXT:      function defaulted ::cxx::Codegen& operator =(::cxx::Codegen&&)
// CHECK-NEXT:        parameters
// CHECK-NEXT:          parameter ::cxx::Codegen&&
// CHECK-NEXT:      function defaulted void ~Codegen()
