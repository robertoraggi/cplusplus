// RUN: %cxx -verify -dump-symbols %s | %filecheck %s

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
// CHECK-NEXT:      constructor defaulted void Codegen(const cxx::Codegen&)
// CHECK-NEXT:      constructor defaulted void Codegen(cxx::Codegen&&)
// CHECK-NEXT:      injected class name Codegen
// CHECK-NEXT:      function void cg_condition(cxx::ExpressionAST*, ir::BasicBlock*, ir::BasicBlock*)
// CHECK-NEXT:      function defaulted cxx::Codegen& operator =(const cxx::Codegen&)
// CHECK-NEXT:      function defaulted cxx::Codegen& operator =(cxx::Codegen&&)
// CHECK-NEXT:      function defaulted void ~Codegen()
