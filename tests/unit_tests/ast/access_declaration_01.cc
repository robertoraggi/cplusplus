// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

class SimpleClass {
 private:
 protected:
 public:
};
// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: class
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: SimpleClass
// CHECK-NEXT:          declaration-list
// CHECK-NEXT:            access-declaration
// CHECK-NEXT:              access-specifier: private
// CHECK-NEXT:            access-declaration
// CHECK-NEXT:              access-specifier: protected
// CHECK-NEXT:            access-declaration
// CHECK-NEXT:              access-specifier: public
