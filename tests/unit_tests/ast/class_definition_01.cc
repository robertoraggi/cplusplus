// RUN: %cxx -verify -ast-dump %s | %filecheck %s --match-full-lines

class EmptyClass {};

class OtherEmptyClass {};

class FinalClass final {};

class DerivedClass : public EmptyClass {};

class DerivedClass2 : public EmptyClass, private OtherEmptyClass {};

class DerivedClass3 : virtual public EmptyClass {};

// clang-format off
//      CHECK:translation-unit
// CHECK-NEXT:  declaration-list
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: class
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: EmptyClass
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: class
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: OtherEmptyClass
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: class
// CHECK-NEXT:          is-final: true
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: FinalClass
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: class
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: DerivedClass
// CHECK-NEXT:          base-specifier-list
// CHECK-NEXT:            base-specifier
// CHECK-NEXT:              access-specifier: public
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: EmptyClass
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: class
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: DerivedClass2
// CHECK-NEXT:          base-specifier-list
// CHECK-NEXT:            base-specifier
// CHECK-NEXT:              access-specifier: public
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: EmptyClass
// CHECK-NEXT:            base-specifier
// CHECK-NEXT:              access-specifier: private
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: OtherEmptyClass
// CHECK-NEXT:    simple-declaration
// CHECK-NEXT:      decl-specifier-list
// CHECK-NEXT:        class-specifier
// CHECK-NEXT:          class-key: class
// CHECK-NEXT:          unqualified-id: name-id
// CHECK-NEXT:            identifier: DerivedClass3
// CHECK-NEXT:          base-specifier-list
// CHECK-NEXT:            base-specifier
// CHECK-NEXT:              is-virtual: true
// CHECK-NEXT:              access-specifier: public
// CHECK-NEXT:              unqualified-id: name-id
// CHECK-NEXT:                identifier: EmptyClass
