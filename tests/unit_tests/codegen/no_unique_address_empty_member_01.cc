// RUN: %cxx -target x86_64-linux-gnu -emit-llvm %s -o - | %filecheck %s --check-prefix=DEFAULT
// RUN: %cxx -target x86_64-linux-gnu -emit-llvm %s -o - | %filecheck %s --check-prefix=PRVALUE

struct Present {};

struct Slot {
  int* key = nullptr;
  [[no_unique_address]] Present value{};
};

struct Tagged {
  int* key;
  [[no_unique_address]] Present value;
};

Present makePresent();

Slot defaultMember(int* key) { return Slot{key}; }

Tagged fromPrvalue(int* key) { return Tagged{key, makePresent()}; }

// DEFAULT-LABEL: define {{.*}}i64 @_Z13defaultMemberPi(
// DEFAULT: getelementptr inbounds %Slot, ptr %{{[0-9]+}}, i32 0, i32 0
// DEFAULT: store ptr
// DEFAULT-NOT: {{store|memset|memcpy}}
// DEFAULT: ret i64

// PRVALUE-LABEL: define {{.*}}i64 @_Z11fromPrvaluePi(
// PRVALUE: call void @_Z11makePresentv()
// PRVALUE-NOT: {{store|memset|memcpy}}
// PRVALUE: ret i64
