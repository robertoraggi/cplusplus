// RUN: %cxx -target x86_64-linux-gnu -emit-llvm %s -o - | %filecheck %s --check-prefix=EMPTY
// RUN: %cxx -target x86_64-linux-gnu -emit-llvm %s -o - | %filecheck %s --check-prefix=ARRAY

struct Present {};

struct Slot {
  int* key = nullptr;
  [[no_unique_address]] Present value{};
};

Present* addressOf(Slot& slot) { return &slot.value; }

struct Value {
  int value;
};

struct Shared : virtual Value {};

struct Cell : Shared {
  [[no_unique_address]] Value values[2][2];
};

struct Tag {};

struct Grid : Shared, Tag, virtual Value {
  [[no_unique_address]] Cell cells[2][2];
};

Cell (*cellsOf(Grid& grid))[2][2] { return &grid.cells; }

// EMPTY-LABEL: define {{.*}}ptr @_Z9addressOfR4Slot(
// EMPTY-NOT: getelementptr
// EMPTY: ret ptr

// ARRAY-LABEL: define {{.*}}ptr @_Z7cellsOfR4Grid(
// ARRAY: getelementptr inbounds i8, ptr %{{[0-9]+}}, i64 8
// ARRAY: ret ptr
