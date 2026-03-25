// RUN: %cxx -emit-mlir %s

char msg[3][10] = {"hello", "world", "test"};

char gammamsg[5][26] = {
    "gamma correction off",     "gamma correction level 1",
    "gamma correction level 2", "gamma correction level 3",
    "gamma correction level 4",
};

int main(void) { return 0; }
