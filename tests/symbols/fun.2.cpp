
struct bar {
};

const char* fun(const char* s) { return s; }
int fun(int x) { return x; }
double fun(double d) { return d; }
int *fun(int* b) { return b; }
bar fun(bar b) { return b; }

int x;
double d;
bar b;

decltype(fun("test")) call1;
decltype(fun(0)) call2;
decltype(fun(d)) call3;
decltype(fun(x)) call4;
decltype(fun(b)) call5;
