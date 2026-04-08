// RUN: %cxx -verify %s

template <bool Flag>
int dispatch(void) {
    void* ptr = Flag ? &&yes : &&no;
    goto *ptr;
yes:
    return 1;
no:
    return 0;
}

int test_true()  { return dispatch<true>(); }
int test_false() { return dispatch<false>(); }
