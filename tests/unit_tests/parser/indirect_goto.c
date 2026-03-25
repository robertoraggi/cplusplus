// RUN: %cxx -verify -fcheck %s

// GCC extension: &&label gives the address of a label as void*
// goto *expr jumps to the address stored in expr

int indirect_goto_basic(void) {
  void* ptr = &&done;
  goto* ptr;
done:
  return 0;
}

int indirect_goto_conditional(int flag) {
  void* ptr = flag ? &&yes : &&no;
  goto* ptr;
yes:
  return 1;
no:
  return 0;
}
