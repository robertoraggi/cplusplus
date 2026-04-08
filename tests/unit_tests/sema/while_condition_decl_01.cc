// RUN: %cxx -verify %s
// expected-no-diagnostics

struct Node {
  int value;
  Node* next;
  static Node* advance(Node* n) { return n ? n->next : nullptr; }
};

int test_while_condition_decl(Node* head) {
  int sum = 0;
  while (Node* p = Node::advance(head)) {
    sum += p->value;
    head = p;
  }
  return sum;
}

int test_while_condition_int() {
  int n = 5;
  while (int x = n--) {
    (void)x;
  }
  return 0;
}
