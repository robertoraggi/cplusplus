// RUN: %cxx -verify %s

int items[2];

// expected-error@1 {{conflicting declaration of 'items'}}
int items[3];
