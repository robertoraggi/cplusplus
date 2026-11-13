// RUN: %cxx -std=c++20 -verify -fsyntax-only %s

int value;
auto initCapture = [Fixed = 1](int Fixed) { return Fixed; }; // expected-error {{parameter 'Fixed' has the same name as a capture of the lambda-expression}}
auto referenceInitCapture = [&Fixed = value](int Fixed) { return Fixed; }; // expected-error {{parameter 'Fixed' has the same name as a capture of the lambda-expression}}
auto templateInitCapture = [Fixed = 1]<class Fixed>(Fixed v) { return v; }; // expected-error {{template parameter 'Fixed' has the same name as a capture of the lambda-expression}}

void simple() {
  int Fixed = 1;
  auto copy = [Fixed](int Fixed) { return Fixed; }; // expected-error {{parameter 'Fixed' has the same name as a capture of the lambda-expression}}
  auto reference = [&Fixed]<int Fixed>() { return Fixed; }; // expected-error {{template parameter 'Fixed' has the same name as a capture of the lambda-expression}}
  auto distinct = [Fixed](int other) { return Fixed + other; };
}
