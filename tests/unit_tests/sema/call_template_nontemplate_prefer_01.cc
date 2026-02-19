// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct NonTemplateTag {};
struct TemplateTag {};

NonTemplateTag pick(long);

template <typename T>
TemplateTag pick(T);

static_assert(__is_same(decltype(pick(1L)), NonTemplateTag),
              "non-template should be preferred on exact tie");
static_assert(__is_same(decltype(pick(1)), TemplateTag),
              "template remains viable and selected when better");
