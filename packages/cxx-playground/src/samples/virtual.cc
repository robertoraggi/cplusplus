extern "C" {
auto printf(const char* format, ...) -> int;
}

class Shape {
 public:
  virtual auto area() const -> double { return 0.0; }
};

class Rectangle : public Shape {
  double width_;
  double height_;

 public:
  Rectangle(double w, double h) : width_(w), height_(h) {}

  auto area() const -> double override { return width_ * height_; }
};

class Circle : public Shape {
  double radius_;

 public:
  Circle(double r) : radius_(r) {}

  auto area() const -> double override {
    return 3.1415926535 * radius_ * radius_;
  }
};

auto print_area(const Shape& s) -> void {
  printf("Shape area: %.2f\n", s.area());
}

auto main() -> int {
  Rectangle rect(4.0, 5.0);
  Circle circle(3.0);

  print_area(rect);
  print_area(circle);

  return 0;
}
