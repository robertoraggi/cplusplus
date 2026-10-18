#define device __attribute__((address_space(1)))
#define constant __attribute__((address_space(2)))

using uint2 = unsigned __attribute__((ext_vector_type(2)));
using float4 = float __attribute__((ext_vector_type(4)));
using float2 = float __attribute__((ext_vector_type(2)));
using float3 = float __attribute__((ext_vector_type(3)));

namespace hash {

constexpr unsigned pcg(unsigned v) {
  unsigned state = v * 747796405u + 2891336453u;
  unsigned word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

constexpr unsigned combine(unsigned a, unsigned b) { return pcg(a ^ pcg(b)); }

inline float unit(unsigned bits) {
  return __builtin_bit_cast(float, 0x3f800000u | (bits >> 9)) - 1.0f;
}

}  // namespace hash

static_assert(hash::pcg(0u) != hash::pcg(1u));

template <class T>
void swap(T& a, T& b) {
  T t = a;
  a = b;
  b = t;
}

struct Site {
  float2 position;
  float3 color;
};

constexpr int kNeighbours = 9;

class Cells {
 public:
  Cells(float2 cell, float time) : cell_(cell) {
    int index = 0;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) fill(sites_[index++], dx, dy, time);
  }

  void nearest(float2 p, float& first, float& second, float3& color) const {
    first = 1e9f;
    second = 1e9f;
    auto visit = [&](const Site& site) {
      float2 d = site.position - p;
      float distance = d.x * d.x + d.y * d.y;
      if (distance < second) second = distance;
      if (second < first) {
        swap(first, second);
        color = site.color;
      }
    };
    for (const Site& site : sites_) visit(site);
    first = __builtin_sqrtf(first);
    second = __builtin_sqrtf(second);
  }

 private:
  void fill(Site& site, int dx, int dy, float time) const {
    float2 neighbour = cell_ + float2{float(dx), float(dy)};
    unsigned key = hash::combine(unsigned(int(neighbour.x) + 1024),
                                 unsigned(int(neighbour.y) + 1024));
    float2 jitter{hash::unit(hash::pcg(key)), hash::unit(hash::pcg(key + 1u))};
    float2 wobble{__builtin_sinf(time + 6.2831853f * jitter.x),
                  __builtin_cosf(time * 0.7f + 6.2831853f * jitter.y)};
    site.position = neighbour + 0.5f + 0.35f * wobble;
    site.color = float3{hash::unit(hash::pcg(key + 2u)),
                        hash::unit(hash::pcg(key + 3u)),
                        hash::unit(hash::pcg(key + 4u))} *
                     0.6f +
                 0.3f;
  }

  float2 cell_;
  Site sites_[kNeighbours];
};

static unsigned pack(float3 color) {
  float3 c = color * 255.0f + 0.5f;
  c = c < 0.0f ? 0.0f : (c > 255.0f ? 255.0f : c);
  return 0xff000000u | unsigned(c.z) << 16 | unsigned(c.y) << 8 | unsigned(c.x);
}

[[cxx::kernel, cxx::workgroup_size(16, 8, 1)]] void voronoi(
    [[cxx::global_id]] uint2 pixel,
    [[cxx::buffer(0)]] device unsigned* image,
    [[cxx::buffer(1)]] constant float3& iResolution,
    [[cxx::buffer(2)]] constant float& iTime,
    [[cxx::buffer(3)]] constant float4& iMouse) {
  float2 size{iResolution[0], iResolution[1]};
  if (pixel.x >= unsigned(size.x) || pixel.y >= unsigned(size.y)) return;

  float scale = iMouse[2] > 0.0f ? 3.0f + 9.0f * iMouse[0] / size.x : 6.0f;
  float2 p = float2{float(pixel.x), float(pixel.y)} / size.y * scale;
  float2 cell{__builtin_floorf(p.x), __builtin_floorf(p.y)};

  Cells cells{cell, iTime};
  float first, second;
  float3 color;
  cells.nearest(p, first, second, color);

  float edge = __builtin_fminf((second - first) * 8.0f, 1.0f);
  float glow = __builtin_expf(-6.0f * first);
  image[pixel.y * unsigned(size.x) + pixel.x] =
      pack(color * edge + float3{1.0f, 0.9f, 0.7f} * glow * 0.35f);
}
