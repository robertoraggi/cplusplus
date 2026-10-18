#define device __attribute__((address_space(1)))
#define constant __attribute__((address_space(2)))

using uint2 = unsigned __attribute__((ext_vector_type(2)));
using float4 = float __attribute__((ext_vector_type(4)));
using float3 = float __attribute__((ext_vector_type(3)));

[[kernel, cxx::workgroup_size(16, 8, 1)]] void circles(
    [[cxx::global_id]] uint2 pixel,
    [[cxx::buffer(0)]] device unsigned* image,
    [[cxx::buffer(1)]] constant float3& iResolution,
    [[cxx::buffer(2)]] constant float& iTime,
    [[cxx::buffer(3)]] constant float4& iMouse) {
  unsigned x = pixel[0];
  unsigned y = pixel[1];
  float width = iResolution[0];
  float height = iResolution[1];
  if (x >= unsigned(width) || y >= unsigned(height)) return;

  float px = float(x) + 0.5f;
  float py = float(y) + 0.5f;

  float red = 0.08f + 0.12f * py / height;
  float green = 0.08f;
  float blue = 0.16f + 0.2f * py / height;

  for (int i = 0; i < 6; ++i) {
    float phaseX = iTime * (0.11f + 0.03f * i) + 0.17f * i;
    float phaseY = iTime * (0.07f + 0.05f * i) + 0.29f * i;
    phaseX -= float(int(phaseX));
    phaseY -= float(int(phaseY));
    float waveX = phaseX < 0.5f ? 2.0f * phaseX : 2.0f - 2.0f * phaseX;
    float waveY = phaseY < 0.5f ? 2.0f * phaseY : 2.0f - 2.0f * phaseY;

    float radius = 14.0f + 5.0f * i;
    float cx = radius + (width - 2.0f * radius) * waveX;
    float cy = radius + (height - 2.0f * radius) * waveY;
    float dx = px - cx;
    float dy = py - cy;

    if (dx * dx + dy * dy < radius * radius) {
      red = 0.35f + 0.11f * i;
      green = 0.9f - 0.12f * i;
      blue = 0.4f + 0.1f * (i % 3);
    }
  }

  float mx = px - iMouse[0];
  float my = py - iMouse[1];
  if (iMouse[2] > 0.0f && mx * mx + my * my < 100.0f) {
    red = 1.0f;
    green = 1.0f;
    blue = 1.0f;
  }

  unsigned r = unsigned(red * 255.0f);
  unsigned g = unsigned(green * 255.0f);
  unsigned b = unsigned(blue * 255.0f);
  image[y * unsigned(width) + x] = 0xff000000u | b << 16 | g << 8 | r;
}
