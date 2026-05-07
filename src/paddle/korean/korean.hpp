#include <string>

static bool is_hangul_syllable_utf8(const unsigned char* s, size_t n, size_t i) {
  if (i + 2 >= n) return false;
  const unsigned char b0 = s[i];
  const unsigned char b1 = s[i + 1];
  const unsigned char b2 = s[i + 2];

  if (!((b0 >= 0xEA && b0 <= 0xED) &&
        ((b1 & 0xC0) == 0x80) &&
        ((b2 & 0xC0) == 0x80))) {
    return false;
  }

  // U+AC00..U+D7A3
  if ((b0 == 0xEA && b1 < 0xB0) ||
      (b0 == 0xED && b1 > 0x9E) ||
      (b0 == 0xED && b1 == 0x9E && b2 > 0xA3)) {
    return false;
  }
  return true;
}