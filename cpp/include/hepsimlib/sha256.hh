// SHA-256 (FIPS 180-4), header-only, for the content hash of hepsimlib.library.hashing.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

namespace hepsimlib {

class Sha256 {
 public:
  Sha256() { Reset(); }

  void Reset() {
    static const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::memcpy(fState, init, sizeof fState);
    fBytes = 0;
    fFill = 0;
  }

  void Update(const void* data, size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    fBytes += n;
    while (n > 0) {
      const size_t take = std::min(n, sizeof fBlock - fFill);
      std::memcpy(fBlock + fFill, p, take);
      fFill += take;
      p += take;
      n -= take;
      if (fFill == sizeof fBlock) {
        Compress(fBlock);
        fFill = 0;
      }
    }
  }

  void Update(const std::string& s) { Update(s.data(), s.size()); }

  // the digest as 64 lower-case hex digits; the object must be Reset() before reuse
  std::string HexDigest() {
    const uint64_t bits = fBytes * 8;
    const unsigned char pad = 0x80;
    Update(&pad, 1);
    const unsigned char zero = 0;
    while (fFill != 56) Update(&zero, 1);
    unsigned char len[8];
    for (int i = 0; i < 8; ++i) len[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
    Update(len, 8);
    static const char* hex = "0123456789abcdef";
    std::string out;
    for (uint32_t w : fState)
      for (int i = 3; i >= 0; --i) {
        const unsigned b = (w >> (8 * i)) & 0xff;
        out += hex[b >> 4];
        out += hex[b & 15];
      }
    return out;
  }

 private:
  static uint32_t Rot(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  void Compress(const unsigned char* b) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(b[4 * i]) << 24) | (uint32_t(b[4 * i + 1]) << 16) | (uint32_t(b[4 * i + 2]) << 8) | b[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = Rot(w[i - 15], 7) ^ Rot(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = Rot(w[i - 2], 17) ^ Rot(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = fState[0], bb = fState[1], c = fState[2], d = fState[3], e = fState[4], f = fState[5],
             g = fState[6], h = fState[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = h + (Rot(e, 6) ^ Rot(e, 11) ^ Rot(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const uint32_t t2 = (Rot(a, 2) ^ Rot(a, 13) ^ Rot(a, 22)) + ((a & bb) ^ (a & c) ^ (bb & c));
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = bb;
      bb = a;
      a = t1 + t2;
    }
    fState[0] += a;
    fState[1] += bb;
    fState[2] += c;
    fState[3] += d;
    fState[4] += e;
    fState[5] += f;
    fState[6] += g;
    fState[7] += h;
  }

  uint32_t fState[8];
  unsigned char fBlock[64];
  size_t fFill = 0;
  uint64_t fBytes = 0;
};

}  // namespace hepsimlib
