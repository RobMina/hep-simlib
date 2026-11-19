// NumPy .npz / .npy reader, header-only (zlib for deflated members). Reads what numpy.savez and savez_compressed
// write: a zip archive (stored or deflated members, zip64 extensions) of C-order .npy arrays with simple
// little-endian dtypes. Anything else (structured or object dtypes, Fortran order, big-endian, encrypted members)
// is refused with std::runtime_error.
#pragma once

#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace hepsimlib {

struct NpyArray {
  std::string descr;           // numpy dtype.str, e.g. "<f8", "<i8", "|b1", "<U12"
  std::vector<uint64_t> shape;  // empty for a 0-d array
  std::string data;            // C-order bytes

  uint64_t Count() const {
    uint64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
  uint64_t ItemSize() const { return ItemSizeOf(descr); }

  template <class T>
  const T* As(const char* want, const std::string& name) const {
    if (descr != want || sizeof(T) != ItemSize())
      throw std::runtime_error("array " + name + " is " + descr + ", expected " + want);
    return reinterpret_cast<const T*>(data.data());
  }

  // a 0-d "<U#" array as UTF-8
  std::string Unicode(const std::string& name) const {
    if (descr.size() < 3 || descr.compare(0, 2, "<U") != 0 || !shape.empty())
      throw std::runtime_error("array " + name + " is not a 0-d unicode string");
    std::string out;
    for (size_t i = 0; i + 4 <= data.size(); i += 4) {
      uint32_t c;
      std::memcpy(&c, data.data() + i, 4);
      if (c == 0) break;  // numpy pads with NUL
      if (c < 0x80) {
        out += char(c);
      } else if (c < 0x800) {
        out += char(0xc0 | (c >> 6));
        out += char(0x80 | (c & 0x3f));
      } else if (c < 0x10000) {
        out += char(0xe0 | (c >> 12));
        out += char(0x80 | ((c >> 6) & 0x3f));
        out += char(0x80 | (c & 0x3f));
      } else {
        out += char(0xf0 | (c >> 18));
        out += char(0x80 | ((c >> 12) & 0x3f));
        out += char(0x80 | ((c >> 6) & 0x3f));
        out += char(0x80 | (c & 0x3f));
      }
    }
    return out;
  }

  static uint64_t ItemSizeOf(const std::string& d) {
    if (d.size() < 3 || (d[0] != '<' && d[0] != '|'))
      throw std::runtime_error("unsupported dtype '" + d + "' (little-endian or byte types only)");
    const std::string kinds = "biufcSUV";
    if (kinds.find(d[1]) == std::string::npos) throw std::runtime_error("unsupported dtype '" + d + "'");
    uint64_t n = 0;
    for (size_t i = 2; i < d.size(); ++i) {
      if (d[i] < '0' || d[i] > '9') throw std::runtime_error("unsupported dtype '" + d + "'");
      n = n * 10 + uint64_t(d[i] - '0');
    }
    return d[1] == 'U' ? 4 * n : n;
  }
};

namespace detail {

inline uint64_t Le(const std::string& b, uint64_t at, int bytes) {
  if (at + bytes > b.size()) throw std::runtime_error("npz: truncated archive");
  uint64_t v = 0;
  for (int i = bytes - 1; i >= 0; --i) v = (v << 8) | static_cast<unsigned char>(b[at + i]);
  return v;
}

inline NpyArray ParseNpy(const std::string& b, const std::string& name) {
  if (b.size() < 10 || b.compare(0, 6, "\x93NUMPY") != 0) throw std::runtime_error("npy " + name + ": bad magic");
  const int major = static_cast<unsigned char>(b[6]);
  uint64_t hlen, start;
  if (major == 1) {
    hlen = Le(b, 8, 2);
    start = 10;
  } else if (major == 2 || major == 3) {
    hlen = Le(b, 8, 4);
    start = 12;
  } else {
    throw std::runtime_error("npy " + name + ": unsupported format version");
  }
  if (start + hlen > b.size()) throw std::runtime_error("npy " + name + ": truncated header");
  const std::string h = b.substr(start, hlen);
  NpyArray a;
  auto field = [&](const char* key) {
    const size_t k = h.find(std::string("'") + key + "'");
    if (k == std::string::npos) throw std::runtime_error("npy " + name + ": header lacks " + key);
    size_t v = h.find(':', k);
    if (v == std::string::npos) throw std::runtime_error("npy " + name + ": malformed header");
    ++v;
    while (v < h.size() && h[v] == ' ') ++v;
    return v;
  };
  size_t v = field("descr");
  if (v >= h.size() || h[v] != '\'') throw std::runtime_error("npy " + name + ": structured dtypes are not supported");
  const size_t e = h.find('\'', v + 1);
  if (e == std::string::npos) throw std::runtime_error("npy " + name + ": malformed descr");
  a.descr = h.substr(v + 1, e - v - 1);
  v = field("fortran_order");
  if (h.compare(v, 5, "False") != 0) throw std::runtime_error("npy " + name + ": Fortran order is not supported");
  v = field("shape");
  if (v >= h.size() || h[v] != '(') throw std::runtime_error("npy " + name + ": malformed shape");
  const size_t close = h.find(')', v);
  if (close == std::string::npos) throw std::runtime_error("npy " + name + ": malformed shape");
  uint64_t cur = 0;
  bool digits = false;
  for (size_t i = v + 1; i < close; ++i) {
    const char c = h[i];
    if (c >= '0' && c <= '9') {
      cur = cur * 10 + uint64_t(c - '0');
      digits = true;
    } else if (c == ',' || c == ' ') {
      if (digits) a.shape.push_back(cur);
      cur = 0;
      digits = false;
    } else {
      throw std::runtime_error("npy " + name + ": malformed shape");
    }
  }
  if (digits) a.shape.push_back(cur);
  const uint64_t bytes = a.Count() * a.ItemSize();
  if (b.size() - (start + hlen) != bytes)
    throw std::runtime_error("npy " + name + ": data size differs from dtype and shape");
  a.data = b.substr(start + hlen);
  return a;
}

}  // namespace detail

// every member "<name>.npy" of the archive, by <name>
inline std::map<std::string, NpyArray> ReadNpz(const std::string& path) {
  {
    const uint16_t probe = 1;
    unsigned char first;
    std::memcpy(&first, &probe, 1);
    if (first != 1) throw std::runtime_error("npz: big-endian hosts are not supported");
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("npz: cannot open " + path);
  const std::string b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  using detail::Le;
  if (b.size() < 22) throw std::runtime_error("npz " + path + ": not a zip archive");
  // end of central directory: last occurrence of its signature within the final 64 KiB + 22 bytes
  uint64_t eocd = b.size();
  for (uint64_t i = b.size() - 22 + 1; i-- > 0;) {
    if (b.size() - i > 65557) break;
    if (Le(b, i, 4) == 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd == b.size()) throw std::runtime_error("npz " + path + ": no end-of-central-directory record");
  uint64_t entries = Le(b, eocd + 10, 2), cdOffset = Le(b, eocd + 16, 4);
  if (entries == 0xffff || cdOffset == 0xffffffff || Le(b, eocd + 12, 4) == 0xffffffff) {
    if (eocd < 20 || Le(b, eocd - 20, 4) != 0x07064b50) throw std::runtime_error("npz " + path + ": bad zip64 locator");
    const uint64_t z = Le(b, eocd - 20 + 8, 8);
    if (Le(b, z, 4) != 0x06064b50) throw std::runtime_error("npz " + path + ": bad zip64 end record");
    entries = Le(b, z + 32, 8);
    cdOffset = Le(b, z + 48, 8);
  }
  std::map<std::string, NpyArray> out;
  uint64_t p = cdOffset;
  for (uint64_t k = 0; k < entries; ++k) {
    if (Le(b, p, 4) != 0x02014b50) throw std::runtime_error("npz " + path + ": bad central directory");
    const uint64_t flags = Le(b, p + 8, 2), method = Le(b, p + 10, 2), crc = Le(b, p + 16, 4);
    uint64_t csize = Le(b, p + 20, 4), usize = Le(b, p + 24, 4);
    const uint64_t nlen = Le(b, p + 28, 2), xlen = Le(b, p + 30, 2), clen = Le(b, p + 32, 2);
    uint64_t local = Le(b, p + 42, 4);
    const std::string member = b.substr(p + 46, nlen);
    // zip64 extra field: the 8-byte values present are those whose 32-bit field is 0xffffffff, in this order
    for (uint64_t x = p + 46 + nlen; x + 4 <= p + 46 + nlen + xlen;) {
      const uint64_t id = Le(b, x, 2), sz = Le(b, x + 2, 2);
      if (id == 1) {
        uint64_t q = x + 4;
        if (usize == 0xffffffff) { usize = Le(b, q, 8); q += 8; }
        if (csize == 0xffffffff) { csize = Le(b, q, 8); q += 8; }
        if (local == 0xffffffff) { local = Le(b, q, 8); q += 8; }
      }
      x += 4 + sz;
    }
    p += 46 + nlen + xlen + clen;
    if (flags & 1) throw std::runtime_error("npz " + path + ": encrypted member " + member);
    if (member.size() < 4 || member.compare(member.size() - 4, 4, ".npy") != 0)
      throw std::runtime_error("npz " + path + ": member " + member + " is not an .npy array");
    if (Le(b, local, 4) != 0x04034b50) throw std::runtime_error("npz " + path + ": bad local header for " + member);
    const uint64_t data = local + 30 + Le(b, local + 26, 2) + Le(b, local + 28, 2);
    if (data + csize > b.size()) throw std::runtime_error("npz " + path + ": truncated member " + member);
    std::string raw;
    if (method == 0) {
      if (csize != usize) throw std::runtime_error("npz " + path + ": stored member sizes differ: " + member);
      raw = b.substr(data, csize);
    } else if (method == 8) {
      raw.assign(usize, '\0');
      z_stream zs{};
      if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) throw std::runtime_error("npz: zlib init failed");
      uint64_t inDone = 0, outDone = 0;
      int rc = Z_OK;
      while (rc == Z_OK) {  // chunks of at most 1 GiB (zlib counts in 32-bit unsigned)
        const uint64_t inChunk = std::min<uint64_t>(csize - inDone, 1u << 30);
        const uint64_t outChunk = std::min<uint64_t>(usize - outDone, 1u << 30);
        zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(b.data() + data + inDone));
        zs.avail_in = static_cast<uInt>(inChunk);
        zs.next_out = reinterpret_cast<Bytef*>(&raw[0] + outDone);
        zs.avail_out = static_cast<uInt>(outChunk);
        rc = inflate(&zs, Z_NO_FLUSH);
        inDone += inChunk - zs.avail_in;
        outDone += outChunk - zs.avail_out;
        if (rc == Z_BUF_ERROR && (inDone == csize || outDone == usize)) break;
      }
      inflateEnd(&zs);
      if (rc != Z_STREAM_END || outDone != usize) throw std::runtime_error("npz " + path + ": inflate failed for " + member);
    } else {
      throw std::runtime_error("npz " + path + ": unsupported compression method for " + member);
    }
    uLong c = crc32(0L, Z_NULL, 0);
    for (uint64_t at = 0; at < raw.size(); at += (1u << 30))
      c = crc32(c, reinterpret_cast<const Bytef*>(raw.data() + at),
                static_cast<uInt>(std::min<uint64_t>(raw.size() - at, 1u << 30)));
    if (c != crc) throw std::runtime_error("npz " + path + ": CRC mismatch for " + member);
    const std::string name = member.substr(0, member.size() - 4);
    if (out.count(name)) throw std::runtime_error("npz " + path + ": duplicate member " + member);
    out.emplace(name, detail::ParseNpy(raw, name));
  }
  return out;
}

}  // namespace hepsimlib
