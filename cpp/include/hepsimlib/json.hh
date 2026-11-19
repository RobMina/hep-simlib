// A small JSON reader (RFC 8259) for library manifests: objects, arrays, strings, numbers, true, false, null.
// Header-only; throws std::runtime_error on malformed input.
#pragma once

#include <cctype>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace hepsimlib {
namespace json {

struct Value {
  enum Type { Null, Bool, Number, String, Array, Object } type = Null;
  bool boolean = false;
  double number = 0;
  std::string text;  // String: the decoded text; Number: the literal as written
  std::vector<Value> array;
  std::map<std::string, Value> object;

  bool Has(const std::string& key) const { return type == Object && object.count(key); }
  const Value& operator[](const std::string& key) const {
    auto it = object.find(key);
    if (type != Object || it == object.end()) throw std::runtime_error("manifest: no field '" + key + "'");
    return it->second;
  }
  const std::string& Str() const {
    if (type != String) throw std::runtime_error("manifest: expected a string");
    return text;
  }
};

namespace detail {

class Parser {
 public:
  explicit Parser(const std::string& s) : fS(s) {}

  Value Parse() {
    Value v = ParseValue(0);
    Space();
    if (fI != fS.size()) Fail("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void Fail(const std::string& what) const {
    throw std::runtime_error("manifest JSON: " + what + " at offset " + std::to_string(fI));
  }
  void Space() {
    while (fI < fS.size() && (fS[fI] == ' ' || fS[fI] == '\n' || fS[fI] == '\r' || fS[fI] == '\t')) ++fI;
  }
  bool Literal(const char* word) {
    const std::string w(word);
    if (fS.compare(fI, w.size(), w) == 0) {
      fI += w.size();
      return true;
    }
    return false;
  }
  static void Utf8(std::string& out, unsigned long c) {
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
  unsigned long Hex4() {
    if (fI + 4 > fS.size()) Fail("short \\u escape");
    unsigned long c = 0;
    for (int k = 0; k < 4; ++k) {
      const char h = fS[fI++];
      c <<= 4;
      if (h >= '0' && h <= '9') c |= h - '0';
      else if (h >= 'a' && h <= 'f') c |= h - 'a' + 10;
      else if (h >= 'A' && h <= 'F') c |= h - 'A' + 10;
      else Fail("bad \\u escape");
    }
    return c;
  }
  std::string ParseString() {
    if (fS[fI] != '"') Fail("expected a string");
    ++fI;
    std::string out;
    while (true) {
      if (fI >= fS.size()) Fail("unterminated string");
      const char c = fS[fI++];
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < 0x20) Fail("control character in string");
      if (c != '\\') {
        out += c;
        continue;
      }
      if (fI >= fS.size()) Fail("unterminated escape");
      const char e = fS[fI++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          unsigned long c1 = Hex4();
          if (c1 >= 0xd800 && c1 < 0xdc00) {
            if (!(fI + 1 < fS.size() && fS[fI] == '\\' && fS[fI + 1] == 'u')) Fail("lone surrogate");
            fI += 2;
            const unsigned long c2 = Hex4();
            if (c2 < 0xdc00 || c2 >= 0xe000) Fail("bad surrogate pair");
            c1 = 0x10000 + ((c1 - 0xd800) << 10) + (c2 - 0xdc00);
          }
          Utf8(out, c1);
          break;
        }
        default: Fail("bad escape");
      }
    }
  }
  Value ParseValue(int depth) {
    if (depth > 64) Fail("nesting too deep");
    Space();
    if (fI >= fS.size()) Fail("unexpected end");
    Value v;
    const char c = fS[fI];
    if (c == '{') {
      v.type = Value::Object;
      ++fI;
      Space();
      if (fI < fS.size() && fS[fI] == '}') {
        ++fI;
        return v;
      }
      while (true) {
        Space();
        std::string key = ParseString();
        Space();
        if (fI >= fS.size() || fS[fI] != ':') Fail("expected ':'");
        ++fI;
        if (v.object.count(key)) Fail("duplicate key '" + key + "'");
        v.object[key] = ParseValue(depth + 1);
        Space();
        if (fI < fS.size() && fS[fI] == ',') { ++fI; continue; }
        if (fI < fS.size() && fS[fI] == '}') { ++fI; return v; }
        Fail("expected ',' or '}'");
      }
    }
    if (c == '[') {
      v.type = Value::Array;
      ++fI;
      Space();
      if (fI < fS.size() && fS[fI] == ']') {
        ++fI;
        return v;
      }
      while (true) {
        v.array.push_back(ParseValue(depth + 1));
        Space();
        if (fI < fS.size() && fS[fI] == ',') { ++fI; continue; }
        if (fI < fS.size() && fS[fI] == ']') { ++fI; return v; }
        Fail("expected ',' or ']'");
      }
    }
    if (c == '"') {
      v.type = Value::String;
      v.text = ParseString();
      return v;
    }
    if (Literal("true")) { v.type = Value::Bool; v.boolean = true; return v; }
    if (Literal("false")) { v.type = Value::Bool; return v; }
    if (Literal("null")) return v;
    const size_t start = fI;
    if (fI < fS.size() && fS[fI] == '-') ++fI;
    while (fI < fS.size() && (std::isdigit(static_cast<unsigned char>(fS[fI])) || fS[fI] == '.' || fS[fI] == 'e' ||
                              fS[fI] == 'E' || fS[fI] == '+' || fS[fI] == '-'))
      ++fI;
    if (fI == start) Fail("unexpected character");
    v.type = Value::Number;
    v.text = fS.substr(start, fI - start);
    char* end = nullptr;
    v.number = std::strtod(v.text.c_str(), &end);
    if (end != v.text.c_str() + v.text.size()) Fail("bad number");
    return v;
  }

  const std::string& fS;
  size_t fI = 0;
};

}  // namespace detail

inline Value Parse(const std::string& text) { return detail::Parser(text).Parse(); }

}  // namespace json
}  // namespace hepsimlib
