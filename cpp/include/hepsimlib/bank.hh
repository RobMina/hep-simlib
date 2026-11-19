// Component banks in C++: load and check a bank file written by hepsimlib.component.save_library, and replay it
// exactly as hepsimlib.component.replay does in float64 (docs/geant4-host.md). Header-only; needs zlib.
//
// Load() refuses, listing every problem, a file whose manifest is not a replay bank, whose arrays differ from the
// manifest's declaration, whose content hash (hepsimlib.library.hashing.content_hash) differs from the manifest's,
// or whose tables fail any structural check of hepsimlib.component.Library.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "hepsimlib/json.hh"
#include "hepsimlib/npz.hh"
#include "hepsimlib/sha256.hh"

namespace hepsimlib {

// the state tolerances of hepsimlib.component
constexpr double kPositionAtol = 1e-7;    // mm
constexpr double kDirectionAtol = 1e-12;  // direction cosine
constexpr double kStateRtol = 1e-12;
constexpr double kStateAtol = 1e-9;       // MeV/c, MeV

using Vec3 = std::array<double, 3>;

class BankError : public std::runtime_error {
 public:
  BankError(const std::string& where, const std::vector<std::string>& problems)
      : std::runtime_error(Join(where, problems)), fProblems(problems) {}
  const std::vector<std::string>& Problems() const { return fProblems; }

 private:
  static std::string Join(const std::string& where, const std::vector<std::string>& p) {
    std::string s = "bank " + where + ":";
    for (size_t i = 0; i < p.size(); ++i) s += (i ? "; " : " ") + p[i];
    return s;
  }
  std::vector<std::string> fProblems;
};

// incoming rows outside the bank's support, refused before any draw (hepsimlib.component.UnsupportedEntrance)
class UnsupportedEntrance : public std::runtime_error {
 public:
  explicit UnsupportedEntrance(std::vector<size_t> rows)
      : std::runtime_error("unsupported local entrance rows (" + std::to_string(rows.size()) + ")"), fRows(std::move(rows)) {}
  const std::vector<size_t>& Rows() const { return fRows; }

 private:
  std::vector<size_t> fRows;
};

struct ComponentType {
  std::string geometry, physics;
  Vec3 half{};
  long volumes = 0;
  bool operator==(const ComponentType& o) const {
    return geometry == o.geometry && physics == o.physics && half == o.half && volumes == o.volumes;
  }
  bool operator!=(const ComponentType& o) const { return !(*this == o); }

  bool Inside(const Vec3& p) const {
    for (int i = 0; i < 3; ++i)
      if (!(std::fabs(p[i]) <= half[i] + kPositionAtol)) return false;
    return true;
  }
  // inside and on at least one face, whatever the direction
  bool OnFace(const Vec3& p) const {
    if (!Inside(p)) return false;
    for (int i = 0; i < 3; ++i)
      if (std::fabs(std::fabs(p[i]) - half[i]) <= kPositionAtol) return true;
    return false;
  }
  // inside, on a face, and moving inward through every face it lies on (entering) or outward through one (exiting)
  bool OnBoundary(const Vec3& p, const Vec3& momentum, bool entering) const {
    if (!Inside(p)) return false;
    const double norm = std::sqrt(momentum[0] * momentum[0] + momentum[1] * momentum[1] + momentum[2] * momentum[2]);
    bool anyOn = false, inward = true, outward = false;
    for (int i = 0; i < 3; ++i) {
      const bool on = std::fabs(std::fabs(p[i]) - half[i]) <= kPositionAtol;
      const double sign = p[i] > 0 ? 1.0 : (p[i] < 0 ? -1.0 : 0.0);
      const double dot = sign * (momentum[i] / norm);
      anyOn = anyOn || on;
      if (on && !(dot < -kDirectionAtol)) inward = false;
      if (on && dot > kDirectionAtol) outward = true;
    }
    return entering ? (anyOn && inward) : outward;
  }
};

// a rigid placement: rotation local -> global (row-major), translation (mm), receiving global volume IDs
struct Placement {
  ComponentType component;
  long instance = 0;
  double rotation[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  Vec3 translation{};
  std::vector<long> volumeIds;

  Vec3 LocalPoint(const Vec3& g) const {  // (g - T) @ R
    const Vec3 d{g[0] - translation[0], g[1] - translation[1], g[2] - translation[2]};
    return LocalVector(d);
  }
  Vec3 LocalVector(const Vec3& g) const {  // g @ R
    Vec3 l;
    for (int i = 0; i < 3; ++i) l[i] = g[0] * rotation[0][i] + g[1] * rotation[1][i] + g[2] * rotation[2][i];
    return l;
  }
  Vec3 GlobalVector(const Vec3& l) const {  // l @ R^T
    Vec3 g;
    for (int i = 0; i < 3; ++i) g[i] = l[0] * rotation[i][0] + l[1] * rotation[i][1] + l[2] * rotation[i][2];
    return g;
  }
  Vec3 GlobalPoint(const Vec3& l) const {  // l @ R^T + T
    Vec3 g = GlobalVector(l);
    for (int i = 0; i < 3; ++i) g[i] += translation[i];
    return g;
  }
};

// sha256 over the arrays (names sorted; name, dtype, shape and bytes, length-prefixed): hashing.content_hash
inline std::string ContentHash(const std::map<std::string, NpyArray>& arrays) {
  Sha256 h;
  auto put = [&h](const std::string& part) {
    unsigned char len[8];
    const uint64_t n = part.size();
    for (int i = 0; i < 8; ++i) len[i] = static_cast<unsigned char>(n >> (8 * i));
    h.Update(len, 8);
    h.Update(part);
  };
  for (const auto& kv : arrays) {  // std::map: byte order = Python's sorted() for ASCII names
    std::string head = "{\"dtype\":\"" + kv.second.descr + "\",\"name\":\"";
    for (unsigned char c : kv.first) {
      if (c >= 0x80 || c < 0x20 || c == '"' || c == '\\')
        throw std::runtime_error("array name '" + kv.first + "' needs JSON escaping, which this reader refuses");
      head += char(c);
    }
    head += "\",\"shape\":[";
    for (size_t i = 0; i < kv.second.shape.size(); ++i) head += (i ? "," : "") + std::to_string(kv.second.shape[i]);
    head += "]}";
    std::string first(8, '\0');
    for (int i = 0; i < 8; ++i) first[i] = char(uint64_t(head.size()) >> (8 * i));
    put(first + head);
    put(kv.second.data);
  }
  return h.HexDigest();
}

// one incoming particle of a replay batch (hepsimlib.component INCOMING)
struct Incoming {
  std::array<long, 3> event{};
  long track = 0, parent = -1;
  int pdg = 0;
  Vec3 position{}, momentum{};
  double energy = 0, time = 0;
};

// hepsimlib.component.Replay in float64
struct ReplayResult {
  struct Particle {
    std::array<long, 3> event;
    long interaction, instance, track, parent;
    int pdg;
    bool exited;
  };
  struct Step {
    std::array<long, 3> event;
    long interaction, instance, track, parent;
    int pdg;
    long volume;
    Vec3 x0, x1, p0, p1;
    double t0, t1, length, edep;
  };
  struct Exit {
    std::array<long, 3> event;
    long interaction, instance, track, parent;
    int pdg;
    Vec3 position, momentum;
    double time, energy;
  };
  std::vector<Particle> particles;
  std::vector<Step> steps;
  std::vector<Exit> exits;
  std::vector<size_t> trials;
  long nextTrack = 0;
};

class Bank {
 public:
  // the typed views point into the loaded arrays: a bank can be moved or shared, not copied
  Bank(const Bank&) = delete;
  Bank& operator=(const Bank&) = delete;
  Bank(Bank&&) = default;
  Bank& operator=(Bank&&) = default;

  static Bank Load(const std::string& path) {
    Bank b;
    b.fPath = path;
    std::vector<std::string> problems;
    std::map<std::string, NpyArray> arrays;
    try {
      arrays = ReadNpz(path);
    } catch (const std::exception& e) {
      throw BankError(path, {e.what()});
    }
    auto m = arrays.find("__manifest__");
    if (m == arrays.end()) throw BankError(path, {"no embedded __manifest__ (a library without a manifest is refused)"});
    try {
      b.fManifestText = m->second.Unicode("__manifest__");
      b.fManifest = json::Parse(b.fManifestText);
    } catch (const std::exception& e) {
      throw BankError(path, {e.what()});
    }
    arrays.erase(m);
    b.CheckManifest(arrays, problems);
    if (!problems.empty()) throw BankError(path, problems);
    b.fArrays = std::move(arrays);
    try {
      b.Bind();
    } catch (const std::exception& e) {
      throw BankError(path, {e.what()});
    }
    b.CheckStructure(problems);
    if (!problems.empty()) throw BankError(path, problems);
    b.Index();
    return b;
  }

  const std::string& Path() const { return fPath; }
  const json::Value& Manifest() const { return fManifest; }
  const std::string& ManifestText() const { return fManifestText; }
  std::string Name() const { return fManifest["name"].Str(); }
  long Version() const { return static_cast<long>(fManifest["version"].number); }
  const std::string& ContentHash() const { return fContentHash; }
  const ComponentType& Component() const { return fComponent; }
  size_t Trials() const { return fN; }
  std::set<int> Species() const {
    std::set<int> s;
    for (size_t t = 0; t < fN; ++t) s.insert(static_cast<int>(fEntPdg[t]));
    return s;
  }

  // entrance state of trial t (local frame)
  int EntrancePdg(size_t t) const { return static_cast<int>(fEntPdg[t]); }
  Vec3 EntrancePosition(size_t t) const { return V(fEntPos, t); }
  Vec3 EntranceMomentum(size_t t) const { return V(fEntMom, t); }
  double EntranceEnergy(size_t t) const { return fEntE[t]; }

  // trial contents: [begin, end) rows of the particle, step and exit tables
  size_t ParticleBegin(size_t t) const { return static_cast<size_t>(fPOff[t]); }
  size_t ParticleEnd(size_t t) const { return static_cast<size_t>(fPOff[t + 1]); }
  size_t StepBegin(size_t t) const { return static_cast<size_t>(fSOff[t]); }
  size_t StepEnd(size_t t) const { return static_cast<size_t>(fSOff[t + 1]); }
  size_t ExitBegin(size_t t) const { return static_cast<size_t>(fEOff[t]); }
  size_t ExitEnd(size_t t) const { return static_cast<size_t>(fEOff[t + 1]); }
  int ParticlePdg(size_t row) const { return static_cast<int>(fPPdg[row]); }
  long ParticleParent(size_t row) const { return fPPar[row]; }
  bool ParticleExited(size_t row) const { return fPExit[row] != 0; }
  long StepParticle(size_t row) const { return fSPart[row]; }
  long StepVolume(size_t row) const { return fSVol[row]; }
  Vec3 StepX0(size_t row) const { return V(fSX0, row); }
  Vec3 StepX1(size_t row) const { return V(fSX1, row); }
  Vec3 StepP0(size_t row) const { return V(fSP0, row); }
  Vec3 StepP1(size_t row) const { return V(fSP1, row); }
  double StepT0(size_t row) const { return fST0[row]; }
  double StepT1(size_t row) const { return fST1[row]; }
  double StepLength(size_t row) const { return fSLen[row]; }
  double StepEdep(size_t row) const { return fSEdep[row]; }
  long ExitParticle(size_t row) const { return fXPart[row]; }
  Vec3 ExitPosition(size_t row) const { return V(fXPos, row); }
  Vec3 ExitMomentum(size_t row) const { return V(fXMom, row); }
  double ExitEnergy(size_t row) const { return fXE[row]; }
  double ExitTime(size_t row) const { return fXT[row]; }

  // inward on the envelope boundary (the replay's entrance condition)
  bool Entering(const Vec3& localPosition, const Vec3& localMomentum) const {
    return fComponent.OnBoundary(localPosition, localMomentum, true);
  }

  // trials whose recorded entrance equals the state within the tolerances, in bank order (component._matches)
  std::vector<size_t> Matches(int pdg, const Vec3& position, const Vec3& momentum, double energy) const {
    std::vector<size_t> out;
    auto lo = std::lower_bound(fOrder.begin(), fOrder.end(), Key{pdg, position[0] - kPositionAtol - 1e-9, 0});
    for (auto it = lo; it != fOrder.end() && it->pdg == pdg && it->x <= position[0] + kPositionAtol + 1e-9; ++it) {
      const size_t t = it->trial;
      bool good = true;
      for (int i = 0; i < 3 && good; ++i) {
        good = std::fabs(position[i] - fEntPos[3 * t + i]) <= kPositionAtol;
        const double b = fEntMom[3 * t + i];
        good = good && std::fabs(momentum[i] - b) <= kStateAtol + kStateRtol * std::fabs(b);
      }
      good = good && std::fabs(energy - fEntE[t]) <= kStateAtol + kStateRtol * std::fabs(fEntE[t]);
      if (good) out.push_back(t);
    }
    std::sort(out.begin(), out.end());
    return out;
  }

  // the trial drawn by one uniform u in [0, 1): floor(u * n) over the matches (component._choose)
  static size_t Choose(const std::vector<size_t>& matches, double u) {
    size_t k = static_cast<size_t>(static_cast<long long>(u * static_cast<double>(matches.size())));
    if (k >= matches.size()) k = matches.size() - 1;
    return matches[k];
  }

  // hepsimlib.component.replay(library, placement, incoming, interaction, rng, next_track=..., dtype=float64), the
  // uniforms given instead of the generator (one per incoming row)
  ReplayResult Replay(const Placement& placement, const std::vector<Incoming>& incoming,
                      const std::vector<long>& interaction, const std::vector<double>& uniforms, long nextTrack) const {
    const size_t n = incoming.size();
    if (interaction.size() != n || uniforms.size() != n) throw std::invalid_argument("one interaction and uniform per row");
    long maxId = 0;
    std::set<std::pair<std::array<long, 3>, long>> keys;
    for (size_t i = 0; i < n; ++i) {
      const auto& r = incoming[i];
      CheckKinematics(r.momentum, r.energy, "incoming");
      if (r.track <= 0 || (r.parent != -1 && r.parent <= 0)) throw std::invalid_argument("track/parent IDs must be positive or parent=-1");
      if (r.parent == r.track) throw std::invalid_argument("particle cannot parent itself");
      maxId = std::max({maxId, r.track, r.parent});
      if (interaction[i] < 0 || !keys.insert({r.event, interaction[i]}).second)
        throw std::invalid_argument("interaction IDs must be nonnegative and unique within each event");
    }
    if (placement.component != fComponent) throw std::invalid_argument("incompatible component geometry/material/physics");
    if (static_cast<long>(placement.volumeIds.size()) != fComponent.volumes) throw std::invalid_argument("volume_ids must have one ID per component volume");
    if (nextTrack <= maxId) throw std::invalid_argument("next_track must exceed input track/parent IDs and reserve unused IDs");
    std::vector<Vec3> lpos(n), lmom(n);
    std::vector<size_t> bad;
    for (size_t i = 0; i < n; ++i) {
      lpos[i] = placement.LocalPoint(incoming[i].position);
      lmom[i] = placement.LocalVector(incoming[i].momentum);
      if (!Entering(lpos[i], lmom[i])) bad.push_back(i);
    }
    if (!bad.empty()) throw UnsupportedEntrance(bad);
    ReplayResult out;
    out.trials.resize(n);
    for (size_t i = 0; i < n; ++i) {
      const auto m = Matches(incoming[i].pdg, lpos[i], lmom[i], incoming[i].energy);
      if (m.empty()) bad.push_back(i);
      else out.trials[i] = Choose(m, uniforms[i]);
    }
    if (!bad.empty()) throw UnsupportedEntrance(bad);
    long cursor = nextTrack;
    for (size_t i = 0; i < n; ++i) {
      const auto& r = incoming[i];
      const size_t t = out.trials[i], p0 = ParticleBegin(t), np = ParticleEnd(t) - p0;
      std::vector<long> track(np), parent(np);
      for (size_t k = 0; k < np; ++k) {
        track[k] = k == 0 ? r.track : cursor++;
        parent[k] = k == 0 ? r.parent : track[static_cast<size_t>(fPPar[p0 + k])];
        out.particles.push_back({r.event, interaction[i], placement.instance, track[k], parent[k], ParticlePdg(p0 + k),
                                 ParticleExited(p0 + k)});
      }
      for (size_t s = StepBegin(t); s < StepEnd(t); ++s) {
        const size_t k = static_cast<size_t>(fSPart[s]);
        out.steps.push_back({r.event, interaction[i], placement.instance, track[k], parent[k], ParticlePdg(p0 + k),
                             placement.volumeIds[static_cast<size_t>(fSVol[s])], placement.GlobalPoint(StepX0(s)),
                             placement.GlobalPoint(StepX1(s)), placement.GlobalVector(StepP0(s)),
                             placement.GlobalVector(StepP1(s)), fST0[s] + r.time, fST1[s] + r.time, fSLen[s], fSEdep[s]});
      }
      for (size_t e = ExitBegin(t); e < ExitEnd(t); ++e) {
        const size_t k = static_cast<size_t>(fXPart[e]);
        out.exits.push_back({r.event, interaction[i], placement.instance, track[k], parent[k], ParticlePdg(p0 + k),
                             placement.GlobalPoint(ExitPosition(e)), placement.GlobalVector(ExitMomentum(e)),
                             fXT[e] + r.time, fXE[e]});
      }
    }
    out.nextTrack = cursor;
    return out;
  }

 private:
  Bank() = default;

  struct Key {
    long pdg;
    double x;
    size_t trial;
    bool operator<(const Key& o) const {
      if (pdg != o.pdg) return pdg < o.pdg;
      if (x != o.x) return x < o.x;
      return trial < o.trial;
    }
  };

  static Vec3 V(const double* a, size_t row) { return {a[3 * row], a[3 * row + 1], a[3 * row + 2]}; }

  static void CheckKinematics(const Vec3& p, double e, const char* what) {
    const double m = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    if (!(m > 0) || e < m - kStateAtol - kStateRtol * m)
      throw std::invalid_argument(std::string(what) + ": propagating particles require nonzero momentum and energy >= |p|");
  }

  // the manifest is a replay bank and declares exactly these arrays; the content hash matches
  void CheckManifest(const std::map<std::string, NpyArray>& arrays, std::vector<std::string>& problems) {
    try {
      for (const char* key : {"schema", "name", "version", "kind", "arrays", "content_hash"})
        if (!fManifest.Has(key)) problems.push_back(std::string("manifest lacks '") + key + "'");
      if (!problems.empty()) return;
      if (fManifest["kind"].Str() != "replay")
        problems.push_back("not a component bank (kind '" + fManifest["kind"].Str() + "')");
      const auto& declared = fManifest["arrays"];
      if (declared.type != json::Value::Object) {
        problems.push_back("manifest 'arrays' is not an object");
        return;
      }
      for (const auto& kv : declared.object)
        if (!arrays.count(kv.first)) problems.push_back("missing array '" + kv.first + "'");
      for (const auto& kv : arrays) {
        if (!declared.object.count(kv.first)) {
          problems.push_back("undeclared array '" + kv.first + "'");
          continue;
        }
        const auto& spec = declared[kv.first];
        std::vector<uint64_t> shape;
        for (const auto& s : spec["shape"].array) shape.push_back(static_cast<uint64_t>(s.number));
        if (spec["dtype"].Str() != kv.second.descr || shape != kv.second.shape)
          problems.push_back("array '" + kv.first + "' differs from its declaration in dtype or shape");
      }
      if (!problems.empty()) return;
      fContentHash = hepsimlib::ContentHash(arrays);
      if (fContentHash != fManifest["content_hash"].Str())
        problems.push_back("content hash " + fContentHash.substr(0, 16) + "... differs from the manifest's " +
                           fManifest["content_hash"].Str().substr(0, 16) + "...");
    } catch (const std::exception& e) {
      problems.push_back(e.what());
    }
  }

  const NpyArray& A(const std::string& name) const {
    auto it = fArrays.find(name);
    if (it == fArrays.end()) throw std::runtime_error("bank lacks array '" + name + "'");
    return it->second;
  }

  // typed views of the bank arrays, shapes checked (component.bank_arrays)
  void Bind() {
    fN = A("entrances_pdg").Count();
    auto vec = [&](const char* name, const char* dtype, uint64_t rows, bool three) {
      const auto& a = A(name);
      const std::vector<uint64_t> want = three ? std::vector<uint64_t>{rows, 3} : std::vector<uint64_t>{rows};
      if (a.shape != want) throw std::runtime_error(std::string("array ") + name + " has the wrong shape");
      (void)dtype;
      return &a;
    };
    auto f8 = [&](const char* name, uint64_t rows, bool three) { return vec(name, "<f8", rows, three)->As<double>("<f8", name); };
    auto i8 = [&](const char* name, uint64_t rows) { return vec(name, "<i8", rows, false)->As<long long>("<i8", name); };
    static_assert(sizeof(long) == 8, "this reader needs 64-bit long");
    fEntPdg = reinterpret_cast<const long*>(i8("entrances_pdg", fN));
    fEntPos = f8("entrances_position", fN, true);
    fEntMom = f8("entrances_momentum", fN, true);
    fEntE = f8("entrances_energy", fN, false);
    fPOff = reinterpret_cast<const long*>(i8("particle_offsets", fN + 1));
    fSOff = reinterpret_cast<const long*>(i8("step_offsets", fN + 1));
    fEOff = reinterpret_cast<const long*>(i8("exit_offsets", fN + 1));
    fNP = A("particles_pdg").Count();
    fNS = A("steps_particle").Count();
    fNE = A("exits_particle").Count();
    fPPdg = reinterpret_cast<const long*>(i8("particles_pdg", fNP));
    fPPar = reinterpret_cast<const long*>(i8("particles_parent", fNP));
    fPExit = vec("particles_exited", "|b1", fNP, false)->As<unsigned char>("|b1", "particles_exited");
    fSPart = reinterpret_cast<const long*>(i8("steps_particle", fNS));
    fSVol = reinterpret_cast<const long*>(i8("steps_volume", fNS));
    fSX0 = f8("steps_x0", fNS, true);
    fSX1 = f8("steps_x1", fNS, true);
    fSP0 = f8("steps_p0", fNS, true);
    fSP1 = f8("steps_p1", fNS, true);
    fST0 = f8("steps_t0", fNS, false);
    fST1 = f8("steps_t1", fNS, false);
    fSLen = f8("steps_length", fNS, false);
    fSEdep = f8("steps_edep", fNS, false);
    fXPart = reinterpret_cast<const long*>(i8("exits_particle", fNE));
    fXPos = f8("exits_position", fNE, true);
    fXMom = f8("exits_momentum", fNE, true);
    fXE = f8("exits_energy", fNE, false);
    fXT = f8("exits_time", fNE, false);
    fComponent.geometry = A("component_geometry").Unicode("component_geometry");
    fComponent.physics = A("component_physics").Unicode("component_physics");
    const auto& h = A("component_half_size");
    if (h.shape != std::vector<uint64_t>{3}) throw std::runtime_error("component_half_size must have shape (3,)");
    const double* hh = h.As<double>("<f8", "component_half_size");
    fComponent.half = {hh[0], hh[1], hh[2]};
    const auto& v = A("component_volumes");
    if (!v.shape.empty()) throw std::runtime_error("component_volumes must be 0-d");
    fComponent.volumes = static_cast<long>(*v.As<long long>("<i8", "component_volumes"));
  }

  // the structural checks of hepsimlib.component.Library and ComponentType
  void CheckStructure(std::vector<std::string>& problems) const {
    auto fail = [&](const std::string& what) { problems.push_back(what); };
    const auto& c = fComponent;
    if (c.geometry.empty() || c.physics.empty() || c.volumes <= 0 || !(c.half[0] > kPositionAtol) ||
        !(c.half[1] > kPositionAtol) || !(c.half[2] > kPositionAtol))
      fail("component requires construction/physics labels, positive sizes and volumes");
    if (fN == 0) fail("library needs at least one incident trial");
    for (const auto& kv : fArrays) {
      if (kv.second.descr != "<f8") continue;
      const double* d = kv.second.As<double>("<f8", kv.first);
      for (uint64_t i = 0; i < kv.second.Count(); ++i)
        if (!std::isfinite(d[i])) {
          fail("invalid finite values for " + kv.first);
          break;
        }
    }
    auto offsets = [&](const char* name, const long* o, size_t rows) {
      bool ok = o[0] == 0 && o[fN] == static_cast<long>(rows);
      for (size_t t = 0; t < fN && ok; ++t) ok = o[t + 1] >= o[t];
      if (!ok) fail(std::string("invalid offsets for ") + name);
      return ok;
    };
    const bool offsetsOk = offsets("particle", fPOff, fNP) & offsets("step", fSOff, fNS) & offsets("exit", fEOff, fNE);
    for (size_t t = 0; t < fN; ++t) {
      try {
        CheckKinematics(EntranceMomentum(t), fEntE[t], "entrance");
      } catch (const std::exception& e) {
        fail(e.what());
        break;
      }
    }
    for (size_t e = 0; e < fNE; ++e) {
      try {
        CheckKinematics(ExitMomentum(e), fXE[e], "exit");
      } catch (const std::exception& ex) {
        fail(ex.what());
        break;
      }
    }
    for (size_t t = 0; t < fN; ++t)
      if (!c.OnBoundary(EntrancePosition(t), EntranceMomentum(t), true)) {
        fail("entrances must be inward on the component boundary");
        break;
      }
    for (size_t e = 0; e < fNE; ++e)
      if (!c.OnFace(ExitPosition(e))) {
        fail("exits must lie on the component boundary");
        break;
      }
    for (size_t s = 0; s < fNS; ++s) {
      if (fSVol[s] < 0 || fSVol[s] >= c.volumes) {
        fail("local step volume outside component");
        break;
      }
    }
    for (size_t s = 0; s < fNS; ++s)
      if (!c.Inside(StepX0(s)) || !c.Inside(StepX1(s))) {
        fail("step endpoint outside component; no clamping is allowed");
        break;
      }
    for (size_t s = 0; s < fNS; ++s)
      if (fSEdep[s] < 0 || fSLen[s] < 0 || fST0[s] < 0 || fST1[s] < fST0[s]) {
        fail("negative deposits/lengths or invalid relative times");
        break;
      }
    for (size_t e = 0; e < fNE; ++e)
      if (fXT[e] < 0) {
        fail("negative deposits/lengths or invalid relative times");
        break;
      }
    if (!offsetsOk) return;
    std::vector<int> exitCount(fNP, 0);
    std::vector<double> exitTime(fNP, INFINITY);
    bool ancestry = true, root = true, index = true;
    for (size_t t = 0; t < fN; ++t) {
      const long p0 = fPOff[t], np = fPOff[t + 1] - p0;
      if (np == 0) {
        fail("every trial needs its incident root particle");
        return;
      }
      for (long k = 0; k < np; ++k) {
        const long par = fPPar[p0 + k];
        if (k == 0 ? par != -1 : (par < 0 || par >= k)) ancestry = false;
      }
      if (fPPdg[p0] != fEntPdg[t]) root = false;
      for (long s = fSOff[t]; s < fSOff[t + 1]; ++s)
        if (fSPart[s] < 0 || fSPart[s] >= np) index = false;
      for (long e = fEOff[t]; e < fEOff[t + 1]; ++e) {
        if (fXPart[e] < 0 || fXPart[e] >= np) {
          index = false;
          continue;
        }
        ++exitCount[static_cast<size_t>(p0 + fXPart[e])];
        exitTime[static_cast<size_t>(p0 + fXPart[e])] = fXT[e];
      }
    }
    if (!ancestry) fail("local ancestry must have root 0 and parents preceding children");
    if (!root) fail("root species differs from incident species");
    if (!index) {
      fail("particle index outside trial");
      return;
    }
    for (size_t p = 0; p < fNP; ++p)
      if (exitCount[p] != (fPExit[p] ? 1 : 0)) {
        fail("exited particles need exactly one first-exit record");
        break;
      }
    for (size_t t = 0; t < fN; ++t)
      for (long s = fSOff[t]; s < fSOff[t + 1]; ++s)
        if (fST1[s] > exitTime[static_cast<size_t>(fPOff[t] + fSPart[s])] + kStateAtol) {
          fail("step after first exit is not component-local");
          return;
        }
  }

  void Index() {
    fOrder.clear();
    fOrder.reserve(fN);
    for (size_t t = 0; t < fN; ++t) fOrder.push_back({fEntPdg[t], fEntPos[3 * t], t});
    std::sort(fOrder.begin(), fOrder.end());
  }

  std::string fPath, fManifestText, fContentHash;
  json::Value fManifest;
  std::map<std::string, NpyArray> fArrays;
  ComponentType fComponent;
  size_t fN = 0, fNP = 0, fNS = 0, fNE = 0;
  const long *fEntPdg = nullptr, *fPOff = nullptr, *fSOff = nullptr, *fEOff = nullptr, *fPPdg = nullptr,
             *fPPar = nullptr, *fSPart = nullptr, *fSVol = nullptr, *fXPart = nullptr;
  const unsigned char* fPExit = nullptr;
  const double *fEntPos = nullptr, *fEntMom = nullptr, *fEntE = nullptr, *fSX0 = nullptr, *fSX1 = nullptr,
               *fSP0 = nullptr, *fSP1 = nullptr, *fST0 = nullptr, *fST1 = nullptr, *fSLen = nullptr,
               *fSEdep = nullptr, *fXPos = nullptr, *fXMom = nullptr, *fXE = nullptr, *fXT = nullptr;
  std::vector<Key> fOrder;
};

}  // namespace hepsimlib
