// hepsimlib-bank BANK.npz
//   loads and checks a bank; prints its identity as JSON (exit 0) or every problem (exit 1).
// hepsimlib-replay BANK.npz INPUT.npz OUTDIR
//   replays the bank at the incoming rows of INPUT.npz with the given uniforms (Bank::Replay, float64) and writes
//   the outputs as .npy files in OUTDIR; exit 3 with OUTDIR/unsupported.npy for unsupported rows, 4 for invalid input.
// INPUT.npz arrays: rotation (3,3), translation (3,), volume_ids (V,), instance (), event (n,3), track, parent, pdg,
// interaction (n,), position, momentum (n,3), energy, time, uniforms (n,), next_track (); integers int64.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "hepsimlib/bank.hh"

using namespace hepsimlib;

namespace {

void WriteNpy(const std::string& path, const std::string& descr, const std::vector<uint64_t>& shape, const void* data,
              size_t bytes) {
  std::string h = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) h += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
  h += "), }";
  while ((10 + h.size() + 1) % 64) h += ' ';
  h += '\n';
  std::ofstream out(path, std::ios::binary);
  out.write("\x93NUMPY\x01\x00", 8);
  const uint16_t n = static_cast<uint16_t>(h.size());
  out.put(char(n & 0xff));
  out.put(char(n >> 8));
  out.write(h.data(), static_cast<std::streamsize>(h.size()));
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!out) throw std::runtime_error("cannot write " + path);
}

void I64(const std::string& dir, const std::string& name, const std::vector<long long>& v, uint64_t cols = 0) {
  std::vector<uint64_t> shape{cols ? v.size() / cols : v.size()};
  if (cols) shape.push_back(cols);
  WriteNpy(dir + "/" + name + ".npy", "<i8", shape, v.data(), v.size() * 8);
}
void F64(const std::string& dir, const std::string& name, const std::vector<double>& v, uint64_t cols = 0) {
  std::vector<uint64_t> shape{cols ? v.size() / cols : v.size()};
  if (cols) shape.push_back(cols);
  WriteNpy(dir + "/" + name + ".npy", "<f8", shape, v.data(), v.size() * 8);
}

const NpyArray& Need(const std::map<std::string, NpyArray>& a, const char* name) {
  auto it = a.find(name);
  if (it == a.end()) throw std::invalid_argument(std::string("input lacks ") + name);
  return it->second;
}

int Check(const std::string& path) {
  try {
    const Bank b = Bank::Load(path);
    const auto& c = b.Component();
    std::cout << "{\"name\": \"" << b.Name() << "\", \"version\": " << b.Version() << ", \"content_hash\": \""
              << b.ContentHash() << "\", \"trials\": " << b.Trials() << ", \"geometry\": \"" << c.geometry
              << "\", \"physics\": \"" << c.physics << "\", \"volumes\": " << c.volumes << ", \"species\": [";
    bool first = true;
    for (int s : b.Species()) {
      std::cout << (first ? "" : ", ") << s;
      first = false;
    }
    std::cout << "]}" << std::endl;
    return 0;
  } catch (const BankError& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}

int Replay(const std::string& bankPath, const std::string& inputPath, const std::string& dir) {
  std::shared_ptr<Bank> bank;
  try {
    bank = std::make_shared<Bank>(Bank::Load(bankPath));
  } catch (const BankError& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  try {
    const auto in = ReadNpz(inputPath);
    Placement p;
    p.component = bank->Component();
    const double* r = Need(in, "rotation").As<double>("<f8", "rotation");
    for (int i = 0; i < 9; ++i) p.rotation[i / 3][i % 3] = r[i];
    const double* t = Need(in, "translation").As<double>("<f8", "translation");
    p.translation = {t[0], t[1], t[2]};
    const auto& vids = Need(in, "volume_ids");
    const long long* v = vids.As<long long>("<i8", "volume_ids");
    p.volumeIds.assign(v, v + vids.Count());
    p.instance = static_cast<long>(*Need(in, "instance").As<long long>("<i8", "instance"));
    const size_t n = Need(in, "track").Count();
    const long long* ev = Need(in, "event").As<long long>("<i8", "event");
    const long long* tr = Need(in, "track").As<long long>("<i8", "track");
    const long long* pa = Need(in, "parent").As<long long>("<i8", "parent");
    const long long* pdg = Need(in, "pdg").As<long long>("<i8", "pdg");
    const long long* it = Need(in, "interaction").As<long long>("<i8", "interaction");
    const double* pos = Need(in, "position").As<double>("<f8", "position");
    const double* mom = Need(in, "momentum").As<double>("<f8", "momentum");
    const double* en = Need(in, "energy").As<double>("<f8", "energy");
    const double* tm = Need(in, "time").As<double>("<f8", "time");
    const double* u = Need(in, "uniforms").As<double>("<f8", "uniforms");
    const long next = static_cast<long>(*Need(in, "next_track").As<long long>("<i8", "next_track"));
    std::vector<Incoming> rows(n);
    std::vector<long> inter(n);
    std::vector<double> uni(u, u + n);
    for (size_t i = 0; i < n; ++i) {
      rows[i].event = {static_cast<long>(ev[3 * i]), static_cast<long>(ev[3 * i + 1]), static_cast<long>(ev[3 * i + 2])};
      rows[i].track = static_cast<long>(tr[i]);
      rows[i].parent = static_cast<long>(pa[i]);
      rows[i].pdg = static_cast<int>(pdg[i]);
      rows[i].position = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
      rows[i].momentum = {mom[3 * i], mom[3 * i + 1], mom[3 * i + 2]};
      rows[i].energy = en[i];
      rows[i].time = tm[i];
      inter[i] = static_cast<long>(it[i]);
    }
    ReplayResult res;
    try {
      res = bank->Replay(p, rows, inter, uni, next);
    } catch (const UnsupportedEntrance& e) {
      std::vector<long long> bad(e.Rows().begin(), e.Rows().end());
      I64(dir, "unsupported", bad);
      return 3;
    }
    std::vector<long long> ievent, iinter, iinst, itrack, iparent, ipdg, iexited, ivol;
    std::vector<double> a, b, c2, d, e1, e2, e3, e4;
    for (const auto& q : res.particles) {
      ievent.insert(ievent.end(), q.event.begin(), q.event.end());
      iinter.push_back(q.interaction); iinst.push_back(q.instance); itrack.push_back(q.track);
      iparent.push_back(q.parent); ipdg.push_back(q.pdg); iexited.push_back(q.exited);
    }
    I64(dir, "particles_event", ievent, 3); I64(dir, "particles_interaction", iinter); I64(dir, "particles_instance", iinst);
    I64(dir, "particles_track", itrack); I64(dir, "particles_parent", iparent); I64(dir, "particles_pdg", ipdg);
    I64(dir, "particles_exited", iexited);
    ievent.clear(); iinter.clear(); iinst.clear(); itrack.clear(); iparent.clear(); ipdg.clear();
    std::vector<double> x0, x1, p0, p1, t0, t1, len, edep;
    for (const auto& s : res.steps) {
      ievent.insert(ievent.end(), s.event.begin(), s.event.end());
      iinter.push_back(s.interaction); iinst.push_back(s.instance); itrack.push_back(s.track);
      iparent.push_back(s.parent); ipdg.push_back(s.pdg); ivol.push_back(s.volume);
      x0.insert(x0.end(), s.x0.begin(), s.x0.end()); x1.insert(x1.end(), s.x1.begin(), s.x1.end());
      p0.insert(p0.end(), s.p0.begin(), s.p0.end()); p1.insert(p1.end(), s.p1.begin(), s.p1.end());
      t0.push_back(s.t0); t1.push_back(s.t1); len.push_back(s.length); edep.push_back(s.edep);
    }
    I64(dir, "steps_event", ievent, 3); I64(dir, "steps_interaction", iinter); I64(dir, "steps_instance", iinst);
    I64(dir, "steps_track", itrack); I64(dir, "steps_parent", iparent); I64(dir, "steps_pdg", ipdg);
    I64(dir, "steps_volume", ivol);
    F64(dir, "steps_x0", x0, 3); F64(dir, "steps_x1", x1, 3); F64(dir, "steps_p0", p0, 3); F64(dir, "steps_p1", p1, 3);
    F64(dir, "steps_t0", t0); F64(dir, "steps_t1", t1); F64(dir, "steps_length", len); F64(dir, "steps_edep", edep);
    ievent.clear(); iinter.clear(); iinst.clear(); itrack.clear(); iparent.clear(); ipdg.clear();
    std::vector<double> xp, xm, xt, xe;
    for (const auto& x : res.exits) {
      ievent.insert(ievent.end(), x.event.begin(), x.event.end());
      iinter.push_back(x.interaction); iinst.push_back(x.instance); itrack.push_back(x.track);
      iparent.push_back(x.parent); ipdg.push_back(x.pdg);
      xp.insert(xp.end(), x.position.begin(), x.position.end());
      xm.insert(xm.end(), x.momentum.begin(), x.momentum.end());
      xt.push_back(x.time); xe.push_back(x.energy);
    }
    I64(dir, "exits_event", ievent, 3); I64(dir, "exits_interaction", iinter); I64(dir, "exits_instance", iinst);
    I64(dir, "exits_track", itrack); I64(dir, "exits_parent", iparent); I64(dir, "exits_pdg", ipdg);
    F64(dir, "exits_position", xp, 3); F64(dir, "exits_momentum", xm, 3); F64(dir, "exits_time", xt);
    F64(dir, "exits_energy", xe);
    std::vector<long long> trials(res.trials.begin(), res.trials.end());
    I64(dir, "trials", trials);
    I64(dir, "next_track", {res.nextTrack});
    return 0;
  } catch (const std::invalid_argument& e) {
    std::cerr << "invalid input: " << e.what() << std::endl;
    return 4;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 4;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string self = argv[0];
  const bool replay = self.size() >= 6 && self.compare(self.size() - 6, 6, "replay") == 0;
  if (!replay && argc == 2) return Check(argv[1]);
  if (replay && argc == 4) return Replay(argv[1], argv[2], argv[3]);
  std::cerr << "usage: hepsimlib-bank BANK.npz | hepsimlib-replay BANK.npz INPUT.npz OUTDIR" << std::endl;
  return 2;
}
