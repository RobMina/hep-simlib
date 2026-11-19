// hepsimlib-hybrid PHYSICS_LIST MACRO
// A Geant4 job with one GDML component and downstream blocks, the component either simulated by Geant4 or replaced
// by its bank (hepsimlib::BankModel); one summary row per event (docs/geant4-host.md, validation/a4).
#include <cmath>
#include <cstdint>
#include <ctime>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>

#include "Config.hh"
#include "G4AnalysisManager.hh"
#include "G4Box.hh"
#include "G4Event.hh"
#include "G4GDMLParser.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PhysListFactory.hh"
#include "G4Region.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIdirectory.hh"
#include "G4UImanager.hh"
#include "G4UserEventAction.hh"
#include "G4UserRunAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Version.hh"
#include "Output.hh"
#include "Randomize.hh"
#include "hepsimlib/g4/BankModel.hh"

namespace hyb {

[[noreturn]] void fail(const std::string& what) {
  G4Exception("hepsimlib-hybrid", "Hybrid001", FatalErrorInArgument, what.c_str());
  std::abort();
}

struct Catcher {
  std::string name, material;
  G4ThreeVector half, centre;  // mm, component frame
  const G4VPhysicalVolume* pv = nullptr;
};

struct Config {
  std::string physicsList, gdml, envelope, output = "hybrid.root", bank;
  bool plumbing = true;
  long run = 1, seed = 1, detail = 0;
  double approach = 1.0;
  G4ThreeVector translation;
  G4RotationMatrix rotation;
  std::vector<hsr::Entrance> entrances;
  std::vector<Catcher> catchers;
};

// the event's random seed from (shard seed, event number): splitmix64, 31 bits
long EventSeed(long seed, long event) {
  uint64_t z = static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(event);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return static_cast<long>(z >> 33);
}

class Messenger : public G4UImessenger {
 public:
  explicit Messenger(Config& c) : fC(c) {
    fDir = new G4UIdirectory("/hybrid/");
    auto str = [this](const char* path, const char* guidance) {
      auto* x = new G4UIcmdWithAString(path, this);
      x->SetGuidance(guidance);
      x->AvailableForStates(G4State_PreInit, G4State_Idle);
      fOwned.emplace_back(x);
      return x;
    };
    auto num = [this](const char* path, const char* guidance) {
      auto* x = new G4UIcmdWithAnInteger(path, this);
      x->SetGuidance(guidance);
      x->AvailableForStates(G4State_PreInit, G4State_Idle);
      fOwned.emplace_back(x);
      return x;
    };
    fGdml = str("/hybrid/gdml", "GDML file with the component");
    fEnvelope = str("/hybrid/envelope", "logical volume of the component envelope (a G4Box)");
    fOutput = str("/hybrid/output", "output file (.root)");
    fBank = str("/hybrid/bank", "bank file (.npz): the component is replaced by hepsimlib::BankModel");
    fRotation = str("/hybrid/rotation", "component rotation, 9 values row-major (local -> global)");
    fEntrance = str("/hybrid/entrance", "append an entrance: PDG LX LY LZ UX UY UZ KINETIC_MEV COUNT (component frame)");
    fCatcher = str("/hybrid/catcher", "append a block: NAME NIST_MATERIAL HX HY HZ CX CY CZ (mm, component frame)");
    fPlumbing = num("/hybrid/plumbing", "1 (default): every particle carries the fast-simulation process; 0: none");
    fRun = num("/hybrid/run", "run number");
    fSeed = num("/hybrid/seed", "shard seed; each event is seeded from (seed, event number)");
    fDetail = num("/hybrid/detail", "write every deposit and exit of the first N events of each entrance");
    fTranslation = new G4UIcmdWith3VectorAndUnit("/hybrid/translation", this);
    fTranslation->SetDefaultUnit("mm");
    fTranslation->AvailableForStates(G4State_PreInit);
    fOwned.emplace_back(fTranslation);
    fApproach = new G4UIcmdWithADoubleAndUnit("/hybrid/approach", this);
    fApproach->SetDefaultUnit("mm");
    fApproach->AvailableForStates(G4State_PreInit);
    fOwned.emplace_back(fApproach);
  }
  ~Messenger() override {
    fOwned.clear();  // commands before their directory
    delete fDir;
  }

  void SetNewValue(G4UIcommand* cmd, G4String v) override {
    if (cmd == fGdml) fC.gdml = v;
    else if (cmd == fEnvelope) fC.envelope = v;
    else if (cmd == fOutput) fC.output = v;
    else if (cmd == fBank) fC.bank = v;
    else if (cmd == fPlumbing) fC.plumbing = fPlumbing->GetNewIntValue(v) != 0;
    else if (cmd == fRun) fC.run = fRun->GetNewIntValue(v);
    else if (cmd == fSeed) fC.seed = fSeed->GetNewIntValue(v);
    else if (cmd == fDetail) fC.detail = fDetail->GetNewIntValue(v);
    else if (cmd == fTranslation) fC.translation = fTranslation->GetNew3VectorValue(v);
    else if (cmd == fApproach) {
      fC.approach = fApproach->GetNewDoubleValue(v);
      if (!(fC.approach > 0)) fail("/hybrid/approach must be positive");
    } else if (cmd == fRotation) {
      std::istringstream in(v);
      double r[9];
      for (double& x : r)
        if (!(in >> x)) fail("/hybrid/rotation needs 9 numbers");
      double worst = 0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          double s = 0;
          for (int k = 0; k < 3; ++k) s += r[3 * i + k] * r[3 * j + k];
          worst = std::max(worst, std::abs(s - (i == j ? 1.0 : 0.0)));
        }
      const double det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) +
                         r[2] * (r[3] * r[7] - r[4] * r[6]);
      if (worst > 1e-12 || std::abs(det - 1) > 1e-12) fail("/hybrid/rotation must be proper orthonormal");
      fC.rotation.set(CLHEP::HepRep3x3(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]));
    } else if (cmd == fEntrance) {
      std::istringstream in(v);
      hsr::Entrance e;
      double lx, ly, lz, ux, uy, uz;
      if (!(in >> e.pdg >> lx >> ly >> lz >> ux >> uy >> uz >> e.kinetic >> e.count) || e.count <= 0 || !(e.kinetic > 0))
        fail("/hybrid/entrance needs PDG LX LY LZ UX UY UZ KINETIC_MEV COUNT with positive energy and count");
      e.position = G4ThreeVector(lx, ly, lz);
      e.direction = G4ThreeVector(ux, uy, uz);
      if (std::abs(e.direction.mag() - 1) > 1e-12) fail("/hybrid/entrance direction must be a unit vector");
      fC.entrances.push_back(e);
    } else if (cmd == fCatcher) {
      std::istringstream in(v);
      Catcher c;
      double hx, hy, hz, cx, cy, cz;
      if (!(in >> c.name >> c.material >> hx >> hy >> hz >> cx >> cy >> cz) || !(hx > 0 && hy > 0 && hz > 0))
        fail("/hybrid/catcher needs NAME NIST_MATERIAL HX HY HZ CX CY CZ with positive half sizes");
      c.half = G4ThreeVector(hx, hy, hz);
      c.centre = G4ThreeVector(cx, cy, cz);
      fC.catchers.push_back(c);
    }
  }

 private:
  Config& fC;
  G4UIdirectory* fDir;
  std::vector<std::unique_ptr<G4UIcommand>> fOwned;
  G4UIcmdWithAString *fGdml, *fEnvelope, *fOutput, *fBank, *fRotation, *fEntrance, *fCatcher;
  G4UIcmdWithAnInteger *fPlumbing, *fRun, *fSeed, *fDetail;
  G4UIcmdWith3VectorAndUnit* fTranslation;
  G4UIcmdWithADoubleAndUnit* fApproach;
};

// per-event tallies shared by the actions and the sink (serial run manager)
struct CatcherTally {
  double edep = 0, wx = 0, wy = 0, wz = 0, first = std::numeric_limits<double>::infinity();
  long enter = 0;
};

struct State {
  Config config;
  std::shared_ptr<const hepsimlib::Bank> bank;
  G4LogicalVolume* envelopeLV = nullptr;
  G4Region* region = nullptr;
  hepsimlib::BankModel* model = nullptr;
  std::set<const G4VPhysicalVolume*> module;
  G4RotationMatrix inverse;
  // event
  long line = -1, within = -1;
  bool detail = false;
  double moduleG4 = 0, modelEdep = 0, uniform = -1;
  long triggers = 0, primaryTriggers = 0, trial = -1, matches = 0, reentrant = 0;
  hepsimlib::DomainCounts countsAtBegin;
  std::vector<CatcherTally> tallies;
  std::clock_t start = 0;
  // output
  hsr::Table events, deposits, exits, trials, runinfo;
  std::map<std::string, int> col;
  long nEvents = 0;

  G4ThreeVector Local(const G4ThreeVector& g) const { return inverse * (g - config.translation); }
};

class Sink : public hepsimlib::DepositSink {
 public:
  explicit Sink(State& s) : fS(s) {}
  void BeginTrial(const hepsimlib::TrialRecord& t) override {
    ++fS.triggers;
    fTrial = static_cast<long>(t.trial);
    if (t.trackID == 1) {
      ++fS.primaryTriggers;
      fS.trial = fTrial;
      fS.matches = static_cast<long>(t.matches);
      fS.uniform = t.uniform;
    }
    if (!fS.detail) return;
    auto& c = fS.col;
    auto& tb = fS.trials;
    tb.Set(c["t.event"], fS.nEvents);
    tb.Set(c["t.trial"], fTrial);
    tb.Set(c["t.matches"], static_cast<long>(t.matches));
    tb.Set(c["t.uniform"], t.uniform);
    tb.Set(c["t.pdg"], static_cast<long>(t.pdg));
    tb.Set(c["t.track"], static_cast<long>(t.trackID));
    const double v[8] = {t.position[0], t.position[1], t.position[2], t.momentum[0], t.momentum[1], t.momentum[2],
                         t.energy, t.time};
    const char* n[8] = {"t.x", "t.y", "t.z", "t.px", "t.py", "t.pz", "t.energy", "t.time"};
    for (int i = 0; i < 8; ++i) tb.Set(c[n[i]], v[i]);
    tb.Add();
  }
  void Deposit(const hepsimlib::DepositRecord& d) override {
    fS.modelEdep += d.edep;
    if (!fS.detail) return;
    auto& c = fS.col;
    auto& t = fS.deposits;
    t.Set(c["d.event"], fS.nEvents);
    t.Set(c["d.trial"], fTrial);
    t.Set(c["d.particle"], d.particle);
    t.Set(c["d.pdg"], static_cast<long>(d.pdg));
    t.Set(c["d.volume"], d.volume);
    const double v[16] = {d.x0[0], d.x0[1], d.x0[2], d.x1[0], d.x1[1], d.x1[2], d.p0[0], d.p0[1],
                          d.p0[2], d.p1[0], d.p1[1], d.p1[2], d.t0,    d.t1,    d.length, d.edep};
    const char* n[16] = {"d.x0", "d.y0", "d.z0", "d.x1", "d.y1", "d.z1", "d.px0", "d.py0",
                         "d.pz0", "d.px1", "d.py1", "d.pz1", "d.t0", "d.t1", "d.length", "d.edep"};
    for (int i = 0; i < 16; ++i) t.Set(c[n[i]], v[i]);
    t.Add();
  }
  void Exit(const hepsimlib::ExitRecord& x) override {
    if (x.reentrant) ++fS.reentrant;
    if (!fS.detail) return;
    auto& c = fS.col;
    auto& t = fS.exits;
    t.Set(c["x.event"], fS.nEvents);
    t.Set(c["x.trial"], fTrial);
    t.Set(c["x.particle"], x.particle);
    t.Set(c["x.pdg"], static_cast<long>(x.pdg));
    t.Set(c["x.reentrant"], x.reentrant);
    const double v[8] = {x.position[0], x.position[1], x.position[2], x.momentum[0], x.momentum[1], x.momentum[2],
                         x.energy, x.time};
    const char* n[8] = {"x.x", "x.y", "x.z", "x.px", "x.py", "x.pz", "x.energy", "x.time"};
    for (int i = 0; i < 8; ++i) t.Set(c[n[i]], v[i]);
    t.Add();
  }

 private:
  State& fS;
  long fTrial = -1;
};

class Geometry : public G4VUserDetectorConstruction {
 public:
  Geometry(State& s, Sink& sink) : fS(s), fSink(sink) {}
  G4VPhysicalVolume* Construct() override {
    auto& c = fS.config;
    if (c.gdml.empty() || c.envelope.empty()) fail("/hybrid/gdml and /hybrid/envelope are required");
    G4GDMLParser parser;
    parser.Read(c.gdml, false);
    fS.envelopeLV = G4LogicalVolumeStore::GetInstance()->GetVolume(c.envelope, false);
    if (!fS.envelopeLV) fail("no logical volume " + c.envelope + " in " + c.gdml);
    auto* box = dynamic_cast<G4Box*>(fS.envelopeLV->GetSolid());
    if (!box) fail("the envelope must be a G4Box");
    double extent = c.translation.mag() + c.approach + 1 * m +
                    G4ThreeVector(box->GetXHalfLength(), box->GetYHalfLength(), box->GetZHalfLength()).mag();
    for (const auto& k : c.catchers) extent = std::max(extent, c.translation.mag() + k.centre.mag() + k.half.mag() + 1 * m);
    auto* nist = G4NistManager::Instance();
    auto* worldLV = new G4LogicalVolume(new G4Box("world", extent, extent, extent),
                                        nist->FindOrBuildMaterial("G4_Galactic"), "world");
    auto* world = new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, "world", nullptr, false, 0);
    auto* envelope = new G4PVPlacement(G4Transform3D(c.rotation, c.translation), fS.envelopeLV, c.envelope, worldLV,
                                       false, 0, true);
    fS.inverse = c.rotation.inverse();
    Walk(envelope);
    for (auto& k : c.catchers) {
      auto* mat = nist->FindOrBuildMaterial(k.material);
      if (!mat) fail("unknown NIST material " + k.material);
      auto* lv = new G4LogicalVolume(new G4Box(k.name, k.half.x(), k.half.y(), k.half.z()), mat, k.name);
      k.pv = new G4PVPlacement(G4Transform3D(c.rotation, c.translation + c.rotation * k.centre), lv, k.name, worldLV,
                               false, 0, true);
    }
    if (!c.bank.empty()) {
      fS.bank = std::make_shared<const hepsimlib::Bank>(hepsimlib::Bank::Load(c.bank));
      const auto& h = fS.bank->Component().half;
      if (std::abs(h[0] - box->GetXHalfLength()) > 1e-9 || std::abs(h[1] - box->GetYHalfLength()) > 1e-9 ||
          std::abs(h[2] - box->GetZHalfLength()) > 1e-9)
        fail("the bank's envelope differs from " + c.envelope);
      if (fS.bank->Component().volumes != static_cast<long>(fS.module.size()) - 1 &&
          fS.bank->Component().volumes != static_cast<long>(fS.module.size()))
        G4cout << "hepsimlib-hybrid: note: the bank has " << fS.bank->Component().volumes << " volumes, the envelope "
               << fS.module.size() << " placements" << G4endl;
      fS.region = new G4Region("hepsimlib_component");
      fS.region->AddRootLogicalVolume(fS.envelopeLV);
    }
    return world;
  }
  void ConstructSDandField() override {
    if (fS.bank) fS.model = new hepsimlib::BankModel("hepsimlib_bank", fS.region, fS.bank, &fSink);
  }

 private:
  void Walk(const G4VPhysicalVolume* pv) {
    fS.module.insert(pv);
    auto* lv = pv->GetLogicalVolume();
    for (size_t i = 0; i < lv->GetNoDaughters(); ++i) Walk(lv->GetDaughter(static_cast<int>(i)));
  }
  State& fS;
  Sink& fSink;
};

class Plumbing : public hepsimlib::BankPhysics {
 public:
  explicit Plumbing(const State& s) : fS(s) {}
  void ConstructProcess() override {
    if (fS.config.plumbing) hepsimlib::BankPhysics::ConstructProcess();
  }

 private:
  const State& fS;
};

class Primaries : public G4VUserPrimaryGeneratorAction {
 public:
  explicit Primaries(State& s) : fS(s), fGun(new G4ParticleGun(1)) {}
  ~Primaries() override { delete fGun; }
  void GeneratePrimaries(G4Event* event) override {
    const auto& c = fS.config;
    G4Random::setTheSeed(EventSeed(c.seed, event->GetEventID()));
    long k = event->GetEventID();
    fS.line = -1;
    for (size_t i = 0; i < c.entrances.size(); ++i) {
      if (k < c.entrances[i].count) {
        fS.line = static_cast<long>(i);
        break;
      }
      k -= c.entrances[i].count;
    }
    if (fS.line < 0) fail("more events than /hybrid/entrance counts; /run/beamOn must equal their total");
    fS.within = k;
    fS.detail = k < c.detail;
    const auto& e = c.entrances[static_cast<size_t>(fS.line)];
    auto* particle = G4ParticleTable::GetParticleTable()->FindParticle(e.pdg);
    if (!particle) fail("unknown PDG code " + std::to_string(e.pdg));
    const G4ThreeVector start = e.position - c.approach * e.direction;
    fGun->SetParticleDefinition(particle);
    fGun->SetParticleEnergy(e.kinetic * MeV);
    fGun->SetParticlePosition(c.rotation * start + c.translation);
    fGun->SetParticleMomentumDirection(c.rotation * e.direction);
    fGun->SetParticleTime(0);
    fGun->GeneratePrimaryVertex(event);
  }

 private:
  State& fS;
  G4ParticleGun* fGun;
};

class RunAction : public G4UserRunAction {
 public:
  explicit RunAction(State& s) : fS(s) {}
  void BeginOfRunAction(const G4Run*) override {
    auto* am = G4AnalysisManager::Instance();
    am->SetDefaultFileType("root");
    am->SetVerboseLevel(0);
    am->SetNtupleDirectoryName("hybrid");
    auto& c = fS.col;
    fS.events.Create("events");
    for (const char* n : {"event", "entrance", "within", "pdg"}) c[n] = fS.events.I(n);
    c["kinetic"] = fS.events.D("kinetic");
    for (const char* n : {"triggers", "primary_triggers", "lookups", "refused", "reentrant", "trial", "matches"})
      c[n] = fS.events.I(n);
    for (const char* n : {"uniform", "module_edep_g4", "model_edep"}) c[n] = fS.events.D(n);
    for (const auto& k : fS.config.catchers) {
      for (const char* f : {"_edep", "_x", "_y", "_z", "_time"}) c[k.name + f] = fS.events.D(k.name + f);
      c[k.name + "_enter"] = fS.events.I(k.name + "_enter");
    }
    c["cpu_s"] = fS.events.D("cpu_s");
    fS.events.Finish();
    fS.deposits.Create("deposits");
    for (const char* n : {"event", "trial", "particle", "pdg", "volume"}) c[std::string("d.") + n] = fS.deposits.I(n);
    for (const char* n : {"x0", "y0", "z0", "x1", "y1", "z1", "px0", "py0", "pz0", "px1", "py1", "pz1", "t0", "t1",
                          "length", "edep"})
      c[std::string("d.") + n] = fS.deposits.D(n);
    fS.deposits.Finish();
    fS.exits.Create("exits");
    for (const char* n : {"event", "trial", "particle", "pdg", "reentrant"}) c[std::string("x.") + n] = fS.exits.I(n);
    for (const char* n : {"x", "y", "z", "px", "py", "pz", "energy", "time"}) c[std::string("x.") + n] = fS.exits.D(n);
    fS.exits.Finish();
    fS.trials.Create("trials");
    for (const char* n : {"event", "trial", "matches", "pdg", "track"}) c[std::string("t.") + n] = fS.trials.I(n);
    for (const char* n : {"uniform", "x", "y", "z", "px", "py", "pz", "energy", "time"})
      c[std::string("t.") + n] = fS.trials.D(n);
    fS.trials.Finish();
    fS.runinfo.Create("runinfo");
    for (const char* n : {"run", "seed", "plumbing", "events", "lookups", "refused", "triggered", "reentrant"})
      c[std::string("r.") + n] = fS.runinfo.I(n);
    for (const char* n : {"physicsList", "gdml", "envelope", "bank", "bankName", "bankContentHash", "g4Version"})
      c[std::string("r.") + n] = fS.runinfo.S(n);
    for (const char* n : {"translation", "rotation"}) c[std::string("r.") + n] = fS.runinfo.V(n);
    fS.runinfo.Finish();
    if (!am->OpenFile(fS.config.output)) fail("cannot open " + fS.config.output);
  }
  void EndOfRunAction(const G4Run*) override {
    auto& c = fS.col;
    auto& r = fS.runinfo;
    const auto& cfg = fS.config;
    const hepsimlib::DomainCounts n = fS.model ? fS.model->Counts() : hepsimlib::DomainCounts{};
    r.Set(c["r.run"], cfg.run);
    r.Set(c["r.seed"], cfg.seed);
    r.Set(c["r.plumbing"], cfg.plumbing);
    r.Set(c["r.events"], fS.nEvents);
    r.Set(c["r.lookups"], n.lookups);
    r.Set(c["r.refused"], n.refused);
    r.Set(c["r.triggered"], n.triggered);
    r.Set(c["r.reentrant"], n.reentrant);
    r.Set(c["r.physicsList"], cfg.physicsList);
    r.Set(c["r.gdml"], cfg.gdml);
    r.Set(c["r.envelope"], cfg.envelope);
    r.Set(c["r.bank"], cfg.bank);
    r.Set(c["r.bankName"], fS.bank ? fS.bank->Name() : std::string());
    r.Set(c["r.bankContentHash"], fS.bank ? fS.bank->ContentHash() : std::string());
    r.Set(c["r.g4Version"], std::string(G4Version));
    r.Set(c["r.translation"], std::vector<double>{cfg.translation.x(), cfg.translation.y(), cfg.translation.z()});
    const auto& m = cfg.rotation;
    r.Set(c["r.rotation"], std::vector<double>{m.xx(), m.xy(), m.xz(), m.yx(), m.yy(), m.yz(), m.zx(), m.zy(), m.zz()});
    r.Add();
    auto* am = G4AnalysisManager::Instance();
    am->Write();
    am->CloseFile();
  }

 private:
  State& fS;
};

class EventAction : public G4UserEventAction {
 public:
  explicit EventAction(State& s) : fS(s) {}
  void BeginOfEventAction(const G4Event*) override {
    fS.moduleG4 = fS.modelEdep = 0;
    fS.uniform = -1;
    fS.triggers = fS.primaryTriggers = fS.matches = fS.reentrant = 0;
    fS.trial = -1;
    fS.tallies.assign(fS.config.catchers.size(), CatcherTally{});
    fS.countsAtBegin = fS.model ? fS.model->Counts() : hepsimlib::DomainCounts{};
    fS.start = std::clock();
  }
  void EndOfEventAction(const G4Event* event) override {
    auto& c = fS.col;
    auto& t = fS.events;
    const auto& e = fS.config.entrances[static_cast<size_t>(fS.line)];
    const hepsimlib::DomainCounts n = fS.model ? fS.model->Counts() : hepsimlib::DomainCounts{};
    t.Set(c["event"], static_cast<long>(event->GetEventID()));
    t.Set(c["entrance"], fS.line);
    t.Set(c["within"], fS.within);
    t.Set(c["pdg"], static_cast<long>(e.pdg));
    t.Set(c["kinetic"], e.kinetic);
    t.Set(c["triggers"], fS.triggers);
    t.Set(c["primary_triggers"], fS.primaryTriggers);
    t.Set(c["lookups"], n.lookups - fS.countsAtBegin.lookups);
    t.Set(c["refused"], n.refused - fS.countsAtBegin.refused);
    t.Set(c["reentrant"], fS.reentrant);
    t.Set(c["trial"], fS.trial);
    t.Set(c["matches"], fS.matches);
    t.Set(c["uniform"], fS.uniform);
    t.Set(c["module_edep_g4"], fS.moduleG4);
    t.Set(c["model_edep"], fS.modelEdep);
    for (size_t i = 0; i < fS.tallies.size(); ++i) {
      const auto& k = fS.config.catchers[i];
      const auto& y = fS.tallies[i];
      const bool hit = y.edep > 0;
      t.Set(c[k.name + "_edep"], y.edep);
      t.Set(c[k.name + "_x"], hit ? y.wx / y.edep : -1e9);
      t.Set(c[k.name + "_y"], hit ? y.wy / y.edep : -1e9);
      t.Set(c[k.name + "_z"], hit ? y.wz / y.edep : -1e9);
      t.Set(c[k.name + "_time"], hit ? y.first : -1.0);
      t.Set(c[k.name + "_enter"], y.enter);
    }
    t.Set(c["cpu_s"], double(std::clock() - fS.start) / CLOCKS_PER_SEC);
    t.Add();
    ++fS.nEvents;
  }

 private:
  State& fS;
};

class SteppingAction : public G4UserSteppingAction {
 public:
  explicit SteppingAction(State& s) : fS(s) {}
  void UserSteppingAction(const G4Step* step) override {
    const auto* pre = step->GetPreStepPoint();
    const auto* post = step->GetPostStepPoint();
    const G4VPhysicalVolume* prePV = pre->GetPhysicalVolume();
    const G4VPhysicalVolume* postPV = post->GetPhysicalVolume();
    const double edep = step->GetTotalEnergyDeposit();
    if (fS.module.count(prePV)) fS.moduleG4 += edep;
    for (size_t i = 0; i < fS.config.catchers.size(); ++i) {
      const auto* pv = fS.config.catchers[i].pv;
      auto& y = fS.tallies[i];
      if (prePV == pv && edep > 0) {
        const G4ThreeVector mid = fS.Local(0.5 * (pre->GetPosition() + post->GetPosition()));
        y.edep += edep;
        y.wx += edep * mid.x();
        y.wy += edep * mid.y();
        y.wz += edep * mid.z();
        y.first = std::min(y.first, pre->GetGlobalTime());
      }
      if (post->GetStepStatus() == fGeomBoundary && postPV == pv && prePV != pv) ++y.enter;
    }
  }

 private:
  State& fS;
};

}  // namespace hyb

int main(int argc, char** argv) {
  if (argc != 3) {
    G4cerr << "usage: hepsimlib-hybrid PHYSICS_LIST MACRO   (e.g. FTFP_BERT run.mac)" << G4endl;
    return 2;
  }
  hyb::State state;
  state.config.physicsList = argv[1];
  hyb::Messenger messenger(state.config);
  hyb::Sink sink(state);
  std::unique_ptr<G4RunManager> run(G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial));
  G4PhysListFactory factory;
  G4VModularPhysicsList* physics = factory.GetReferencePhysList(state.config.physicsList);
  if (!physics) {
    G4cerr << "hepsimlib-hybrid: unknown reference physics list " << state.config.physicsList << G4endl;
    return 2;
  }
  physics->RegisterPhysics(new hyb::Plumbing(state));
  run->SetUserInitialization(new hyb::Geometry(state, sink));
  run->SetUserInitialization(physics);
  run->SetUserAction(new hyb::Primaries(state));
  run->SetUserAction(new hyb::RunAction(state));
  run->SetUserAction(new hyb::EventAction(state));
  run->SetUserAction(new hyb::SteppingAction(state));
  try {
    if (G4UImanager::GetUIpointer()->ApplyCommand(G4String("/control/execute ") + argv[2]) != 0) {
      G4cerr << "hepsimlib-hybrid: the macro failed" << G4endl;
      return 2;
    }
  } catch (const hepsimlib::BankError& e) {
    G4cerr << "hepsimlib-hybrid: " << e.what() << G4endl;
    return 3;
  }
  run.reset();
  return 0;
}
