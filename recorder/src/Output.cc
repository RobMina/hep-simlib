#include "Output.hh"

#include "G4AnalysisManager.hh"
#include "G4Box.hh"

namespace hsr {

namespace {
G4AnalysisManager* am() { return G4AnalysisManager::Instance(); }
std::vector<double> vec(const G4ThreeVector& v) { return {v.x(), v.y(), v.z()}; }
std::vector<double> vec(const G4RotationMatrix& r) {
  return {r.xx(), r.xy(), r.xz(), r.yx(), r.yy(), r.yz(), r.zx(), r.zy(), r.zz()};
}
}  // namespace

// ------------------------------------------------------------------------------------------------------ Table
void Table::Create(const std::string& name) { fId = am()->CreateNtuple(name, name); }

int Table::I(const std::string& name) {
  fVectorOf.push_back(-1);
  return am()->CreateNtupleIColumn(fId, name);
}
int Table::D(const std::string& name) {
  fVectorOf.push_back(-1);
  return am()->CreateNtupleDColumn(fId, name);
}
int Table::S(const std::string& name) {
  fVectorOf.push_back(-1);
  return am()->CreateNtupleSColumn(fId, name);
}
int Table::V(const std::string& name) {
  fVectors.emplace_back();
  fVectorOf.push_back(static_cast<int>(fVectors.size()) - 1);
  return am()->CreateNtupleDColumn(fId, name, fVectors.back());
}
void Table::Finish() { am()->FinishNtuple(fId); }
void Table::Set(int column, long value) { am()->FillNtupleIColumn(fId, column, static_cast<int>(value)); }
void Table::Set(int column, double value) { am()->FillNtupleDColumn(fId, column, value); }
void Table::Set(int column, const std::string& value) { am()->FillNtupleSColumn(fId, column, value); }
void Table::Set(int column, const std::vector<double>& value) { fVectors.at(fVectorOf.at(column)) = value; }
void Table::Add() { am()->AddNtupleRow(fId); }

// ------------------------------------------------------------------------------------------------- states
void StateColumns::Book(Table& t, const std::string& p) {
  int i = 0;
  for (const char* f : {"x", "y", "z", "px", "py", "pz", "lx", "ly", "lz", "lpx", "lpy", "lpz", "energy", "kinetic",
                        "mass", "charge", "time", "properTime", "weight"})
    c[i++] = t.D(p + f);
  c[i++] = t.I(p + "volume");
  c[i++] = t.S(p + "placement");
  c[i++] = t.S(p + "material");
  c[i++] = t.S(p + "process");
  c[i++] = t.I(p + "stepStatus");
  c[i++] = -1;  // unused slot
}

void StateColumns::Fill(Table& t, const State& s) const {
  const double v[19] = {s.x,  s.y,  s.z,      s.px,      s.py,   s.pz,     s.lx,   s.ly,          s.lz,  s.lpx,
                        s.lpy, s.lpz, s.energy, s.kinetic, s.mass, s.charge, s.time, s.properTime, s.weight};
  for (int i = 0; i < 19; ++i) t.Set(c[i], v[i]);
  t.Set(c[19], s.volume);
  t.Set(c[20], s.placement);
  t.Set(c[21], s.material);
  t.Set(c[22], s.process);
  t.Set(c[23], s.stepStatus);
}

// ------------------------------------------------------------------------------------------------- Output
Output::Output() {
  am()->SetDefaultFileType("root");
  am()->SetVerboseLevel(0);
  am()->SetNtupleDirectoryName("capture");
  auto id = [](Table& t) {
    for (const char* f : {"run", "subrun", "event"}) t.I(f);
  };
  fEvents.Create("events");
  id(fEvents);
  for (const char* f : {"schema"}) fEvents.I(f);
  for (const char* f : {"geometryTag", "physicsTag", "envelope"}) fEvents.S(f);
  for (const char* f : {"translation", "rotation", "halfSize"}) fEvents.V(f);
  for (const char* f : {"complete", "bankAdmissible"}) fEvents.I(f);
  fEvents.S("errors");
  for (const char* f : {"nTracks", "nSteps", "nClassifications", "nVolumes", "nErrors", "nG4Tracks", "nResourceKills"})
    fEvents.I(f);
  for (const char* f : {"cpuTime", "realTime"}) fEvents.D(f);
  fEvents.Finish();

  fTracks.Create("tracks");
  id(fTracks);
  for (const char* f : {"index", "track", "parent", "steps", "secondaries", "pdg", "status", "reason", "entered", "ended"})
    fTracks.I(f);
  for (const char* f : {"creator", "stopProcess"}) fTracks.S(f);
  fBegin.Book(fTracks, "begin_");
  fEnd.Book(fTracks, "end_");
  fEntrance.Book(fTracks, "entrance_");
  fTracks.Finish();

  fSteps.Create("steps");
  id(fSteps);
  for (const char* f : {"index", "track", "number", "secondaries"}) fSteps.I(f);
  for (const char* f : {"deposit", "nonIonizingDeposit", "length"}) fSteps.D(f);
  for (const char* f : {"trackStatus", "preInside", "postInside", "firstExit"}) fSteps.I(f);
  fPre.Book(fSteps, "pre_");
  fPost.Book(fSteps, "post_");
  fSteps.Finish();

  fClassifications.Create("classifications");
  id(fClassifications);
  for (const char* f : {"index", "track", "parent", "pdg", "status", "killed"}) fClassifications.I(f);
  fState.Book(fClassifications, "state_");
  fClassifications.Finish();

  fVolumes.Create("volumes");
  id(fVolumes);
  for (const char* f : {"index", "id", "copy"}) fVolumes.I(f);
  for (const char* f : {"name", "logical", "motherLogical", "material", "solid"}) fVolumes.S(f);
  fVolumes.D("density");
  for (const char* f : {"translation", "rotation"}) fVolumes.V(f);
  fVolumes.Finish();

  fLinks.Create("links");
  id(fLinks);
  for (const char* f : {"index", "track", "parent", "creatorStep", "bornOnSurface", "exitAtBirth"}) fLinks.I(f);
  fLinks.Finish();

  fManifest.Create("manifest");
  id(fManifest);
  for (const char* f : {"linksPresent", "linksIDsMatch", "schema"}) fManifest.I(f);
  for (const char* f : {"g4Version", "manifest", "manifestHash"}) fManifest.S(f);
  for (const char* f : {"nLinks", "nBirthExits"}) fManifest.I(f);
  fManifest.Finish();

  fRunInfo.Create("runinfo");
  for (const char* f : {"run", "subrun", "seed", "events"}) fRunInfo.I(f);
  for (const char* f : {"physicsList", "geometryTag", "physicsTag", "gdml"}) fRunInfo.S(f);
  fRunInfo.Finish();
}

void Output::Open(const std::string& file) { am()->OpenFile(file); }

void Output::Close() {
  am()->Write();
  am()->CloseFile();
}

void Output::WriteEvent(long run, long subrun, long event, const Config& config, const Component& component,
                        const std::vector<TrackRow>& tracks, const std::vector<LinkRow>& links,
                        const std::vector<StepRow>& steps, const std::vector<ClassificationRow>& classifications,
                        const std::vector<std::string>& errors, long resourceKills, double cpu, double real,
                        const std::string& g4Version, const std::string& manifest, const std::string& manifestHash) {
  auto id = [&](Table& t) {
    t.Set(0, run);
    t.Set(1, subrun);
    t.Set(2, event);
  };
  long births = 0;
  for (const auto& l : links) births += l.exitAtBirth ? 1 : 0;
  std::string joined;
  for (const auto& e : errors) joined += (joined.empty() ? "" : "; ") + e;

  // events: columns 0-2 identity, then in booking order
  int c = 3;
  id(fEvents);
  fEvents.Set(c++, 1L);
  fEvents.Set(c++, config.geometryTag);
  fEvents.Set(c++, config.physicsTag);
  fEvents.Set(c++, config.envelope);
  fEvents.Set(c++, vec(component.Translation()));
  fEvents.Set(c++, vec(component.Rotation()));
  fEvents.Set(c++, vec(component.HalfSize()));
  fEvents.Set(c++, errors.empty());
  fEvents.Set(c++, false);
  fEvents.Set(c++, joined);
  fEvents.Set(c++, static_cast<long>(tracks.size()));
  fEvents.Set(c++, static_cast<long>(steps.size()));
  fEvents.Set(c++, static_cast<long>(classifications.size()));
  fEvents.Set(c++, static_cast<long>(component.Volumes().size()));
  fEvents.Set(c++, static_cast<long>(errors.size()));
  fEvents.Set(c++, static_cast<long>(tracks.size()) - births);
  fEvents.Set(c++, resourceKills);
  fEvents.Set(c++, cpu);
  fEvents.Set(c++, real);
  fEvents.Add();

  for (const auto& t : tracks) {
    id(fTracks);
    c = 3;
    for (long v : {t.index, t.track, t.parent, t.steps, t.secondaries, static_cast<long>(t.pdg),
                   static_cast<long>(t.status), static_cast<long>(t.reason), static_cast<long>(t.entered),
                   static_cast<long>(t.ended)})
      fTracks.Set(c++, v);
    fTracks.Set(c++, t.creator);
    fTracks.Set(c++, t.stopProcess);
    fBegin.Fill(fTracks, t.begin);
    fEnd.Fill(fTracks, t.end);
    fEntrance.Fill(fTracks, t.entrance);
    fTracks.Add();
  }
  for (const auto& s : steps) {
    id(fSteps);
    c = 3;
    for (long v : {s.index, s.track, s.number, s.secondaries}) fSteps.Set(c++, v);
    for (double v : {s.deposit, s.nonIonizingDeposit, s.length}) fSteps.Set(c++, v);
    for (long v : {static_cast<long>(s.trackStatus), static_cast<long>(s.preInside), static_cast<long>(s.postInside),
                   static_cast<long>(s.firstExit)})
      fSteps.Set(c++, v);
    fPre.Fill(fSteps, s.pre);
    fPost.Fill(fSteps, s.post);
    fSteps.Add();
  }
  for (const auto& k : classifications) {
    id(fClassifications);
    c = 3;
    for (long v : {k.index, k.track, k.parent, static_cast<long>(k.pdg), static_cast<long>(k.status),
                   static_cast<long>(k.killed)})
      fClassifications.Set(c++, v);
    fState.Fill(fClassifications, k.state);
    fClassifications.Add();
  }
  for (const auto& v : component.Volumes()) {
    id(fVolumes);
    c = 3;
    fVolumes.Set(c++, v.index);
    fVolumes.Set(c++, v.id);
    fVolumes.Set(c++, static_cast<long>(v.copy));
    for (const std::string* s : {&v.name, &v.logical, &v.motherLogical, &v.material, &v.solid}) fVolumes.Set(c++, *s);
    fVolumes.Set(c++, v.density);
    fVolumes.Set(c++, v.translation);
    fVolumes.Set(c++, v.rotation);
    fVolumes.Add();
  }
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    id(fLinks);
    c = 3;
    for (long v : {tracks[i].index, tracks[i].track, tracks[i].parent, links[i].creatorStep,
                   static_cast<long>(links[i].bornOnSurface), static_cast<long>(links[i].exitAtBirth)})
      fLinks.Set(c++, v);
    fLinks.Add();
  }
  id(fManifest);
  c = 3;
  for (long v : {1L, 1L, 1L}) fManifest.Set(c++, v);
  fManifest.Set(c++, g4Version);
  fManifest.Set(c++, manifest);
  fManifest.Set(c++, manifestHash);
  fManifest.Set(c++, static_cast<long>(links.size()));
  fManifest.Set(c++, births);
  fManifest.Add();
}

void Output::WriteRun(long run, long subrun, long events, const Config& config) {
  std::string gdml = config.gdml;
  auto slash = gdml.find_last_of('/');
  if (slash != std::string::npos) gdml = gdml.substr(slash + 1);  // the file name only: no site paths
  int c = 0;
  for (long v : {run, subrun, config.seed, events}) fRunInfo.Set(c++, v);
  const std::string* names[] = {&config.physicsList, &config.geometryTag, &config.physicsTag, &gdml};
  for (const std::string* s : names) fRunInfo.Set(c++, *s);
  fRunInfo.Add();
}

}  // namespace hsr
