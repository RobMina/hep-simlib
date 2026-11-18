#include "Capture.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>

#include "G4Box.hh"
#include "G4Element.hh"
#include "G4Event.hh"
#include "G4Exception.hh"
#include "G4GDMLParser.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4ProductionCuts.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4TouchableHistory.hh"
#include "G4Transform3D.hh"
#include "G4VProcess.hh"
#include "G4Version.hh"
#include "Output.hh"

namespace hsr {

namespace {
[[noreturn]] void fatal(const std::string& what) {
  G4Exception("hsr::Recorder", "Recorder002", FatalException, what.c_str());
  std::abort();
}

std::string processName(const G4VProcess* p) { return p ? std::string(p->GetProcessName()) : std::string(); }

// FNV-1a 64-bit of the manifest text, 16 lower-case hex digits (hepsimlib.capture.tables.fnv1a).
std::string fnv1a(const std::string& text) {
  unsigned long long h = 14695981039346656037ULL;
  for (unsigned char byte : text) {
    h ^= byte;
    h *= 1099511628211ULL;
  }
  char out[17];
  std::snprintf(out, sizeof out, "%016llx", h);
  return out;
}

std::string number(double v) {
  char out[32];
  std::snprintf(out, sizeof out, "%.17g", v);
  return out;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ Component
void Component::Set(G4VPhysicalVolume* world, G4VPhysicalVolume* envelope, const G4ThreeVector& halfSize) {
  fWorld = world;
  fEnvelope = envelope;
  fHalf = halfSize;
  fTranslation = envelope->GetObjectTranslation();
  fRotation = envelope->GetObjectRotationValue();
  fInverse = fRotation.inverse();
  fIds.clear();
  fVolumes.clear();
  Walk(world, "");
  // only the world, the envelope and the envelope's descendants: other daughters of the world are not allowed
  if (world->GetLogicalVolume()->GetNoDaughters() != 1) fatal("the world must hold only the envelope");
}

void Component::Walk(G4VPhysicalVolume* pv, const std::string& mother) {
  if (pv->IsReplicated() || pv->IsParameterised())
    fatal("replicated or parameterised volume " + std::string(pv->GetName()) + " is not supported");
  VolumeRow row;
  row.index = row.id = static_cast<long>(fVolumes.size());
  row.copy = pv->GetCopyNo();
  row.name = pv->GetName();
  row.logical = pv->GetLogicalVolume()->GetName();
  row.motherLogical = mother;
  row.material = pv->GetLogicalVolume()->GetMaterial()->GetName();
  row.solid = pv->GetLogicalVolume()->GetSolid()->GetEntityType();
  row.density = pv->GetLogicalVolume()->GetMaterial()->GetDensity() / (g / cm3);
  const G4ThreeVector t = pv->GetObjectTranslation();
  const G4RotationMatrix r = pv->GetObjectRotationValue();
  row.translation = {t.x(), t.y(), t.z()};
  row.rotation = {r.xx(), r.xy(), r.xz(), r.yx(), r.yy(), r.yz(), r.zx(), r.zy(), r.zz()};
  if (fIds.count(pv)) fatal("physical volume " + row.name + " is placed twice; give every placement its own volume");
  fIds[pv] = row.id;
  fVolumes.push_back(row);
  auto* lv = pv->GetLogicalVolume();
  for (std::size_t i = 0; i < lv->GetNoDaughters(); ++i) Walk(lv->GetDaughter(i), row.logical);
}

bool Component::Inside(const G4VTouchable* touchable) const {
  if (!touchable) return false;
  for (int d = 0; d <= touchable->GetHistoryDepth(); ++d)
    if (touchable->GetVolume(d) == fEnvelope) return true;
  return false;
}

long Component::VolumeId(const G4VPhysicalVolume* pv) const {
  auto it = fIds.find(pv);
  return it == fIds.end() ? -1 : it->second;
}

bool Component::OnFace(const G4ThreeVector& l) const {
  bool inside = true, on = false;
  for (int i = 0; i < 3; ++i) {
    const double d = std::abs(l[i]) - fHalf[i];
    inside = inside && d <= kPositionTol;
    on = on || std::abs(d) <= kPositionTol;
  }
  return inside && on;
}

int Component::FaceDirection(const G4ThreeVector& l, const G4ThreeVector& u) const {
  bool outward = false, inward = true, any = false;
  for (int i = 0; i < 3; ++i) {
    if (std::abs(std::abs(l[i]) - fHalf[i]) > kPositionTol) continue;
    any = true;
    const double normal = (l[i] >= 0 ? 1.0 : -1.0) * u[i];
    outward = outward || normal > kDirectionTol;
    inward = inward && normal < -kDirectionTol;
  }
  if (!any) return 0;
  return outward ? 1 : (inward ? -1 : 0);
}

std::vector<const G4Material*> Component::Materials() const {
  std::vector<const G4Material*> out;
  std::set<std::string> seen;
  for (const auto& v : fVolumes) {
    if (v.index == 0 || !seen.insert(v.material).second) continue;  // the world's vacuum is not the component's
    out.push_back(G4Material::GetMaterial(v.material));
  }
  return out;
}

// ------------------------------------------------------------------------------------------------- Geometry
G4VPhysicalVolume* Geometry::Construct() {
  if (fConfig.gdml.empty() || fConfig.envelope.empty() || fConfig.geometryTag.empty() || fConfig.physicsTag.empty())
    fatal("/recorder/gdml, /recorder/envelope, /recorder/geometryTag and /recorder/physicsTag are required");
  G4GDMLParser parser;
  parser.Read(fConfig.gdml, false);
  auto* lv = G4LogicalVolumeStore::GetInstance()->GetVolume(fConfig.envelope, false);
  if (!lv) fatal("no logical volume " + fConfig.envelope + " in " + fConfig.gdml);
  auto* box = dynamic_cast<G4Box*>(lv->GetSolid());
  if (!box) fatal("the envelope " + fConfig.envelope + " must be a G4Box");
  const G4ThreeVector half(box->GetXHalfLength(), box->GetYHalfLength(), box->GetZHalfLength());
  const double extent = fConfig.translation.mag() + half.mag() + fConfig.approach + 1 * m;
  auto* vacuum = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
  auto* worldLV = new G4LogicalVolume(new G4Box("world", extent, extent, extent), vacuum, "world");
  auto* world = new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, "world", nullptr, false, 0);
  auto* envelope = new G4PVPlacement(G4Transform3D(fConfig.rotation, fConfig.translation), lv, fConfig.envelope,
                                     worldLV, false, 0, true);
  fComponent.Set(world, envelope, half);
  return world;
}

// ------------------------------------------------------------------------------------------------ Primaries
Primaries::Primaries(Config& config, Component& component)
    : fConfig(config), fComponent(component), fGun(new G4ParticleGun(1)) {}

Primaries::~Primaries() { delete fGun; }

void Primaries::GeneratePrimaries(G4Event* event) {
  long k = event->GetEventID();
  const Entrance* e = nullptr;
  for (const auto& candidate : fConfig.entrances) {
    if (k < candidate.count) {
      e = &candidate;
      break;
    }
    k -= candidate.count;
  }
  if (!e) fatal("more events than /recorder/entrance counts; /run/beamOn must equal their total");
  if (!fComponent.OnFace(e->position) || fComponent.FaceDirection(e->position, e->direction) != -1)
    fatal("an entrance must lie on an envelope face with an inward direction");
  auto* particle = G4ParticleTable::GetParticleTable()->FindParticle(e->pdg);
  if (!particle) fatal("unknown PDG code " + std::to_string(e->pdg));
  const G4ThreeVector start = e->position - fConfig.approach * e->direction;
  fGun->SetParticleDefinition(particle);
  fGun->SetParticleEnergy(e->kinetic * MeV);
  fGun->SetParticlePosition(fComponent.Rotation() * start + fComponent.Translation());
  fGun->SetParticleMomentumDirection(fComponent.Rotation() * e->direction);
  fGun->SetParticleTime(0);
  fGun->GeneratePrimaryVertex(event);
}

// ------------------------------------------------------------------------------------------------- Recorder
Recorder::Recorder(Config& config, Component& component)
    : fConfig(config), fComponent(component), fOut(new Output) {}

Recorder::~Recorder() = default;

void Recorder::Error(const std::string& what) { fErrors.push_back(what); }

void Recorder::Local(State& s) const {
  const G4ThreeVector l = fComponent.LocalPoint({s.x, s.y, s.z});
  const G4ThreeVector lp = fComponent.LocalVector({s.px, s.py, s.pz});
  s.lx = l.x(), s.ly = l.y(), s.lz = l.z(), s.lpx = lp.x(), s.lpy = lp.y(), s.lpz = lp.z();
}

State Recorder::FromPoint(const G4StepPoint* p) const {
  State s;
  const G4ThreeVector x = p->GetPosition(), q = p->GetMomentum();
  s.x = x.x(), s.y = x.y(), s.z = x.z(), s.px = q.x(), s.py = q.y(), s.pz = q.z();
  s.energy = p->GetTotalEnergy(), s.kinetic = p->GetKineticEnergy(), s.mass = p->GetMass();
  s.charge = p->GetCharge(), s.time = p->GetGlobalTime(), s.properTime = p->GetProperTime();
  s.weight = p->GetWeight();
  const G4VPhysicalVolume* pv = p->GetPhysicalVolume();
  s.volume = fComponent.VolumeId(pv);
  s.placement = pv ? std::string(pv->GetName()) : std::string();
  s.material = p->GetMaterial() ? std::string(p->GetMaterial()->GetName()) : std::string();
  s.process = processName(p->GetProcessDefinedStep());
  s.stepStatus = p->GetStepStatus();
  Local(s);
  return s;
}

State Recorder::FromTrack(const G4Track* t) {
  State s;
  const G4ThreeVector x = t->GetPosition(), q = t->GetMomentum();
  s.x = x.x(), s.y = x.y(), s.z = x.z(), s.px = q.x(), s.py = q.y(), s.pz = q.z();
  s.energy = t->GetTotalEnergy(), s.kinetic = t->GetKineticEnergy();
  s.mass = t->GetDynamicParticle()->GetMass(), s.charge = t->GetDynamicParticle()->GetCharge();
  s.time = t->GetGlobalTime(), s.properTime = t->GetProperTime(), s.weight = t->GetWeight();
  G4ThreeVector direction = t->GetMomentumDirection();
  const G4VPhysicalVolume* pv = fLocator.LocateGlobalPointAndSetup(x, &direction, false, false);
  s.volume = fComponent.VolumeId(pv);
  s.placement = pv ? std::string(pv->GetName()) : std::string();
  s.material = pv ? std::string(pv->GetLogicalVolume()->GetMaterial()->GetName()) : std::string();
  s.process = "";
  s.stepStatus = 0;
  Local(s);
  return s;
}

std::string Recorder::BuildManifest() const {
  std::ostringstream m;
  m << "manifest|1\n";
  m << "geant4|" << G4Version << "\n";
  m << "physics|" << fConfig.physicsList << "|" << fConfig.physicsTag << "\n";
  m << "geometry|" << fConfig.geometryTag << "|" << fConfig.envelope << "\n";
  for (const char* name : {"gamma", "e-", "e+", "mu-", "mu+", "pi-", "pi+", "kaon-", "kaon+", "proton", "anti_proton",
                           "neutron", "deuteron", "triton", "He3", "alpha", "GenericIon"}) {
    auto* p = G4ParticleTable::GetParticleTable()->FindParticle(name);
    if (!p || !p->GetProcessManager()) continue;
    m << "processes|" << name;
    G4ProcessVector* v = p->GetProcessManager()->GetProcessList();
    for (std::size_t i = 0; i < v->size(); ++i) m << "|" << (*v)[i]->GetProcessName();
    m << "\n";
  }
  auto* cuts = G4ProductionCutsTable::GetProductionCutsTable();
  for (std::size_t i = 0; i < cuts->GetTableSize(); ++i) {
    auto* couple = cuts->GetMaterialCutsCouple(static_cast<int>(i));
    if (!couple->IsUsed()) continue;
    m << "cuts|" << couple->GetMaterial()->GetName();
    for (const char* particle : {"gamma", "e-", "e+", "proton"})
      m << "|" << particle << "=" << number(couple->GetProductionCuts()->GetProductionCut(particle) / mm) << "mm";
    m << "\n";
  }
  for (const G4Material* mat : fComponent.Materials()) {
    m << "material|" << mat->GetName() << "|density=" << number(mat->GetDensity() / (g / cm3)) << "g/cm3"
      << "|I=" << number(mat->GetIonisation()->GetMeanExcitationEnergy() / eV) << "eV";
    for (std::size_t i = 0; i < mat->GetNumberOfElements(); ++i)
      m << "|" << mat->GetElement(static_cast<int>(i))->GetSymbol() << "=" << number(mat->GetFractionVector()[i]);
    m << "\n";
  }
  return m.str();
}

void Recorder::BeginOfRunAction(const G4Run*) {
  if (fConfig.entrances.empty()) fatal("no /recorder/entrance states");
  fLocator.SetWorldVolume(fComponent.World());
  fManifest = BuildManifest();
  fManifestHash = fnv1a(fManifest);
  fEvents = 0;
  fOut->Open(fConfig.output);
}

void Recorder::EndOfRunAction(const G4Run* run) {
  if (run->GetNumberOfEvent() != fConfig.totalEntrances()) {
    G4cerr << "hepsimlib-recorder: " << run->GetNumberOfEvent() << " events but " << fConfig.totalEntrances()
           << " entrance states; the capture is incomplete" << G4endl;
    ++fRunErrors;
  }
  fOut->WriteRun(fConfig.run, fConfig.subrun, fEvents, fConfig);
  fOut->Close();
}

void Recorder::BeginOfEventAction(const G4Event* event) {
  fEventId = event->GetEventID();
  fTracks.clear();
  fLinks.clear();
  fSteps.clear();
  fClassifications.clear();
  fTrackRow.clear();
  fCreatorStep.clear();
  fErrors.clear();
  fResourceKills = 0;
  fTimer.Start();
}

void Recorder::EndOfEventAction(const G4Event*) {
  fTimer.Stop();
  long primaries = 0;
  for (const auto& t : fTracks) {
    if (t.parent == 0) ++primaries;
    if (!t.ended) Error("track " + std::to_string(t.track) + " was stacked but never ended");
  }
  if (primaries != 1) Error("expected one primary");
  if (!fCreatorStep.empty()) Error("secondaries created but never stacked");
  fRunErrors += static_cast<long>(fErrors.size());
  fOut->WriteEvent(fConfig.run, fConfig.subrun, fEventId, fConfig, fComponent, fTracks, fLinks, fSteps,
                   fClassifications, fErrors, fResourceKills, fTimer.GetUserElapsed() + fTimer.GetSystemElapsed(),
                   fTimer.GetRealElapsed(), G4Version, fManifest, fManifestHash);
  ++fEvents;
}

G4ClassificationOfNewTrack Recorder::ClassifyNewTrack(const G4Track* track) {
  const bool primary = track->GetParentID() == 0;
  ClassificationRow c;
  c.index = static_cast<long>(fClassifications.size());
  c.track = track->GetTrackID();
  c.parent = track->GetParentID();
  c.pdg = track->GetDefinition()->GetPDGEncoding();
  c.status = track->GetTrackStatus();
  c.state = FromTrack(track);

  TrackRow t;
  LinkRow link;
  t.index = static_cast<long>(fTracks.size());
  t.track = c.track;
  t.parent = c.parent;
  t.pdg = c.pdg;
  t.creator = processName(track->GetCreatorProcess());
  t.begin = c.state;
  t.entrance = c.state;
  if (fTrackRow.count(t.track)) Error("track " + std::to_string(t.track) + " classified twice");
  if (!primary) {
    auto it = fCreatorStep.find(track);
    if (it == fCreatorStep.end()) {
      Error("secondary " + std::to_string(t.track) + " has no creator step");
    } else {
      link.creatorStep = it->second;
      fCreatorStep.erase(it);
    }
    const G4ThreeVector l(c.state.lx, c.state.ly, c.state.lz), u = G4ThreeVector(c.state.lpx, c.state.lpy, c.state.lpz).unit();
    if (fComponent.OnFace(l)) {
      const int direction = fComponent.FaceDirection(l, u);
      if (direction > 0) {
        link.exitAtBirth = true;
        c.killed = true;
      } else if (direction < 0) {
        link.bornOnSurface = true;
      } else {
        Error("secondary " + std::to_string(t.track) + " born tangent to the envelope surface");
      }
    } else if (c.state.volume < 0 || !(std::abs(l.x()) < fComponent.HalfSize().x() &&
                                       std::abs(l.y()) < fComponent.HalfSize().y() &&
                                       std::abs(l.z()) < fComponent.HalfSize().z())) {
      Error("secondary " + std::to_string(t.track) + " born outside the component");
    }
    t.entered = !link.exitAtBirth;
  }
  if (c.killed) {  // an exit at birth: recorded, never tracked
    t.ended = true;
    t.reason = 4;
    t.status = 2;
    t.end = c.state;
  }
  fTrackRow[t.track] = t.index;
  fTracks.push_back(t);
  fLinks.push_back(link);
  fClassifications.push_back(c);
  if (!primary) {
    auto parent = fTrackRow.find(t.parent);
    if (parent == fTrackRow.end()) Error("secondary " + std::to_string(t.track) + " without a recorded parent");
    else ++fTracks[parent->second].secondaries;
  }
  return c.killed ? fKill : fUrgent;
}

void Recorder::PreUserTrackingAction(const G4Track* track) {
  auto it = fTrackRow.find(track->GetTrackID());
  if (it == fTrackRow.end()) {
    Error("track " + std::to_string(track->GetTrackID()) + " tracked without a classification");
    return;
  }
  // the begin state recorded at classification is the track's state now (nothing moves a stacked track)
}

void Recorder::UserSteppingAction(const G4Step* step) {
  G4Track* track = step->GetTrack();
  auto it = fTrackRow.find(track->GetTrackID());
  if (it == fTrackRow.end()) return;  // reported in PreUserTrackingAction
  TrackRow& t = fTracks[it->second];
  StepRow s;
  s.index = static_cast<long>(fSteps.size());
  s.track = t.track;
  s.number = track->GetCurrentStepNumber();
  s.deposit = step->GetTotalEnergyDeposit();
  s.nonIonizingDeposit = step->GetNonIonizingEnergyDeposit();
  s.length = step->GetStepLength();
  s.trackStatus = track->GetTrackStatus();
  s.preInside = fComponent.Inside(step->GetPreStepPoint()->GetTouchable());
  s.postInside = fComponent.Inside(step->GetPostStepPoint()->GetTouchable());
  s.pre = FromPoint(step->GetPreStepPoint());
  s.post = FromPoint(step->GetPostStepPoint());
  if (const auto* secondaries = step->GetSecondaryInCurrentStep()) {
    s.secondaries = static_cast<long>(secondaries->size());
    for (const G4Track* child : *secondaries) fCreatorStep[child] = s.index;
  }
  if (!s.preInside && s.postInside) {  // entry: only a primary approaching through vacuum
    if (t.parent != 0 || t.entered) Error("track " + std::to_string(t.track) + " entered the component twice");
    t.entered = true;
    t.entrance = s.post;
  }
  // the vacuum can leave a rounding-level deposit; it is kept as recorded (the audit reports it as approach deposit)
  if (!s.preInside && (t.parent != 0 || s.secondaries != 0)) Error("nontrivial transport outside the component");
  if (s.preInside && !s.postInside) {  // first exit: record it, keep its children, stop the track here
    s.firstExit = true;
    t.exited = true;
    t.statusBeforeStop = s.trackStatus;
    track->SetTrackStatus(fStopAndKill);
  }
  t.steps += 1;
  t.stopProcess = s.post.process;
  t.end = s.post;
  fSteps.push_back(s);
}

void Recorder::PostUserTrackingAction(const G4Track* track) {
  auto it = fTrackRow.find(track->GetTrackID());
  if (it == fTrackRow.end()) return;
  TrackRow& t = fTracks[it->second];
  t.ended = true;
  if (t.steps == 0) t.end = t.begin;
  if (t.exited) {
    t.reason = 2;
    t.status = t.statusBeforeStop;
  } else if (!t.entered) {
    Error("track " + std::to_string(t.track) + " ended without entering the component");
  } else if (track->GetTrackStatus() == fStopAndKill) {
    t.reason = 1;
    t.status = fStopAndKill;
    // a transport-limited stop with kinetic energy left is a resource kill (e.g. a looping track), not physics
    if (t.stopProcess == "Transportation" && t.end.kinetic > 0) ++fResourceKills;
  } else {
    Error("track " + std::to_string(t.track) + " ended with status " + std::to_string(track->GetTrackStatus()));
  }
}

}  // namespace hsr
