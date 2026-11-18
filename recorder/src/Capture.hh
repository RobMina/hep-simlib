// The recorder: geometry (GDML component in a vacuum world), primaries (exact local entrance states), and the
// actions that record every in-component step, track, classification and creator link (docs/recorder.md).
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Config.hh"
#include "G4ClassificationOfNewTrack.hh"
#include "G4Navigator.hh"
#include "G4ThreeVector.hh"
#include "G4Timer.hh"
#include "G4UserEventAction.hh"
#include "G4UserRunAction.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"

class G4Event;
class G4Material;
class G4ParticleGun;
class G4Run;
class G4Step;
class G4StepPoint;
class G4Track;
class G4VPhysicalVolume;
class G4VTouchable;

namespace hsr {

constexpr double kPositionTol = 1e-7;   // mm, the bank's POSITION_ATOL
constexpr double kDirectionTol = 1e-12;  // direction cosine, the bank's DIRECTION_ATOL

struct State {
  double x = 0, y = 0, z = 0, px = 0, py = 0, pz = 0, lx = 0, ly = 0, lz = 0, lpx = 0, lpy = 0, lpz = 0;
  double energy = 0, kinetic = 0, mass = 0, charge = 0, time = 0, properTime = 0, weight = 1;
  long volume = -1;
  std::string placement, material, process;
  int stepStatus = 0;
};

struct TrackRow {
  long index = 0, track = 0, parent = 0, steps = 0, secondaries = 0;
  int pdg = 0, status = 0, reason = 0;
  bool entered = false, ended = false;
  std::string creator, stopProcess;
  State begin, end, entrance;
  // recorder bookkeeping, not written
  bool exited = false;
  int statusBeforeStop = 0;
};

struct StepRow {
  long index = 0, track = 0, number = 0, secondaries = 0;
  double deposit = 0, nonIonizingDeposit = 0, length = 0;
  int trackStatus = 0;
  bool preInside = false, postInside = false, firstExit = false;
  State pre, post;
};

struct ClassificationRow {
  long index = 0, track = 0, parent = 0;
  int pdg = 0, status = 0;
  bool killed = false;
  State state;
};

struct LinkRow {
  long creatorStep = -1;
  bool bornOnSurface = false, exitAtBirth = false;
};

struct VolumeRow {
  long index = 0, id = 0;
  int copy = 0;
  std::string name, logical, motherLogical, material, solid;
  double density = 0;
  std::vector<double> translation, rotation;
};

// The placed component: envelope placement, local frame, inside test and volume inventory.
class Component {
 public:
  void Set(G4VPhysicalVolume* world, G4VPhysicalVolume* envelope, const G4ThreeVector& halfSize);
  bool Inside(const G4VTouchable* touchable) const;
  long VolumeId(const G4VPhysicalVolume* pv) const;
  G4ThreeVector LocalPoint(const G4ThreeVector& global) const { return fInverse * (global - fTranslation); }
  G4ThreeVector LocalVector(const G4ThreeVector& global) const { return fInverse * global; }
  // on an envelope face (within kPositionTol) and inside it
  bool OnFace(const G4ThreeVector& local) const;
  // +1 outward through a face it lies on, -1 inward through every face it lies on, 0 tangent
  int FaceDirection(const G4ThreeVector& local, const G4ThreeVector& localDirection) const;
  const std::vector<VolumeRow>& Volumes() const { return fVolumes; }
  G4VPhysicalVolume* World() const { return fWorld; }
  G4VPhysicalVolume* Envelope() const { return fEnvelope; }
  const G4ThreeVector& HalfSize() const { return fHalf; }
  const G4ThreeVector& Translation() const { return fTranslation; }
  const G4RotationMatrix& Rotation() const { return fRotation; }
  std::vector<const G4Material*> Materials() const;

 private:
  void Walk(G4VPhysicalVolume* pv, const std::string& mother);
  G4VPhysicalVolume *fWorld = nullptr, *fEnvelope = nullptr;
  G4ThreeVector fHalf, fTranslation;
  G4RotationMatrix fRotation, fInverse;
  std::map<const G4VPhysicalVolume*, long> fIds;
  std::vector<VolumeRow> fVolumes;
};

class Geometry : public G4VUserDetectorConstruction {
 public:
  Geometry(Config& config, Component& component) : fConfig(config), fComponent(component) {}
  G4VPhysicalVolume* Construct() override;

 private:
  Config& fConfig;
  Component& fComponent;
};

class Primaries : public G4VUserPrimaryGeneratorAction {
 public:
  Primaries(Config& config, Component& component);
  ~Primaries() override;
  void GeneratePrimaries(G4Event* event) override;

 private:
  Config& fConfig;
  Component& fComponent;
  G4ParticleGun* fGun;
};

class Output;

// The recording logic, shared by the five thin user actions below (serial run manager: one event at a time).
// Geant4 deletes every registered user action, so the actions are separate objects and the Recorder is owned by main.
class Recorder {
 public:
  Recorder(Config& config, Component& component);
  ~Recorder();
  void BeginOfRunAction(const G4Run* run);
  void EndOfRunAction(const G4Run* run);
  void BeginOfEventAction(const G4Event* event);
  void EndOfEventAction(const G4Event* event);
  void PreUserTrackingAction(const G4Track* track);
  void PostUserTrackingAction(const G4Track* track);
  void UserSteppingAction(const G4Step* step);
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* track);
  long Errors() const { return fRunErrors; }

 private:
  State FromPoint(const G4StepPoint* point) const;
  State FromTrack(const G4Track* track);
  void Local(State& s) const;
  void Error(const std::string& what);
  std::string BuildManifest() const;

  Config& fConfig;
  Component& fComponent;
  std::unique_ptr<Output> fOut;
  G4Navigator fLocator;
  G4Timer fTimer;
  std::string fManifest, fManifestHash;
  long fEventId = 0, fRunErrors = 0, fEvents = 0;
  // per event
  std::vector<TrackRow> fTracks;
  std::vector<LinkRow> fLinks;
  std::vector<StepRow> fSteps;
  std::vector<ClassificationRow> fClassifications;
  std::map<long, long> fTrackRow;               // G4 track ID -> row
  std::map<const G4Track*, long> fCreatorStep;  // secondary (not yet stacked) -> step row
  std::vector<std::string> fErrors;
  long fResourceKills = 0;
};

struct RunAction : G4UserRunAction {
  explicit RunAction(Recorder& r) : rec(r) {}
  void BeginOfRunAction(const G4Run* run) override { rec.BeginOfRunAction(run); }
  void EndOfRunAction(const G4Run* run) override { rec.EndOfRunAction(run); }
  Recorder& rec;
};
struct EventAction : G4UserEventAction {
  explicit EventAction(Recorder& r) : rec(r) {}
  void BeginOfEventAction(const G4Event* e) override { rec.BeginOfEventAction(e); }
  void EndOfEventAction(const G4Event* e) override { rec.EndOfEventAction(e); }
  Recorder& rec;
};
struct TrackingAction : G4UserTrackingAction {
  explicit TrackingAction(Recorder& r) : rec(r) {}
  void PreUserTrackingAction(const G4Track* t) override { rec.PreUserTrackingAction(t); }
  void PostUserTrackingAction(const G4Track* t) override { rec.PostUserTrackingAction(t); }
  Recorder& rec;
};
struct SteppingAction : G4UserSteppingAction {
  explicit SteppingAction(Recorder& r) : rec(r) {}
  void UserSteppingAction(const G4Step* s) override { rec.UserSteppingAction(s); }
  Recorder& rec;
};
struct StackingAction : G4UserStackingAction {
  explicit StackingAction(Recorder& r) : rec(r) {}
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override { return rec.ClassifyNewTrack(t); }
  Recorder& rec;
};

}  // namespace hsr
