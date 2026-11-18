#include "Config.hh"

#include <cmath>
#include <cstdlib>
#include <sstream>

#include "G4Exception.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIdirectory.hh"
#include "Randomize.hh"

namespace hsr {

namespace {
G4UIcmdWithAString* stringCommand(const char* path, const char* guidance, G4UImessenger* owner) {
  auto* c = new G4UIcmdWithAString(path, owner);
  c->SetGuidance(guidance);
  c->AvailableForStates(G4State_PreInit, G4State_Idle);
  return c;
}
G4UIcmdWithAnInteger* intCommand(const char* path, const char* guidance, G4UImessenger* owner) {
  auto* c = new G4UIcmdWithAnInteger(path, owner);
  c->SetGuidance(guidance);
  c->AvailableForStates(G4State_PreInit, G4State_Idle);
  return c;
}
[[noreturn]] void fail(const std::string& what) {
  G4Exception("hsr::Messenger", "Recorder001", FatalErrorInArgument, what.c_str());
  std::abort();  // not reached: a fatal G4Exception aborts
}
}  // namespace

Messenger::Messenger(Config& config) : fConfig(config) {
  fDir = new G4UIdirectory("/recorder/");
  fDir->SetGuidance("hep-simlib component recorder (docs/recorder.md)");
  fGdml = stringCommand("/recorder/gdml", "GDML file with the component", this);
  fEnvelope = stringCommand("/recorder/envelope", "logical volume of the envelope (a G4Box)", this);
  fGeometryTag = stringCommand("/recorder/geometryTag", "construction label kept by the bank", this);
  fPhysicsTag = stringCommand("/recorder/physicsTag", "physics label kept by the bank", this);
  fOutput = stringCommand("/recorder/output", "capture file (.root)", this);
  fRotation = stringCommand("/recorder/rotation", "envelope rotation, 9 values row-major (local -> global)", this);
  fEntrance = stringCommand("/recorder/entrance",
                            "append an entrance state: PDG LX LY LZ UX UY UZ KINETIC_MEV COUNT (mm, envelope frame)",
                            this);
  fRun = intCommand("/recorder/run", "run number of the event identities", this);
  fSubrun = intCommand("/recorder/subrun", "subrun number of the event identities", this);
  fSeed = intCommand("/recorder/seed", "random seed (recorded in runinfo)", this);
  fTranslation = new G4UIcmdWith3VectorAndUnit("/recorder/translation", this);
  fTranslation->SetGuidance("envelope position in the world");
  fTranslation->SetDefaultUnit("mm");
  fTranslation->AvailableForStates(G4State_PreInit);
  fApproach = new G4UIcmdWithADoubleAndUnit("/recorder/approach", this);
  fApproach->SetGuidance("distance behind the face where the primary starts");
  fApproach->SetDefaultUnit("mm");
  fApproach->AvailableForStates(G4State_PreInit, G4State_Idle);
}

Messenger::~Messenger() {
  delete fGdml; delete fEnvelope; delete fGeometryTag; delete fPhysicsTag; delete fOutput; delete fRotation;
  delete fEntrance; delete fRun; delete fSubrun; delete fSeed; delete fTranslation; delete fApproach; delete fDir;
}

void Messenger::SetNewValue(G4UIcommand* command, G4String value) {
  if (command == fGdml) fConfig.gdml = value;
  else if (command == fEnvelope) fConfig.envelope = value;
  else if (command == fGeometryTag) fConfig.geometryTag = value;
  else if (command == fPhysicsTag) fConfig.physicsTag = value;
  else if (command == fOutput) fConfig.output = value;
  else if (command == fRun) fConfig.run = fRun->GetNewIntValue(value);
  else if (command == fSubrun) fConfig.subrun = fSubrun->GetNewIntValue(value);
  else if (command == fSeed) {
    fConfig.seed = fSeed->GetNewIntValue(value);
    G4Random::setTheSeed(fConfig.seed);
  } else if (command == fTranslation) fConfig.translation = fTranslation->GetNew3VectorValue(value);
  else if (command == fApproach) {
    fConfig.approach = fApproach->GetNewDoubleValue(value);
    if (!(fConfig.approach > 0)) fail("/recorder/approach must be positive");
  } else if (command == fRotation) {
    std::istringstream in(value);
    double r[9];
    for (double& x : r)
      if (!(in >> x)) fail("/recorder/rotation needs 9 numbers");
    G4RotationMatrix m;
    m.set(CLHEP::HepRep3x3(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]));
    // proper orthonormal: R R^T = 1, det R = +1 (CLHEP::HepRotation does not check what rep3x3 is given)
    double worst = 0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        double s = 0;
        for (int k = 0; k < 3; ++k) s += r[3 * i + k] * r[3 * j + k];
        worst = std::max(worst, std::abs(s - (i == j ? 1.0 : 0.0)));
      }
    double det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) +
                 r[2] * (r[3] * r[7] - r[4] * r[6]);
    if (worst > 1e-12 || std::abs(det - 1) > 1e-12) fail("/recorder/rotation must be proper orthonormal");
    fConfig.rotation = m;
  } else if (command == fEntrance) {
    std::istringstream in(value);
    Entrance e;
    double ux, uy, uz, lx, ly, lz;
    if (!(in >> e.pdg >> lx >> ly >> lz >> ux >> uy >> uz >> e.kinetic >> e.count) || e.count <= 0 || !(e.kinetic > 0))
      fail("/recorder/entrance needs PDG LX LY LZ UX UY UZ KINETIC_MEV COUNT with positive energy and count");
    e.position = G4ThreeVector(lx, ly, lz);
    e.direction = G4ThreeVector(ux, uy, uz);
    if (std::abs(e.direction.mag() - 1) > 1e-12) fail("/recorder/entrance direction must be a unit vector");
    fConfig.entrances.push_back(e);
  }
}

}  // namespace hsr
