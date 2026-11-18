// Configuration of one recorder process, filled by /recorder/ commands (Messenger) and read by the actions.
#pragma once

#include <string>
#include <vector>

#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"
#include "G4UIcommand.hh"
#include "G4UImessenger.hh"

class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithADoubleAndUnit;
class G4UIdirectory;

namespace hsr {

// An exact local entrance state, repeated `count` times: species, a point on an envelope face (mm, envelope frame),
// an inward unit direction (envelope frame) and a kinetic energy (MeV).
struct Entrance {
  int pdg = 0;
  G4ThreeVector position, direction;
  double kinetic = 0;
  long count = 0;
};

struct Config {
  std::string physicsList;                  // from the command line
  std::string gdml, envelope, geometryTag, physicsTag, output = "capture.root";
  G4ThreeVector translation;                // envelope position in the world (mm)
  G4RotationMatrix rotation;                // envelope rotation, local -> global
  long run = 1, subrun = 0, seed = 1;
  double approach = 1.0;                    // mm behind the face where the primary starts
  std::vector<Entrance> entrances;
  long totalEntrances() const {
    long n = 0;
    for (const auto& e : entrances) n += e.count;
    return n;
  }
};

class Messenger : public G4UImessenger {
 public:
  explicit Messenger(Config& config);
  ~Messenger() override;
  void SetNewValue(G4UIcommand* command, G4String value) override;

 private:
  Config& fConfig;
  G4UIdirectory* fDir;
  G4UIcmdWithAString *fGdml, *fEnvelope, *fGeometryTag, *fPhysicsTag, *fOutput, *fRotation, *fEntrance;
  G4UIcmdWithAnInteger *fRun, *fSubrun, *fSeed;
  G4UIcmdWith3VectorAndUnit* fTranslation;
  G4UIcmdWithADoubleAndUnit* fApproach;
};

}  // namespace hsr
