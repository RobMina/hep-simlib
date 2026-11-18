// hepsimlib-recorder PHYSICS_LIST MACRO
// Records complete trials of one GDML component (docs/recorder.md). Exits 1 if any event recorded an error.
#include <memory>

#include "Capture.hh"
#include "Config.hh"
#include "G4PhysListFactory.hh"
#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4VModularPhysicsList.hh"
#include "G4ios.hh"

int main(int argc, char** argv) {
  if (argc != 3) {
    G4cerr << "usage: hepsimlib-recorder PHYSICS_LIST MACRO   (e.g. FTFP_BERT run.mac)" << G4endl;
    return 2;
  }
  hsr::Config config;
  config.physicsList = argv[1];
  hsr::Messenger messenger(config);
  hsr::Component component;
  // one Geant4 thread per process: batch work scales by running more processes
  std::unique_ptr<G4RunManager> run(G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial));
  G4PhysListFactory factory;
  G4VModularPhysicsList* physics = factory.GetReferencePhysList(config.physicsList);
  if (!physics) {
    G4cerr << "hepsimlib-recorder: unknown reference physics list " << config.physicsList << G4endl;
    return 2;
  }
  run->SetUserInitialization(new hsr::Geometry(config, component));
  run->SetUserInitialization(physics);
  hsr::Recorder recorder(config, component);
  run->SetUserAction(new hsr::Primaries(config, component));
  run->SetUserAction(new hsr::RunAction(recorder));
  run->SetUserAction(new hsr::EventAction(recorder));
  run->SetUserAction(new hsr::TrackingAction(recorder));
  run->SetUserAction(new hsr::SteppingAction(recorder));
  run->SetUserAction(new hsr::StackingAction(recorder));
  if (G4UImanager::GetUIpointer()->ApplyCommand(G4String("/control/execute ") + argv[2]) != 0) {
    G4cerr << "hepsimlib-recorder: the macro failed" << G4endl;
    return 2;
  }
  const long errors = recorder.Errors();
  run.reset();  // the run manager deletes the user actions; the recorder outlives them
  if (errors) G4cerr << "hepsimlib-recorder: " << errors << " recorded error(s); the capture fails its audit" << G4endl;
  return errors ? 1 : 0;
}
