// hepsimlib::BankModel: a component bank as a Geant4 fast-simulation model (docs/geant4-host.md). Header-only.
//
// Attach it to a G4Region whose root logical volume is the component's G4Box envelope, and give every particle
// species the fast-simulation process (BankPhysics below, or G4FastSimulationPhysics). A particle on an envelope
// face moving inward is an entrance (a lookup); the model triggers only if the bank holds its exact local state,
// otherwise Geant4 transports it as usual (refused). On a trigger the model draws one trial with G4UniformRand(),
// hands its deposit rows to the DepositSink in global coordinates and times, continues the incoming particle from
// its recorded exit (or kills it), and creates every other exit as a Geant4 secondary.
//
// One model per thread (create it in ConstructSDandField); the Bank is shared read-only.
#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "G4AffineTransform.hh"
#include "G4DynamicParticle.hh"
#include "G4Exception.hh"
#include "G4FastSimulationHelper.hh"
#include "G4FastStep.hh"
#include "G4FastTrack.hh"
#include "G4IonTable.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4Track.hh"
#include "G4VFastSimulationModel.hh"
#include "G4VPhysicsConstructor.hh"
#include "Randomize.hh"
#include "hepsimlib/bank.hh"

namespace hepsimlib {

// the entrance that triggered, in the envelope frame and globally
struct TrialRecord {
  size_t trial = 0, matches = 0;
  double uniform = 0;
  int pdg = 0;
  G4int trackID = 0;
  Vec3 localPosition{}, localMomentum{}, position{}, momentum{};
  double energy = 0, time = 0;
};

// one deposit row of the drawn trial, global (mm, MeV/c, ns, MeV); volume is the local index, or volumeIds[local]
struct DepositRecord {
  size_t trial = 0;
  long particle = 0, volume = 0;
  int pdg = 0;
  Vec3 x0{}, x1{}, p0{}, p1{};
  double t0 = 0, t1 = 0, length = 0, edep = 0;
};

// one exit of the drawn trial as handed to Geant4 (global); particle 0 continues the incoming track
struct ExitRecord {
  size_t trial = 0;
  long particle = 0;
  int pdg = 0;
  Vec3 position{}, momentum{};
  double energy = 0, time = 0;
  bool reentrant = false;  // recorded direction points back into the envelope: Geant4 transports it inside
};

class DepositSink {
 public:
  virtual ~DepositSink() = default;
  virtual void BeginTrial(const TrialRecord&) {}
  virtual void Deposit(const DepositRecord&) = 0;
  virtual void Exit(const ExitRecord&) {}
  virtual void EndTrial() {}
};

struct DomainCounts {
  long lookups = 0, refused = 0, triggered = 0, reentrant = 0;
};

class BankModel : public G4VFastSimulationModel {
 public:
  BankModel(const G4String& name, G4Region* envelope, std::shared_ptr<const Bank> bank, DepositSink* sink,
            std::vector<long> volumeIds = {})
      : G4VFastSimulationModel(name, envelope), fBank(std::move(bank)), fSink(sink), fVolumeIds(std::move(volumeIds)),
        fSpecies(fBank->Species()) {
    if (!fVolumeIds.empty() && static_cast<long>(fVolumeIds.size()) != fBank->Component().volumes)
      G4Exception("hepsimlib::BankModel", "HSL001", FatalException, "volumeIds needs one ID per component volume");
  }

  G4bool IsApplicable(const G4ParticleDefinition& p) override { return fSpecies.count(p.GetPDGEncoding()) > 0; }

  G4bool ModelTrigger(const G4FastTrack& fastTrack) override {
    fMatches.clear();
    const Vec3 pos = V(fastTrack.GetPrimaryTrackLocalPosition());
    const Vec3 mom = V(fastTrack.GetPrimaryTrackLocalMomentum());
    if (!fBank->Entering(pos, mom)) return false;
    ++fCounts.lookups;
    const G4Track* track = fastTrack.GetPrimaryTrack();
    fMatches = fBank->Matches(track->GetDefinition()->GetPDGEncoding(), pos, mom, track->GetTotalEnergy());
    if (fMatches.empty()) {
      ++fCounts.refused;
      return false;
    }
    fLocalPosition = pos;
    fLocalMomentum = mom;
    return true;
  }

  void DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep) override {
    const G4Track* track = fastTrack.GetPrimaryTrack();
    const G4AffineTransform& toGlobal = *fastTrack.GetInverseAffineTransformation();
    TrialRecord tr;
    tr.uniform = G4UniformRand();
    tr.trial = Bank::Choose(fMatches, tr.uniform);
    tr.matches = fMatches.size();
    tr.pdg = track->GetDefinition()->GetPDGEncoding();
    tr.trackID = track->GetTrackID();
    tr.localPosition = fLocalPosition;
    tr.localMomentum = fLocalMomentum;
    tr.position = V(track->GetPosition());
    tr.momentum = V(track->GetMomentum());
    tr.energy = track->GetTotalEnergy();
    tr.time = track->GetGlobalTime();
    ++fCounts.triggered;
    const Bank& b = *fBank;
    const size_t t = tr.trial, p0 = b.ParticleBegin(t);
    if (fSink) fSink->BeginTrial(tr);
    for (size_t s = b.StepBegin(t); s < b.StepEnd(t); ++s) {
      DepositRecord d;
      d.trial = t;
      d.particle = b.StepParticle(s);
      d.pdg = b.ParticlePdg(p0 + static_cast<size_t>(d.particle));
      d.volume = fVolumeIds.empty() ? b.StepVolume(s) : fVolumeIds[static_cast<size_t>(b.StepVolume(s))];
      d.x0 = Point(toGlobal, b.StepX0(s));
      d.x1 = Point(toGlobal, b.StepX1(s));
      d.p0 = Axis(toGlobal, b.StepP0(s));
      d.p1 = Axis(toGlobal, b.StepP1(s));
      d.t0 = b.StepT0(s) + tr.time;
      d.t1 = b.StepT1(s) + tr.time;
      d.length = b.StepLength(s);
      d.edep = b.StepEdep(s);
      if (fSink) fSink->Deposit(d);
    }
    int secondaries = 0;
    for (size_t e = b.ExitBegin(t); e < b.ExitEnd(t); ++e)
      if (b.ExitParticle(e) != 0) ++secondaries;
    fastStep.SetNumberOfSecondaryTracks(secondaries);
    bool rootExited = false;
    for (size_t e = b.ExitBegin(t); e < b.ExitEnd(t); ++e) {
      ExitRecord x;
      x.trial = t;
      x.particle = b.ExitParticle(e);
      x.pdg = b.ParticlePdg(p0 + static_cast<size_t>(x.particle));
      x.position = Point(toGlobal, b.ExitPosition(e));
      x.momentum = Axis(toGlobal, b.ExitMomentum(e));
      x.energy = b.ExitEnergy(e);
      x.time = b.ExitTime(e) + tr.time;
      x.reentrant = b.Component().OnBoundary(b.ExitPosition(e), b.ExitMomentum(e), true);
      if (x.reentrant) ++fCounts.reentrant;
      const G4ParticleDefinition* def = Definition(x.pdg);
      const G4ThreeVector pos(x.position[0], x.position[1], x.position[2]);
      const G4ThreeVector mom(x.momentum[0], x.momentum[1], x.momentum[2]);
      if (x.particle == 0) {
        rootExited = true;
        const double m = def->GetPDGMass();
        fastStep.ProposePrimaryTrackFinalPosition(pos, false);
        fastStep.ProposePrimaryTrackFinalTime(x.time);
        fastStep.ProposePrimaryTrackFinalMomentumDirection(mom.unit(), false);
        fastStep.ProposePrimaryTrackFinalKineticEnergy(std::sqrt(mom.mag2() + m * m) - m);
      } else {
        fastStep.CreateSecondaryTrack(G4DynamicParticle(def, mom), pos, x.time, false);
      }
      if (fSink) fSink->Exit(x);
    }
    if (!rootExited) fastStep.KillPrimaryTrack();
    fastStep.ProposeTotalEnergyDeposited(0);
    if (fSink) fSink->EndTrial();
  }

  const DomainCounts& Counts() const { return fCounts; }
  const Bank& GetBank() const { return *fBank; }

 private:
  static Vec3 V(const G4ThreeVector& v) { return {v.x(), v.y(), v.z()}; }
  static Vec3 Point(const G4AffineTransform& a, const Vec3& v) { return V(a.TransformPoint(G4ThreeVector(v[0], v[1], v[2]))); }
  static Vec3 Axis(const G4AffineTransform& a, const Vec3& v) { return V(a.TransformAxis(G4ThreeVector(v[0], v[1], v[2]))); }

  static const G4ParticleDefinition* Definition(int pdg) {
    const G4ParticleDefinition* d = G4ParticleTable::GetParticleTable()->FindParticle(pdg);
    if (!d && pdg > 1000000000) d = G4IonTable::GetIonTable()->GetIon(pdg);
    if (!d) G4Exception("hepsimlib::BankModel", "HSL002", FatalException, ("unknown PDG code in the bank: " + std::to_string(pdg)).c_str());
    return d;
  }

  std::shared_ptr<const Bank> fBank;
  DepositSink* fSink;
  std::vector<long> fVolumeIds;
  std::set<int> fSpecies;
  std::vector<size_t> fMatches;
  Vec3 fLocalPosition{}, fLocalMomentum{};
  DomainCounts fCounts;
};

// gives every particle that has a process manager the fast-simulation process (mass geometry); the models decide
// which species they apply to
class BankPhysics : public G4VPhysicsConstructor {
 public:
  BankPhysics() : G4VPhysicsConstructor("hepsimlib_fast_simulation") {}
  void ConstructParticle() override {}
  void ConstructProcess() override {
    auto* it = GetParticleIterator();
    it->reset();
    while ((*it)()) {
      G4ProcessManager* pm = it->value()->GetProcessManager();
      if (pm) G4FastSimulationHelper::ActivateFastSimulation(pm);
    }
  }
};

}  // namespace hepsimlib
