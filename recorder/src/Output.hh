// The capture tables as Geant4 ntuples (G4AnalysisManager, ROOT format, directory "capture"); docs/recorder.md.
#pragma once

#include <deque>
#include <string>
#include <vector>

#include "Capture.hh"

namespace hsr {

// One ntuple: columns are created in order, then rows are filled column by column and added.
class Table {
 public:
  void Create(const std::string& name);
  int I(const std::string& name);
  int D(const std::string& name);
  int S(const std::string& name);
  int V(const std::string& name);  // std::vector<double> column
  void Finish();
  void Set(int column, long value);
  void Set(int column, int value) { Set(column, static_cast<long>(value)); }
  void Set(int column, bool value) { Set(column, static_cast<long>(value ? 1 : 0)); }
  void Set(int column, double value);
  void Set(int column, const std::string& value);
  void Set(int column, const std::vector<double>& value);
  void Add();

 private:
  int fId = -1;
  std::deque<std::vector<double>> fVectors;  // storage bound to the vector columns
  std::vector<int> fVectorOf;                // column -> index in fVectors, or -1
};

// The columns of one recorded state under a prefix (begin_, pre_, ...).
struct StateColumns {
  void Book(Table& t, const std::string& prefix);
  void Fill(Table& t, const State& s) const;
  int c[25];
};

class Output {
 public:
  Output();
  void Open(const std::string& file);
  void Close();
  void WriteEvent(long run, long subrun, long event, const Config& config, const Component& component,
                  const std::vector<TrackRow>& tracks, const std::vector<LinkRow>& links,
                  const std::vector<StepRow>& steps, const std::vector<ClassificationRow>& classifications,
                  const std::vector<std::string>& errors, long resourceKills, double cpu, double real,
                  const std::string& g4Version, const std::string& manifest, const std::string& manifestHash);
  void WriteRun(long run, long subrun, long events, const Config& config);

 private:
  Table fEvents, fTracks, fSteps, fClassifications, fVolumes, fLinks, fManifest, fRunInfo;
  StateColumns fBegin, fEnd, fEntrance, fPre, fPost, fState;
};

}  // namespace hsr
