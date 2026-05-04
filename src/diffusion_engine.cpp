#include "diffusion_engine.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <sys/stat.h>

using namespace diffusion_engine;

// ---------------------------------------------------------------------------
// DiffusionEngine
// ---------------------------------------------------------------------------

DiffusionEngine::DiffusionEngine(AtomSpace& atomSpace)
    : m_atomSpace(atomSpace), m_temperature(1.0) {}

// --- Temperature ---

void DiffusionEngine::decayTemperature(double factor) {
    m_temperature = max(0.05, m_temperature * factor);
}

void DiffusionEngine::resetTemperature(double tau) {
    m_temperature = max(0.0, min(1.0, tau));
}

// --- Trust propagation ---

void DiffusionEngine::propagateTrustFromAtom(
    shared_ptr<Atom> origin, double fraction, int maxDepth)
{
    if (!origin) return;
    // Scale depth by current temperature so novelty spreads further at
    // the start of a session and stays local once patterns are settled.
    int effectiveDepth = max(1, (int)ceil(maxDepth * m_temperature));
    m_atomSpace.propagateTrust(origin, fraction, effectiveDepth);
}

// --- Conceptual interpolation ---

string DiffusionEngine::interpolateConcepts(const string& conceptA,
                                             const string& conceptB)
{
    if (conceptA.empty() || conceptB.empty()) return "";
    if (conceptA == conceptB) return conceptA;

    auto blendNode = m_atomSpace.interpolateConcepts(conceptA, conceptB,
                                                      0.5);
    if (!blendNode) return "";

    string blendName = blendNode->getName();
    m_blendedConcepts.insert(blendName);
    return blendName;
}

void DiffusionEngine::reinforceBlend(const string& blendName, double reward) {
    auto atom = m_atomSpace.getAtom(opencog::CONCEPT_NODE, blendName);
    if (!atom) return;
    double newTV = min(1.0, atom->getTruthValue() + reward);
    atom->setTruthValue(newTV);
}

void DiffusionEngine::garbageCollectBlends(double minConfidence) {
    m_atomSpace.garbageCollectBlends(minConfidence);
    // Remove stale entries from our local tracking set.
    for (auto it = m_blendedConcepts.begin(); it != m_blendedConcepts.end(); ) {
        auto atom = m_atomSpace.getAtom(opencog::CONCEPT_NODE, *it);
        if (!atom || atom->getTruthValue() < minConfidence)
            it = m_blendedConcepts.erase(it);
        else
            ++it;
    }
}

// --- Category consolidation ---

int DiffusionEngine::consolidateLearnedCategories(
    LearnableCategoryList* learnedCL,
    const string& outputDir,
    double threshold)
{
    if (!learnedCL) return 0;

    auto candidates = learnedCL->consolidate(threshold);
    if (candidates.empty()) return 0;

    // Ensure output directory exists (best-effort).
    mkdir(outputDir.c_str(), 0755);

    // Group candidates that share the same generalised pattern key to merge
    // semantically redundant categories.
    // Simple merge strategy: same pattern → unify templates under <random>.
    map<string, vector<string>> patternToTemplates;
    for (Category* cat : candidates) {
        if (!cat->pattern() || !cat->templ()) continue;
        string pat = cat->pattern()->toString();
        string tmpl = cat->templ()->toString();
        if (!pat.empty() && !tmpl.empty())
            patternToTemplates[pat].push_back(tmpl);
    }

    // Write a single AIML file for this consolidation pass.
    string filename = outputDir + "/learned_" +
                      to_string((long long)time(nullptr)) + ".aiml";
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "[DiffusionEngine] Cannot write to " << filename << endl;
        return 0;
    }

    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        << "<aiml version=\"1.0\">\n";

    int exported = 0;
    for (const auto& kv : patternToTemplates) {
        const string& pat = kv.first;
        const vector<string>& tmpls = kv.second;

        out << "  <category>\n"
            << "    <pattern>" << pat << "</pattern>\n";

        if (tmpls.size() == 1) {
            out << "    <template>" << tmpls[0] << "</template>\n";
        } else {
            out << "    <template>\n      <random>\n";
            for (const auto& t : tmpls)
                out << "        <li>" << t << "</li>\n";
            out << "      </random>\n    </template>\n";
        }

        out << "  </category>\n";
        exported++;
    }

    out << "</aiml>\n";
    out.close();

    cout << "[DiffusionEngine] Consolidated " << exported
         << " learned categories -> " << filename << endl;
    return exported;
}

// --- Diagnostics ---

void DiffusionEngine::printDiffusionStats() const {
    cout << "\n=== DiffusionEngine Stats ===" << endl;
    cout << "Temperature (tau): " << m_temperature << endl;
    cout << "Blended concepts:  " << m_blendedConcepts.size() << endl;
    cout << "AtomSpace size:    " << m_atomSpace.size() << endl;
    cout << "============================\n" << endl;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

string DiffusionEngine::formatAIMLCategory(const string& pattern,
                                            const string& templateText) const {
    ostringstream ss;
    ss << "  <category>\n"
       << "    <pattern>" << pattern << "</pattern>\n"
       << "    <template>" << templateText << "</template>\n"
       << "  </category>\n";
    return ss.str();
}
