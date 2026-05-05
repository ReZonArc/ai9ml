#ifndef __DIFFUSION_ENGINE_H__
#define __DIFFUSION_ENGINE_H__

/**
 * diffusion_engine.h — Variability Diffusion Mechanism (Phase 4)
 *
 * DiffusionEngine controls the spread of novelty through the AtomSpace:
 *
 *  1. Trust propagation (§4.1):
 *     When a new atom is added, propagate a fraction of its truth value
 *     to neighbouring atoms through link connections.  A temperature
 *     parameter τ controls diffusion depth: high τ (cold-start) spreads
 *     novelty widely; low τ (settled session) limits propagation.
 *
 *  2. Conceptual interpolation (§4.2):
 *     Given two salient concepts A and B, create a temporary blended
 *     ConceptNode "A~B" connected to both by SimilarityLinks.  Blends
 *     that receive positive reinforcement are stabilised; ignored ones
 *     decay and are garbage-collected.
 *
 *  3. Category consolidation (§4.3):
 *     A LearnableCategoryList is inspected; categories with confidence
 *     above a threshold are serialised as AIML XML into a target directory.
 *     Semantically similar soft categories are merged before export.
 */

#include "atomspace.h"
#include "categorylist.h"
#include <string>
#include <unordered_set>

using namespace std;
using namespace opencog;
using namespace aiml;

namespace diffusion_engine {

    class DiffusionEngine {
    public:
        explicit DiffusionEngine(AtomSpace& atomSpace);
        ~DiffusionEngine() = default;

        // --- Temperature management ---
        void   setTemperature(double tau) { m_temperature = tau; }
        double getTemperature()     const { return m_temperature; }
        // Multiply temperature by factor (called each turn, converges to 0).
        void decayTemperature(double factor = 0.95);
        // Reset to initial high temperature (new session / topic shift).
        void resetTemperature(double tau = 1.0);

        // --- Trust propagation (§4.1) ---
        // Propagate trust from origin up to ceil(maxDepth * temperature) hops.
        void propagateTrustFromAtom(shared_ptr<Atom> origin,
                                    double fraction = 0.3,
                                    int    maxDepth = 4);

        // --- Conceptual interpolation (§4.2) ---
        // Blend two concepts; returns the blended ConceptNode name or "".
        string interpolateConcepts(const string& conceptA,
                                   const string& conceptB);

        // Reward a blended concept — increases its truth value.
        void reinforceBlend(const string& blendName, double reward = 0.1);

        // Remove blended concepts below minConfidence.
        void garbageCollectBlends(double minConfidence = 0.05);

        // --- Category consolidation (§4.3) ---
        // Write high-confidence soft categories as AIML XML into outputDir.
        // Returns number of categories exported.
        int consolidateLearnedCategories(LearnableCategoryList* learnedCL,
                                         const string& outputDir,
                                         double threshold = 0.7);

        // Print diffusion diagnostics.
        void printDiffusionStats() const;

    private:
        AtomSpace& m_atomSpace;
        double     m_temperature;
        int        m_consolidationCounter; // prevents filename collisions in consolidation

        // Names of concepts that were created by interpolation.
        unordered_set<string> m_blendedConcepts;

        // Build an AIML <category> XML string from pattern/template strings.
        string formatAIMLCategory(const string& pattern,
                                   const string& templateText) const;
    };

} // namespace diffusion_engine

#endif // __DIFFUSION_ENGINE_H__
