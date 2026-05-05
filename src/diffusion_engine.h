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

        // --- Nested learning loops (§5) ---
        //
        // Call innerLoopStep() once per conversation turn from
        // Chatmachine::updateNSVDState().  It automatically triggers the
        // middle loop every innerThreshold calls, and the outer loop every
        // outerThreshold middle-loop completions.
        //
        //   Inner  loop (per-turn, default every 1):
        //     - trust propagation, temperature decay, blend reinforcement.
        //
        //   Middle loop (default every 10 inner steps):
        //     - blend garbage-collection, concept generalisation pass,
        //       temperature plateau check.
        //
        //   Outer  loop (default every 5 middle-loop completions = 50 turns):
        //     - AIML category consolidation, full blend GC,
        //       temperature reset for next session segment.
        //
        // The Chatmachine may attach additional callbacks to these hooks by
        // checking getInnerLoopCount() / getMiddleLoopCount() modulo thresholds.

        void innerLoopStep();
        void middleLoopStep();
        void outerLoopStep();

        void setInnerThreshold(int n) { m_innerThreshold = n; }
        void setOuterThreshold(int n) { m_outerThreshold = n; }
        int  getInnerThreshold() const { return m_innerThreshold; }
        int  getOuterThreshold() const { return m_outerThreshold; }

        int getInnerLoopCount()  const { return m_innerCount;  }
        int getMiddleLoopCount() const { return m_middleCount; }
        int getOuterLoopCount()  const { return m_outerCount;  }

    private:
        AtomSpace& m_atomSpace;
        double     m_temperature;
        int        m_consolidationCounter; // prevents filename collisions in consolidation

        // Names of concepts that were created by interpolation.
        unordered_set<string> m_blendedConcepts;

        // Nested loop counters and thresholds.
        int m_innerCount;      // total inner-loop steps executed
        int m_middleCount;     // total middle-loop steps executed
        int m_outerCount;      // total outer-loop steps executed
        int m_innerThreshold;  // inner steps per middle-loop trigger (default 10)
        int m_outerThreshold;  // middle steps per outer-loop trigger  (default 5)

        // Build an AIML <category> XML string from pattern/template strings.
        string formatAIMLCategory(const string& pattern,
                                   const string& templateText) const;
    };

} // namespace diffusion_engine

#endif // __DIFFUSION_ENGINE_H__
