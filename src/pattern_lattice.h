#ifndef __PATTERN_LATTICE_H__
#define __PATTERN_LATTICE_H__

/**
 * pattern_lattice.h — Probabilistic Pattern Lattice (Phase 1.2)
 *
 * Organises AIML Category objects into a DAG where wildcard patterns are
 * ancestors of specific patterns.  Replaces pure Levenshtein-distance ranking
 * with variational scoring:
 *
 *   score = semantic_similarity(input, pattern) * tv.strength * tv.confidence
 *           + salience_boost(pattern, contextVector)
 *
 * The salience_boost is driven by the ContextVector held in
 * OpenCogAIMLIntegration (passed in at query time).
 */

#include "aimlcategory.h"
#include <string>
#include <vector>
#include <map>
#include <set>

using namespace std;
using namespace aiml;

namespace pattern_lattice {

    // Scored pair returned by findBestCategories().
    struct ScoredCategory {
        Category* category;
        double    score;
        bool      wildcardMatch; // true if the match required a wildcard

        ScoredCategory()
            : category(nullptr), score(0.0), wildcardMatch(false) {}

        ScoredCategory(Category* c, double s, bool wc = false)
            : category(c), score(s), wildcardMatch(wc) {}
    };

    /**
     * PatternLattice — builds a lightweight DAG over a flat list of categories.
     *
     * Wildcard categories (containing "*" tokens) become ancestor nodes;
     * fully-specific categories become leaf nodes.  At query time the lattice
     * traversal starts at leaves and falls back through wildcard ancestors,
     * collecting ScoredCategory candidates.
     */
    class PatternLattice {
    public:
        PatternLattice();
        ~PatternLattice() = default;

        // Build / rebuild the lattice from a flat category vector.
        // All categories added here receive immutable TruthValues (anchored).
        void build(const vector<Category*>& categories);

        // Add a soft (learned) category at runtime.
        void addLearnedCategory(Category* category);

        // Primary query: returns topK candidates scored by the variational metric.
        // contextVector: concept → salience (from ContextVector in OpenCog layer).
        vector<ScoredCategory> findBestCategories(
            const string& input,
            const map<string, double>& contextVector,
            int topK = 5) const;

        // Return the single best category (convenience wrapper).
        Category* findBest(
            const string& input,
            const map<string, double>& contextVector) const;

        // Decay confidence on all soft (learnable) categories.
        void decayAll(double factor = 0.99);

        // Statistics.
        size_t size() const { return m_categories.size(); }
        size_t wildcardCount() const { return m_wildcardCategories.size(); }
        size_t specificCount() const { return m_specificCategories.size(); }

    private:
        vector<Category*> m_categories;          // all registered categories
        vector<Category*> m_wildcardCategories;  // patterns containing "*"
        vector<Category*> m_specificCategories;  // fully-specific patterns

        // Compute variational score for one (input, category) pair.
        double computeScore(const string& input,
                            Category* category,
                            const map<string, double>& contextVector) const;

        // Semantic similarity between input and pattern (0.0–1.0).
        double semanticSimilarity(const string& input, const string& pattern) const;

        // Salience boost from the context vector (0.0–1.0).
        double salienceBoost(const string& pattern,
                             const map<string, double>& contextVector) const;

        // Helpers
        bool hasWildcard(const string& pattern) const;
        vector<string> tokenize(const string& text) const;
    };

} // namespace pattern_lattice

#endif // __PATTERN_LATTICE_H__
