#ifndef __CATEGORYLIST_H__
#define __CATEGORYLIST_H__

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <map>
#include "aimlelement.h"
#include "aimlcategory.h"

using namespace std;

namespace aiml {

    class CategoryList {
    public:
        CategoryList() :m_uSize(0) {}
        CategoryList(string file) :m_uSize(0), m_sFile(file) {}
        virtual ~CategoryList() {}

        virtual void append(Category* category);
        Category* child(int index);
        string file();
        unsigned int size();

        // Access to categories vector
        const vector<Category*>& getCategories() const { return m_vChildren; }

    protected:
        vector<Category*> m_vChildren;
        string m_sFile;
        unsigned int m_uSize;
    };

    /**
     * LearnableCategoryList — a mutable extension of CategoryList for
     * dynamically synthesised (soft) categories.
     *
     * Soft categories have TruthValue(strength=0.5, confidence=0.1, immutable=false).
     * They are kept separate from the statically loaded AIML files and can be:
     *   - reinforced when they are reused,
     *   - decayed each conversation turn,
     *   - consolidated (exported to AIML XML) when confidence > threshold,
     *   - pruned when confidence falls below a minimum.
     */
    class LearnableCategoryList : public CategoryList {
    public:
        LearnableCategoryList();
        ~LearnableCategoryList();

        // Synthesise a new soft category from a raw input string and response.
        // The pattern is a generalised form (low-frequency words replaced by *).
        // Returns the new category, which is owned by this list.
        Category* synthesize(const string& input, const string& response);

        // Reinforce the first soft category whose pattern string equals patternStr.
        void reinforce(const string& patternStr, double amount = 0.05);

        // Decay the confidence of every soft category by factor.
        void decayAll(double factor = 0.99);

        // Return categories whose confidence >= threshold (for consolidation).
        vector<Category*> consolidate(double threshold = 0.7) const;

        // Remove categories with confidence < minConfidence.
        void prune(double minConfidence = 0.02);

    private:
        // Generalise input into a pattern by replacing non-keyword tokens with *.
        string generaliseInput(const string& input) const;

        // Minimal stop-word list used during generalisation.
        static bool isStopWord(const string& word);
    };

} // namespace aiml

#endif
