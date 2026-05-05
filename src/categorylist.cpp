#include "categorylist.h"
#include "aimlpattern.h"
#include "aimltemplate.h"
#include "aimltext.h"
#include <map>
#include <sstream>
#include <algorithm>
#include <cctype>

using namespace std;
using namespace aiml;

// ---------------------------------------------------------------------------
// CategoryList
// ---------------------------------------------------------------------------

void CategoryList::append(Category* category) {
    m_vChildren.push_back(category);
    m_uSize++;
}

Category* CategoryList::child(int index) {
	return m_vChildren[index];
}

string CategoryList::file() {
	return m_sFile;
}

unsigned int CategoryList::size() {
	return m_uSize;
}

// ---------------------------------------------------------------------------
// LearnableCategoryList
// ---------------------------------------------------------------------------

LearnableCategoryList::LearnableCategoryList()
    : CategoryList("__learned__") {}

LearnableCategoryList::~LearnableCategoryList() {
    // m_vChildren owns its Category* pointers; delete them here since
    // the base class destructor is not virtual-delete-aware.
    for (Category* cat : m_vChildren)
        delete cat;
    m_vChildren.clear();
    m_uSize = 0;
}

Category* LearnableCategoryList::synthesize(const string& input,
                                             const string& response)
{
    if (input.empty() || response.empty()) return nullptr;

    string patternStr = generaliseInput(input);
    if (patternStr.empty()) return nullptr;

    // Build Pattern and Template objects from raw strings.
    Pattern*  pat   = new Pattern(patternStr);
    Text*     tText = new Text(response);
    Template* templ = new Template();
    templ->appendChild(tText);

    TruthValue tv(0.5, 0.1, false);
    Category* cat = new Category(pat, templ, tv);

    append(cat);
    return cat;
}

void LearnableCategoryList::reinforce(const string& patternStr, double amount) {
    for (Category* cat : m_vChildren) {
        if (cat && cat->pattern() &&
            cat->pattern()->toString() == patternStr)
        {
            cat->reinforceMatch(amount);
            break;
        }
    }
}

void LearnableCategoryList::decayAll(double factor) {
    for (Category* cat : m_vChildren)
        cat->decayConfidence(factor);
}

vector<Category*> LearnableCategoryList::consolidate(double threshold) const {
    vector<Category*> result;
    for (Category* cat : m_vChildren) {
        if (cat && cat->getTruthValue().confidence >= threshold)
            result.push_back(cat);
    }
    return result;
}

void LearnableCategoryList::prune(double minConfidence) {
    vector<Category*> survivors;
    for (Category* cat : m_vChildren) {
        if (cat && cat->getTruthValue().confidence >= minConfidence) {
            survivors.push_back(cat);
        } else {
            delete cat;
        }
    }
    m_vChildren = survivors;
    m_uSize = (unsigned int)survivors.size();
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

string LearnableCategoryList::generaliseInput(const string& input) const {
    istringstream ss(input);
    string tok;
    ostringstream result;
    bool first = true;

    while (ss >> tok) {
        // Strip punctuation.
        tok.erase(remove_if(tok.begin(), tok.end(), ::ispunct), tok.end());
        if (tok.empty()) continue;

        string lower = tok;
        transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if (!first) result << " ";
        first = false;

        // Replace stop words and short filler tokens with wildcards.
        if (isStopWord(lower) || lower.size() <= 2)
            result << "*";
        else
            result << lower;
    }

    return result.str();
}

bool LearnableCategoryList::isStopWord(const string& word) {
    static const string stops[] = {
        "a","an","the","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","i","you","he","she",
        "it","we","they","me","him","her","us","them","my","your",
        "his","its","our","their","this","that","these","those",
        "and","but","or","so","if","of","in","on","at","to","for",
        "with","about","what","how","why","who","when","where"
    };
    static const int N = sizeof(stops) / sizeof(stops[0]);
    for (int i = 0; i < N; ++i)
        if (stops[i] == word) return true;
    return false;
}

