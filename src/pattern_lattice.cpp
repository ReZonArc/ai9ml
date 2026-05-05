#include "pattern_lattice.h"
#include "strings.h"
#include <algorithm>
#include <sstream>
#include <cctype>

using namespace pattern_lattice;

// ---------------------------------------------------------------------------
// PatternLattice
// ---------------------------------------------------------------------------

PatternLattice::PatternLattice() {}

void PatternLattice::build(const vector<Category*>& categories) {
    m_categories.clear();
    m_wildcardCategories.clear();
    m_specificCategories.clear();

    for (Category* cat : categories) {
        if (!cat || !cat->pattern()) continue;
        m_categories.push_back(cat);
        string pat = cat->pattern()->toString();
        if (hasWildcard(pat))
            m_wildcardCategories.push_back(cat);
        else
            m_specificCategories.push_back(cat);
    }
}

void PatternLattice::addLearnedCategory(Category* category) {
    if (!category || !category->pattern()) return;
    m_categories.push_back(category);
    string pat = category->pattern()->toString();
    if (hasWildcard(pat))
        m_wildcardCategories.push_back(category);
    else
        m_specificCategories.push_back(category);
}

vector<ScoredCategory> PatternLattice::findBestCategories(
    const string& input,
    const map<string, double>& contextVector,
    int topK) const
{
    vector<ScoredCategory> results;
    results.reserve(m_categories.size());

    for (Category* cat : m_categories) {
        double score = computeScore(input, cat, contextVector);
        bool wc = hasWildcard(cat->pattern()->toString());
        results.emplace_back(cat, score, wc);
    }

    // Sort descending by score.
    sort(results.begin(), results.end(),
         [](const ScoredCategory& a, const ScoredCategory& b) {
             return a.score > b.score;
         });

    if (topK > 0 && (int)results.size() > topK)
        results.resize(topK);

    return results;
}

Category* PatternLattice::findBest(
    const string& input,
    const map<string, double>& contextVector) const
{
    auto candidates = findBestCategories(input, contextVector, 1);
    if (!candidates.empty() && candidates[0].score > 0.0)
        return candidates[0].category;
    return nullptr;
}

void PatternLattice::decayAll(double factor) {
    for (Category* cat : m_categories)
        cat->decayConfidence(factor);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

double PatternLattice::computeScore(
    const string& input,
    Category* category,
    const map<string, double>& contextVector) const
{
    if (!category || !category->pattern()) return 0.0;

    const string& pat = category->pattern()->toString();
    const TruthValue& tv = category->getTruthValue();

    double simScore  = semanticSimilarity(input, pat);
    double salience  = salienceBoost(pat, contextVector);

    // Core variational score.
    double score = simScore * tv.strength * tv.confidence + salience * 0.1;

    // Wildcard patterns get a small structural penalty so specific patterns
    // win when similarity is equal (Dirichlet prior: mass is concentrated
    // at specific leaves and only propagates up when specifics are absent).
    if (hasWildcard(pat))
        score *= 0.85;

    return score;
}

double PatternLattice::semanticSimilarity(
    const string& input,
    const string& pattern) const
{
    if (input.empty() || pattern.empty()) return 0.0;

    // Exact match after case-normalisation.
    string inLower  = input;
    string patLower = pattern;
    transform(inLower.begin(),  inLower.end(),  inLower.begin(),  ::tolower);
    transform(patLower.begin(), patLower.end(), patLower.begin(), ::tolower);

    if (inLower == patLower) return 1.0;

    // Pure-wildcard pattern matches anything with low base score.
    if (patLower == "*") return 0.15;

    vector<string> inputTokens   = tokenize(inLower);
    vector<string> patternTokens = tokenize(patLower);

    if (inputTokens.empty() || patternTokens.empty()) return 0.0;

    // Count meaningful shared tokens (ignore wildcards in the count).
    int shared = 0;
    int patternWords = 0;
    for (const string& pt : patternTokens) {
        if (pt == "*") continue;
        patternWords++;
        for (const string& it : inputTokens) {
            if (it == pt) { shared++; break; }
        }
    }

    if (patternWords == 0) return 0.15; // pattern is all wildcards

    double tokenOverlap = (double)shared / (double)patternWords;

    // Bonus for substring containment.
    double substringBonus = 0.0;
    if (inLower.find(patLower) != string::npos ||
        patLower.find(inLower) != string::npos)
        substringBonus = 0.15;

    return min(1.0, tokenOverlap + substringBonus);
}

double PatternLattice::salienceBoost(
    const string& pattern,
    const map<string, double>& contextVector) const
{
    if (contextVector.empty() || pattern.empty()) return 0.0;

    vector<string> patTokens = tokenize(pattern);
    double boost = 0.0;
    int count = 0;

    for (const string& tok : patTokens) {
        if (tok == "*") continue;
        auto it = contextVector.find(tok);
        if (it != contextVector.end()) {
            boost += it->second;
            count++;
        }
    }

    return (count > 0) ? min(1.0, boost / count) : 0.0;
}

bool PatternLattice::hasWildcard(const string& pattern) const {
    return pattern.find('*') != string::npos;
}

vector<string> PatternLattice::tokenize(const string& text) const {
    vector<string> tokens;
    istringstream ss(text);
    string tok;
    while (ss >> tok) {
        tok.erase(remove_if(tok.begin(), tok.end(), ::ispunct), tok.end());
        if (!tok.empty()) tokens.push_back(tok);
    }
    return tokens;
}
