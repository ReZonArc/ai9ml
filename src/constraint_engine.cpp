#include "constraint_engine.h"
#include <algorithm>
#include <sstream>
#include <cctype>

using namespace constraint_engine;

// ---------------------------------------------------------------------------
// ConstraintEngine — public interface
// ---------------------------------------------------------------------------

ResponseCandidate ConstraintEngine::selectBestCandidate(
    vector<ResponseCandidate>& candidates,
    const ResponseConstraints& constraints,
    const map<string, double>& contextVector,
    const vector<string>& recentResponses) const
{
    if (candidates.empty()) return ResponseCandidate();

    // Score and potentially eliminate each candidate.
    for (auto& cand : candidates) {
        if (cand.text.empty()) {
            cand.finalScore = -1.0;
            continue;
        }
        cand.topicalityScore = computeTopicalityScore(cand.text, contextVector);
        cand.finalScore = applyPenalties(cand, constraints, contextVector,
                                         recentResponses);
    }

    // Pick the surviving candidate with the highest final score.
    ResponseCandidate best;
    double bestScore = -2.0;
    for (const auto& cand : candidates) {
        if (cand.finalScore > bestScore) {
            bestScore = cand.finalScore;
            best = cand;
        }
    }

    if (bestScore < 0.0) return ResponseCandidate(); // all eliminated
    return best;
}

string ConstraintEngine::buildGPT4oConstraintPrompt(
    const ResponseConstraints& constraints,
    const map<string, double>& contextVector,
    const vector<string>& recentResponses) const
{
    ostringstream prompt;
    prompt << "You are a helpful AI assistant integrated into an AIML chatbot.";

    // Inject topic context.
    if (!contextVector.empty()) {
        // Collect top-5 concepts by salience.
        vector<pair<string, double>> sorted(contextVector.begin(),
                                            contextVector.end());
        sort(sorted.begin(), sorted.end(),
             [](const pair<string, double>& a,
                const pair<string, double>& b) {
                 return a.second > b.second;
             });
        if ((int)sorted.size() > 5) sorted.resize(5);

        prompt << " The current conversation is focused on: ";
        for (size_t i = 0; i < sorted.size(); ++i) {
            if (i) prompt << ", ";
            prompt << sorted[i].first;
        }
        prompt << ".";
    }

    // Anti-repetition hint.
    if (constraints.antiRepetition && !recentResponses.empty()) {
        int window = min(constraints.recentResponseWindow,
                         (int)recentResponses.size());
        prompt << " Do not repeat or closely paraphrase these recent responses:";
        for (int i = (int)recentResponses.size() - window;
             i < (int)recentResponses.size(); ++i) {
            if (i >= 0 && !recentResponses[i].empty()) {
                prompt << " [" << recentResponses[i].substr(
                    0, min((int)recentResponses[i].size(), 60)) << "...]";
            }
        }
    }

    // Length guidance.
    if (constraints.maxLength > 0)
        prompt << " Keep the response under " << constraints.maxLength
               << " characters.";

    prompt << " Provide a clear, helpful, and contextually relevant response.";
    return prompt.str();
}

double ConstraintEngine::computeTopicalityScore(
    const string& response,
    const map<string, double>& contextVector) const
{
    if (contextVector.empty() || response.empty()) return 0.0;

    auto tokens = tokenize(response);
    double score = 0.0;
    int    count = 0;
    for (const auto& tok : tokens) {
        auto it = contextVector.find(tok);
        if (it != contextVector.end()) {
            score += it->second;
            count++;
        }
    }
    return (count > 0) ? min(1.0, score / count) : 0.0;
}

ResponseConstraints ConstraintEngine::defaultConstraints() {
    return ResponseConstraints();
}

ResponseConstraints ConstraintEngine::strictConstraints() {
    ResponseConstraints c;
    c.topicalityFloor         = 0.1;
    c.strictTopicality        = true;
    c.antiRepetition          = true;
    c.recentResponseWindow    = 8;
    c.confidenceFloor         = 0.05;
    c.topicalityPenaltyWeight = 0.5;
    c.repetitionPenaltyWeight = 0.7;
    c.lengthPenaltyWeight     = 0.2;
    return c;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

double ConstraintEngine::applyPenalties(
    ResponseCandidate& candidate,
    const ResponseConstraints& constraints,
    const map<string, double>& contextVector,
    const vector<string>& recentResponses) const
{
    double score = candidate.baseScore;

    // Confidence floor (hard constraint).
    if (constraints.confidenceFloor > 0.0 &&
        candidate.confidence < constraints.confidenceFloor)
        return -1.0;

    // Topicality (hard or soft).
    if (constraints.topicalityFloor > 0.0 && !contextVector.empty()) {
        double topicality = candidate.topicalityScore;
        if (topicality < constraints.topicalityFloor) {
            if (constraints.strictTopicality) return -1.0;
            double deficit = constraints.topicalityFloor - topicality;
            score -= deficit * constraints.topicalityPenaltyWeight;
        }
    }

    // Anti-repetition (soft).
    if (constraints.antiRepetition) {
        int window = min(constraints.recentResponseWindow,
                         (int)recentResponses.size());
        for (int i = (int)recentResponses.size() - window;
             i < (int)recentResponses.size(); ++i) {
            if (i >= 0 && recentResponses[i] == candidate.text) {
                score -= constraints.repetitionPenaltyWeight;
                break;
            }
        }
    }

    // Length (soft).
    int len = (int)candidate.text.size();
    if (constraints.minLength > 0 && len < constraints.minLength)
        score -= (1.0 - (double)len / constraints.minLength) *
                 constraints.lengthPenaltyWeight;
    if (constraints.maxLength > 0 && len > constraints.maxLength)
        score -= ((double)(len - constraints.maxLength) / constraints.maxLength) *
                 constraints.lengthPenaltyWeight;

    return score;
}

vector<string> ConstraintEngine::tokenize(const string& text) const {
    vector<string> tokens;
    istringstream ss(text);
    string tok;
    while (ss >> tok) {
        tok.erase(remove_if(tok.begin(), tok.end(), ::ispunct), tok.end());
        transform(tok.begin(), tok.end(), tok.begin(), ::tolower);
        if (tok.size() > 2) tokens.push_back(tok);
    }
    return tokens;
}
