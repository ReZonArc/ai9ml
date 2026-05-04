#ifndef __CONSTRAINT_ENGINE_H__
#define __CONSTRAINT_ENGINE_H__

/**
 * constraint_engine.h — Response Constraint Optimisation (Phase 3)
 *
 * Defines:
 *   ResponseConstraints  — specification of hard/soft limits on responses.
 *   ResponseCandidate    — a response with its provenance and scores.
 *   ConstraintEngine     — penalty-based re-ranking and GPT-4o prompt builder.
 *
 * Hard constraints eliminate candidates entirely (topicality, contradiction).
 * Soft constraints subtract a weighted penalty from the candidate's score.
 */

#include <string>
#include <vector>
#include <map>

using namespace std;

namespace constraint_engine {

    // -----------------------------------------------------------------------
    // Constraint specification
    // -----------------------------------------------------------------------

    struct ResponseConstraints {
        // Topicality: minimum overlap between response concepts and the
        // current context vector (0.0 = disabled).
        double topicalityFloor;

        // When true, topicality below topicalityFloor *eliminates* the
        // candidate (hard constraint).  Otherwise it merely penalises.
        bool strictTopicality;

        // Anti-repetition: do not allow a response identical to any of the
        // last recentResponseWindow responses.
        bool antiRepetition;
        int  recentResponseWindow;

        // Length bounds (in characters; 0 = no limit).
        int minLength;
        int maxLength;

        // Minimum source confidence to accept a candidate (0.0 = disabled).
        double confidenceFloor;

        // Penalty weights for soft constraint violations (0.0–1.0 each).
        double topicalityPenaltyWeight;
        double repetitionPenaltyWeight;
        double lengthPenaltyWeight;

        ResponseConstraints()
            : topicalityFloor(0.0),
              strictTopicality(false),
              antiRepetition(true),
              recentResponseWindow(5),
              minLength(1),
              maxLength(0),
              confidenceFloor(0.0),
              topicalityPenaltyWeight(0.3),
              repetitionPenaltyWeight(0.5),
              lengthPenaltyWeight(0.2) {}
    };

    // -----------------------------------------------------------------------
    // Response candidate
    // -----------------------------------------------------------------------

    struct ResponseCandidate {
        string text;
        double baseScore;    // score from the originating subsystem
        double finalScore;   // after constraint penalties
        string source;       // "aiml" | "opencog" | "learned" | "gpt4o"
        double confidence;   // originating subsystem confidence
        double topicalityScore; // computed by ConstraintEngine

        ResponseCandidate()
            : baseScore(0.0), finalScore(0.0), confidence(1.0),
              topicalityScore(0.0) {}

        ResponseCandidate(const string& t, double bs, const string& src,
                          double conf = 1.0)
            : text(t), baseScore(bs), finalScore(bs), source(src),
              confidence(conf), topicalityScore(0.0) {}
    };

    // -----------------------------------------------------------------------
    // ConstraintEngine
    // -----------------------------------------------------------------------

    class ConstraintEngine {
    public:
        ConstraintEngine() = default;
        ~ConstraintEngine() = default;

        /**
         * Score all candidates against the constraints, eliminate violators
         * of hard constraints, and return the single best candidate.
         * Returns an empty ResponseCandidate (text == "") if all candidates
         * are eliminated.
         */
        ResponseCandidate selectBestCandidate(
            vector<ResponseCandidate>& candidates,
            const ResponseConstraints& constraints,
            const map<string, double>& contextVector,
            const vector<string>& recentResponses) const;

        /**
         * Build a system-prompt prefix that injects the active constraints
         * and top-salience concepts into a GPT-4o request.
         */
        string buildGPT4oConstraintPrompt(
            const ResponseConstraints& constraints,
            const map<string, double>& contextVector,
            const vector<string>& recentResponses) const;

        // Compute topicality score for a response against the context vector.
        double computeTopicalityScore(
            const string& response,
            const map<string, double>& contextVector) const;

        // Preset constraint profiles.
        static ResponseConstraints defaultConstraints();
        static ResponseConstraints strictConstraints();

    private:
        // Apply penalties and return the adjusted score.
        // May set score to -1.0 to mark hard-constraint elimination.
        double applyPenalties(ResponseCandidate& candidate,
                              const ResponseConstraints& constraints,
                              const map<string, double>& contextVector,
                              const vector<string>& recentResponses) const;

        vector<string> tokenize(const string& text) const;
    };

} // namespace constraint_engine

#endif // __CONSTRAINT_ENGINE_H__
