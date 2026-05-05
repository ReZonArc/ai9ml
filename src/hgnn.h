#ifndef __HGNN_H__
#define __HGNN_H__

/**
 * hgnn.h — Spatially-Aware Hyper-Graph Neural Network (Phase 5)
 *
 * Runs spatial message-passing over the OpenCog AtomSpace hypergraph.
 * Each ConceptNode maintains a HGNN_FEAT_DIM-dimensional embedding that is
 * updated by aggregating its structural neighbours' embeddings through
 * InheritanceLinks and SimilarityLinks.
 *
 * Complementary to DTESNN (temporal): HGNN captures the *structural*
 * neighbourhood of concepts independent of when they appeared in the
 * conversation.
 *
 * The forward pass (forwardPass()) is called from the NSVD middle loop
 * (every K conversation turns).  The async hgnn response-scoring path
 * uses the cached embeddings produced by the last forwardPass() call,
 * so the async path is read-only and thread-safe.
 */

#include "atomspace.h"
#include <vector>
#include <string>
#include <map>

using namespace std;
using namespace opencog;

namespace hgnn {

    static const int HGNN_FEAT_DIM   = 16;  // per-node embedding dimension
    static const int HGNN_OUTPUT_DIM = 8;   // projected output fed to MLP
    static const int HGNN_LAYERS     = 2;   // message-passing rounds

    class HyperGraphNeuralNet {
    public:
        explicit HyperGraphNeuralNet(AtomSpace& atomSpace);
        ~HyperGraphNeuralNet() = default;

        // Run HGNN_LAYERS rounds of spatial message passing.
        // All ConceptNodes in the AtomSpace are lazily initialised if needed.
        // Call from the NSVD middle loop; not thread-safe with concurrent writes.
        void forwardPass();

        // Read-only: get the HGNN_FEAT_DIM embedding for a named concept.
        // Returns a zero vector if the concept is unknown.
        vector<double> getEmbedding(const string& conceptName) const;

        // Read-only: mean-pool embeddings over a set of concepts, then project
        // to HGNN_OUTPUT_DIM.  Unknown concepts are ignored.
        vector<double> aggregateEmbeddings(const vector<string>& concepts) const;

        // Read-only: score a response candidate given input concepts.
        // Score = cosine similarity(agg(inputConcepts), agg(responseConcepts)).
        double scoreResponse(const string& response,
                             const vector<string>& inputConcepts) const;

        // Positive reinforcement: nudge a concept embedding toward the reward
        // direction.  Used by the outer learning loop.
        void reinforce(const string& conceptName, double reward = 0.05);

        // Number of concept embeddings currently maintained.
        size_t size() const { return m_embeddings.size(); }

    private:
        AtomSpace& m_atomSpace;

        // concept name → HGNN_FEAT_DIM feature vector.
        map<string, vector<double>> m_embeddings;

        // Learnable projection: HGNN_OUTPUT_DIM × HGNN_FEAT_DIM.
        vector<vector<double>> m_projection;

        // Initialise projection with random values if not done yet.
        void ensureProjection();

        // Get-or-create embedding for a concept name.
        vector<double>& ensureEmbedding(const string& name);

        // One round: update every embedding by aggregating its neighbours.
        void messagePassingRound();

        // Aggregate neighbour embeddings via mean pooling.
        vector<double> aggregateNeighbours(const string& conceptName) const;

        // Project HGNN_FEAT_DIM → HGNN_OUTPUT_DIM via m_projection.
        vector<double> project(const vector<double>& embedding) const;

        // Tokenise text into lowercase words.
        static vector<string> tokenize(const string& text);

        // Numeric helpers.
        static vector<double> randomVec(int dim, double scale);
        static vector<double> meanPool(const vector<vector<double>>& vecs);
        static double         cosine(const vector<double>& a,
                                     const vector<double>& b);
        static vector<double> relu(const vector<double>& x);
        static double         dot(const vector<double>& a,
                                  const vector<double>& b);
    };

} // namespace hgnn

#endif // __HGNN_H__
