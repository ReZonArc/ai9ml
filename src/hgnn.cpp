#include "hgnn.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <cctype>

using namespace hgnn;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

HyperGraphNeuralNet::HyperGraphNeuralNet(AtomSpace& atomSpace)
    : m_atomSpace(atomSpace)
{
    ensureProjection();
}

// ---------------------------------------------------------------------------
// Forward pass (spatial message passing)
// ---------------------------------------------------------------------------

void HyperGraphNeuralNet::forwardPass()
{
    // Ensure every ConceptNode in the AtomSpace has an embedding.
    auto conceptAtoms = m_atomSpace.getAtomsByType(opencog::CONCEPT_NODE);
    for (const auto& atom : conceptAtoms)
        ensureEmbedding(atom->getName());

    // Run HGNN_LAYERS rounds of message passing.
    for (int layer = 0; layer < HGNN_LAYERS; ++layer)
        messagePassingRound();
}

// ---------------------------------------------------------------------------
// Read-only queries
// ---------------------------------------------------------------------------

vector<double> HyperGraphNeuralNet::getEmbedding(const string& name) const
{
    auto it = m_embeddings.find(name);
    if (it != m_embeddings.end()) return it->second;
    return vector<double>(HGNN_FEAT_DIM, 0.0);
}

vector<double> HyperGraphNeuralNet::aggregateEmbeddings(
    const vector<string>& concepts) const
{
    vector<vector<double>> vecs;
    for (const auto& c : concepts) {
        auto it = m_embeddings.find(c);
        if (it != m_embeddings.end() && !it->second.empty())
            vecs.push_back(it->second);
    }
    if (vecs.empty())
        return vector<double>(HGNN_OUTPUT_DIM, 0.0);

    auto pooled = meanPool(vecs);
    return project(pooled);
}

double HyperGraphNeuralNet::scoreResponse(const string& response,
                                           const vector<string>& inputConcepts) const
{
    if (inputConcepts.empty()) return 0.0;
    auto inputAgg = aggregateEmbeddings(inputConcepts);

    auto responseTokens = tokenize(response);
    auto responseAgg    = aggregateEmbeddings(responseTokens);

    double sim = cosine(inputAgg, responseAgg);
    // Map [-1,1] to [0,1].
    return 0.5 * (1.0 + sim);
}

// ---------------------------------------------------------------------------
// Positive reinforcement
// ---------------------------------------------------------------------------

void HyperGraphNeuralNet::reinforce(const string& name, double reward)
{
    auto& emb = ensureEmbedding(name);
    // Nudge embedding toward a unit direction proportional to reward.
    for (int i = 0; i < HGNN_FEAT_DIM; ++i)
        emb[i] = min(1.0, emb[i] + reward * 0.1);
}

// ---------------------------------------------------------------------------
// Private: message-passing round
// ---------------------------------------------------------------------------

void HyperGraphNeuralNet::messagePassingRound()
{
    // Compute new embeddings from current (two-pass to avoid in-place corruption).
    map<string, vector<double>> newEmbs = m_embeddings;

    for (auto& kv : m_embeddings) {
        const string& conceptName = kv.first;

        // Gather structural neighbours from AtomSpace.
        auto neighbours = aggregateNeighbours(conceptName);

        // Mix: new = 0.6 * self + 0.4 * ReLU(neighbours)
        auto activated = relu(neighbours);
        for (int i = 0; i < HGNN_FEAT_DIM; ++i)
            newEmbs[conceptName][i] = 0.6 * kv.second[i] + 0.4 * activated[i];
    }

    m_embeddings = newEmbs;
}

vector<double> HyperGraphNeuralNet::aggregateNeighbours(const string& name) const
{
    vector<string> neighbourNames;

    // Parents (is-a hierarchy).
    auto parents = m_atomSpace.getParentConcepts(name);
    neighbourNames.insert(neighbourNames.end(), parents.begin(), parents.end());

    // Children.
    auto children = m_atomSpace.getChildConcepts(name);
    neighbourNames.insert(neighbourNames.end(), children.begin(), children.end());

    // Semantically similar concepts.
    auto similar = m_atomSpace.findSimilarConcepts(name, 0.5);
    for (const auto& a : similar)
        neighbourNames.push_back(a->getName());

    if (neighbourNames.empty())
        return vector<double>(HGNN_FEAT_DIM, 0.0);

    // Collect embeddings of known neighbours.
    vector<vector<double>> vecs;
    for (const auto& n : neighbourNames) {
        auto it = m_embeddings.find(n);
        if (it != m_embeddings.end())
            vecs.push_back(it->second);
    }

    return vecs.empty() ? vector<double>(HGNN_FEAT_DIM, 0.0) : meanPool(vecs);
}

// ---------------------------------------------------------------------------
// Private: projection
// ---------------------------------------------------------------------------

void HyperGraphNeuralNet::ensureProjection()
{
    if (!m_projection.empty()) return;
    // Random projection matrix: HGNN_OUTPUT_DIM × HGNN_FEAT_DIM.
    double scale = sqrt(2.0 / (HGNN_FEAT_DIM + HGNN_OUTPUT_DIM));
    m_projection.assign(HGNN_OUTPUT_DIM, vector<double>(HGNN_FEAT_DIM));
    for (auto& row : m_projection)
        for (double& v : row)
            v = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
}

vector<double> HyperGraphNeuralNet::project(const vector<double>& embedding) const
{
    vector<double> out(HGNN_OUTPUT_DIM, 0.0);
    int cols = min((int)embedding.size(), HGNN_FEAT_DIM);
    for (int i = 0; i < HGNN_OUTPUT_DIM; ++i)
        for (int j = 0; j < cols; ++j)
            out[i] += m_projection[i][j] * embedding[j];
    // Apply ReLU on projection output.
    for (double& v : out) v = max(0.0, v);
    return out;
}

// ---------------------------------------------------------------------------
// Private: get-or-create embedding
// ---------------------------------------------------------------------------

vector<double>& HyperGraphNeuralNet::ensureEmbedding(const string& name)
{
    auto it = m_embeddings.find(name);
    if (it == m_embeddings.end()) {
        m_embeddings[name] = randomVec(HGNN_FEAT_DIM, 0.1);
        it = m_embeddings.find(name);
    }
    return it->second;
}

// ---------------------------------------------------------------------------
// Private: numeric helpers
// ---------------------------------------------------------------------------

vector<double> HyperGraphNeuralNet::randomVec(int dim, double scale)
{
    vector<double> v(dim);
    for (double& x : v)
        x = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
    return v;
}

vector<double> HyperGraphNeuralNet::meanPool(
    const vector<vector<double>>& vecs)
{
    if (vecs.empty()) return vector<double>(HGNN_FEAT_DIM, 0.0);
    int dim = (int)vecs[0].size();
    vector<double> out(dim, 0.0);
    for (const auto& v : vecs)
        for (int i = 0; i < (int)min((int)v.size(), dim); ++i)
            out[i] += v[i];
    double n = (double)vecs.size();
    for (double& v : out) v /= n;
    return out;
}

double HyperGraphNeuralNet::cosine(const vector<double>& a,
                                    const vector<double>& b)
{
    double ab = dot(a, b);
    double na = sqrt(dot(a, a));
    double nb = sqrt(dot(b, b));
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return ab / (na * nb);
}

vector<double> HyperGraphNeuralNet::relu(const vector<double>& x)
{
    vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = max(0.0, x[i]);
    return out;
}

double HyperGraphNeuralNet::dot(const vector<double>& a,
                                 const vector<double>& b)
{
    double s = 0.0;
    int n = min(a.size(), b.size());
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

vector<string> HyperGraphNeuralNet::tokenize(const string& text)
{
    vector<string> tokens;
    istringstream iss(text);
    string tok;
    while (iss >> tok) {
        string lower;
        for (char c : tok)
            if (isalpha((unsigned char)c))
                lower += tolower((unsigned char)c);
        if (!lower.empty()) tokens.push_back(lower);
    }
    return tokens;
}
