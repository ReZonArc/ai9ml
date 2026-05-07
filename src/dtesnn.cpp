#include "dtesnn.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cctype>
#include <functional>

using namespace dtesnn;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

DeepTreeEchoStateNet::DeepTreeEchoStateNet()
    : m_stepCount(0)
{
    // Leak rates: fast → medium → slow (tree leaf → root).
    double leakRates[DTESNN_TREE_DEPTH]     = {0.90, 0.50, 0.10};
    double spectralRadii[DTESNN_TREE_DEPTH] = {0.95, 0.90, 0.85};

    m_levels.resize(DTESNN_TREE_DEPTH);
    for (int k = 0; k < DTESNN_TREE_DEPTH; ++k) {
        // Level 0 receives the external input; higher levels receive the
        // previous level's state, so their input dimension = DTESNN_RESERVOIR.
        int inputDim = (k == 0) ? DTESNN_INPUT_DIM : DTESNN_RESERVOIR;
        initLevel(m_levels[k], inputDim, leakRates[k], spectralRadii[k]);
    }

    // Initialise readout projection:
    // DTESNN_OUTPUT_DIM × (TREE_DEPTH * RESERVOIR)
    int fullStateDim = DTESNN_TREE_DEPTH * DTESNN_RESERVOIR;
    double scale = 1.0 / sqrt((double)fullStateDim);
    m_Wout = randMatrix(DTESNN_OUTPUT_DIM, fullStateDim, scale);
}

// ---------------------------------------------------------------------------
// Reservoir step
// ---------------------------------------------------------------------------

void DeepTreeEchoStateNet::step(const vector<double>& inputFeatures)
{
    // Pad / truncate to DTESNN_INPUT_DIM.
    vector<double> u = inputFeatures;
    u.resize(DTESNN_INPUT_DIM, 0.0);

    // Feed through levels: each level receives the previous level's new state.
    for (int k = 0; k < DTESNN_TREE_DEPTH; ++k) {
        vector<double> levelInput = (k == 0) ? u : m_levels[k - 1].state;
        m_levels[k].state = updateLevel(m_levels[k], levelInput);
    }

    m_stepCount++;
}

// ---------------------------------------------------------------------------
// Readout
// ---------------------------------------------------------------------------

vector<double> DeepTreeEchoStateNet::getReadout() const
{
    auto fullState = getFullState();
    vector<double> out(DTESNN_OUTPUT_DIM, 0.0);
    int cols = min((int)fullState.size(), DTESNN_TREE_DEPTH * DTESNN_RESERVOIR);
    for (int i = 0; i < DTESNN_OUTPUT_DIM; ++i)
        for (int j = 0; j < cols; ++j)
            out[i] += m_Wout[i][j] * fullState[j];
    // Tanh squash to [-1, 1], then remap to [0, 1] for scoring.
    for (double& v : out) v = 0.5 * (1.0 + tanh(v));
    return out;
}

double DeepTreeEchoStateNet::scoreResponse(
    const string& response,
    const map<string, double>& contextVector) const
{
    if (m_stepCount == 0) return 0.0;

    // Encode the response text using the same hashed-binning as context vectors.
    // First build a pseudo context-vector from the response tokens.
    map<string, double> respCtx;
    for (const auto& tok : tokenize(response))
        respCtx[tok] += 1.0;

    // Encode both.
    vector<double> reservoirReadout = getReadout();     // DTESNN_OUTPUT_DIM
    vector<double> respEncoded = encodeContextVector(respCtx);

    // Project respEncoded to DTESNN_OUTPUT_DIM via dot products with readout.
    // Score = cosine(readout, encoded_response[:OUTPUT_DIM]).
    respEncoded.resize(DTESNN_OUTPUT_DIM, 0.0);

    double dot_rr = dot(reservoirReadout, respEncoded);
    double nr = sqrt(dot(reservoirReadout, reservoirReadout));
    double ne = sqrt(dot(respEncoded,     respEncoded));
    if (nr < 1e-12 || ne < 1e-12) return 0.0;

    // Map cosine [-1,1] to score [0,1].
    return 0.5 * (1.0 + dot_rr / (nr * ne));
}

// ---------------------------------------------------------------------------
// Context-vector encoding
// ---------------------------------------------------------------------------

vector<double> DeepTreeEchoStateNet::encodeContextVector(
    const map<string, double>& contextVector)
{
    vector<double> feat(DTESNN_INPUT_DIM, 0.0);

    for (const auto& kv : contextVector) {
        // Hash the concept name into a bin.
        size_t h = hash<string>{}(kv.first) % DTESNN_INPUT_DIM;
        feat[h] += kv.second;
    }

    // L2 normalise.
    double norm = 0.0;
    for (double v : feat) norm += v * v;
    if (norm > 1e-12) {
        norm = sqrt(norm);
        for (double& v : feat) v /= norm;
    }
    return feat;
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

void DeepTreeEchoStateNet::resetState()
{
    for (auto& level : m_levels)
        fill(level.state.begin(), level.state.end(), 0.0);
    m_stepCount = 0;
}

// ---------------------------------------------------------------------------
// Temporal salience
// ---------------------------------------------------------------------------

double DeepTreeEchoStateNet::temporalSalience(int turnsAgo) const
{
    static const double DECAY = 0.80;
    if (turnsAgo < 0) return 0.0;
    return pow(DECAY, (double)turnsAgo);
}

// ---------------------------------------------------------------------------
// Private: full reservoir state
// ---------------------------------------------------------------------------

vector<double> DeepTreeEchoStateNet::getFullState() const
{
    vector<double> full;
    full.reserve(DTESNN_TREE_DEPTH * DTESNN_RESERVOIR);
    for (const auto& level : m_levels)
        full.insert(full.end(), level.state.begin(), level.state.end());
    return full;
}

// ---------------------------------------------------------------------------
// Private: reservoir initialisation
// ---------------------------------------------------------------------------

void DeepTreeEchoStateNet::initLevel(ReservoirLevel& level,
                                      int inputDim,
                                      double leakRate,
                                      double targetSpectralRadius) const
{
    level.reservoirSize   = DTESNN_RESERVOIR;
    level.leakRate        = leakRate;
    level.spectralRadius  = targetSpectralRadius;

    // Random reservoir matrix scaled to target spectral radius.
    auto W = randMatrix(DTESNN_RESERVOIR, DTESNN_RESERVOIR, 1.0);
    level.W = scaleToSpectralRadius(W, targetSpectralRadius);

    // Random input matrix (dense).
    level.Win = randMatrix(DTESNN_RESERVOIR, inputDim, 0.5);

    // Zero initial state.
    level.state.assign(DTESNN_RESERVOIR, 0.0);
}

// ---------------------------------------------------------------------------
// Private: reservoir update
// ---------------------------------------------------------------------------

vector<double> DeepTreeEchoStateNet::updateLevel(
    ReservoirLevel& level, const vector<double>& input) const
{
    int n = level.reservoirSize;
    int m = (int)input.size();

    // Pre-activation: W * x(t-1) + Win * u(t)
    vector<double> pre(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            pre[i] += level.W[i][j] * level.state[j];
        for (int j = 0; j < min(m, (int)level.Win[i].size()); ++j)
            pre[i] += level.Win[i][j] * input[j];
    }
    auto tanh_pre = tanhVec(pre);

    // Leaky integration: x(t) = (1-α)*x(t-1) + α*tanh(pre)
    double alpha = level.leakRate;
    vector<double> newState(n);
    for (int i = 0; i < n; ++i)
        newState[i] = (1.0 - alpha) * level.state[i] + alpha * tanh_pre[i];

    return newState;
}

// ---------------------------------------------------------------------------
// Private: numeric helpers
// ---------------------------------------------------------------------------

double DeepTreeEchoStateNet::dot(const vector<double>& a,
                                  const vector<double>& b)
{
    double s = 0.0;
    int n = (int)min(a.size(), b.size());
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

vector<double> DeepTreeEchoStateNet::tanhVec(const vector<double>& x)
{
    vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = tanh(x[i]);
    return out;
}

vector<double> DeepTreeEchoStateNet::randVec(int dim, double scale)
{
    vector<double> v(dim);
    for (double& x : v)
        x = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
    return v;
}

vector<vector<double>> DeepTreeEchoStateNet::randMatrix(
    int rows, int cols, double scale)
{
    vector<vector<double>> M(rows, vector<double>(cols));
    for (auto& row : M)
        for (double& v : row)
            v = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
    return M;
}

double DeepTreeEchoStateNet::estimateSpectralRadius(
    const vector<vector<double>>& W)
{
    int n = (int)W.size();
    if (n == 0) return 0.0;

    // Power iteration (50 steps).
    vector<double> v = randVec(n, 1.0);
    double norm = sqrt(dot(v, v));
    if (norm < 1e-12) return 0.0;
    for (double& x : v) x /= norm;

    double rho = 0.0;
    for (int iter = 0; iter < 50; ++iter) {
        vector<double> Wv(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Wv[i] += W[i][j] * v[j];
        rho = sqrt(dot(Wv, Wv));
        if (rho < 1e-12) break;
        for (double& x : Wv) x /= rho;
        v = Wv;
    }
    return rho;
}

vector<vector<double>> DeepTreeEchoStateNet::scaleToSpectralRadius(
    vector<vector<double>> W, double targetRadius)
{
    double rho = estimateSpectralRadius(W);
    if (rho < 1e-12) return W;
    double factor = targetRadius / rho;
    for (auto& row : W)
        for (double& v : row)
            v *= factor;
    return W;
}

vector<string> DeepTreeEchoStateNet::tokenize(const string& text)
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
