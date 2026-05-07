#include "mlp_engine.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace mlp_engine;

// ---------------------------------------------------------------------------
// Construction / initialisation
// ---------------------------------------------------------------------------

MLPEngine::MLPEngine()
    : m_lr(0.01), m_updateCount(0)
{
    xavierInit(m_W1, m_b1, MLP_INPUT_DIM,  MLP_HIDDEN1);
    xavierInit(m_W2, m_b2, MLP_HIDDEN1,    MLP_HIDDEN2);
    xavierInit(m_W3, m_b3, MLP_HIDDEN2,    MLP_OUTPUT_DIM);

    m_h1.assign(MLP_HIDDEN1,    0.0);
    m_h2.assign(MLP_HIDDEN2,    0.0);
    m_out.assign(MLP_OUTPUT_DIM, 0.0);
    m_lastInput.assign(MLP_INPUT_DIM, 0.0);
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

vector<double> MLPEngine::forward(const vector<double>& input)
{
    // Pad / truncate input to MLP_INPUT_DIM.
    m_lastInput = input;
    m_lastInput.resize(MLP_INPUT_DIM, 0.0);

    m_h1  = tanh_vec(affine(m_W1, m_b1, m_lastInput));
    m_h2  = tanh_vec(affine(m_W2, m_b2, m_h1));
    m_out = softmax(affine(m_W3, m_b3, m_h2));
    return m_out;
}

// ---------------------------------------------------------------------------
// Backward pass (online SGD, cross-entropy + softmax loss)
// ---------------------------------------------------------------------------

void MLPEngine::backward(const vector<double>& input, int targetIndex,
                          double learningRate)
{
    if (targetIndex < 0 || targetIndex >= MLP_OUTPUT_DIM) return;

    // Re-run forward to refresh cached activations if input changed.
    if (input != m_lastInput)
        forward(input);

    // ---- Output layer gradient (softmax + cross-entropy): dL/dz3 = out - y ----
    vector<double> dz3(MLP_OUTPUT_DIM, 0.0);
    for (int i = 0; i < MLP_OUTPUT_DIM; ++i)
        dz3[i] = m_out[i] - (i == targetIndex ? 1.0 : 0.0);

    // ---- Gradient of W3, b3 ----
    for (int i = 0; i < MLP_OUTPUT_DIM; ++i) {
        m_b3[i] -= learningRate * dz3[i];
        for (int j = 0; j < MLP_HIDDEN2; ++j)
            m_W3[i][j] -= learningRate * dz3[i] * m_h2[j];
    }

    // ---- Backprop through hidden layer 2 ----
    // dh2 = W3^T * dz3 ∘ dtanh(h2)
    vector<double> dh2(MLP_HIDDEN2, 0.0);
    for (int j = 0; j < MLP_HIDDEN2; ++j) {
        double sum = 0.0;
        for (int i = 0; i < MLP_OUTPUT_DIM; ++i)
            sum += m_W3[i][j] * dz3[i];
        dh2[j] = sum * (1.0 - m_h2[j] * m_h2[j]); // dtanh
    }

    for (int i = 0; i < MLP_HIDDEN2; ++i) {
        m_b2[i] -= learningRate * dh2[i];
        for (int j = 0; j < MLP_HIDDEN1; ++j)
            m_W2[i][j] -= learningRate * dh2[i] * m_h1[j];
    }

    // ---- Backprop through hidden layer 1 ----
    vector<double> dh1(MLP_HIDDEN1, 0.0);
    for (int j = 0; j < MLP_HIDDEN1; ++j) {
        double sum = 0.0;
        for (int i = 0; i < MLP_HIDDEN2; ++i)
            sum += m_W2[i][j] * dh2[i];
        dh1[j] = sum * (1.0 - m_h1[j] * m_h1[j]);
    }

    for (int i = 0; i < MLP_HIDDEN1; ++i) {
        m_b1[i] -= learningRate * dh1[i];
        for (int j = 0; j < MLP_INPUT_DIM; ++j)
            m_W1[i][j] -= learningRate * dh1[i] * m_lastInput[j];
    }

    m_updateCount++;
}

// ---------------------------------------------------------------------------
// Feature encoding
// ---------------------------------------------------------------------------

vector<double> MLPEngine::encodeFeatures(
    double symbolicScore,    double subSymbolicScore,
    double hgnnScore,        double dtesnnScore,
    const vector<double>& hgnnFeatures,
    const vector<double>& dtesnnFeatures)
{
    vector<double> feat(MLP_INPUT_DIM, 0.0);

    // [0..3] — path scores clamped to [0,1].
    auto clamp01 = [](double v) { return max(0.0, min(1.0, v)); };
    feat[PATH_SYMBOLIC]    = clamp01(symbolicScore);
    feat[PATH_SUBSYMBOLIC] = clamp01(subSymbolicScore);
    feat[PATH_HGNN]        = clamp01(hgnnScore);
    feat[PATH_DTESNN]      = clamp01(dtesnnScore);

    // [4..11] — HGNN 8-D features (zero-pad / truncate).
    for (int i = 0; i < 8; ++i)
        feat[4 + i] = (i < (int)hgnnFeatures.size()) ? hgnnFeatures[i] : 0.0;

    // [12..19] — DTESNN 8-D features.
    for (int i = 0; i < 8; ++i)
        feat[12 + i] = (i < (int)dtesnnFeatures.size()) ? dtesnnFeatures[i] : 0.0;

    return feat;
}

// ---------------------------------------------------------------------------
// Learning rate decay
// ---------------------------------------------------------------------------

void MLPEngine::decayLearningRate(double factor)
{
    m_lr = max(1e-5, m_lr * factor);
}

// ---------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------

void MLPEngine::saveWeights(const string& filename) const
{
    ofstream f(filename);
    if (!f.is_open()) {
        cerr << "[MLPEngine] Cannot write weights to " << filename << endl;
        return;
    }

    auto writeMatrix = [&](const vector<vector<double>>& W,
                           const vector<double>& b) {
        f << W.size() << " " << (W.empty() ? 0 : W[0].size()) << "\n";
        for (const auto& row : W) {
            for (double v : row) f << v << " ";
            f << "\n";
        }
        for (double v : b) f << v << " ";
        f << "\n";
    };

    writeMatrix(m_W1, m_b1);
    writeMatrix(m_W2, m_b2);
    writeMatrix(m_W3, m_b3);
    f << m_lr << "\n";
}

bool MLPEngine::loadWeights(const string& filename)
{
    ifstream f(filename);
    if (!f.is_open()) return false;

    auto readMatrix = [&](vector<vector<double>>& W, vector<double>& b) {
        int rows, cols;
        f >> rows >> cols;
        W.assign(rows, vector<double>(cols));
        for (auto& row : W)
            for (double& v : row) f >> v;
        b.resize(rows);
        for (double& v : b) f >> v;
    };

    readMatrix(m_W1, m_b1);
    readMatrix(m_W2, m_b2);
    readMatrix(m_W3, m_b3);
    f >> m_lr;
    return true;
}

// ---------------------------------------------------------------------------
// Private: activation functions
// ---------------------------------------------------------------------------

vector<double> MLPEngine::tanh_vec(const vector<double>& x)
{
    vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = tanh(x[i]);
    return out;
}

vector<double> MLPEngine::dtanh_vec(const vector<double>& activated)
{
    vector<double> out(activated.size());
    for (size_t i = 0; i < activated.size(); ++i)
        out[i] = 1.0 - activated[i] * activated[i];
    return out;
}

vector<double> MLPEngine::softmax(const vector<double>& x)
{
    double maxVal = *max_element(x.begin(), x.end());
    vector<double> out(x.size());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = exp(x[i] - maxVal);
        sum   += out[i];
    }
    if (sum > 0) for (auto& v : out) v /= sum;
    return out;
}

vector<double> MLPEngine::affine(const vector<vector<double>>& W,
                                  const vector<double>& b,
                                  const vector<double>& x)
{
    int rows = (int)W.size();
    int cols = (int)(rows > 0 ? W[0].size() : 0);
    vector<double> out(rows, 0.0);
    for (int i = 0; i < rows; ++i) {
        out[i] = b[i];
        int lim = min(cols, (int)x.size());
        for (int j = 0; j < lim; ++j)
            out[i] += W[i][j] * x[j];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Private: Xavier initialisation
// ---------------------------------------------------------------------------

void MLPEngine::xavierInit(vector<vector<double>>& W,
                            vector<double>& b,
                            int fanIn, int fanOut)
{
    double scale = sqrt(2.0 / (fanIn + fanOut));
    W.assign(fanOut, vector<double>(fanIn));
    for (auto& row : W)
        for (double& v : row)
            v = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
    b.assign(fanOut, 0.0);
}
