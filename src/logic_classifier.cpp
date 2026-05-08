#include "logic_classifier.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <cctype>

using namespace logic_classifier;
using namespace logic_meta_patterns;

LogicClassifier::LogicClassifier()
    : m_updateCount(0)
{
    auto initMatrix = [](int rows, int cols) {
        vector<vector<double>> w(rows, vector<double>(cols, 0.0));
        double scale = sqrt(2.0 / (rows + cols));
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                w[i][j] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * scale;
        return w;
    };

    m_W1 = initMatrix(HIDDEN_DIM, INPUT_DIM);
    m_W2 = initMatrix(OUTPUT_DIM, HIDDEN_DIM);
    m_b1.assign(HIDDEN_DIM, 0.0);
    m_b2.assign(OUTPUT_DIM, 0.0);

    m_lastInput.assign(INPUT_DIM, 0.0);
    m_lastH.assign(HIDDEN_DIM, 0.0);
    m_lastOut.assign(OUTPUT_DIM, 0.0);
}

vector<string> LogicClassifier::tokenize(const string& text) {
    vector<string> out;
    istringstream iss(text);
    string tok;
    while (iss >> tok) {
        string w;
        for (char c : tok) {
            if (isalpha((unsigned char)c) || c == '_')
                w += (char)tolower((unsigned char)c);
        }
        if (!w.empty()) out.push_back(w);
    }
    return out;
}

vector<double> LogicClassifier::tanhVec(const vector<double>& x) {
    vector<double> y(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) y[i] = tanh(x[i]);
    return y;
}

vector<double> LogicClassifier::softmax(const vector<double>& x) {
    double m = *max_element(x.begin(), x.end());
    vector<double> y(x.size(), 0.0);
    double s = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = exp(x[i] - m);
        s += y[i];
    }
    if (s > 0.0) {
        for (double& v : y) v /= s;
    }
    return y;
}

vector<double> LogicClassifier::buildFeatures(
    const string& input,
    const map<string, double>& contextVector,
    hgnn::HyperGraphNeuralNet* hgnnNet,
    dtesnn::DeepTreeEchoStateNet* dtesnnNet) const
{
    vector<double> feat(INPUT_DIM, 0.0);
    auto tokens = tokenize(input);

    map<LogicSystem, vector<string>> keywords;
    keywords[PL] = {"cnf", "dnf", "proposition", "resolution", "implies", "iff"};
    keywords[FOL] = {"forall", "exists", "unify", "predicate", "herbrand", "robinson"};
    keywords[MODAL] = {"modal", "world", "necessarily", "possibly", "kripke", "access"};
    keywords[LINEAR] = {"linear", "resource", "tensor", "bang", "consume", "sequent"};
    keywords[DTT] = {"type", "pi", "sigma", "identity", "dependent", "context"};
    keywords[CAT_THEORY] = {"functor", "morphism", "adjunction", "monad", "natural", "category"};

    vector<LogicSystem> systems = {PL, FOL, MODAL, LINEAR, DTT, CAT_THEORY};
    for (size_t i = 0; i < systems.size(); ++i) {
        LogicSystem s = systems[i];
        int rawHits = 0;
        double sal = 0.0;
        for (const auto& kw : keywords[s]) {
            if (find(tokens.begin(), tokens.end(), kw) != tokens.end()) rawHits++;
            auto it = contextVector.find(kw);
            if (it != contextVector.end()) sal += it->second;
        }
        feat[i * 2] = min(1.0, rawHits / 6.0);
        feat[i * 2 + 1] = min(1.0, sal / 6.0);
    }

    // HGNN 8-D [12..19]
    if (hgnnNet) {
        auto emb = hgnnNet->aggregateEmbeddings(tokens);
        for (int i = 0; i < 8; ++i)
            feat[12 + i] = (i < (int)emb.size()) ? emb[i] : 0.0;
    }

    // DTESNN 8-D [20..27]
    if (dtesnnNet) {
        auto tmp = dtesnnNet->getReadout();
        for (int i = 0; i < 8; ++i)
            feat[20 + i] = (i < (int)tmp.size()) ? tmp[i] : 0.0;
    }

    return feat;
}

vector<double> LogicClassifier::forward(const vector<double>& input) {
    m_lastInput = input;

    vector<double> h(HIDDEN_DIM, 0.0);
    for (int i = 0; i < HIDDEN_DIM; ++i) {
        double v = m_b1[i];
        for (int j = 0; j < INPUT_DIM; ++j)
            v += m_W1[i][j] * m_lastInput[j];
        h[i] = v;
    }

    m_lastH = tanhVec(h);

    vector<double> out(OUTPUT_DIM, 0.0);
    for (int i = 0; i < OUTPUT_DIM; ++i) {
        double v = m_b2[i];
        for (int j = 0; j < HIDDEN_DIM; ++j)
            v += m_W2[i][j] * m_lastH[j];
        out[i] = v;
    }

    m_lastOut = softmax(out);
    return m_lastOut;
}

void LogicClassifier::backward(const vector<double>& input, int targetIndex, double learningRate) {
    if (targetIndex < 0 || targetIndex >= OUTPUT_DIM) return;
    auto out = forward(input);

    vector<double> dz2(OUTPUT_DIM, 0.0);
    for (int i = 0; i < OUTPUT_DIM; ++i)
        dz2[i] = out[i] - (i == targetIndex ? 1.0 : 0.0);

    for (int i = 0; i < OUTPUT_DIM; ++i) {
        m_b2[i] -= learningRate * dz2[i];
        for (int j = 0; j < HIDDEN_DIM; ++j)
            m_W2[i][j] -= learningRate * dz2[i] * m_lastH[j];
    }

    vector<double> dh(HIDDEN_DIM, 0.0);
    for (int j = 0; j < HIDDEN_DIM; ++j) {
        double s = 0.0;
        for (int i = 0; i < OUTPUT_DIM; ++i)
            s += m_W2[i][j] * dz2[i];
        dh[j] = s * (1.0 - m_lastH[j] * m_lastH[j]);
    }

    for (int i = 0; i < HIDDEN_DIM; ++i) {
        m_b1[i] -= learningRate * dh[i];
        for (int j = 0; j < INPUT_DIM; ++j)
            m_W1[i][j] -= learningRate * dh[i] * m_lastInput[j];
    }

    m_updateCount++;
}

int LogicClassifier::systemToIndex(LogicSystem s) {
    switch (s) {
        case LOGIC_NONE: return 0;
        case PL: return 1;
        case FOL: return 2;
        case MODAL: return 3;
        case LINEAR: return 4;
        case DTT: return 5;
        case CAT_THEORY: return 6;
        default: return 0;
    }
}

LogicSystem LogicClassifier::indexToSystem(int idx) {
    switch (idx) {
        case 1: return PL;
        case 2: return FOL;
        case 3: return MODAL;
        case 4: return LINEAR;
        case 5: return DTT;
        case 6: return CAT_THEORY;
        default: return LOGIC_NONE;
    }
}

Classification LogicClassifier::classify(const string& input,
                                         const map<string, double>& contextVector,
                                         hgnn::HyperGraphNeuralNet* hgnnNet,
                                         dtesnn::DeepTreeEchoStateNet* dtesnnNet)
{
    Classification c;
    auto feat = buildFeatures(input, contextVector, hgnnNet, dtesnnNet);
    auto probs = forward(feat);

    int bestIdx = 0;
    double bestP = 0.0;
    for (int i = 0; i < (int)probs.size(); ++i) {
        if (probs[i] > bestP) {
            bestP = probs[i];
            bestIdx = i;
        }
    }

    c.system = indexToSystem(bestIdx);
    c.confidence = bestP;
    c.probabilities = probs;
    c.isLogic = (c.system != LOGIC_NONE);
    m_lastClassification = c;
    return c;
}

bool LogicClassifier::parseOverride(const string& input,
                                    LogicSystem& system,
                                    string& remainingInput) const
{
    string s = input;
    string prefix = "USE ";
    if (s.size() < prefix.size()) return false;

    string head = s.substr(0, prefix.size());
    transform(head.begin(), head.end(), head.begin(), ::toupper);
    if (head != prefix) return false;

    size_t colon = s.find(':');
    if (colon == string::npos) return false;

    string systemText = s.substr(prefix.size(), colon - prefix.size());
    while (!systemText.empty() && isspace((unsigned char)systemText.back())) systemText.pop_back();
    while (!systemText.empty() && isspace((unsigned char)systemText.front())) systemText.erase(systemText.begin());

    system = logicSystemFromString(systemText);
    if (system == LOGIC_NONE) return false;

    remainingInput = s.substr(colon + 1);
    while (!remainingInput.empty() && isspace((unsigned char)remainingInput.front())) remainingInput.erase(remainingInput.begin());
    return true;
}

void LogicClassifier::reinforce(const string& input,
                                const map<string, double>& contextVector,
                                hgnn::HyperGraphNeuralNet* hgnnNet,
                                dtesnn::DeepTreeEchoStateNet* dtesnnNet,
                                LogicSystem target,
                                double learningRate)
{
    auto feat = buildFeatures(input, contextVector, hgnnNet, dtesnnNet);
    backward(feat, systemToIndex(target), learningRate);
}
