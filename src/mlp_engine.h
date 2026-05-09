#ifndef __MLP_ENGINE_H__
#define __MLP_ENGINE_H__

/**
 * mlp_engine.h — Emergent Multi-Layer Perceptron (Phase 5)
 *
 * An online-learning MLP that integrates features from four response paths:
 *   - PatternLattice symbolic scoring      (score only)
 *   - AtomSpace sub-symbolic scoring       (score only)
 *   - HGNN spatial embedding path          (score + 8-D feature vector)
 *   - DTESNN temporal reservoir path       (score + 8-D feature vector)
 *
 * Architecture:  MLP_INPUT_DIM(21) → H1(16) → H2(8) → OUTPUT(5)
 *   Input layout:
 *     [0..4]   normalised path scores: symbolic, subsymbolic, hgnn, dtesnn, workflow
 *     [5..12]  HGNN aggregated embedding  (8-D)
 *     [13..20] DTESNN readout features    (8-D)
 *
 * Output: softmax over 5 blend weights (one per response path).
 *
 * Weight updates use online SGD with a cross-entropy loss against a
 * one-hot target built from whichever path produced the accepted response.
 * The outer learning loop decays the learning rate to stabilise training.
 */

#include <vector>
#include <string>

using namespace std;

namespace mlp_engine {

    static const int MLP_INPUT_DIM  = 21;
    static const int MLP_HIDDEN1    = 16;
    static const int MLP_HIDDEN2    = 8;
    static const int MLP_OUTPUT_DIM = 5;

    // Path indices into blend-weight vector.
    static const int PATH_SYMBOLIC    = 0;
    static const int PATH_SUBSYMBOLIC = 1;
    static const int PATH_HGNN        = 2;
    static const int PATH_DTESNN      = 3;
    static const int PATH_WORKFLOW    = 4;

    class MLPEngine {
    public:
        MLPEngine();
        ~MLPEngine() = default;

        // Forward pass — returns softmax blend weights (sum to 1).
        vector<double> forward(const vector<double>& input);

        // Online SGD backward pass with cross-entropy loss.
        // targetIndex: one of PATH_* constants above.
        void backward(const vector<double>& input,
                      int targetIndex,
                      double learningRate = 0.01);

        // Convenience: encode a 20-D feature vector from the four path scores
        // and the two 8-D feature vectors.  Scores are clamped to [0,1] before
        // insertion; feature vectors are zero-padded / truncated to 8 elements.
        static vector<double> encodeFeatures(
            double symbolicScore,    double subSymbolicScore,
            double hgnnScore,        double dtesnnScore,
            double workflowScore,
            const vector<double>& hgnnFeatures,
            const vector<double>& dtesnnFeatures);

        // Serialise / deserialise weights to a plain-text file.
        void saveWeights(const string& filename) const;
        bool loadWeights(const string& filename);

        // Outer-loop learning-rate decay.
        void decayLearningRate(double factor = 0.95);
        double getLearningRate() const { return m_lr; }

        // Number of backward passes executed (for diagnostics).
        int getUpdateCount() const { return m_updateCount; }

    private:
        // Weight matrices and bias vectors.
        vector<vector<double>> m_W1;  // [H1 × INPUT]
        vector<double>         m_b1;  // [H1]
        vector<vector<double>> m_W2;  // [H2 × H1]
        vector<double>         m_b2;  // [H2]
        vector<vector<double>> m_W3;  // [OUTPUT × H2]
        vector<double>         m_b3;  // [OUTPUT]

        // Cached activations from the last forward pass (used in backward).
        vector<double> m_h1;
        vector<double> m_h2;
        vector<double> m_out;
        vector<double> m_lastInput;

        double m_lr;
        int    m_updateCount;

        // Activation helpers.
        static vector<double> tanh_vec(const vector<double>& x);
        static vector<double> dtanh_vec(const vector<double>& activated); // d/dx tanh from activated value
        static vector<double> softmax(const vector<double>& x);

        // Affine transform: W * x + b.
        static vector<double> affine(const vector<vector<double>>& W,
                                     const vector<double>& b,
                                     const vector<double>& x);

        // Xavier/Glorot weight initialisation.
        static void xavierInit(vector<vector<double>>& W,
                               vector<double>& b,
                               int fanIn, int fanOut);
    };

} // namespace mlp_engine

#endif // __MLP_ENGINE_H__
