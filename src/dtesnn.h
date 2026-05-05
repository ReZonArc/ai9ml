#ifndef __DTESNN_H__
#define __DTESNN_H__

/**
 * dtesnn.h — Temporally-Aware Deep-Tree Echo State Neural Network (Phase 5)
 *
 * A multi-level reservoir computing network where each level operates at a
 * different temporal resolution, forming a hierarchical "tree" of timescales:
 *
 *   Level 0 (leaves / fast):  leakRate ≈ 0.90 — immediate, token-level context
 *   Level 1 (branches / med): leakRate ≈ 0.50 — turn-level conversation patterns
 *   Level 2 (root / slow):    leakRate ≈ 0.10 — session-level long-range memory
 *
 * The "deep" aspect: each level's reservoir activation drives the input to
 * the next (coarser) level, creating a hierarchy of temporal abstractions.
 *
 * Update rule per level k:
 *   x_k(t) = (1 - α_k) * x_k(t-1)
 *            + α_k * tanh(W_k * x_k(t-1) + Win_k * u_k(t))
 * where u_0(t) = inputFeatures,  u_k(t) = x_{k-1}(t)  for k > 0.
 *
 * Complementary to HGNN (spatial): DTESNN captures *when* concepts appeared
 * and how conversation context evolved over time.
 *
 * The async dtesnn path in nsvd_respond() is read-only (uses cached state).
 * The reservoir step is called once per turn from Chatmachine::updateNSVDState.
 */

#include <vector>
#include <string>
#include <map>

using namespace std;

namespace dtesnn {

    static const int DTESNN_INPUT_DIM   = 16;  // matches HGNN_FEAT_DIM
    static const int DTESNN_RESERVOIR   = 32;  // reservoir neurons per level
    static const int DTESNN_TREE_DEPTH  = 3;   // number of hierarchy levels
    static const int DTESNN_OUTPUT_DIM  = 8;   // readout dimension fed to MLP

    struct ReservoirLevel {
        int    reservoirSize;
        double leakRate;
        double spectralRadius;
        vector<vector<double>> W;    // [reservoirSize × reservoirSize]
        vector<vector<double>> Win;  // [reservoirSize × inputDim]
        vector<double>         state;// [reservoirSize] current activation
    };

    class DeepTreeEchoStateNet {
    public:
        DeepTreeEchoStateNet();
        ~DeepTreeEchoStateNet() = default;

        // Feed one input step through the full tree reservoir.
        // Call once per conversation turn from Chatmachine::updateNSVDState.
        // inputFeatures must have DTESNN_INPUT_DIM elements (zero-padded if shorter).
        void step(const vector<double>& inputFeatures);

        // Read-only: get DTESNN_OUTPUT_DIM readout from current state.
        vector<double> getReadout() const;

        // Read-only: score a candidate response against current temporal state.
        double scoreResponse(const string& response,
                             const map<string, double>& contextVector) const;

        // Encode a context vector (concept→salience map) into DTESNN_INPUT_DIM
        // features via hashed binning + L2 normalisation.
        static vector<double> encodeContextVector(
            const map<string, double>& contextVector);

        // Reset all reservoir states (call on topic shift / new session).
        void resetState();

        // Geometric temporal-salience weight for an item k turns ago.
        // w(k) = decayBase^k,  decayBase ≈ 0.8.
        double temporalSalience(int turnsAgo) const;

        // Number of step() calls since last resetState().
        int getStepCount() const { return m_stepCount; }

    private:
        vector<ReservoirLevel> m_levels;
        int m_stepCount;

        // Fixed readout projection: DTESNN_OUTPUT_DIM × (TREE_DEPTH * RESERVOIR).
        vector<vector<double>> m_Wout;

        // Concatenation of all level states.
        vector<double> getFullState() const;

        // Initialise one reservoir level.
        void initLevel(ReservoirLevel& level,
                       int inputDim,
                       double leakRate,
                       double targetSpectralRadius) const;

        // Update one level: returns new state.
        vector<double> updateLevel(ReservoirLevel& level,
                                   const vector<double>& input) const;

        // Numeric helpers.
        static double         dot(const vector<double>& a,
                                  const vector<double>& b);
        static vector<double> tanhVec(const vector<double>& x);
        static vector<double> randVec(int dim, double scale);
        static vector<vector<double>> randMatrix(int rows, int cols, double scale);
        static vector<vector<double>> scaleToSpectralRadius(
            vector<vector<double>> W, double targetRadius);
        static double estimateSpectralRadius(const vector<vector<double>>& W);
        static vector<string> tokenize(const string& text);
    };

} // namespace dtesnn

#endif // __DTESNN_H__
