"""Tests for Viterbi sequence decoding."""

import numpy as np
import metalmom


class TestViterbi:
    """Tests for metalmom.viterbi()."""

    def test_basic_2state(self):
        """Clear observations should produce expected path."""
        # 5 frames, 2 states
        prob = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.85, 0.15],
            [0.1, 0.9],
            [0.15, 0.85],
        ], dtype=np.float32)

        path = metalmom.viterbi(prob)
        assert path.dtype == np.int64
        assert path.shape == (5,)
        # First 3 frames: state 0, last 2: state 1
        np.testing.assert_array_equal(path[:3], [0, 0, 0])
        np.testing.assert_array_equal(path[3:], [1, 1])

    def test_uniform_transition(self):
        """With uniform transition, path follows strongest observations."""
        n_frames = 8
        n_states = 3
        prob = np.full((n_frames, n_states), 0.1, dtype=np.float32)
        # State 2 is strong for all frames
        prob[:, 2] = 0.8

        path = metalmom.viterbi(prob)
        assert path.shape == (n_frames,)
        np.testing.assert_array_equal(path, np.full(n_frames, 2))

    def test_custom_transition(self):
        """Custom transition matrix should influence path."""
        prob = np.array([
            [0.6, 0.4],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.4, 0.6],
        ], dtype=np.float32)

        # Strong self-loop (0.99): path should stay in one state
        transition = np.array([[0.99, 0.01],
                                [0.01, 0.99]], dtype=np.float32)

        path = metalmom.viterbi(prob, transition=transition)
        assert path.shape == (4,)
        # With very strong self-loop, path should be constant
        assert len(np.unique(path)) == 1

    def test_custom_initial(self):
        """Initial distribution affects first frame state choice."""
        prob = np.array([
            [0.5, 0.5],
            [0.3, 0.7],
        ], dtype=np.float32)

        # Force start in state 1
        initial = np.array([0.01, 0.99], dtype=np.float32)
        path = metalmom.viterbi(prob, initial=initial)
        assert path[0] == 1

    def test_weather_hmm(self):
        """Classic Weather HMM: walk/shop/clean."""
        # Emissions: walk|Rain=0.1, shop|Rain=0.4, clean|Rain=0.5
        #            walk|Sun=0.6, shop|Sun=0.3, clean|Sun=0.1
        prob = np.array([
            [0.1, 0.6],  # walk
            [0.4, 0.3],  # shop
            [0.5, 0.1],  # clean
        ], dtype=np.float32)

        transition = np.array([
            [0.7, 0.3],
            [0.4, 0.6],
        ], dtype=np.float32)

        initial = np.array([0.6, 0.4], dtype=np.float32)

        path = metalmom.viterbi(prob, transition=transition, initial=initial)
        # Expected: Sun(1), Rain(0), Rain(0)
        np.testing.assert_array_equal(path, [1, 0, 0])

    def test_valid_state_indices(self):
        """All path values should be valid state indices."""
        n_frames = 20
        n_states = 5
        rng = np.random.default_rng(42)
        prob = rng.random((n_frames, n_states)).astype(np.float32)
        # Normalize rows
        prob /= prob.sum(axis=1, keepdims=True)

        path = metalmom.viterbi(prob)
        assert path.shape == (n_frames,)
        assert np.all(path >= 0)
        assert np.all(path < n_states)

    def test_single_frame(self):
        """Single frame should pick highest-probability state."""
        prob = np.array([[0.1, 0.3, 0.6]], dtype=np.float32)
        path = metalmom.viterbi(prob)
        assert path.shape == (1,)
        assert path[0] == 2

    def test_single_state(self):
        """Single state: path is always 0."""
        prob = np.array([[0.5], [0.3], [0.8]], dtype=np.float32)
        path = metalmom.viterbi(prob)
        np.testing.assert_array_equal(path, [0, 0, 0])

    def test_invalid_dimensions(self):
        """1D input should raise ValueError."""
        try:
            metalmom.viterbi(np.array([0.5, 0.5]))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestViterbiDiscriminative:
    """Tests for metalmom.viterbi_discriminative()."""

    def test_basic(self):
        """Discriminative Viterbi with clear posteriors."""
        prob = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ], dtype=np.float32)

        path = metalmom.viterbi_discriminative(prob)
        assert path.shape == (4,)
        assert path[0] == 0
        assert path[1] == 0
        assert path[2] == 1
        assert path[3] == 1

    def test_with_transition(self):
        """Custom transition should influence discriminative path."""
        prob = np.array([
            [0.6, 0.4],
            [0.4, 0.6],
            [0.6, 0.4],
        ], dtype=np.float32)

        # Very strong self-loop
        transition = np.array([[0.99, 0.01],
                                [0.01, 0.99]], dtype=np.float32)

        path = metalmom.viterbi_discriminative(prob, transition=transition)
        assert path.shape == (3,)
        # With such strong self-loop, path should stay constant
        assert len(np.unique(path)) == 1

    def test_uniform_transition_default(self):
        """Default transition is uniform (all zeros in log domain)."""
        prob = np.array([
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
        ], dtype=np.float32)

        path = metalmom.viterbi_discriminative(prob)
        assert path.shape == (3,)
        # Should follow observations
        assert path[0] == 1
        assert path[1] == 1
        assert path[2] == 0


class TestViterbiBinary:
    """Tests for metalmom.viterbi_binary()."""

    def test_basic(self):
        """Clear binary probabilities with moderate self-loop."""
        # Use weaker self-loop so observations dominate
        prob = np.array([0.05, 0.05, 0.95, 0.95, 0.05], dtype=np.float32)
        path = metalmom.viterbi_binary(prob, transition=0.6)
        assert path.shape == (5,)
        assert path.dtype == np.int64
        # Middle frames should be active (1)
        assert path[2] == 1
        assert path[3] == 1
        # First and last should be inactive (0)
        assert path[0] == 0
        assert path[4] == 0

    def test_all_active(self):
        """All-high probabilities should give all-active path."""
        prob = np.full(10, 0.95, dtype=np.float32)
        path = metalmom.viterbi_binary(prob)
        np.testing.assert_array_equal(path, np.ones(10, dtype=np.int64))

    def test_all_inactive(self):
        """All-low probabilities should give all-inactive path."""
        prob = np.full(10, 0.05, dtype=np.float32)
        path = metalmom.viterbi_binary(prob)
        np.testing.assert_array_equal(path, np.zeros(10, dtype=np.int64))

    def test_scalar_transition(self):
        """Scalar self-loop probability."""
        prob = np.array([0.9, 0.4, 0.4, 0.4, 0.9], dtype=np.float32)
        path = metalmom.viterbi_binary(prob, transition=0.95)
        assert path.shape == (5,)
        # With very strong self-loop, once started in a state, should stay
        assert len(np.unique(path)) <= 2

    def test_custom_2x2_transition(self):
        """Custom 2x2 transition matrix."""
        prob = np.array([0.9, 0.1, 0.1, 0.9], dtype=np.float32)
        trans = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
        path = metalmom.viterbi_binary(prob, transition=trans)
        assert path.shape == (4,)
        # Values should be 0 or 1
        assert np.all((path == 0) | (path == 1))

    def test_invalid_1d_required(self):
        """2D input should raise ValueError."""
        try:
            metalmom.viterbi_binary(np.array([[0.5, 0.5]]))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_empty_input(self):
        """Empty input returns empty array."""
        path = metalmom.viterbi_binary(np.array([], dtype=np.float32))
        assert path.shape == (0,)
