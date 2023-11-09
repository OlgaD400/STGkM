import unittest
import numpy as np
from stgkm.distance_functions import s_journey
from stgkm.STGKM import similarity_measure, similarity_matrix, STGKM


class Tests(unittest.TestCase):
    """
    Test class for tkm
    """

    def __init__(self, *args, **kwargs):
        """Initialize test class."""
        super(Tests, self).__init__(*args, **kwargs)

        self.two_cluster_connectivity_matrix = np.array(
            [
                [
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 1],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                ],
                [
                    [0, 1, 0, 0, 0, 0],
                    [1, 1, 1, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 1, 1],
                ],
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 0, 1, 1],
                    [0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 1, 1, 0],
                ],
                [
                    [0, 1, 1, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1, 0],
                ],
            ]
        )

    def test_temporal_graph_distance(self):
        """
        Test temporal graph distance
        """

        connectivity_matrix = np.array(
            [
                [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
                [[0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 1], [0, 1, 1, 0]],
                [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 0]],
                [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
            ]
        )

        timesteps, _, _ = connectivity_matrix.shape
        # Ensure test cases are symmetric
        for i in range(timesteps):
            assert np.all(
                connectivity_matrix[i, :, :] == connectivity_matrix[i, :, :].T
            )

        distance_matrix = s_journey(connectivity_matrix)
        assert np.all(
            distance_matrix
            == np.array(
                [
                    [[0, 1, 4, 2], [1, 0, 1, 2], [2, 1, 0, 1], [3, 3, 1, 0]],
                    [[0, 1, 2, 2], [1, 0, 3, 1], [2, 2, 0, 1], [3, 1, 1, 0]],
                    [
                        [0, np.inf, np.inf, 1],
                        [2, 0, 1, 1],
                        [2, 1, 0, np.inf],
                        [1, 1, 2, 0],
                    ],
                    [
                        [0, 1, 1, np.inf],
                        [1, 0, 1, np.inf],
                        [1, 1, 0, np.inf],
                        [np.inf, np.inf, np.inf, 0],
                    ],
                ]
            )
        )

        connectivity_matrix = np.array(
            [
                [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0]],
                [[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]],
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            ]
        )
        distance_matrix = s_journey(connectivity_matrix)
        assert np.all(
            distance_matrix
            == np.array(
                [
                    [[0, 1, 2, 2], [1, 0, 1, 1], [2, 1, 0, 1], [2, 1, 1, 0]],
                    [[0, 2, 1, 2], [np.inf, 0, 1, 1], [1, 1, 0, np.inf], [2, 1, 2, 0]],
                    [
                        [0, 1, np.inf, np.inf],
                        [1, 0, np.inf, np.inf],
                        [np.inf, np.inf, 0, 1],
                        [np.inf, np.inf, 1, 0],
                    ],
                ]
            )
        )

    def test_similarity_measure(self):
        """Test similarity measure."""
        x = np.arange(10)
        y = np.arange(10)[::-1]
        assert np.isclose(similarity_measure(vector_1=x, vector_2=y), 0)

        x = np.arange(10)
        y = np.arange(10)
        assert np.isclose(similarity_measure(vector_1=x, vector_2=y), 1)

        x = np.arange(10)
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        assert np.isclose(similarity_measure(vector_1=x, vector_2=y), 0.50)

        x = np.arange(10)
        y = np.arange(5)

        self.assertRaises(AssertionError, similarity_measure, vector_1=x, vector_2=y)

    def test_similarity_matrix(self):
        """Test similarity matrix."""
        weights = np.array([[1, 0, 1, 1, 1], [0, 1, 0, 0, 0], [1, 1, 0, 1, 1]])
        sim_mat = similarity_matrix(
            weights=weights, similarity_function=similarity_measure
        )
        target_mat = np.array([[1, 0, 3 / 5], [0, 1, 2 / 5], [3 / 5, 2 / 5, 1]])
        np.testing.assert_array_almost_equal(sim_mat, target_mat)

    def test_init_stgkm(self):
        """Test STGkM initialization."""
        distance_matrix = s_journey(self.two_cluster_connectivity_matrix)
        self.assertRaises(
            AssertionError,
            STGKM,
            distance_matrix=distance_matrix,
            penalty=5,
            max_drift=1,
            num_clusters=7,
            iterations=10,
            drift_time_window=1,
            tie_breaker=False,
        )

        self.assertRaises(
            AssertionError,
            STGKM,
            distance_matrix=distance_matrix,
            penalty=5,
            max_drift=0,
            num_clusters=7,
            iterations=10,
            drift_time_window=1,
            tie_breaker=False,
        )

        self.assertRaises(
            AssertionError,
            STGKM,
            distance_matrix=distance_matrix,
            penalty=5,
            max_drift=1,
            num_clusters=7,
            iterations=10,
            drift_time_window=0,
            tie_breaker=False,
        )

    def test_stgkm_penalize_distance(self):
        """Test STGkM distance penalization pre-processing step."""

        distance_matrix = s_journey(self.two_cluster_connectivity_matrix)
        stgkm = STGKM(
            distance_matrix=distance_matrix,
            penalty=5,
            max_drift=1,
            num_clusters=2,
            iterations=10,
            drift_time_window=1,
            tie_breaker=False,
        )
        target_matrix = np.array(
            [
                [
                    [0.0, 2.0, 1.0, 5, 5, 5],
                    [4.0, 0.0, 1.0, 1.0, 3.0, 2.0],
                    [1.0, 1.0, 0.0, 3.0, 2.0, 3.0],
                    [2.0, 1.0, 2.0, 0.0, 1.0, 1.0],
                    [5, 5, 5, 1.0, 0.0, 2.0],
                    [5, 5, 5, 1.0, 3.0, 0.0],
                ],
                [
                    [0.0, 1.0, 2.0, 5, 5, 5],
                    [1.0, 0.0, 1.0, 2.0, 1.0, 2.0],
                    [3.0, 1.0, 0.0, 5, 5, 5],
                    [5, 5, 5, 0.0, 2.0, 1.0],
                    [3.0, 1.0, 2.0, 2.0, 0.0, 1.0],
                    [3.0, 3.0, 2.0, 1.0, 1.0, 0.0],
                ],
                [
                    [0.0, 2.0, 2.0, 5, 5, 5],
                    [2.0, 0.0, 1.0, 5, 5, 5],
                    [2.0, 1.0, 0.0, 1.0, 5, 2.0],
                    [2.0, 2.0, 1.0, 0.0, 1.0, 1.0],
                    [5, 5, 5, 1.0, 0.0, 1.0],
                    [5, 5, 5, 1.0, 1.0, 0.0],
                ],
                [
                    [0.0, 1.0, 1.0, 5, 5, 5],
                    [1.0, 0.0, 1.0, 5, 5, 5],
                    [1.0, 1.0, 0.0, 5, 5, 5],
                    [5, 5, 5, 0.0, 5, 1.0],
                    [5, 5, 5, 5, 0.0, 1.0],
                    [5, 5, 5, 1.0, 1.0, 0.0],
                ],
            ]
        )

        penalized_matrix = stgkm.penalize_distance()

        assert np.all(np.isclose(penalized_matrix, target_matrix))

    def test_stgkm_assign_points(self):
        """Test point assignment"""
        distance_matrix = np.array(
            [
                [
                    [0, 3, 1, 2],
                    [3, 0, 2, 3],
                    [1, 2, 0, 3],
                    [2, 3, 3, 0],
                ],
                [
                    [0, 3, 1, 2],
                    [3, 0, 2, 3],
                    [1, 2, 0, 3],
                    [2, 3, 3, 0],
                ],
            ]
        )

        stgkm = STGKM(
            distance_matrix=distance_matrix,
            penalty=5,
            max_drift=1,
            num_clusters=2,
            iterations=10,
            drift_time_window=1,
            tie_breaker=False,
        )

        membership = stgkm.assign_points(
            distance_matrix=distance_matrix[0], centers=np.array([2, 3])
        )

        target_membership = np.array([[1, 1, 1, 0], [0, 0, 0, 1]])

        assert np.all(np.isclose(membership, target_membership))

    def test_stgkm_choose_centers(self):
        """Test STGkM center assignment."""
        distance_matrix = np.array(
            [
                [
                    [0, 3, 1, 2],
                    [3, 0, 2, 3],
                    [1, 2, 0, 3],
                    [2, 3, 3, 0],
                ],
                [
                    [0, 3, 1, 2],
                    [3, 0, 2, 3],
                    [1, 2, 0, 3],
                    [2, 3, 3, 0],
                ],
            ]
        )
        stgkm = STGKM(
            distance_matrix=distance_matrix,
            penalty=5,
            max_drift=1,
            num_clusters=2,
            iterations=10,
            drift_time_window=1,
            tie_breaker=False,
        )
        membership = stgkm.assign_points(
            distance_matrix=distance_matrix[0], centers=np.array([2, 3])
        )
        centers = stgkm.choose_centers(
            distance_matrix=distance_matrix[0],
            membership=membership,
            centers=np.array([2, 3]),
        )
        assert np.all(np.isclose(centers, np.array([2, 3])))

    def test_stgkm_calculate_intra_cluster_distance(self):
        """Test STGKM intra cluster distance calculation."""
        distance_matrix = np.array(
            [
                [
                    [0, 3, 1, 2],
                    [3, 0, 2, 3],
                    [1, 2, 0, 3],
                    [2, 3, 3, 0],
                ],
                [
                    [0, 3, 1, 2],
                    [3, 0, 2, 3],
                    [1, 2, 0, 3],
                    [2, 3, 3, 0],
                ],
            ]
        )
        stgkm = STGKM(
            distance_matrix=distance_matrix,
            penalty=5,
            max_drift=1,
            num_clusters=2,
            iterations=10,
            drift_time_window=1,
            tie_breaker=False,
        )
        membership = stgkm.assign_points(
            distance_matrix=distance_matrix[0], centers=np.array([2, 3])
        )
        centers = np.array([2, 3])

        intra_cluster_sum = stgkm.calculate_intra_cluster_distance(
            distance_matrix=distance_matrix[0], membership=membership, centers=centers
        )
        target_intra_cluster_sum = 3

        assert np.isclose(intra_cluster_sum, target_intra_cluster_sum)

    def test_stgkm_find_center_connections(self):
        """Test STGKM's search of potential centers."""
        distance_matrix = np.array(
            [
                [
                    [0, 3, 1, 2],
                    [3, 0, 2, 3],
                    [1, 2, 0, 3],
                    [2, 3, 3, 0],
                ],
                [
                    [0, 3, 1, 2],
                    [3, 0, 2, 3],
                    [1, 2, 0, 3],
                    [2, 3, 3, 0],
                ],
                [
                    [0, 3, 1, 2],
                    [3, 0, 2, 3],
                    [1, 2, 0, 3],
                    [2, 3, 3, 0],
                ],
                [
                    [0, 3, 1, 2],
                    [3, 0, 2, 3],
                    [1, 2, 0, 3],
                    [2, 3, 3, 0],
                ],
            ]
        )

        stgkm = STGKM(
            distance_matrix=distance_matrix,
            penalty=5,
            max_drift=2,
            num_clusters=2,
            iterations=10,
            drift_time_window=2,
            tie_breaker=False,
        )
        # For the previous two time steps, which points have been
        # within a max distance of 2 of the current centers
        center_connections = stgkm.find_center_connections(
            current_centers=np.array([2, 3]), distance_matrix=distance_matrix, time=2
        )
        target_center_connections = [np.array([0, 1, 2]), np.array([0, 3])]

        for index in range(2):
            assert np.all(
                np.isclose(center_connections[index], target_center_connections[index])
            )

    def test_stgkm_next_assignment(self):
        """Test stgkm assignment of points at current time step."""
        pass

    def run_tests(self):
        """Run all tests."""
        self.test_temporal_graph_distance()
        self.test_similarity_measure()
        self.test_similarity_matrix()
        self.test_init_stgkm()
        self.test_stgkm_penalize_distance()
        self.test_stgkm_assign_points()
        self.test_stgkm_choose_centers()
        self.test_stgkm_calculate_intra_cluster_distance()
        self.test_stgkm_find_center_connections()


tests = Tests()
tests.run_tests()
