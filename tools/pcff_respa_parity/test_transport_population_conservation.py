"""Regression: a large neutral cluster must not disappear from population statistics."""
import unittest
from types import SimpleNamespace
import numpy as np
import analyze_polygen_transport as a


class PopulationConservationTest(unittest.TestCase):
    def setUp(self):
        self.top = SimpleNamespace(
            cat_mols=np.arange(1, 13), anion_mols=np.arange(13, 25),
            cat_atom_sel=np.arange(12), anion_ref_atom_sel=np.arange(12, 24),
            anion_trace_sel_by_mol={m: np.array([m - 1]) for m in range(13, 25)},
        )
        # One giant neutral cluster, then all 24 ions separated.
        x = np.zeros((2, 24, 3))
        x[1, :, 0] = np.arange(24)
        self.cache = {'wrapped_nm': x, 'box_nm': np.full((2, 3), 100.)}

    def test_auto_preserves_large_neutral_cluster_and_free_ions(self):
        p, n = a.htp_atom_population_matrix(self.cache, self.top, None, .34, 1)
        self.assertEqual(n, 2)
        self.assertEqual(p.shape, (13, 13))
        self.assertEqual(p[12, 12], .5)
        self.assertEqual(p[1, 0], 6.)
        self.assertEqual(p[0, 1], 6.)
        i, j = np.indices(p.shape)
        np.testing.assert_allclose([(p*i).sum(), (p*j).sum()], [12, 12])
        explicit, _ = a.htp_atom_population_matrix(self.cache, self.top, 13, .34, 1)
        np.testing.assert_array_equal(p, explicit)

    def test_exclusive_boundary_rejects_silent_loss(self):
        for size in [10, 12]:
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'refusing to drop ions'):
                a.htp_atom_population_matrix(self.cache, self.top, size, .34, 1)

    def test_small_matrix_allowed_when_all_observed_clusters_fit(self):
        cache = {k: v[1:] for k, v in self.cache.items()}
        p, n = a.htp_atom_population_matrix(cache, self.top, 2, .34, 1)
        self.assertEqual((n, p[1, 0], p[0, 1]), (1, 12., 12.))

    def test_empty_or_invalid_sampling_is_not_zero_conductivity(self):
        with self.assertRaisesRegex(ValueError, 'No frames'):
            a.htp_atom_population_matrix({k: v[:0] for k, v in self.cache.items()}, self.top, None, .34, 1)
        with self.assertRaisesRegex(ValueError, 'stride'):
            a.htp_atom_population_matrix(self.cache, self.top, None, .34, 0)


class CacheTopologyTest(unittest.TestCase):
    def test_stale_atom_selection_cannot_be_reused(self):
        top = SimpleNamespace(ion_atom_ids=np.array([2, 3]), ion_indices0=np.array([1, 2]),
            mol_ids=np.array([1, 2, 3]), masses=np.array([12., 7., 14.]),
            charges=np.array([0., 1., -1.]), cat_mols=np.array([2]), anion_mols=np.array([3]))
        cache = dict(ion_atom_ids=top.ion_atom_ids.copy(), ion_mol_ids=np.array([2, 3]),
            ion_masses=np.array([7., 14.]), ion_charges=np.array([1., -1.]),
            cat_mols=np.array([2]), anion_mols=np.array([3]))
        a.validate_cache_topology(cache, top)
        cache['ion_atom_ids'] = np.array([1, 2])
        with self.assertRaisesRegex(ValueError, 'force-cache'):
            a.validate_cache_topology(cache, top)


if __name__ == '__main__':
    unittest.main()
