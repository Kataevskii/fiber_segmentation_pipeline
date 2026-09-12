import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.extract_individual_fibers import (
    extract_fibers_from_patch,
    remove_extracted_fibers_for_patch,
    get_available_patches,
    get_extracted_patch_counts,
    sync_morpher_library,
    normalize_patch_name
)
from core.dataset import ensure_individual_fibers_extracted
from core.fiber_morpher import RealFiberLibrary


class TestSelectiveFiberExtraction(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.curated_dir = os.path.join(self.tmp_dir.name, 'patches')
        self.indiv_dir = os.path.join(self.tmp_dir.name, 'individual_fibers')
        os.makedirs(self.curated_dir, exist_ok=True)
        os.makedirs(self.indiv_dir, exist_ok=True)

        # Create two dummy patches: patch_0001 (has 2 fibers) and patch_0002 (has 1 fiber)
        for p_idx, n_fibs in [(1, 2), (2, 1)]:
            prefix = os.path.join(self.curated_dir, f"patch_{p_idx:04d}")
            vol = np.zeros((32, 32, 32), dtype=bool)
            inst = np.zeros((32, 32, 32), dtype=np.uint16)
            intensity = np.zeros((32, 32, 32), dtype=np.float32)
            ori = np.zeros((3, 32, 32, 32), dtype=np.float32)
            skel = np.zeros((32, 32, 32), dtype=np.uint16)

            for fid in range(1, n_fibs + 1):
                # Put some voxels for this fiber
                coords_z = np.arange(5, 25)
                coords_y = np.full_like(coords_z, 10 * fid)
                coords_x = np.full_like(coords_z, 10 * fid)
                for z, y, x in zip(coords_z, coords_y, coords_x):
                    vol[z, y, x] = True
                    inst[z, y, x] = fid
                    intensity[z, y, x] = 1.0
                    ori[0, z, y, x] = 1.0
                    skel[z, y, x] = fid

            np.save(f"{prefix}_vol.npy", vol)
            np.save(f"{prefix}_instance.npy", inst)
            np.save(f"{prefix}_intensity.npy", intensity)
            np.save(f"{prefix}_ori.npy", ori)
            np.save(f"{prefix}_centerline.npy", skel)

            meta = {
                'patch_index': p_idx,
                'num_fibers': n_fibs,
                'cube_size': 32,
                'seeds': []
            }
            import json
            with open(f"{prefix}_meta.json", 'w') as f:
                json.dump(meta, f)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_normalize_patch_name(self):
        self.assertEqual(normalize_patch_name("1"), "patch_0001")
        self.assertEqual(normalize_patch_name("patch_0001"), "patch_0001")
        self.assertEqual(normalize_patch_name("patch_0001_vol.npy"), "patch_0001")
        self.assertEqual(normalize_patch_name("data/curated/patches/patch_0002_vol.npy"), "patch_0002")

    def test_ensure_individual_fibers_does_not_auto_extract_by_default(self):
        # By default auto_extract=False, so individual_fibers should remain empty!
        count = ensure_individual_fibers_extracted(real_data_dir=self.tmp_dir.name, auto_extract=False, verbose=False)
        self.assertEqual(count, 0)
        self.assertEqual(len(os.listdir(self.indiv_dir)), 0)

    def test_extract_and_remove_single_patch(self):
        # Extract fibers ONLY from patch_0001
        n_extracted = extract_fibers_from_patch(
            "patch_0001",
            curated_dir=self.curated_dir,
            indiv_dir=self.indiv_dir,
            padding=2,
            verbose=False
        )
        self.assertEqual(n_extracted, 2)

        # Check that files for patch_0001 exist
        self.assertTrue(os.path.exists(os.path.join(self.indiv_dir, "patch_0001_fiber_01_vol.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.indiv_dir, "patch_0001_fiber_02_vol.npy")))
        # Check that patch_0002 files do NOT exist
        self.assertFalse(os.path.exists(os.path.join(self.indiv_dir, "patch_0002_fiber_01_vol.npy")))

        # Check counts
        counts = get_extracted_patch_counts(self.indiv_dir)
        self.assertEqual(counts.get("patch_0001"), 2)
        self.assertNotIn("patch_0002", counts)

        # Test removal
        n_removed = remove_extracted_fibers_for_patch("patch_0001", indiv_dir=self.indiv_dir, verbose=False)
        self.assertEqual(n_removed, 6)  # 2 fibers * 3 files (vol, int, ori)
        self.assertEqual(len(os.listdir(self.indiv_dir)), 0)

    def test_morpher_library_sync(self):
        # Extract patch_0002 only
        extract_fibers_from_patch("patch_0002", curated_dir=self.curated_dir, indiv_dir=self.indiv_dir, verbose=False)
        cache_path = os.path.join(self.tmp_dir.name, 'fiber_library.pkl')

        # Sync library
        sync_morpher_library(curated_dir=self.curated_dir, indiv_dir=self.indiv_dir, cache_path=cache_path, verbose=False)
        self.assertTrue(os.path.exists(cache_path))

        # RealFiberLibrary loads only patch_0002
        lib = RealFiberLibrary(curated_dir=self.curated_dir, cache_path=cache_path)
        for f in lib.fibers:
            self.assertEqual(f.patch_index, 2)


if __name__ == '__main__':
    unittest.main()

