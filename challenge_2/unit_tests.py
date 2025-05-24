import unittest
import os
from vidyut.prakriya import Vyakarana, Pada, Prayoga, Lakara, Purusha, Vacana, Data
from vidyut.lipi import transliterate, Scheme
from unittest.mock import patch

# --- Configuration ---
# IMPORTANT: Please set this path to where your Vidyut data is located.
MORPHOLOGICAL_DATA_PATH = "vidyut-0.4.0/prakriya/"

# --- Attempt to Import ---
# We try to import the function we want to test.
try:
    from make_dataset import get_human_readable_dhatu as imported_ghrd
    CAN_IMPORT_GHRD = True
    print("Successfully imported 'get_human_readable_dhatu' from make_dataset.py.")
except ImportError as e:
    imported_ghrd = None
    CAN_IMPORT_GHRD = False
    print(f"Warning: Could not import 'get_human_readable_dhatu': {e}")
    print("Tests requiring this function will be skipped.")


# --- Unit Test Class ---
class TestMakeDataset(unittest.TestCase):

    v = None
    found_dhatus = {}

    @classmethod
    def setUpClass(cls):
        """
        Set up the Vyakarana instance and load Dhatu objects once.
        It will patch 'make_dataset.v' while loading Dhatus
        if 'get_human_readable_dhatu' was successfully imported.
        """
        print("\nSetting up test class...")
        cls.v = Vyakarana(log_steps=False) # Create our own 'v' instance.

        # Check if the data path exists before trying to load.
        if not os.path.exists(MORPHOLOGICAL_DATA_PATH):
            raise unittest.SkipTest(
                f"Vidyut data path not found: {MORPHOLOGICAL_DATA_PATH}. "
                "Skipping tests that require Dhatu objects."
            )

        # Only load Dhatus if we can process them (i.e., if we imported ghrd)
        if not CAN_IMPORT_GHRD:
            print("Skipping Dhatu loading as 'get_human_readable_dhatu' could not be imported.")
            return

        print(f"Loading Dhatu data from: {MORPHOLOGICAL_DATA_PATH}")
        data = Data(MORPHOLOGICAL_DATA_PATH)
        dhatu_list = [e.dhatu for e in data.load_dhatu_entries()]
        print(f"Loaded {len(dhatu_list)} Dhatu entries.")

        desired_dhatus_set = {
            "bhāṣ", "gam", "bhū", "dṛś", "śru", "car", "han"
        }
        
        print("Searching for desired Dhatus (using imported function with patch)...")
        
        # We MUST patch 'make_dataset.v' here so that the imported
        # 'get_human_readable_dhatu' uses *our* 'cls.v' instance.
        with patch('make_dataset.v', cls.v):
            for i in dhatu_list:
                # Use the imported function (it will use cls.v due to patch)
                hrd = imported_ghrd(i)
                if hrd and hrd in desired_dhatus_set:
                    if hrd not in cls.found_dhatus: # Store only the first match
                       cls.found_dhatus[hrd] = i

        print(f"Found {len(cls.found_dhatus)} unique desired Dhatus.")

        # Assign specific dhatus for easier access in tests
        cls.dhatu_bhu = cls.found_dhatus.get("bhū")
        cls.dhatu_gam = cls.found_dhatus.get("gam")
        cls.dhatu_drs = cls.found_dhatus.get("dṛś")
        cls.dhatu_bhas = cls.found_dhatus.get("bhāṣ")
        
        if not all([cls.dhatu_bhu, cls.dhatu_gam, cls.dhatu_drs, cls.dhatu_bhas]):
             print("Warning: Not all essential test Dhatus were found.")

    # ==================================
    # Tests for transliterate
    # ==================================

    def test_transliterate_slp1_to_iast(self):
        """Test SLP1 to IAST transliteration."""
        self.assertEqual(transliterate("Bavati", Scheme.Slp1, Scheme.Iast), "bhavati")
        self.assertEqual(transliterate("gacCati", Scheme.Slp1, Scheme.Iast), "gacchati")
        self.assertEqual(transliterate("kftaH", Scheme.Slp1, Scheme.Iast), "kṛtaḥ")

    def test_transliterate_iast_to_slp1(self):
        """Test IAST to SLP1 transliteration."""
        self.assertEqual(transliterate("bhavāmi", Scheme.Iast, Scheme.Slp1), "BavAmi")
        self.assertEqual(transliterate("kṛtaḥ", Scheme.Iast, Scheme.Slp1), "kftaH")

    # ==================================
    # Test for get_human_readable_dhatu (Imported)
    # ==================================

    @unittest.skipIf(not CAN_IMPORT_GHRD, "Skipping test: 'get_human_readable_dhatu' could not be imported.")
    def test_get_human_readable_dhatu_imported(self):
        """Test the imported get_human_readable_dhatu with real Dhatu objects."""
        if not self.dhatu_bhu or not self.dhatu_gam:
            self.skipTest("Required Dhatu objects (bhū, gam) not loaded.")
            
        # Patch 'make_dataset.v' to use our 'self.v' instance for this test
        with patch('make_dataset.v', self.v):
            # Test 'bhū'
            hrd_bhu = imported_ghrd(self.dhatu_bhu)
            self.assertEqual(hrd_bhu, "bhū")

            # Test 'gam'
            hrd_gam = imported_ghrd(self.dhatu_gam)
            self.assertEqual(hrd_gam, "gam")

    # ==================================
    # Tests for vyakarana.derive (using real Dhatus)
    # ==================================

    def test_vyakarana_derive_lat_real_dhatu(self):
        """Test v.derive for Lat lakara using real Dhatu objects."""
        if not self.dhatu_bhu or not self.dhatu_gam:
            self.skipTest("Required Dhatu objects not loaded for Lat test.")

        # bhū -> bhavati (using our local 'self.v')
        pada_bhu = Pada.Tinanta(
            dhatu=self.dhatu_bhu, prayoga=Prayoga.Kartari, lakara=Lakara.Lat,
            purusha=Purusha.Prathama, vacana=Vacana.Eka
        )
        prakriyas_bhu = self.v.derive(pada_bhu)
        self.assertTrue(prakriyas_bhu, "Should derive 'Bavati'")
        self.assertEqual(prakriyas_bhu[0].text, "Bavati")

        # gam -> gacchathaḥ (using our local 'self.v')
        pada_gam = Pada.Tinanta(
            dhatu=self.dhatu_gam, prayoga=Prayoga.Kartari, lakara=Lakara.Lat,
            purusha=Purusha.Madhyama, vacana=Vacana.Dvi
        )
        prakriyas_gam = self.v.derive(pada_gam)
        self.assertTrue(prakriyas_gam, "Should derive 'gacCaTaH'")
        self.assertEqual(prakriyas_gam[0].text, "gacCaTaH")

    def test_vyakarana_derive_lit_real_dhatu(self):
        """Test v.derive for Lit lakara using real Dhatu objects."""
        if not self.dhatu_bhu or not self.dhatu_gam:
            self.skipTest("Required Dhatu objects not loaded for Lit test.")
            
        # bhū -> babhūva (using our local 'self.v')
        pada_bhu = Pada.Tinanta(
            dhatu=self.dhatu_bhu, prayoga=Prayoga.Kartari, lakara=Lakara.Lit,
            purusha=Purusha.Prathama, vacana=Vacana.Eka
        )
        prakriyas_bhu = self.v.derive(pada_bhu)
        self.assertTrue(prakriyas_bhu, "Should derive 'baBUva'")
        self.assertEqual(prakriyas_bhu[0].text, "baBUva")

        # gam -> jagāma (using our local 'self.v')
        pada_gam = Pada.Tinanta(
            dhatu=self.dhatu_gam, prayoga=Prayoga.Kartari, lakara=Lakara.Lit,
            purusha=Purusha.Prathama, vacana=Vacana.Eka
        )
        prakriyas_gam = self.v.derive(pada_gam)
        self.assertTrue(prakriyas_gam, "Should derive 'jagAma'")
        self.assertEqual(prakriyas_gam[0].text, "jagAma")

# --- Run Tests ---
if __name__ == '__main__':
    unittest.main()
