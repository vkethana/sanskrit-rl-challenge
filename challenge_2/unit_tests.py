import unittest
from vidyut.prakriya import *
from vidyut.lipi import *
from vidyut.kosha import DhatuEntry, PadaEntry

class TestSanskritMorphology(unittest.TestCase):
    def setUp(self):
        self.v = Vyakarana(log_steps=False)
        self.translit = lambda x: transliterate(str(x), Scheme.Slp1, Scheme.Iast)

    def test_bhavati(self):
        """Test the classic bhavati example"""
        prakriyas = self.v.derive(Pada.Tinanta(
            dhatu=Dhatu.mula(aupadeshika="BU", gana=Gana.Bhvadi),
            prayoga=Prayoga.Kartari,
            lakara=Lakara.Lat,
            purusha=Purusha.Prathama,
            vacana=Vacana.Eka,
        ))
        self.assertTrue(prakriyas)
        self.assertEqual(self.translit(prakriyas[0].text), "bhavati")

    def test_dual_plural(self):
        """Test dual and plural forms"""
        # Dual
        prakriyas = self.v.derive(Pada.Tinanta(
            dhatu=Dhatu.mula(aupadeshika="BU", gana=Gana.Bhvadi),
            prayoga=Prayoga.Kartari,
            lakara=Lakara.Lat,
            purusha=Purusha.Prathama,
            vacana=Vacana.Dvi,
        ))
        self.assertTrue(prakriyas)
        self.assertEqual(self.translit(prakriyas[0].text), "bhavataḥ")

        # Plural
        prakriyas = self.v.derive(Pada.Tinanta(
            dhatu=Dhatu.mula(aupadeshika="BU", gana=Gana.Bhvadi),
            prayoga=Prayoga.Kartari,
            lakara=Lakara.Lat,
            purusha=Purusha.Prathama,
            vacana=Vacana.Bahu,
        ))
        self.assertTrue(prakriyas)
        self.assertEqual(self.translit(prakriyas[0].text), "bhavanti")

if __name__ == '__main__':
    unittest.main()
