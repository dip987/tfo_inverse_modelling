"""
Unit test for tools/parameter_generation.py
"""
import unittest
from random import randint
from inverse_modelling_tfo.tools.parameter_generation import MuAGenerator, ProximityMuAGenerator


class MuAGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.m_s_range = (0.9, 1.0)
        self.m_s_count = 5
        self.m_c_range = (5, 12)
        self.m_c_count = 4
        self.f_s_range = (0.1, 0.6)
        self.f_s_count = 5
        self.f_c_range = (0.1, 1)
        self.f_c_count = 5
        self.wave_int = 1

    def test_mu_a_generator_fetal_mu_count(self):
        f_s_count = randint(1, 10)
        f_c_count = randint(1, 10)
        gen = MuAGenerator(
            self.m_s_range,
            self.m_s_count,
            self.m_c_range,
            self.m_c_count,
            self.f_s_range,
            f_s_count,
            self.f_c_range,
            f_c_count,
            self.wave_int,
        )
        _, fetal_mu_a = gen.generate()
        self.assertEqual(len(fetal_mu_a), f_s_count * f_c_count)

    def test_mu_a_generator_maternal_mu_count(self):
        m_s_count = randint(1, 10)
        m_c_count = randint(1, 10)
        gen = MuAGenerator(
            self.m_s_range,
            m_s_count,
            self.m_c_range,
            m_c_count,
            self.f_s_range,
            self.f_s_count,
            self.f_c_range,
            self.f_c_count,
            self.wave_int,
        )
        maternal_mu_a, _ = gen.generate()
        self.assertEqual(len(maternal_mu_a), m_s_count * m_c_count)

    def test_proximity_mu_a_generator_fetal_mu_count(self):
        f_s_count = randint(1, 10)
        f_c_count = randint(1, 10)
        proximity_fractions = [0.05, -0.05]
        gen = ProximityMuAGenerator(
            self.m_s_range,
            self.m_s_count,
            self.m_c_range,
            self.m_c_count,
            self.f_s_range,
            f_s_count,
            self.f_c_range,
            f_c_count,
            self.wave_int,
            0.1,
            0.1,
            proximity_fractions,
            proximity_fractions,
        )
        _, fetal_mu_a = gen.generate()
        self.assertEqual(len(fetal_mu_a), f_s_count * f_c_count * (1 + len(proximity_fractions)))

    def test_proximity_mu_a_generator_maternal_mu_count(self):
        proximity_fractions = [0.05, -0.05]
        m_s_count = randint(1, 10)
        m_c_count = randint(1, 10)
        gen = ProximityMuAGenerator(
            self.m_s_range,
            m_s_count,
            self.m_c_range,
            m_c_count,
            self.f_s_range,
            self.f_s_count,
            self.f_c_range,
            self.f_c_count,
            self.wave_int,
            0.1,
            0.1,
            proximity_fractions,
            proximity_fractions,
        )
        maternal_mu_a, _ = gen.generate()
        self.assertEqual(len(maternal_mu_a), m_s_count * m_c_count * (1 + len(proximity_fractions)))
