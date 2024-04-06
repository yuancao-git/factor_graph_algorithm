import os
import unittest
import math
from collections import Counter

import numpy as np

from pgmpy.utils import get_example_model

from explainbn.utilities import *

class TestPGMPYUtilities(unittest.TestCase):
    def setUp(self, epsilon = 1e-6):
      # Create model
      network_name = 'asia'
      self.model = get_example_model(network_name)
      
      # Get factor
      self.mono_factor = self.model.get_cpds('asia').to_factor()
      self.extreme_factor = init_factor(self.model, 'asia', 'yes', eps=0)
      self.unif_factor = init_factor(self.model, 'asia')
      self.cpd_factor = self.model.get_cpds('dysp').to_factor()
      
      self.epsilon = epsilon
    
    # Factor tests
    def test_init_factor(self):
      # Test uniform factor creation
      asia_factor = init_factor(self.model, "asia")
      
      # Test lopsided factor creation
      asia_yes = init_factor(self.model, "asia", "yes")
    
    def test_factor_to_logodds(self):
      # Test with single assigment
      logodds = factor_to_logodds(self.mono_factor, ('asia', 'yes'))
      self.assertLess(logodds, 0)
      
      # Test with multiple assigment
      logodds = factor_to_logodds(self.cpd_factor, [('dysp', 'yes'), ('bronc', 'yes'), ('either', 'yes')])
    
    def test_factor_argmax(self):
      argmax = factor_argmax(self.mono_factor)
      self.assertCountEqual(argmax, [('asia', 'no')])
    
    def test_factor_to_outcomes(self):
      outcome, strength = factor_to_outcomes(self.mono_factor)
      self.assertCountEqual(outcome, [('asia', 'no')])
      self.assertGreater(strength, 0)
    
    def test_factor_distance(self):
      distance = factor_distance(self.mono_factor, self.unif_factor)
      self.assertGreater(distance, 0)
      
      distance = factor_distance(self.mono_factor, self.mono_factor)
      self.assertLess(abs(distance), self.epsilon)
    
    def test_desextremize(self):
      self.extreme_factor.values = desextremize(self.extreme_factor.values)
      s = factor_to_logodds(self.extreme_factor, ('asia', 'yes'))
      self.assertFalse(math.isinf(s))
      
      desextremized_factor_values = desextremize(self.mono_factor.values)
      self.assertTrue(np.allclose(desextremized_factor_values, self.mono_factor.values))
    
    # Test other PGMPY utilities
    
    def test_get_child_from_factor_scope(self):
      child = get_child_from_factor_scope(self.model, self.cpd_factor.scope())
      self.assertEqual(child, 'dysp')
    
    def test_to_factor_graph(self):
      factor_graph = to_factor_graph(self.model)
      
      expected_nodes = list(self.model.nodes) + \
        [frozenset(self.model.get_cpds(node).to_factor().scope()) 
         for node in self.model.nodes]
      
      self.assertCountEqual(factor_graph.nodes, expected_nodes)
    
    def test_probability_manipulation(self):
      s = prob_to_logodds(0.5)
      self.assertAlmostEqual(s, 0.0)
      
      p = logodds_to_prob(0.0)
      self.assertAlmostEqual(p, 0.5)
      
      s1 = 0.1245
      s2 = prob_to_logodds(logodds_to_prob(s1))
      self.assertAlmostEqual(s1,s2)
      
      p1 = 0.9876
      p2 = logodds_to_prob(prob_to_logodds(p1))
      self.assertAlmostEqual(p1,p2)
      

class TestIterationUtilities(unittest.TestCase):

    def test_powerset(self):
      s = list(powerset([1,2,3]))
      self.assertCountEqual(s, [(), (1,), (2,), (3,), (2,3), (1,3), (1,2), (1,2,3)])
    
    def test_limited_powerset(self):
      s = list(limited_powerset([1,2,3], k=2))
      self.assertCountEqual(s, [(), (1,), (2,), (3,), (2,3), (1,3), (1,2)])
    
    def test_nwise(self):
      s = list(nwise([1,2,3,4]))
      self.assertCountEqual(s, [(1,2), (2,3), (3,4)])
      
      s = list(nwise([1,2,3,4], n=3))
      self.assertCountEqual(s, [(1,2,3), (2,3,4)])
    
    def test_partitions(self):
      s = list(partitions([1,2,3]))
      foo = lambda s : [frozenset(frozenset(part) for part in partition) for partition in s]
      self.assertCountEqual(foo(s), foo([
        [[1,2,3]],
        [[1], [2,3]],
        [[2], [1,3]],
        [[3], [1,2]],
        [[1], [2], [3]]
      ]))

class TestArgumentUtilities(unittest.TestCase):
    def setUp(self):
      # Create factor argument
      argument = nx.DiGraph()
      argument.observations = {'tub' : 'no', 
                               'xray' : 'yes'}
      argument.target = ('lung', 'yes')

      edges = [
        ('tub', frozenset({'tub', 'lung', 'either'})),

        ('xray', frozenset({'xray', 'either'})),
        (frozenset({'xray', 'either'}), 'either'),

        ('either', frozenset({'tub', 'lung', 'either'})),
        (frozenset({'tub', 'lung', 'either'}), 'lung'),
      ]

      argument.add_edges_from(edges)
      
      self.argument = argument
    
    def make_argument_from_stack(self):
      argument = make_argument_from_stack(reversed(['xray', frozenset(['xray', 'either']), 
                                          'either', frozenset(['either', 'tub', 'lung']),
                                          'lung']))
    
    def test_iterate_argument(self):
      traversal = list(iterate_argument(self.argument))
      self.assertEqual(traversal, ['xray', 'tub', 'either', 'lung'])
    
    def test_is_subargument(self):
      arg = make_argument_from_stack(reversed(['xray', frozenset(['xray', 'either']), 
                                     'either', frozenset(['either', 'tub', 'lung']),
                                     'lung']))
      self.assertTrue(is_subargument(arg, self.argument))
      self.assertTrue(is_subargument(self.argument, self.argument))
      self.assertFalse(is_subargument(self.argument, arg))
    
    def test_compose_arguments(self):
      arg1 = make_argument_from_stack(reversed(['xray', frozenset(['xray', 'either']), 
                                      'either', frozenset(['either', 'tub', 'lung']),
                                      'lung']), observations = {'xray': 'yes'},
                                      target = ('lung', 'yes'))
      arg2 = make_argument_from_stack(reversed(['tub', frozenset(['either', 'tub', 'lung']), 'lung']),
                                      observations = {'either': 'yes'},
                                      target = ('lung', 'yes'))
      
      arg = compose_arguments([arg1, arg2])
      self.assertEqual(frozenset(arg.edges), frozenset(self.argument.edges))