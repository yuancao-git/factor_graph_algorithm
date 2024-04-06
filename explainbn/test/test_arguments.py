import os
import unittest

from pgmpy.utils import get_example_model
import networkx as nx

from explainbn.arguments import (
  compute_step_effect, 
  compute_argument_effects, 
  compute_argument_strength, 
  all_simple_arguments, 
  all_local_arguments,
  random_argument
)
from explainbn.utilities import init_factor, factor_to_logodds

class TestArguments(unittest.TestCase):
    def setUp(self):
      # Create context
      self.model = get_example_model('asia')
      self.target = ('lung', 'yes')
      self.evidence = {'asia': 'yes', 'tub': 'yes', 'xray': 'yes'}
      
      # Large context
      self.alarm_model = get_example_model('alarm')
      self.alarm_target = ('PVSAT', 'NORMAL')
      self.alarm_evidence = {'VENTALV': 'LOW', 'SHUNT': 'HIGH', 
        'CATECHOL': 'NORMAL', 'PULMEMBOLUS': 'FALSE', 'VENTTUBE': 'LOW', 
        'PRESS': 'HIGH', 'DISCONNECT': 'TRUE', 'PCWP': 'HIGH', 'FIO2': 'NORMAL', 
        'HRBP': 'HIGH', 'ERRCAUTER': 'TRUE', 'LVEDVOLUME': 'HIGH', 'CO': 'LOW', 
        'HISTORY': 'FALSE', 'HYPOVOLEMIA': 'TRUE', 'ANAPHYLAXIS': 'FALSE', 
        'EXPCO2': 'NORMAL', 'LVFAILURE': 'TRUE'}
      
      # Conditional probability table
      self.factor = self.model.get_cpds('either').to_factor()
      
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
    
    def test_compute_step_effect(self, epsilon=0.00001):
      # Test 
      incoming_messages = {'either' : init_factor(self.model, 'either', 'yes'), 
                           'tub'    : init_factor(self.model, 'tub', 'no'),}
      target = 'lung'
      delta = compute_step_effect(self.model, self.factor, incoming_messages, target)
      logodds = factor_to_logodds(delta, ('lung', 'yes'))
      self.assertGreater(logodds, 0)
      
      # Compare interaction to evidential case
      incoming_messages = {'either' : init_factor(self.model, 'either', 'yes')}
      target = 'lung'
      delta = compute_step_effect(self.model, self.factor, incoming_messages, target)
      logodds2 = factor_to_logodds(delta, ('lung', 'yes'))
      self.assertGreater(logodds2, 0)
      self.assertLess(logodds2, logodds)
      
      # D-separated link
      incoming_messages = {'tub'    : init_factor(self.model, 'tub', 'no'),}
      target = 'lung'
      delta = compute_step_effect(self.model, self.factor, incoming_messages, target)
      logodds = factor_to_logodds(delta, ('lung', 'yes'))
      self.assertLess(logodds, epsilon)
      
      # Negative case
      incoming_messages = {'either' : init_factor(self.model, 'either', 'no'), 
                           'tub'    : init_factor(self.model, 'tub', 'no'),}
      target = 'lung'
      delta = compute_step_effect(self.model, self.factor, incoming_messages, target)
      logodds = factor_to_logodds(delta, ('lung', 'yes'))
      self.assertLess(logodds, 0)
    
    def test_compute_argument_effects(self):
      argument = compute_argument_effects(self.model, self.argument)
    
    def test_compute_argument_effects(self):
      # Call without effects computed
      strength = compute_argument_strength(self.model, self.argument)
      
      # Call with effects computed
      strength = compute_argument_strength(self.model, self.argument)
    
    def test_all_simple_arguments(self):
      simple_arguments = all_simple_arguments(self.model, 
                                              self.target, 
                                              self.evidence)
    
    def test_all_simple_arguments_short(self):
      simple_arguments = all_simple_arguments(self.alarm_model, 
                                              self.alarm_target, 
                                              self.alarm_evidence, 
                                              length_limit=5)
      
      for simple_argument in simple_arguments:
        len_argument = len(simple_argument)
        self.assertLessEqual(len_argument, 5)
    
    def test_all_local_arguments(self):
      arguments = all_local_arguments(self.model, self.target, self.evidence)
    
    def test_all_local_arguments_heuristic(self):
      arguments = all_local_arguments(self.alarm_model, 
                                      self.alarm_target, 
                                      self.alarm_evidence, 
                                      path_length_limit=7, 
                                      argument_complexity_limit=3)
      
      for argument in arguments:
        pass
    
    def test_random_argument(self):
      argument = random_argument(self.model)