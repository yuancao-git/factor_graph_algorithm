import os
import unittest

from pgmpy.utils import get_example_model
import networkx as nx

from explainbn.explanations import explain_step, explain_argument, explain_evidence
from explainbn.utilities import init_factor
from explainbn.arguments import compute_argument_effects

class TestExplanations(unittest.TestCase):
    def setUp(self):
      # Create model
      network_name = 'asia'
      self.model = get_example_model(network_name)
      
      # Add node and state descriptions
      self.model.descriptions = \
      {
       ('asia', 'no'): 'the patient has not recently visited Asia',
       ('asia', 'yes'): 'the patient has recently visited Asia',
       ('bronc', 'no'): 'the patient does not have bronchitis',
       ('bronc', 'yes'): 'the patient has bronchitis',
       ('dysp', 'no'): 'the patient is not experiencing shortness of breath',
       ('dysp', 'yes'): 'the patient is experiencing shortness of breath',
       ('either', 'no'): 'the patient does not have a lung disease',
       ('either', 'yes'): 'the patient has a lung disease',
       ('lung', 'no'): 'the patient does not have lung cancer',
       ('lung', 'yes'): 'the patient has lung cancer',
       ('smoke', 'no'): 'the patient does not smoke',
       ('smoke', 'yes'): 'the patient smokes',
       ('tub', 'no'): 'the patient does not have tuberculosis',
       ('tub', 'yes'): 'the patient has tuberculosis',
       ('xray', 'no'): 'the lung xray is normal',
       ('xray', 'yes'): 'the lung xray shows an abnormality'
      }
      
      # Create rule
      self.rule = self.model.get_cpds('dysp').to_factor()
      
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
      
      self.argument = compute_argument_effects(self.model, argument)
    
    def test_explain_step(self):
      premises = {'dysp' : init_factor(self.model, 'dysp', 'yes'),
                  'either': init_factor(self.model, 'either', 'no')}
      conclusion = 'bronc'
      explanation = explain_step(self.model, self.rule, premises, conclusion, mode='direct')
      
      explanation = explain_step(self.model, self.rule, premises, conclusion, mode='contrastive')

    def test_explain_argument(self):
      explanation = explain_argument(self.model, self.argument, mode='direct')
      
      explanation = explain_argument(self.model, self.argument, mode='contrastive')
      
      explanation = explain_argument(self.model, self.argument, mode='overview')
      
      # Test after deleting descriptions
      del self.model.descriptions
      with self.assertWarns(Warning):
        explanation = explain_argument(self.model, self.argument)
    
    def test_explain_evidence(self):
      explanation, strength = explain_evidence(self.model, self.argument.effects['lung'])
      self.assertEqual(strength, "strong")