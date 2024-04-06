
"""
Utility functions used by the other modules
Includes utlities for
1 - Interfacing with the PGMPY library
2 - Creating iterables
3 - Interacting with arguments
4 - Generating textual and visual explanations
"""

from collections import deque
from functools import lru_cache
from itertools import product, combinations, chain, tee, islice

import networkx as nx
import numpy as np

from pgmpy.inference import BeliefPropagation, VariableElimination
from pgmpy.factors.discrete import DiscreteFactor

# PGMPY utilities
def init_factor(model, node, observed_state = None, 
               eps = 0.0000001):
  """ Initializes a factor to an extreme state 
      corresponding to an observation.

      If observed_state is None, the factor is
      initialized as a constant factor
  """
  possible_states = model.states[node]
  '''
  #uniform distribution
  m = [1. if state == observed_state else eps
       for state in possible_states]
  '''
  #prior probability
  if observed_state:
    m = [1. if state == observed_state else eps
         for state in possible_states]
  else:
    infer = VariableElimination(model)
    q = infer.query(variables=[node], evidence={})
    #if possible_states[0] == 'yes':
    m = q.values
    #else:
     #   m = [q.values[1],q.values[0]]
  
  '''
  else:
    if model.get_cpds(node).values.size == len(possible_states):
        m = model.get_cpds(node).values
    else:
        m = [eps for state in possible_states]
  '''
  
      
  delta = DiscreteFactor([node], 
                         [len(possible_states)], 
                         [m], 
                         {node:possible_states})
  delta.normalize()
  return delta

def factor_to_logodds(factor, outcome):
  """
  Summarize a factor update as a logodd scalar.
  """
  
  # If the outcome is a single variable assignment 
  # we coerce it as a list
  if type(outcome[0]) == str and len(outcome) == 2:
    outcome = [outcome]
  
  factor.normalize()
  p = factor.get_value(**dict(outcome))
  n = np.product(factor.values.shape)
  s = np.log(p / (1. - p) * (n - 1))
  
  return s

def factor_argmax(factor):
  """ Return the outcome favored by the factor
      Input: factor
      Output: [(node, state), ...]
  """
  possible_scope_values = product(*[[(node, value) 
                                  for value in factor.state_names[node]]
                                  for node in factor.variables])
  max_value = 0.
  max_scope = None
  for scope_value in possible_scope_values:

    v = factor.get_value(**dict(scope_value))
    if v > max_value:
      max_value = v
      max_scope = scope_value
      
  return max_scope

def factor_to_outcomes(factor, threshold = 1):

  # Normalize factor
  factor.normalize()

  # Find outcomes that are close to the maximum probability
  max_logodds = prob_to_logodds(factor.values.max())
  favored_outcomes = []
  possible_scope_values = product(*[[(node, value) 
                                  for value in factor.state_names[node]]
                                  for node in factor.variables])
  for scope_value in possible_scope_values:
    logodds = prob_to_logodds(factor.get_value(**dict(scope_value)))
    if logodds + threshold >= max_logodds:
      favored_outcomes.append(scope_value[0])
  
  # Compute combined strength of favoured outcomes
  strength = prob_to_logodds(np.sum([factor.get_value(**dict([outcome]))
                                     for outcome in favored_outcomes]
                                    ))

  return favored_outcomes, strength

def factor_distance(factor1, factor2):
  """ Computes the similarity between the factors
  """
  
  if factor1.scope() != factor2.scope():
    raise ValueError("The factors must have the same scope!")
  
  delta = factor1.divide(factor2, inplace = False)
  delta.normalize()
  max_delta = 0.0
  for node in delta.variables:
    for state in delta.state_names[node]:
      s = factor_to_logodds(delta, [(node, state)])
      if np.abs(s) > max_delta: max_delta = np.abs(s)
  return max_delta

# Modify factor values back into the eps to 1.-eps range
eps = 0.0000001
desextremize = np.vectorize(lambda f : 
                            eps if f < eps 
                            else 1. - eps 
                            if f > 1. - eps 
                            else f)

def get_child_from_factor_scope(model, factor_scope):
  for node1 in factor_scope:
    for node2 in factor_scope:
      if node1 != node2 and node2 in model.get_children(node1):
        break
    else:
      return node1

@lru_cache(maxsize = 128)
def get_factor_from_scope(model, scope):
  child = get_child_from_factor_scope(model, scope)
  f = model.get_cpds(child).to_factor()
  return f
  
def to_factor_graph(model):
  """ Returns the factor graph associated with a Bayesian Network model
      Factor nodes are represented as a frozenset of their scope
  """
  factors = [frozenset(model.get_cpds(node).to_factor().scope()) for node in model.nodes]
  factor_graph = nx.Graph()
  edges = [(node, factor) for factor in factors for node in factor] + \
          [(factor, node) for factor in factors for node in factor]
  factor_graph.add_edges_from(edges)
  return factor_graph

prob_to_logodds = lambda p : np.log(p / (1. - p))
logodds_to_prob = lambda s : 1. / (1. + 1. / np.exp(s))

random_evidence = lambda model, evidence_nodes : \
   {node : np.random.choice(list(model.states[node])) \
    for node in evidence_nodes if np.random.rand() < 0.5}

def random_outcome(model, nodes=None):
  if nodes is None: nodes = model.nodes
  node = str(np.random.choice(nodes))
  state = np.random.choice(model.states[node])
  return (node, state)

# General iteration utilities

def powerset(iterable):
    """
    powerset([1,2,3]) --> 
    () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def limited_powerset(iterable, k):
  """ Returns all combinations up to size k
  """
  return chain.from_iterable(combinations(iterable, j) for j in range(0, k+1))

def nwise(iterable, n=2):
  """
  Return an iterable that returns the adjacent n-uples of the input
  """
  iters = tee(iterable, n)
  for i, it in enumerate(iters):
      next(islice(it, i, i), None)
  return zip(*iters)

def partitions(collection):
  """ Iterator over partitions of collection
      https://stackoverflow.com/a/30134039/4841832
  """
  if len(collection) == 1:
    yield [ collection ]
    return

  first = collection[0]
  for smaller in partitions(collection[1:]):
    # insert `first` in each of the subpartition's subsets
    for n, subset in enumerate(smaller):
      yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
    # put `first` in its own subset 
    yield [ [ first ] ] + smaller
    
    
# Argument utilities

def make_argument_from_stack(stack, observations = {}, target = None):
  argument = nx.DiGraph()
  edges = ((node2, node1) for node1, node2 in nwise(stack))
  argument.add_edges_from(edges)
  argument.observations = observations
  if target is not None:
    argument.target = target
  return argument

def iterate_argument(argument):
  """ Iterate all variables in the argument from the sources to the sink
  """
  to_explore = deque(node for node in argument.nodes 
                    if len(list(argument.predecessors(node))) == 0)
  added = set()
  explored = set()

  while len(to_explore) > 0:
    next_node = to_explore.pop()
    if type(next_node) == str: # If the node is a variable
      yield next_node
    explored.add(next_node)

    # Add children if ready
    for child in argument.successors(next_node):
      if child not in added and child not in explored and \
         all(parent in explored for parent in argument.predecessors(child)):
        to_explore.appendleft(child)
        added.add(child)

""" Function representing the subargument relation
"""
is_subargument = lambda arg1, arg2: \
 set(arg1.observations.items()) <= set(arg2.observations.items()) \
 and set(arg1.nodes) <= set(arg2.nodes) \
 and set(arg1.edges) <= set(arg2.edges)
 
def compose_arguments(arguments):
  """ Return the union of the arguments
  """
  argument = nx.compose_all(arguments)
  argument.observations = {k:v for argument in arguments 
                           for k,v in argument.observations.items()}
  argument.target = arguments[0].target
  return argument

