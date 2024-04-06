#!/usr/bin/env python3

"""
Facilities to evaluate the strength of arguments,
and identify all relevant arguments that apply given
a Bayesian Network, a target node and some evidence
"""

from functools import reduce
from itertools import combinations
import networkx as nx
import numpy as np
from pgmpy.inference import VariableElimination

from .utilities import (
  desextremize, 
  init_factor, 
  get_factor_from_scope, 
  factor_to_logodds, 
  to_factor_graph, 
  factor_distance, 
  make_argument_from_stack, 
  compose_arguments, 
  iterate_argument, 
  is_subargument, 
  limited_powerset, 
  partitions,
  random_evidence,
  random_outcome,
  )

from itertools import product
def are_sets_dconnected(model, set_1, set_2, observed=None):
    for node1, node2 in product(set_1, set_2):
        if model.is_dconnected(node1, node2, observed=observed):
            return True
    return False

# Definition of link effect
def compute_step_effect(model, factor, incoming_messages, target, observed):
  """ Compute the effect of the shift of beliefs described by 
      incoming_messages on variable target, 
      according to the input factor.
  """
  other_nodes = set(factor.variables) - {target}
  keys = {key for key in incoming_messages}

            
  f1 = reduce(lambda a,b: a.product(b, inplace=False),
              incoming_messages.values(),
              factor)
  f2 = factor.copy()
  
  #marginalization with prior probabilities
  if other_nodes-keys != None:
    for node in other_nodes-keys:
        infer = VariableElimination(model)
        prior = infer.query(variables=[node])
        f = f1.product(prior,inplace=False)
        #f = f1.product(model.get_cpds(node),inplace=False)
        f1 = f

  f1.normalize()
  f2.normalize()

  f1.marginalize(other_nodes)
  f2.marginalize(other_nodes)

  f1.normalize()
  f2.normalize()

  f1.values = desextremize(f1.values)
  f2.values = desextremize(f2.values)
  
  delta = f1.divide(f2, inplace=False)
  #delta = f1
  step_model = model.copy()
  step_model.remove_nodes_from(model.nodes()-factor.variables)    
  if not are_sets_dconnected(step_model,list({target}), list(keys)):
    delta = delta.divide(delta, inplace=False)
  #if not model.is_dconnected(target, list(keys)[0],observed = observed):
   # delta = delta.divide(delta, inplace=False)
  delta.normalize()

  assert set(delta.scope()) == {target}, delta.scope()
  
  return delta
  

def compute_argument_effects(model, argument):
  """ Computes the effect of the argument on each of its variable nodes
  """
  argument.effects = {}
  argument.d_separated = {}
  # We iterate the nodes from the evidence up to the target
  for node in iterate_argument(argument):

    # If the effect was already computed skip
    if node in argument.effects: continue
    observed = []
      
    if len(list(argument.predecessors(node))) == 0: # If the node is an evidence node
      # The effect on the evidence nodes is initialized as a lopsided factor
      argument.effects[node] = init_factor(model, node, argument.observations[node])
      observed.append(node)

    else:
      # The rest of the nodes is computed as the product of the link effects
      # for each factor feeding into the node
      effect = init_factor(model, node)
      effect_init = init_factor(model, node)
      for factor_scope in argument.predecessors(node):
        factor = get_factor_from_scope(model, factor_scope)
        deltas = {parent : argument.effects[parent] for parent in argument.predecessors(factor_scope)}
        effect *= compute_step_effect(model, factor, deltas, node,observed)
        #if node is the child, then do not need to multiply prior probability
        if node == factor.variables[0]:
          effect = effect / effect_init
          #make sure divide only once for the same node
          effect_init = np.ones(effect.values.shape)

        keys = {key for key in deltas}
        #delete all other nodes to determine d-separation
        step_model = model.copy()
        step_model.remove_nodes_from(model.nodes()-factor.variables)    
        if not are_sets_dconnected(step_model,list({node}), list(keys)):
          #effect = effect.divide(effect, inplace=False)
          argument.d_separated = node
          

      effect.normalize()
      argument.effects[node] = effect
    
  return argument
  

def compute_argument_strength(model, argument):
  """ Returns the implied logodds of the argument on its target outcome
  """
  if not hasattr(argument, 'effects'):
    argument = compute_argument_effects(model, argument)
    
  effect = argument.effects[argument.target[0]]
  return factor_to_logodds(effect, [argument.target])
  

# Argument finding
def all_simple_arguments(model, target, evidence,
                         length_limit=None):
  """ Returns all simple argumental chains from the evidence to the target
      If a length_limit is provided, paths with a longer length_limit are discarded.
  """
  target_node, target_state = target
  factor_graph = to_factor_graph(model)
  arguments = []
  for path in nx.all_simple_paths(factor_graph, target_node, evidence.keys()):

    # We skip paths over the length limit
    if length_limit is not None and len(path) > length_limit:
      continue

    # We skip paths whose intermediate nodes are observed
    if any(evidence_node in path[1:-1] for evidence_node in evidence.keys()):
      continue

    # Make an argument from the path
    argument = make_argument_from_stack(path)
    last_node = path[-1]
    argument.observations = {last_node : evidence[last_node]}
    argument.target = target

    # Compute argument effects
    argument = compute_argument_effects(model, argument)

    # Add argument to output
    arguments.append(argument)

  return arguments
  

def all_local_arguments(
    model, target, evidence,
    dependence_threshold = 10,
    path_length_limit = None,
    argument_complexity_limit = None,):
  """ model : a PGMPy bayesian network model
      evidence : {node : state} dictionary with observations
      target: (node, state) tuple with the outcome of interest
      dependence_threshold: sensitivity to interactions. A higher
                            threshold will result in more 
                            thorughly decomposed arguments
      path_length_limit: max length of any chain in an argument 
                         includes both variables and factors 
      argument_complexity_limit: max amount of components any argument
                                 will be made of in the first pass of
                                 the identification of proper arguments
                                 Note that more complex arguments can be 
                                 produced later when pruning pairwise 
                                 dependent arguments.

      returns a list of all proper, maximal arguments from evidence to target
      ordered by decreasing absolute strength
  """
  target_node, target_state = target

  simple_arguments = \
    all_simple_arguments(model, target, evidence, 
                         path_length_limit)

  proper_arguments = {} # map of argument edge hashes 
                        # to arguments with cached effects

  if argument_complexity_limit is None:
    argument_complexity_limit = len(simple_arguments)

  for components in limited_powerset(simple_arguments, 
                                     argument_complexity_limit):
    # Skip empty set of components
    if len(components) == 0: continue

    # Compute the union of the components
    argument = compose_arguments(components)

    # The union of components cannot contain loops
    if len(list(nx.simple_cycles(argument))) > 0:
      continue

    # Compute the effects of the union of components
    argument = compute_argument_effects(model, argument)
    total_effect = argument.effects[target_node]

    # Try to find a partition of the components
    # such that the effect of the total argument
    # equals the effect of the union of subarguments
    # formed by each group in the partition 
    components = list(components)
    for partition in partitions(components):
      
      # Skip trivial partition
      if len(partition) == 1: continue

      subarguments = [compose_arguments(subargument_components) 
                      for subargument_components in partition]
      
      # We retrieve the cached effects
      try:
        subarguments = \
          [proper_arguments[frozenset(subargument.edges)]
           for subargument in subarguments]
      except KeyError:
        # If a subargument is not in the cache is not proper.
        # We only need to check for combinations of 
        # proper subarguments
        continue

      subargument_effects = [subargument.effects[target_node] 
                              for subargument in subarguments]

      product_of_effects = reduce(lambda e1, e2 : e1*e2, 
                                  subargument_effects)
      
      # If the effect of the composite argument is the same as 
      # the subarguments then the argument is not proper
      #print(factor_distance(total_effect, product_of_effects))

      print(factor_distance(total_effect, product_of_effects))
        
      if factor_distance(total_effect, product_of_effects) < \
           dependence_threshold:
        break
    else:
      # If no partition can emulate the effect of the argument
      # Add the argument to the set of proper arguments
      proper_arguments[frozenset(argument.edges)] = argument
  
  arguments = proper_arguments.values()
  
  # Filter non-maximal arguments
  maximal_arguments = []
  for arg1 in arguments:
    for arg2 in arguments:
      if arg1 == arg2: continue
      if is_subargument(arg1, arg2):
        break
    else:
      maximal_arguments.append(arg1)
  arguments = maximal_arguments
  
  # Combine non-independent pairs of arguments
  # until all arguments are pairwise independent
  refinement_is_possible = True
  while refinement_is_possible:
    refinement_is_possible = False
    for arg1, arg2 in combinations(arguments,2):

      composite_argument = compose_arguments([arg1, arg2])

      # The union of components cannot contain loops
      if len(list(nx.simple_cycles(composite_argument))) > 0:
        continue

      composite_argument = \
        compute_argument_effects(model, composite_argument)
      
      try:
        composite_effect = composite_argument.effects[target_node]
      except KeyError:
        return [arg1, arg2, composite_argument]

      effect_product = \
        arg1.effects[target_node] * arg2.effects[target_node]

      if factor_distance(composite_effect, effect_product) > \
            dependence_threshold:
        arguments.remove(arg1)
        arguments.remove(arg2)
        arguments.append(composite_argument)
        refinement_is_possible = True
        break
  
  # Order arguments by decreasing absolute strength
  arguments.sort(key=lambda arg : 
                     abs(compute_argument_strength(model, arg)), 
                 reverse=True)

  return arguments
  

def random_argument(model, 
                    max_complexity = None, 
                    length_limit = None
                    ):
  """ Generate a random argument
  """
  
  simple_arguments = []
  while len(simple_arguments) == 0:
    target = random_outcome(model)
    evidence = random_evidence(model, model.nodes)
    simple_arguments = all_simple_arguments(model, 
                                            target, 
                                            evidence,
                                            length_limit)
                                          
  if max_complexity is None: 
    max_complexity = len(simple_arguments)
  elif max_complexity > len(simple_arguments): 
    max_complexity = len(simple_arguments)
    
  if max_complexity > 1:                                        
    n = np.random.randint(1, max_complexity)
  else: n = 1
  idx = np.random.choice(range(len(simple_arguments)), n, replace=False)
  components = [simple_arguments[i] for i in idx]
  
  argument = compose_arguments(components)
  argument = compute_argument_effects(model, argument)
  return argument