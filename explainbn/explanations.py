#!/usr/bin/env python3

"""
Facilities to explain arguments
with natural language explanations 
and visualizations

The main function is explain_argument(model, argument)
which produces a textual explanation of an argument.
"""

import warnings

import numpy as np

from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete.CPD import TabularCPD
from gensim.matutils import hellinger

from .arguments import compute_step_effect, compute_argument_effects
from .utilities import iterate_argument, factor_to_outcomes, get_factor_from_scope, factor_argmax

def explain_step(observed, model, rule, premises, conclusion, 
                 mode='direct', include_qualifiers=True):
  """
  Produce a textual explanation of an inference step
  of an argument.

  Available modes include:
      - 'contrastive' - produces a contrastive explanation
                        of intercausal reasoning steps.
      - 'direct' - disables contrastive explanations.
  
  Set include_qualifiers to False to omit the inference strength.  
  """
  # Identify type of relation
  if all(premise in model.predecessors(conclusion)
         for premise in premises.keys()):
    type_inference = 'causal'
  elif len(premises) == 1:
    type_inference = 'evidential'
  else:
    type_inference = 'intercausal'

  # Compute link effect
  effect = compute_step_effect(model, rule, premises, conclusion,observed)
  
  # Generate a description of the premises
  premises_description = [explain_evidence(model, premise_effect)[0]
                          for premise_effect in premises.values()]
  premises_description = describe_list(premises_description)

  # Generate a description of the favored outcomes
  conclusion_description, qualifier = explain_evidence(model, effect)
  
  # Build a counterfactual for intercausal reasoning
  if type_inference == 'intercausal':
    # Identify child node
    child = [premise for premise in premises.keys()
             if premise in model.successors(conclusion)][0]
             
    # Build descriptions of the child and parent premises
    child_description, _ = explain_evidence(model, premises[child])
    
    parents_description = [explain_evidence(model, parent_effect)[0]
                           for parent, parent_effect in premises.items()
                           if parent != child]
    parents_description = describe_list(parents_description)
    
    # Compute counterfactual effect
    counterfactual_premise = {child : premises[child]}
    counterfactual_effect =\
      compute_step_effect(model, rule, 
                          counterfactual_premise, 
                          conclusion,observed)

    counterfactual_outcome_description, _ =\
      explain_evidence(model, counterfactual_effect)
    
    # Compare counterfactual to actual effect
    actual_outcome, actual_strength =\
      factor_to_outcomes(effect)
    counterfactual_outcome, counterfactual_strength =\
      factor_to_outcomes(counterfactual_effect)
    
    if counterfactual_outcome != actual_outcome:
      factual_description = f"we infer {conclusion_description} instead"
    elif np.abs(actual_strength) > np.abs(counterfactual_strength):
      factual_description = "we can be more certain than this is the case"
    else:
      factual_description = "we cannot be as certain that this is the case"

  # Build explanation
  if type_inference == 'causal':
    explanation = \
      f"That {premises_description} causes that {conclusion_description}"
  elif type_inference == 'evidential' or mode == 'direct':
    explanation = \
      f"That {premises_description} is evidence that {conclusion_description}"
  else: # if type_inference is 'intercausal':
    explanation = \
      f"Usually, if {child_description} then {counterfactual_outcome_description}.  \n" + \
      f"Since {parents_description}, {factual_description}"
      
  if include_qualifiers:
     explanation += f" ({qualifier} inference)."
  else:
     explanation += "."

  return explanation


def explain_argument(model, argument, evidence,
      mode='direct', 
      include_qualifiers=True, 
      return_list = False
    ):
  """
  Produce a textual explanation of an argument.
  Has three modes:
  
  - 'direct' - A step-by-step explanation of the argument
  - 'contrastive' - An explanation alluding to counterfactuals
  - 'overview' - A succint explanation listing the premises and the conclusion
  
  Set include_qualifiers to False to remove the inference strength qualifiers.
  
  Set return_list to True to receive a list of step-by-step explanations.
  
  Input: PGMPy model, factor argument, string mode, boolean include_qualifiers
  Output: String or list of strings
  """
  observed = [node for node, _ in argument.observations.items()]
  # If the argument effects haven't yet been computed
  if not hasattr(argument, "effects"):
    argument = compute_argument_effects(model, argument)

  # If the model has not been extended with descriptions
  artificial_extension = False
  if not hasattr(model, "descriptions"):
    artificial_extension = True
    warnings.warn("Warning. The model has no state or node descriptions. Automatically generated descriptions will be used instead.")
    model.descriptions = {(node, state) : f"{node} is {state}" 
                          for node in model.nodes 
                          for state in model.states[node]}
    
    model.descriptions.update({node : node for node in model.nodes})
    
  
  # Generate description of the observations
  observations_description =\
      describe_list([model.descriptions[(node, state)] 
                     for node, state 
                     in argument.observations.items()])
  
  # In this mode we provide a very short explanation
  if mode == 'overview':
    
    argument_effect = argument.effects[argument.target[0]]
    conclusion_description, strength_qualifier = explain_evidence(model, argument_effect)
    strength_qualifier_text = f" ({strength_qualifier} inference)"
    explanation = f"Since {observations_description}, we infer that {conclusion_description}{strength_qualifier_text if include_qualifiers else ''}."
    
    if return_list: return [explanation]
    else: return explanation
  
  # In the other modes we provide a step-by-step explanation
  valuation = {}
  explanation = [f"We have observed that {observations_description}."]

        
  infer = VariableElimination(model)



  for node in iterate_argument(argument):

    # We already explained the observations
    if node in argument.observations:
      node_previous = node
      continue
    
    # Explain other nodes
    else:
      # Find rules that have this node as conclusion
      rules_scopes = list(argument.predecessors(node))

      # Explain each rule
      for rule_scope in rules_scopes:
        # Retrive rule
        rule = get_factor_from_scope(model, rule_scope)

        # Retrieve premises of the rule
        premises = {premise_node : argument.effects[premise_node]
                    for premise_node 
                    in argument.predecessors(rule_scope)}

        # Generate explanation of the inference step
        step_explanation =\
          explain_step(observed, model, rule, premises, node, mode, include_qualifiers)
        
        # Accumulate the explanations
        explanation.append(step_explanation)


        q = infer.query(variables=[node],evidence=evidence)
        dist = hellinger(argument.effects[node].values, q.values)
        #explanation.append(f"The distance of {node} is {dist}.")
        
        if node == argument.d_separated:
          explanation.append(f"Because {node} and {node_previous} are d-separated, this argument alone cannot influence the target node.")
          #break
             
        node_previous = node
      # If multiple rules are involved, explain the cumulative effect
      if len(rules_scopes) > 1:
        outcome_description, strength_qualifier = explain_evidence(model, argument.effects[node])
        strength_qualifier_text = f" ({strength_qualifier} inference)"
        cumulative_effect_explanation = f"All in all, this is evidence that {outcome_description}{strength_qualifier_text if include_qualifiers else ''}."
        explanation.append(cumulative_effect_explanation)
  
  # If we added artificial descriptions we delete them
  if artificial_extension:
    del model.descriptions
  
  # Join the explanations
  if not return_list:
    explanation = "\n".join(explanation)
  
  return explanation

def explain_evidence(model, factor, threshold = 0):
  """ Returns a description in natural language of 
      a factor representing evidence about a single node
      and a logodd score for the strength of the evidence
  """

  outcomes, strength = factor_to_outcomes(factor, threshold)

  # Translate the outcomes into descriptions
  outcomes_descriptions =\
    [model.descriptions[outcome] for outcome in outcomes]
  explanation = describe_list(outcomes_descriptions, connector = "or")
  
  strength_description = describe_strength(strength) 
  
  return explanation, strength_description
  
# Explanation utilities
def describe_list(input_list, connector = 'and'):
  """ Returns a string with the enumeration of the list elements
      ['one', 'two', 'three'] => 'one, two and three'
  """
  if len(input_list) > 1:
    input_list[-2] = f"{input_list[-2]} {connector} {input_list[-1]}"
    input_list.pop()
  return ", ".join(input_list)
"""
def describe_strength(strength):
  qualifier = \
    'certain' if np.abs(strength) > 10.0 else \
    'strong' if np.abs(strength) > 1.0 else \
    'moderate' if np.abs(strength) > 0.5 else \
    'weak' if np.abs(strength) > 0.1 else \
    'tenuous'
  qualifier = qualifier+str(' ')+str(strength)
  return qualifier
"""
def describe_strength(strength):
  qualifier = \
    'equal effect' if np.abs(strength) == np.Inf else \
    'certain' if np.abs(strength) > 10.0 else \
    'strong' if np.abs(strength) > 1.0 else \
    'moderate' if np.abs(strength) > 0.5 else \
    'weak' if np.abs(strength) > 0.1 else \
    'tenuous'
  #qualifier = qualifier if np.abs(strength) == np.Inf else qualifier+str(' ')+"{:.2f}".format(strength)
  return qualifier