# factor_graph_algorithm
This is our enhanced version of the factor gragh algorithm based on the explainbn package from J. Sevilla (https://gitlab.nl4xai.eu/jaime.sevilla/explainbn/-/tree/main).

Given some evidence and a query node, the algorithm can generate arguments. Additionally, the outcomes contain explanation in words for arguments. Here is an example of given Quinn, Emerson and Sawyer as evidence, we would like to know the influence on Spider in the Bayesian Belief Network: The Spider Network.

```python
from explainbn.arguments import all_local_arguments
from explainbn.explanations import explain_argument
from pgmpy.readwrite import XMLBIFReader
reader = XMLBIFReader("spider.xml")
model = reader.get_model()

target = ('Spider','False')
evidence = {'Quinns':'False', 'Emersons':'False', 'Sawyer':'True'}


# Find all arguments
arguments = all_local_arguments(model, target, evidence)

# Generate a textual explanation of each argument
for argument in arguments:
  explanation = explain_argument(model, argument,evidence)
  print(explanation)
  print("")
```

```
We have observed that Emersons is False.
That Emersons is False is evidence that Spider is False (strong inference).

We have observed that Quinns is False.
That Quinns is False is evidence that Spider is False (strong inference).

We have observed that Quinns is False.
That Quinns is False is evidence that Both is False (strong inference).
That Both is False is evidence that Spider is False or Spider is True (equal effect inference).
Because Spider and Both are d-separated, this argument alone cannot influence the target node.

We have observed that Emersons is False.
That Emersons is False is evidence that Both is False (strong inference).
That Both is False is evidence that Spider is False or Spider is True (equal effect inference).
Because Spider and Both are d-separated, this argument alone cannot influence the target node.

We have observed that Sawyer is True.
That Sawyer is True is evidence that Spider is True (strong inference).
```

For more information, please refer to our papers available at the following link:
