# factor_graph_algorithm
This is our enhanced version of the factor gragh algorithm based on the explainbn package from J. Sevilla.

With some evidence and a query node, the algorithm can generate arguments. Besides, the outcomes contain explanation in words for arguments. Here is an example of given Quinn as evidence, we would like to know the influence on Spider in the Bayesian Belief Network: The Spider Network.

```
from explainbn.arguments import all_local_arguments
from explainbn.explanations import explain_argument
from pgmpy.readwrite import XMLBIFReader
reader = XMLBIFReader("spider.xml")
model = reader.get_model()

target = ('Spider','False')
evidence = {'Quinns':'False'}


# Find all arguments
arguments = all_local_arguments(model, target, evidence)

# Generate a textual explanation of each argument
for argument in arguments:
  explanation = explain_argument(model, argument)
  print(explanation)
  print("")
```

```
We have observed that Quinns is False.
That Quinns is False is evidence that Spider is True (weak 0.10008343850685929 inference).

We have observed that Quinns is False.
That Quinns is False is evidence that Both is False or Both is True (certain inf inference).
That Both is False or Both is True is evidence that Spider is False or Spider is True (certain inf inference).
```
