# Basics of strtree

**strtree** is a Python package for binary and multiclass classification of strings. It is based on regular expressions automatically placed in a binary tree.

Github repo: [stretree](https://github.com/AntonSarr/strtree)

With strtree you can:

- Do a binary and multiclass classification of your strings using automatically extracted regular expressions
- Find shortest regular expressions which cover strings with positive labels in the most accurate way

Look at a quick example.

## Example
Firstly, let's build a tree from strings and their labels.
```python
import strtree


strings = ['Samsung X-500', 'Samsung SM-10', 'Samsung X-1100', 'Samsung F-10', 'Samsung X-2200',
           'AB Nokia 1', 'DG Nokia 2', 'THGF Nokia 3', 'SFSD Nokia 4', 'Nokia XG', 'Nokia YO']
labels = [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]

tree = strtree.StringTree()
tree.build(strings, labels, min_precision=0.75, min_token_length=1)
```
Let's see what regular expressions were extracted.
```python
for leaf in tree.leaves:
    print(leaf)

# Output:
# PatternNode(".+ .+a.+", right=None, left=PatternNode(.+0.+), n_strings=11, precision=1.0, recall=0.57)
# PatternNode(".+0.+", right=None, left=None, n_strings=7, precision=1.0, recall=1.0)
```
You may need to check the precision and recall of the whole tree for a given set of strings and true labels.
```python
print('Precision: {}'.format(tree.precision_score(strings, labels)))
# Precision: 1.0

print('Recall: {}'.format(tree.recall_score(strings, labels)))
# Recall: 1.0
```
Finally, you can pass any strings you want and see if they match to extracted regular expressions or not.
```python
matches = tree.match(other_strings)

# You will receive a vector of the same size as other_strings containing 0's (no match) or 1's (match)
```

# Installing
2. Use PyPI:
`pip install strtree`
1. Use a distribution file located in the `dist` folder: 
`pip install strtree-x.x.x-py3-none-any.whl`

# Contribution

You are very welcome to participate in the project. You may solve the current issues or add new functionality - it is up to you.
