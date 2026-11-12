# Sample chapter

Test fixture for the fence runner in `tests/test_fences.py`. It is not a chapter and is not in the table of contents.

A plain fence with an exact output block (the float is compared to one unit in the last digit shown):

```python
print(7)
print(3.14159)
print("done")
```

% output

```text
7
3.1416
done
```

A fence whose output is compared by shape only:

```python
print([1.5, 2.5])
print("loss 0.123")
```

% output-shape

```text
[9.9, 8.1]
loss 4.5
```

Pseudo-code is never run:

```python
# pseudo-code
model = load("something that does not exist"
```

Network fences are left to the live tests:

```python
# network
raise RuntimeError("network fence ran")
```

A plotting fence must not trip the warnings-as-errors setting:

```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.show()
```

A function the page shares with its notebook:

```python
def average(xs):
    """Arithmetic mean of a list of numbers."""
    return sum(xs) / len(xs)


print(average([1, 2, 3, 4]))
```

% output

```text
2.5
```

```{admonition} The notebook
The notebook section {ref}`nb-sample-mean` reports a mean of 2.5 over the stored values. It also lists every value, which needs no claim.
```
