# Sample chapter

## Chapter Overview

Test fixture for the page-level checks and the fence runner in `tests/`. It is not a chapter and is not in the table of contents. It passes every check, so each test can be shown to fail on a deliberately broken copy. Section {ref}`sec-ch0-fences` holds the fence cases and Section {ref}`sec-ch0-figures` the figures.

## Learning Objectives

- Run a fence and compare its output with the block shown.
- Reproduce a figure from the fence that plots it.

## Hands-On Learning

The notebook next to this page works through the same average; see the admonition in the last numbered section.

(sec-ch0-fences)=
## 1. Fence cases

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

% output
```text
fetched
```

(sec-ch0-figures)=
## 2. Figures

A plotting fence must not trip the warnings-as-errors setting, and the figure it produces follows it. The page of a real chapter would cite a source here, for example the position of Demo {cite:t}`demo2020` that a ramp is the simplest trend, as in earlier work {cite}`demo2020`.

```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.show()
print("plotted 3 points")
```

% output

```text
plotted 3 points
```

```{figure} figures/ramp.*
:name: fig-ch0-ramp
:alt: A straight line rising from one to three.

A ramp, drawn by the fence above and written by `figures/make_figures.py`.
```

Figure {numref}`fig-ch0-ramp` shows the ramp. The schematic in {numref}`fig-ch0-schematic` is drawn from a Graphviz source.

```{figure} figures/schematic.*
:name: fig-ch0-schematic
:alt: Two boxes joined by an arrow, labeled input and output.

Input flows to output.
```

## 3. Shared function

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

## Summary

The page runs its fences, shows their output, and draws two figures.

- Every runnable fence prints a result and is followed by the output it produced.
- Every figure has a label, alt text, a caption, an SVG and a PDF.
- Every figure is referenced from the text.
- Every exercise has a solution that stays folded until the reader opens it.

## Exercises

1. Work out the mean of 1, 2, 3 and 4 by hand and compare it with the output above.

```{admonition} Solution 1
:class: dropdown

The sum is ten and there are four values, so the mean is 2.5.
```

## Further Reading

- The Graphviz documentation, for the schematic source.
- The Matplotlib documentation, for the settings that make a figure reproducible.
- The MyST documentation, for the figure and admonition directives.
- The pytest documentation, for how the checks are collected.
- The notebook beside this page, for the stored values.
