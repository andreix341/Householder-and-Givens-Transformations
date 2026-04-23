# Householder and Givens Transformations Visualization

A Python GUI application for visualizing Householder reflections and Givens rotations, built with Tkinter and Matplotlib.

## Overview

This project demonstrates two fundamental orthogonal transformation techniques used in numerical linear algebra:

- **Householder Reflections**: Reflect a vector across a hyperplane defined by a Householder vector `v`, with an option to compute `v⊥` via two methods
- **Givens Rotations**: Rotate a vector in the plane spanned by two basis vectors using cosine/sine parameters

## Features

- Interactive GUI for inputting vectors and transformation parameters
- Support for 2D and 3D visualizations
- Toggle between Householder and Givens transformations
- Method selection for computing perpendicular vectors (Householder)
- Real-time 3D plotting with Matplotlib

## Requirements

```
numpy
matplotlib
tkinter (included with Python)
```

## Transformations

### Householder Reflection

Reflects vector `x` across a hyperplane, with reflection direction controlled by index `k`. Supports two methods for computing `v⊥`:

- **Method1**: Uses cross product with coordinate basis vectors
- **Method2**: Projects `x` onto `v` to find perpendicular component

### Givens Rotation

Rotates vector `x` in the `(i, k)` plane using computed cos/sin values. Useful for zeroing out specific vector components.

