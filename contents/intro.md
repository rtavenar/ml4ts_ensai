# Introduction

When learning from structured data, special attention has to be paid to the
models used.
Indeed, designing machine learning models requires to think of the invariants
to be learned {cite}`arjovsky2019invariant`, and either encode them in the model
or design the model so that it is able to discover such invariants and encode
them.

In this course, we will focus on time series and will dig into two main ways
of encoding / learning these invariants.
First, we will cover the design of alignment-based metrics that tackle the
problem of (temporal) localization invariance.
Standard similarity measures will be introduced and their use at the core of
machine learning models will be discussed.
Second, we will discuss standard neural network architectures and the kind of
invariants they encode.


## References

```{bibliography}
:filter: docname in docnames
```
