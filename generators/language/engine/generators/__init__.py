"""Shared machinery the lesson generators draw on.

Every module here is private. It holds the constructions that more than one
lesson needs — scene builders, nonce-word factories, proof search, small
solvers — so that a lesson module contains the lesson and nothing else.

The modules are grouped by the kind of world they build — ontologies, causal
graphs, proof calculi — not by any curriculum's ordering, which lives in
:mod:`langcurriculum.curricula`. :mod:`langcurriculum.generators.base` holds the
vocabulary and scene helpers the whole curriculum shares.
"""
