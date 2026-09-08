"""``open_world_research_agent`` — discover, theorize, experiment, revise and report over a long horizon.

Ultimate transfer and open-world capstones.
"""

from __future__ import annotations

import random

from ..lesson import Lesson, LessonNotImplemented
from ..generators.capstone import _OPEN_WORLD_NOTE


def gen_open_world_research_agent(rng: random.Random):
    """Deliberately unimplemented, and kept in the registry saying so.

    This lesson's content is a loop, not a question, and a single-step version
    of it would grade something easier while wearing this name. The full
    reasoning is on the class as ``note``; raising is the honest behaviour.
    """
    raise LessonNotImplemented(_OPEN_WORLD_NOTE)


class OpenWorldResearchAgent(Lesson):
    """Discover, theorize, experiment, revise and report over a long horizon."""

    id = "open_world_research_agent"
    level = 170
    tags = ("transfer", "capstone", "open-world")
    teaches = "discover, theorize, experiment, revise and report over a long horizon"
    capabilities = ('open_ended_discovery', 'scientific_induction', 'metareasoning', 'self_modeling', 'architecture_adaptation')
    axes = {'lexical_novelty': 5, 'world_complexity': 5, 'reasoning_depth': 5, 'discourse_horizon': 5, 'ambiguity': 5}
    status = "spec"
    note = "Not reducible to a one-step exactly-graded episode, and not faked. The lesson's content is a *loop*: discover an ontology, acquire the language that describes it, propose competing theories, DESIGN an experiment, run it, revise under criticism, and report with provenance. Everything that distinguishes it from lessons 150-169 lives in the parts a single question cannot contain: (1) the agent must choose interventions, so the environment has to accept experiment programs as actions and return outcomes, i.e. a multi-step interactive environment with a hidden generative world model rather than a generate()->answer pair; (2) the score is a trajectory functional \u2014 sample efficiency, whether each claim is supported by an experiment the agent itself ran, calibration of stated confidence, and whether a refuted theory is actually abandoned \u2014 none of which is a function of one answer; (3) grading requires an adversarial critic and a provenance ledger, both of which are additional environments. A real implementation needs: a parameterized hidden world (latent laws + nuisance parameters) with an exact simulator; an action language for experiments, assertions and retractions; an evaluator that checks each asserted law against the true one and each claim against the agent's own evidence ledger; and a scalar built from (discovery completeness x provenance validity x calibration) over the whole trajectory. Until that harness exists, any single-step version would grade a different, easier lesson while wearing this one's name."

    generate = staticmethod(gen_open_world_research_agent)
