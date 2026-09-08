"""The curriculum this package shipped as its only one, preserved exactly.

Before lessons were flat, the ordering lived on the lessons themselves: a number
from 1 to 170 and a section key. That was one curriculum's opinion written onto
the material, and it is now written here instead, where it can be disagreed with.

Nothing else changed. ``order_hint`` carries the old number, so
``linearize()`` reproduces the old ``LESSON_CLASSES`` order exactly — there is a
test that asserts it against an order captured before the move.

There are no edges. The old numbering was a reading order, not a set of
prerequisite claims, and turning a hundred and seventy consecutive integers into
a hundred and sixty-nine "must come before" assertions would be inventing
structure that was never measured. :data:`~langcurriculum.curricula.PROGRESSIVE`
derives real edges from the declared axes; this one states only what was
actually known.
"""

from __future__ import annotations

from .graph import Curriculum, Node

__all__ = ["CORE170", "SUPPLEMENTARY", "CANONICAL"]

CORE170 = Curriculum(
    id="core170",
    title="the numbered curriculum",
    description=(
        "The 170 lessons of the original sequence, in their original order. "
        "A reading order rather than a dependency graph."
    ),
    nodes=(
        Node("symbol_grounding", order_hint=1),
        Node("symbol_equivalence", order_hint=2),
        Node("symbol_discrimination", order_hint=3),
        Node("sequence_copy", order_hint=4),
        Node("next_symbol", order_hint=5),
        Node("finite_state_language", order_hint=6),
        Node("context_free_language", order_hint=7),
        Node("parse_depth", order_hint=8),
        Node("tree_to_sequence", order_hint=9),
        Node("variable_binding", order_hint=10),
        Node("unification", order_hint=11),
        Node("predicate_logic", order_hint=12),
        Node("quantification", order_hint=13),
        Node("scope_ambiguity", order_hint=14),
        Node("compositional_reference", order_hint=15),
        Node("spatial_language", order_hint=16),
        Node("temporal_language", order_hint=17),
        Node("event_semantics", order_hint=18),
        Node("thematic_roles", order_hint=19),
        Node("instruction_following_micro", order_hint=20),
        Node("instruction_composition", order_hint=21),
        Node("pronoun_coreference", order_hint=22),
        Node("discourse_state", order_hint=23),
        Node("ellipsis", order_hint=24),
        Node("presupposition", order_hint=25),
        Node("implicature", order_hint=26),
        Node("speaker_listener_game", order_hint=27),
        Node("lexicon_induction", order_hint=28),
        Node("grammar_induction", order_hint=29),
        Node("few_shot_language_learning", order_hint=30),
        Node("translation", order_hint=31),
        Node("paraphrase", order_hint=32),
        Node("entailment", order_hint=33),
        Node("knowledge_update", order_hint=34),
        Node("belief_state", order_hint=35),
        Node("question_answering", order_hint=36),
        Node("question_generation", order_hint=37),
        Node("interactive_reference", order_hint=38),
        Node("definitions", order_hint=39),
        Node("concept_invention", order_hint=40),
        Node("analogy", order_hint=41),
        Node("causal_language", order_hint=42),
        Node("counterfactuals", order_hint=43),
        Node("planning_language", order_hint=44),
        Node("procedural_language", order_hint=45),
        Node("program_synthesis", order_hint=46),
        Node("program_explanation", order_hint=47),
        Node("dialogue_game", order_hint=48),
        Node("negotiation_game", order_hint=49),
        Node("deception_detection", order_hint=50),
        Node("social_convention_learning", order_hint=51),
        Node("document_world", order_hint=52),
        Node("compressed_language", order_hint=53),
        Node("noisy_channel_language", order_hint=54),
        Node("multimodal_symbolization", order_hint=55),
        Node("open_world_language", order_hint=56),
        Node("continual_language", order_hint=57),
        Node("language_culture", order_hint=58),
        Node("general_language_agent", order_hint=59),
        Node("natural_language_bridge", order_hint=60),
        Node("ontology_construction", order_hint=61),
        Node("ontology_revision", order_hint=62),
        Node("ontology_alignment", order_hint=63),
        Node("representation_selection", order_hint=64),
        Node("representation_invention", order_hint=65),
        Node("abstraction_ladder", order_hint=66),
        Node("conceptual_chunking", order_hint=67),
        Node("latent_rule_discovery", order_hint=68),
        Node("scientific_model_induction", order_hint=69),
        Node("experimental_design", order_hint=70),
        Node("theory_comparison", order_hint=71),
        Node("falsification", order_hint=72),
        Node("anomaly_resolution", order_hint=73),
        Node("mechanism_discovery", order_hint=74),
        Node("multiscale_modeling", order_hint=75),
        Node("emergence_discovery", order_hint=76),
        Node("invariance_discovery", order_hint=77),
        Node("symmetry_reasoning", order_hint=78),
        Node("conservation_law_discovery", order_hint=79),
        Node("dimensional_analysis", order_hint=80),
        Node("mathematical_definition_learning", order_hint=81),
        Node("conjecture_generation", order_hint=82),
        Node("theorem_proving", order_hint=83),
        Node("lemma_invention", order_hint=84),
        Node("proof_compression", order_hint=85),
        Node("proof_translation", order_hint=86),
        Node("counterexample_generation", order_hint=87),
        Node("logic_discovery", order_hint=88),
        Node("logic_selection", order_hint=89),
        Node("uncertain_symbolic_reasoning", order_hint=90),
        Node("default_reasoning", order_hint=91),
        Node("belief_revision", order_hint=92),
        Node("contradiction_tolerance", order_hint=93),
        Node("source_provenance", order_hint=94),
        Node("source_reliability_learning", order_hint=95),
        Node("argumentation", order_hint=96),
        Node("adversarial_argumentation", order_hint=97),
        Node("explanation", order_hint=98),
        Node("explanation_repair", order_hint=99),
        Node("teaching", order_hint=100),
        Node("curriculum_design", order_hint=101),
        Node("knowledge_gap_detection", order_hint=102),
        Node("problem_formulation", order_hint=103),
        Node("problem_reformulation", order_hint=104),
        Node("decomposition", order_hint=105),
        Node("hierarchical_planning", order_hint=106),
        Node("long_horizon_projects", order_hint=107),
        Node("resource_bounded_reasoning", order_hint=108),
        Node("anytime_reasoning", order_hint=109),
        Node("metareasoning", order_hint=110),
        Node("strategy_discovery", order_hint=111),
        Node("strategy_transfer", order_hint=112),
        Node("algorithm_discovery", order_hint=113),
        Node("algorithm_analysis", order_hint=114),
        Node("recursive_self_application", order_hint=115),
        Node("metalinguistic_reasoning", order_hint=116),
        Node("language_design", order_hint=117),
        Node("dsl_invention", order_hint=118),
        Node("compiler_construction", order_hint=119),
        Node("interpreter_learning", order_hint=120),
        Node("protocol_discovery", order_hint=121),
        Node("institution_learning", order_hint=122),
        Node("institution_design", order_hint=123),
        Node("norm_reasoning", order_hint=124),
        Node("contract_reasoning", order_hint=125),
        Node("mechanism_design", order_hint=126),
        Node("coalition_formation", order_hint=127),
        Node("distributed_knowledge", order_hint=128),
        Node("collective_theory_building", order_hint=129),
        Node("cultural_evolution", order_hint=130),
        Node("historical_reconstruction", order_hint=131),
        Node("narrative_modeling", order_hint=132),
        Node("multi_perspective_modeling", order_hint=133),
        Node("identity_continuity", order_hint=134),
        Node("self_model", order_hint=135),
        Node("capability_estimation", order_hint=136),
        Node("self_error_diagnosis", order_hint=137),
        Node("self_repair", order_hint=138),
        Node("architecture_selection", order_hint=139),
        Node("architecture_composition", order_hint=140),
        Node("tool_construction", order_hint=141),
        Node("external_memory_design", order_hint=142),
        Node("knowledge_refactoring", order_hint=143),
        Node("semantic_compression", order_hint=144),
        Node("minimum_description_learning", order_hint=145),
        Node("open_ended_concept_discovery", order_hint=146),
        Node("open_ended_question_generation", order_hint=147),
        Node("research_program", order_hint=148),
        Node("paradigm_shift", order_hint=149),
        Node("world_model_synthesis", order_hint=150),
        Node("cross_domain_unification", order_hint=151),
        Node("theory_transfer", order_hint=152),
        Node("formalization", order_hint=153),
        Node("deformalization", order_hint=154),
        Node("ambiguity_preservation", order_hint=155),
        Node("underspecification_reasoning", order_hint=156),
        Node("value_learning", order_hint=157),
        Node("multi_objective_reasoning", order_hint=158),
        Node("goal_inference", order_hint=159),
        Node("goal_revision", order_hint=160),
        Node("goal_generation", order_hint=161),
        Node("reflective_goal_reasoning", order_hint=162),
        Node("civilization_simulator", order_hint=163),
        Node("scientific_civilization", order_hint=164),
        Node("symbolic_world_builder", order_hint=165),
        Node("curriculum_invention", order_hint=166),
        Node("universal_interface_transfer", order_hint=167),
        Node("unknown_game", order_hint=168),
        Node("symbolic_generalist", order_hint=169),
        Node("open_world_research_agent", order_hint=170),
    ),
)

SUPPLEMENTARY = Curriculum(
    id="supplementary",
    title="supplementary syntax and semantics",
    description=(
        "Lessons that sat outside the numbered sequence: sharper, narrower "
        "probes that were never meant to be walked in order."
    ),
    nodes=(
        Node("center_embedding", order_hint=0),
        Node("comparatives", order_hint=1),
        Node("counting_quantifier", order_hint=2),
        Node("expression_eval", order_hint=3),
        Node("long_range_agreement", order_hint=4),
        Node("negation", order_hint=5),
        Node("nesting_depth_compare", order_hint=6),
        Node("palindrome", order_hint=7),
        Node("set_operations", order_hint=8),
        Node("string_reversal", order_hint=9),
    ),
)

#: Both of the above, in the order the package used to declare them: the numbered
#: sequence, then the supplementary lessons. This is the only shipped curriculum
#: that covers the whole registry in a considered order, which is why the site
#: reads it — an index that omits ten lessons is an index with ten unreachable
#: pages.
CANONICAL = Curriculum(
    id="canonical",
    title="every lesson, in the order the package used to have",
    description=(
        "The numbered sequence followed by the supplementary lessons. Complete, "
        "and still only one opinion: see 'progressive' for one derived from the "
        "difficulty axes instead."
    ),
    nodes=tuple(list(CORE170.nodes)
                + [Node(n.lesson, order_hint=1000 + n.order_hint)
                   for n in SUPPLEMENTARY.nodes]),
)
