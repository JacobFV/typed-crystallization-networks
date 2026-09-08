"""Every lesson, one module each, in one flat namespace.

A lesson does not know where it sits in any curriculum. Ordering, grouping and
prerequisites are properties of a :class:`~langcurriculum.curricula.Curriculum`,
of which there may be many drawing on the same lessons, so the only order this
module has is alphabetical — which is to say, none.

The imports are written out rather than discovered by scanning the directory, so
what is in the registry is a fact you can read off the source. A test checks the
two agree, which catches both a module that failed to import and a module nobody
remembered to list.
"""

from __future__ import annotations

from .abstraction_ladder import AbstractionLadder
from .adversarial_argumentation import AdversarialArgumentation
from .algorithm_analysis import AlgorithmAnalysis
from .algorithm_discovery import AlgorithmDiscovery
from .ambiguity_preservation import AmbiguityPreservation
from .analogy import Analogy
from .anomaly_resolution import AnomalyResolution
from .anytime_reasoning import AnytimeReasoning
from .architecture_composition import ArchitectureComposition
from .architecture_selection import ArchitectureSelection
from .argumentation import Argumentation
from .belief_revision import BeliefRevision
from .belief_state import BeliefState
from .capability_estimation import CapabilityEstimation
from .causal_language import CausalLanguage
from .center_embedding import CenterEmbedding
from .civilization_simulator import CivilizationSimulator
from .coalition_formation import CoalitionFormation
from .collective_theory_building import CollectiveTheoryBuilding
from .comparatives import Comparatives
from .compiler_construction import CompilerConstruction
from .compositional_reference import CompositionalReference
from .compressed_language import CompressedLanguage
from .concept_invention import ConceptInvention
from .conceptual_chunking import ConceptualChunking
from .conjecture_generation import ConjectureGeneration
from .conservation_law_discovery import ConservationLawDiscovery
from .context_free_language import ContextFreeLanguage
from .continual_language import ContinualLanguage
from .contract_reasoning import ContractReasoning
from .contradiction_tolerance import ContradictionTolerance
from .counterexample_generation import CounterexampleGeneration
from .counterfactuals import Counterfactuals
from .counting_quantifier import CountingQuantifier
from .cross_domain_unification import CrossDomainUnification
from .cultural_evolution import CulturalEvolution
from .curriculum_design import CurriculumDesign
from .curriculum_invention import CurriculumInvention
from .deception_detection import DeceptionDetection
from .decomposition import Decomposition
from .default_reasoning import DefaultReasoning
from .definitions import Definitions
from .deformalization import Deformalization
from .dialogue_game import DialogueGame
from .dimensional_analysis import DimensionalAnalysis
from .discourse_state import DiscourseState
from .distributed_knowledge import DistributedKnowledge
from .document_world import DocumentWorld
from .dsl_invention import DslInvention
from .ellipsis import Ellipsis
from .emergence_discovery import EmergenceDiscovery
from .entailment import Entailment
from .event_semantics import EventSemantics
from .experimental_design import ExperimentalDesign
from .explanation import Explanation
from .explanation_repair import ExplanationRepair
from .expression_eval import ExpressionEval
from .external_memory_design import ExternalMemoryDesign
from .falsification import Falsification
from .few_shot_language_learning import FewShotLanguageLearning
from .finite_state_language import FiniteStateLanguage
from .formalization import Formalization
from .general_language_agent import GeneralLanguageAgent
from .goal_generation import GoalGeneration
from .goal_inference import GoalInference
from .goal_revision import GoalRevision
from .grammar_induction import GrammarInduction
from .hierarchical_planning import HierarchicalPlanning
from .historical_reconstruction import HistoricalReconstruction
from .identity_continuity import IdentityContinuity
from .implicature import Implicature
from .institution_design import InstitutionDesign
from .institution_learning import InstitutionLearning
from .instruction_composition import InstructionComposition
from .instruction_following_micro import InstructionFollowingMicro
from .interactive_reference import InteractiveReference
from .interpreter_learning import InterpreterLearning
from .invariance_discovery import InvarianceDiscovery
from .knowledge_gap_detection import KnowledgeGapDetection
from .knowledge_refactoring import KnowledgeRefactoring
from .knowledge_update import KnowledgeUpdate
from .language_culture import LanguageCulture
from .language_design import LanguageDesign
from .latent_rule_discovery import LatentRuleDiscovery
from .lemma_invention import LemmaInvention
from .lexicon_induction import LexiconInduction
from .logic_discovery import LogicDiscovery
from .logic_selection import LogicSelection
from .long_horizon_projects import LongHorizonProjects
from .long_range_agreement import LongRangeAgreement
from .mathematical_definition_learning import MathematicalDefinitionLearning
from .mechanism_design import MechanismDesign
from .mechanism_discovery import MechanismDiscovery
from .metalinguistic_reasoning import MetalinguisticReasoning
from .metareasoning import Metareasoning
from .minimum_description_learning import MinimumDescriptionLearning
from .multi_objective_reasoning import MultiObjectiveReasoning
from .multi_perspective_modeling import MultiPerspectiveModeling
from .multimodal_symbolization import MultimodalSymbolization
from .multiscale_modeling import MultiscaleModeling
from .narrative_modeling import NarrativeModeling
from .natural_language_bridge import NaturalLanguageBridge
from .negation import Negation
from .negotiation_game import NegotiationGame
from .nesting_depth_compare import NestingDepthCompare
from .next_symbol import NextSymbol
from .noisy_channel_language import NoisyChannelLanguage
from .norm_reasoning import NormReasoning
from .ontology_alignment import OntologyAlignment
from .ontology_construction import OntologyConstruction
from .ontology_revision import OntologyRevision
from .open_ended_concept_discovery import OpenEndedConceptDiscovery
from .open_ended_question_generation import OpenEndedQuestionGeneration
from .open_world_language import OpenWorldLanguage
from .open_world_research_agent import OpenWorldResearchAgent
from .palindrome import Palindrome
from .paradigm_shift import ParadigmShift
from .paraphrase import Paraphrase
from .parse_depth import ParseDepth
from .planning_language import PlanningLanguage
from .predicate_logic import PredicateLogic
from .presupposition import Presupposition
from .problem_formulation import ProblemFormulation
from .problem_reformulation import ProblemReformulation
from .procedural_language import ProceduralLanguage
from .program_explanation import ProgramExplanation
from .program_synthesis import ProgramSynthesis
from .pronoun_coreference import PronounCoreference
from .proof_compression import ProofCompression
from .proof_translation import ProofTranslation
from .protocol_discovery import ProtocolDiscovery
from .quantification import Quantification
from .question_answering import QuestionAnswering
from .question_generation import QuestionGeneration
from .recursive_self_application import RecursiveSelfApplication
from .reflective_goal_reasoning import ReflectiveGoalReasoning
from .representation_invention import RepresentationInvention
from .representation_selection import RepresentationSelection
from .research_program import ResearchProgram
from .resource_bounded_reasoning import ResourceBoundedReasoning
from .scientific_civilization import ScientificCivilization
from .scientific_model_induction import ScientificModelInduction
from .scope_ambiguity import ScopeAmbiguity
from .self_error_diagnosis import SelfErrorDiagnosis
from .self_model import SelfModel
from .self_repair import SelfRepair
from .semantic_compression import SemanticCompression
from .sequence_copy import SequenceCopy
from .set_operations import SetOperations
from .social_convention_learning import SocialConventionLearning
from .source_provenance import SourceProvenance
from .source_reliability_learning import SourceReliabilityLearning
from .spatial_language import SpatialLanguage
from .speaker_listener_game import SpeakerListenerGame
from .strategy_discovery import StrategyDiscovery
from .strategy_transfer import StrategyTransfer
from .string_reversal import StringReversal
from .symbol_discrimination import SymbolDiscrimination
from .symbol_equivalence import SymbolEquivalence
from .symbol_grounding import SymbolGrounding
from .symbolic_generalist import SymbolicGeneralist
from .symbolic_world_builder import SymbolicWorldBuilder
from .symmetry_reasoning import SymmetryReasoning
from .teaching import Teaching
from .temporal_language import TemporalLanguage
from .thematic_roles import ThematicRoles
from .theorem_proving import TheoremProving
from .theory_comparison import TheoryComparison
from .theory_transfer import TheoryTransfer
from .tool_construction import ToolConstruction
from .translation import Translation
from .tree_to_sequence import TreeToSequence
from .uncertain_symbolic_reasoning import UncertainSymbolicReasoning
from .underspecification_reasoning import UnderspecificationReasoning
from .unification import Unification
from .universal_interface_transfer import UniversalInterfaceTransfer
from .unknown_game import UnknownGame
from .value_learning import ValueLearning
from .variable_binding import VariableBinding
from .world_model_synthesis import WorldModelSynthesis

#: every lesson class, alphabetically by module
LESSON_CLASSES = (
    AbstractionLadder,
    AdversarialArgumentation,
    AlgorithmAnalysis,
    AlgorithmDiscovery,
    AmbiguityPreservation,
    Analogy,
    AnomalyResolution,
    AnytimeReasoning,
    ArchitectureComposition,
    ArchitectureSelection,
    Argumentation,
    BeliefRevision,
    BeliefState,
    CapabilityEstimation,
    CausalLanguage,
    CenterEmbedding,
    CivilizationSimulator,
    CoalitionFormation,
    CollectiveTheoryBuilding,
    Comparatives,
    CompilerConstruction,
    CompositionalReference,
    CompressedLanguage,
    ConceptInvention,
    ConceptualChunking,
    ConjectureGeneration,
    ConservationLawDiscovery,
    ContextFreeLanguage,
    ContinualLanguage,
    ContractReasoning,
    ContradictionTolerance,
    CounterexampleGeneration,
    Counterfactuals,
    CountingQuantifier,
    CrossDomainUnification,
    CulturalEvolution,
    CurriculumDesign,
    CurriculumInvention,
    DeceptionDetection,
    Decomposition,
    DefaultReasoning,
    Definitions,
    Deformalization,
    DialogueGame,
    DimensionalAnalysis,
    DiscourseState,
    DistributedKnowledge,
    DocumentWorld,
    DslInvention,
    Ellipsis,
    EmergenceDiscovery,
    Entailment,
    EventSemantics,
    ExperimentalDesign,
    Explanation,
    ExplanationRepair,
    ExpressionEval,
    ExternalMemoryDesign,
    Falsification,
    FewShotLanguageLearning,
    FiniteStateLanguage,
    Formalization,
    GeneralLanguageAgent,
    GoalGeneration,
    GoalInference,
    GoalRevision,
    GrammarInduction,
    HierarchicalPlanning,
    HistoricalReconstruction,
    IdentityContinuity,
    Implicature,
    InstitutionDesign,
    InstitutionLearning,
    InstructionComposition,
    InstructionFollowingMicro,
    InteractiveReference,
    InterpreterLearning,
    InvarianceDiscovery,
    KnowledgeGapDetection,
    KnowledgeRefactoring,
    KnowledgeUpdate,
    LanguageCulture,
    LanguageDesign,
    LatentRuleDiscovery,
    LemmaInvention,
    LexiconInduction,
    LogicDiscovery,
    LogicSelection,
    LongHorizonProjects,
    LongRangeAgreement,
    MathematicalDefinitionLearning,
    MechanismDesign,
    MechanismDiscovery,
    MetalinguisticReasoning,
    Metareasoning,
    MinimumDescriptionLearning,
    MultiObjectiveReasoning,
    MultiPerspectiveModeling,
    MultimodalSymbolization,
    MultiscaleModeling,
    NarrativeModeling,
    NaturalLanguageBridge,
    Negation,
    NegotiationGame,
    NestingDepthCompare,
    NextSymbol,
    NoisyChannelLanguage,
    NormReasoning,
    OntologyAlignment,
    OntologyConstruction,
    OntologyRevision,
    OpenEndedConceptDiscovery,
    OpenEndedQuestionGeneration,
    OpenWorldLanguage,
    OpenWorldResearchAgent,
    Palindrome,
    ParadigmShift,
    Paraphrase,
    ParseDepth,
    PlanningLanguage,
    PredicateLogic,
    Presupposition,
    ProblemFormulation,
    ProblemReformulation,
    ProceduralLanguage,
    ProgramExplanation,
    ProgramSynthesis,
    PronounCoreference,
    ProofCompression,
    ProofTranslation,
    ProtocolDiscovery,
    Quantification,
    QuestionAnswering,
    QuestionGeneration,
    RecursiveSelfApplication,
    ReflectiveGoalReasoning,
    RepresentationInvention,
    RepresentationSelection,
    ResearchProgram,
    ResourceBoundedReasoning,
    ScientificCivilization,
    ScientificModelInduction,
    ScopeAmbiguity,
    SelfErrorDiagnosis,
    SelfModel,
    SelfRepair,
    SemanticCompression,
    SequenceCopy,
    SetOperations,
    SocialConventionLearning,
    SourceProvenance,
    SourceReliabilityLearning,
    SpatialLanguage,
    SpeakerListenerGame,
    StrategyDiscovery,
    StrategyTransfer,
    StringReversal,
    SymbolDiscrimination,
    SymbolEquivalence,
    SymbolGrounding,
    SymbolicGeneralist,
    SymbolicWorldBuilder,
    SymmetryReasoning,
    Teaching,
    TemporalLanguage,
    ThematicRoles,
    TheoremProving,
    TheoryComparison,
    TheoryTransfer,
    ToolConstruction,
    Translation,
    TreeToSequence,
    UncertainSymbolicReasoning,
    UnderspecificationReasoning,
    Unification,
    UniversalInterfaceTransfer,
    UnknownGame,
    ValueLearning,
    VariableBinding,
    WorldModelSynthesis,
)

__all__ = ["LESSON_CLASSES"] + [c.__name__ for c in LESSON_CLASSES]
