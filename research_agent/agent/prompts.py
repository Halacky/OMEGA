"""Prompt templates for the research agent."""

ANALYZE_EXPERIMENTS_PROMPT = """\
You are a research analyst specializing in EMG (electromyography) gesture recognition.

## Research History Context
The following is the complete history of our research project, including all phases,
findings, causal chains, and known invariants:

{research_history}

## Current Experiment Results
{experiments_text}

Analyze the experiments above IN THE CONTEXT of the research history. Your analysis must include:
1. RANKING: Rank all model+pipeline combinations by accuracy (best to worst)
2. PATTERNS: What patterns emerge? Which architectures work best with which feature types?
3. AUGMENTATION EFFECT: Where augmentation was used, did it help?
4. SUBJECT VARIANCE: Which experiments show the highest/lowest variance across subjects?
5. GAPS: What combinations haven't been tested yet? What directions from the research history remain unexplored?
6. KEY INSIGHTS: 3-5 actionable insights for improving results, building on the causal chain from the research history
7. AVOID REPEATS: Identify approaches that the research history marks as FAILED or NEGATIVE — do NOT suggest these again

Respond in a structured format. Be specific with numbers and experiment names.
"""

EXTRACT_PAPER_INSIGHTS_PROMPT = """\
You are a research analyst specializing in EMG gesture recognition.

## Research History Context
{research_history}

Given the following arXiv papers, extract insights relevant to improving EMG gesture classification
in a cross-subject (LOSO) setting using deep learning and ML models.

Papers:
{papers_text}

Current best results in our pipeline:
{best_results}

For each paper, extract:
1. The key technique or method proposed
2. How it could be applied to our EMG gesture recognition pipeline
3. Whether it addresses cross-subject generalization
4. Whether it addresses any of the OPEN QUESTIONS from our research history

IMPORTANT: Cross-reference with the research history to avoid suggesting techniques
that we have already tried (see "Failed Approaches" section).

Focus only on actionable insights that could be implemented with these available models:
{constraints}

Respond with a numbered list of insights, each referencing the paper title and arxiv_id.
"""

ANALYZE_ERRORS_PROMPT = """\
You are a research analyst specializing in EMG gesture recognition.

## Research History Context
{research_history}

Analyze the following worst-performing subject/model combinations:

{worst_subjects}

And the overall experiment landscape:
{experiment_analysis}

Using the research history as context (especially the "Key Findings & Invariants" and
"Per-Subject Performance Pattern" sections), identify:
1. Are there subjects that consistently perform poorly across ALL models?
2. Are there models that fail specifically on certain subjects?
3. What might explain the failures (subject variability, gesture difficulty, data quality)?
4. What targeted improvements could address these failures WITHOUT repeating failed approaches?

Available models and augmentations:
{constraints}

Provide specific, actionable recommendations that build on the research history.
"""

EXPLOITATION_PROMPT = """\
You are a research hypothesis generator for EMG gesture recognition experiments.

## Research History Context
{research_history}

Your task: Generate a hypothesis to IMPROVE the best existing configuration through incremental changes.

Current best results:
{best_results}

Full experiment analysis:
{experiment_analysis}

Previously rejected hypotheses this session (avoid similar ideas):
{rejected}

{constraints}

CRITICAL RULES:
- You MUST read the "Failed Approaches & Negative Results" section of the research history
  and NEVER propose anything that has already been shown to fail.
- You MUST read the "Open Questions & Promising Directions" section for ideas that haven't been tried.
- Build on the causal chain: understand WHY the best model works, then propose a targeted improvement.

Generate a hypothesis that makes a small, targeted change to one of the top-performing configurations.
Examples of valid changes:
- Adding or modifying augmentation (noise, time_warp, rotation) to a top model
- Adjusting hyperparameters (dropout, learning_rate, batch_size)
- Combining features (e.g., adding TD features to a raw-signal model)
- Using a slightly different model variant

The hypothesis must be specific, actionable, and different from all existing experiments.
"""

EXPLORATION_PROMPT = """\
You are a research hypothesis generator for EMG gesture recognition experiments.

## Research History Context
{research_history}

Your task: Generate a hypothesis to EXPLORE an untested area of the experiment space.

Untested combinations:
{untested_combinations}

Full experiment analysis:
{experiment_analysis}

Paper insights (if available):
{paper_insights}

Previously rejected hypotheses this session:
{rejected}

{constraints}

CRITICAL RULES:
- Check the research history "Open Questions & Promising Directions" section for unexplored ideas.
- Do NOT repeat directions that the research history marks as failed (see Section 6).
- Prioritize directions that address the KEY FINDINGS, especially F1 (ML > DL), F4 (augmentation doesn't help), F6 (domain adaptation fails).

Generate a hypothesis that tests a novel combination of model, features, or augmentation
that has NOT been tried before. Prioritize combinations that could yield insights
even if they don't achieve the best accuracy.
"""

LITERATURE_PROMPT = """\
You are a research hypothesis generator for EMG gesture recognition experiments.

## Research History Context
{research_history}

Your task: Generate a hypothesis based on INSIGHTS FROM RECENT LITERATURE.

Paper insights:
{paper_insights}

Current experiment landscape:
{experiment_analysis}

Previously rejected hypotheses this session:
{rejected}

{constraints}

CRITICAL RULES:
- Cross-reference paper insights with the research history to identify TRULY novel techniques.
- Do NOT propose techniques that overlap with failed approaches (Section 6).
- Prefer papers that address the specific challenges identified in our research:
  * Subject domain gap (F4, F6)
  * Class bias in fusion models (F5)
  * Feature engineering improvements (Direction A)
  * Test-time adaptation (Direction B)

Generate a hypothesis that adapts a technique or finding from the papers to our pipeline.
You MUST reference specific papers (by title and arxiv_id) in the motivation.
The proposed changes must use only models and features available in our codebase.
"""

ERROR_DRIVEN_PROMPT = """\
You are a research hypothesis generator for EMG gesture recognition experiments.

## Research History Context
{research_history}

Your task: Generate a hypothesis to address SPECIFIC FAILURE PATTERNS.

Error analysis:
{error_analysis}

Full experiment analysis:
{experiment_analysis}

Previously rejected hypotheses this session:
{rejected}

{constraints}

CRITICAL RULES:
- Consult the research history's Appendix A (per-subject performance patterns) and
  Section 5 (Key Findings) to understand WHY certain subjects/models fail.
- Do NOT propose domain adaptation approaches that have already failed (GRL, MMD, contrastive — see Section 6).
- Focus on data quality improvements (F8) or subject clustering (Direction C) instead.

Generate a hypothesis that specifically targets the identified failure patterns.
For example:
- If certain subjects fail, propose subject-specific augmentation or normalization
- If certain gestures are confused, propose architecture changes to disambiguate them
- If variance is high, propose regularization or ensemble approaches
"""

GENERATE_SEARCH_QUERIES_PROMPT = """\
You are a research assistant specializing in EMG gesture recognition and biosignal processing.

## Research History Context
{research_history}

## Current Research Gaps
Based on the experiment analysis:
{experiment_analysis}

## Current Strategy: {strategy}

Generate 4-6 targeted arXiv search queries that would find papers relevant to
our CURRENT research needs. The queries should:

1. Address specific open questions from the research history (Section 7)
2. Target techniques that could solve our identified problems:
   - Cross-subject domain gap (best accuracy is only ~35%)
   - Class bias in fusion models (high accuracy, low F1)
   - Feature engineering that captures subject-invariant information
   - Effective data augmentation for EMG signals
3. Be specific enough to find relevant papers (not too broad)
4. Cover different aspects of the problem

DO NOT search for techniques that have already been tried and failed:
- Generic contrastive learning for EMG (already failed: exp_15)
- MMD-based domain adaptation (already failed: exp_19)
- Gradient reversal layer (already failed: exp_5)

Return ONLY a JSON array of query strings, nothing else.
Example: ["query 1", "query 2", "query 3"]
"""
