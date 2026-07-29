# Results: Domain-Adaptive Fine-Tuning of Whisper for Swahili ASR

This section reports the outcome of Phases 3–5 of the methodology: baseline
evaluation of pre-trained Whisper, domain-adaptive fine-tuning on Swahili speech,
and comparative evaluation on a held-out benchmark. It directly addresses the
project's research question: *can domain-specific fine-tuning of the Whisper ASR
model improve transcription accuracy for multilingual African conference meetings
compared to the original pre-trained model?*

## 1. Evaluation setup

**Models compared.**

- **Baseline**: `openai/whisper-small` (244M parameters), used zero-shot with no
  task-specific adaptation.
- **Fine-tuned**: the same checkpoint, fine-tuned for 3 epochs on the Swahili
  (`sw`) train split of `fsicoli/common_voice_15_0` — a community mirror of
  Mozilla Common Voice 15.0 (44,015 training utterances; see §4 for a note on
  data provenance). Training used the Hugging Face `Seq2SeqTrainer` with
  mixed-precision (fp16), gradient checkpointing, a learning rate of 1e-5, and a
  per-device batch size of 8.

**Held-out evaluation set.** Both models were evaluated on the **test split of
Google FLEURS (`sw_ke`, 487 utterances)**, which was deliberately excluded from
the fine-tuning data entirely — it contributes only its `test` split to the
pipeline and is never seen during training. This keeps the evaluation set
independent of the training distribution and keeps reported metrics comparable
to Whisper's own published FLEURS benchmarks.

**Metrics.**

- **Word Error Rate (WER)**, computed with `jiwer` after normalizing both
  reference and hypothesis text (lowercasing, punctuation removal, whitespace
  collapsing).
- **Character Error Rate (CER)**, computed with `jiwer` on raw (non-normalized)
  text. This asymmetry with WER's normalization is a known limitation of the
  current evaluation script and is noted in §4.
- **Mean inference latency** per utterance (wall-clock, single-example batches).
- **Median WER**, reported alongside the mean because WER's arithmetic mean is
  sensitive to outliers (see §3).
- **Hallucination rate**, defined here as the proportion of utterances with
  WER > 2.0 (i.e., the hypothesis is more than twice as long, in edit-distance
  terms, as the reference) — a proxy for catastrophic decoding failures rather
  than ordinary transcription errors.

## 2. Quantitative results

| Metric | Baseline (zero-shot) | Fine-tuned | Relative improvement |
|---|---:|---:|---:|
| Mean WER | 110.2% | 21.5% | 80.5% |
| Median WER | 75.7% | 18.2% | 75.9% |
| Mean CER | 45.7% | 8.2% | 82.2% |
| Hallucination rate (WER > 2.0) | 2.5% | 0.0% | — |
| Mean latency / utterance | 0.957 s | 0.892 s | — |

*n = 487 for both conditions (FLEURS `sw_ke` test split).*

Fine-tuning reduced mean WER by 80.5% relative (110.2% → 21.5%) and mean CER by
82.2% relative (45.7% → 8.2%). Critically, the **median-based improvement
(75.9%) closely tracks the mean-based improvement (80.5%)**, indicating that the
gain is a genuine, distribution-wide effect rather than an artifact of a small
number of extreme outliers being corrected.

## 3. Robustness: elimination of catastrophic decoding failures

Inspection of per-utterance results showed that a subset of baseline
transcriptions failed catastrophically rather than merely inaccurately — two
recognizable Whisper failure modes:

- **Repetition loops**: the decoder emits a single token or short phrase
  hundreds of times (e.g., a hypothesis consisting of one word repeated
  ~300 times against a normal-length reference sentence), producing WER values
  exceeding 1,700% on individual utterances.
- **Premature truncation**: the decoder terminates after a small fragment of a
  much longer utterance (e.g., a two-word hypothesis for a full-sentence
  reference).

These failures occurred on approximately **2.5% of baseline utterances**
(≈12/487) and are consistent with known behavior of greedy/beam decoding under
low model confidence, particularly on lower-resource languages. The evaluation
pipeline calls the Hugging Face `generate()` API directly and does not apply the
fallback heuristics used by OpenAI's reference `whisper` implementation
(temperature fallback, compression-ratio thresholding, no-speech thresholding),
so these failure modes are not suppressed at inference time — they are observed
as the model actually behaves under standard Hugging Face decoding.

The fine-tuned model exhibited **zero such failures (0.0%)** on the same 487
utterances. This is reported as a distinct finding from the accuracy
improvement in §2: domain-adaptive fine-tuning did not merely shift the error
distribution downward on average, it also improved decoding *stability*,
eliminating a specific, severe failure mode entirely on this benchmark.

## 4. Limitations

- **Training data provenance.** As of October 2025, Mozilla discontinued
  distribution of the official `mozilla-foundation/common_voice_*` datasets via
  Hugging Face, moving to a separate "Mozilla Data Collective" platform not
  accessible through the `datasets` library. Training data was instead sourced
  from `fsicoli/common_voice_15_0`, an actively-used but unofficial community
  mirror of Common Voice 15.0. Results should be understood as reflecting this
  data source, not an official Mozilla release.
- **Domain gap between evaluation and target use case.** FLEURS consists of
  read, single-speaker sentences sourced from Wikipedia-style text, not
  conference or institutional meeting speech. It was chosen as the evaluation
  benchmark for its comparability to published Whisper results and its
  guaranteed independence from the training data, but it does not capture the
  multi-speaker, code-switching, and domain-specific terminology conditions
  described in the project's problem statement. Evaluation on genuine
  conference/institutional recordings (the "Custom Dataset" described in the
  methodology) remains future work.
- **WER/CER normalization asymmetry.** WER is computed on normalized text; CER
  is computed on raw text. Both metrics move in the same direction here, but
  the two are not computed on identical strings.
- **Single training run.** Results reflect one fine-tuning run with one
  hyperparameter configuration and one random seed; no variance across seeds or
  runs is reported.
- **Model scale.** Only `whisper-small` was evaluated, due to compute and time
  constraints on free-tier Colab GPU access. Larger Whisper checkpoints may
  exhibit different baseline performance and different absolute gains from
  fine-tuning.
- **Latency measurement.** Latency was measured on a Colab-allocated T4 GPU and
  is not necessarily representative of latency in a production deployment
  environment.

## 5. Summary

Domain-adaptive fine-tuning of Whisper-small on Swahili Common Voice data
produced a substantial, distribution-wide reduction in transcription error
(≈76–80% relative WER reduction, mean and median in close agreement) on a
held-out benchmark never seen during training, and separately eliminated a
class of catastrophic decoding failures observed in the zero-shot baseline.
This provides direct evidence in support of the project's central hypothesis:
domain-specific fine-tuning meaningfully improves ASR accuracy for Swahili
relative to a general-purpose pre-trained model, motivating its use as the
speech-recognition foundation for the AfriMeet AI meeting assistant.
