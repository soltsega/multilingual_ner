# NER Project Todo — CoNLL-2003 + MasakhaNER (10 languages)
Due: Thu 20 Aug, 09:00

---

## 1. Environment & Data Acquisition

- [ ] `pip install datasets transformers seqeval torch accelerate`
- [ ] Load CoNLL-2003: `load_dataset("conll2003")`
- [ ] Load all 10 MasakhaNER v1 configs: `amh, hau, ibo, kin, lug, luo, pcm, swa, wol, yor`
- [ ] Load supplemental Amharic dataset: `load_dataset("rasyosef/amharic-named-entity-recognition")`
- [ ] Confirm every dataset actually downloaded (check row counts against known totals — see below)

**⚠️ Watch out for:**
- MasakhaNER v1 may use a legacy loading script and error out on newer `datasets` versions.
  - **Fix:** pin `datasets==2.14.0`, or pass `trust_remote_code=True` if supported.
- Silent partial downloads / cache corruption (row counts don't match expected).
  - **Fix:** print `len(dataset["train"])` for each language and compare to the known table (e.g. Amharic 1750, Hausa 1903, etc.) before moving on.

**Expected totals:** ~18,842 train / 2,604 val / 5,219 test across the 10 MasakhaNER languages; +14,041/3,250/3,453 for CoNLL English; +3,470 extra Amharic rows (train-only).

---

## 2. Explore & Profile the Data

- [ ] Print label distribution (count of B-PER/B-ORG/B-LOC/B-DATE) per language
- [ ] Check sentence length distribution per language
- [ ] Confirm tag schema matches across all MasakhaNER languages (`O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-DATE, I-DATE`)
- [ ] Check the tag schema of the supplemental Amharic dataset against MasakhaNER's
- [ ] Spot-check a few raw examples per language manually (read the tokens + tags side by side)

**⚠️ Watch out for:**
- Supplemental Amharic dataset using a different or smaller tag set.
  - **Fix:** map its tags onto MasakhaNER's schema before merging; drop or relabel anything that doesn't correspond cleanly.
- CoNLL-2003 has a `MISC` tag that MasakhaNER doesn't.
  - **Fix:** decide explicitly — either drop MISC entities from CoNLL, or keep MISC as a class only scored within English, and document the choice.
- Severe class imbalance (O is ~85-90% of all tokens) — expected, but confirm you're aware of the scale, since it's the root cause of the all-O trap in step 6.

---

## 3. Deduplicate & Merge Amharic Data

- [ ] Check for near-duplicate/overlapping sentences between the supplemental Amharic dataset and MasakhaNER's Amharic **test/validation** splits
- [ ] Merge supplemental Amharic data **only into the Amharic train split**
- [ ] Tag merged rows with a `source` column (`masakhaner` vs `rasyosef`) for later ablation
- [ ] Leave MasakhaNER's Amharic validation/test untouched

**⚠️ Watch out for:**
- Test-set leakage inflating your Amharic score if overlapping sentences exist.
  - **Fix:** run a simple string-overlap/hash check between the two sources before merging; drop any exact or near-duplicate matches from the training addition.
- Accidentally merging into validation/test too, breaking comparability with the other 9 languages.
  - **Fix:** keep merge logic scoped explicitly to `dataset["train"]`, never touch other splits.

---

## 4. Build the Combined Training Set (10 languages + English)

- [ ] Add a `language` column to every dataset before combining
- [ ] Decide: flat `concatenate_datasets` vs. `interleave_datasets` with sampling weights
- [ ] Keep every language's validation/test split **separate and untouched** — never merge these
- [ ] Confirm final combined train set size matches sum of individual language train sets

**⚠️ Watch out for:**
- Flat concatenation lets high-resource languages (English, Igbo, Yoruba) dominate gradient updates, starving low-resource ones (Luo).
  - **Fix:** use `interleave_datasets` with per-language sampling probabilities (e.g. inverse-proportional to size) if you want more balanced learning; if you use flat concatenation, note this explicitly as a limitation in your writeup.

---

## 5. Preprocess: Tokenize & Align BIO Labels

- [ ] Load a **fast** tokenizer (`AutoTokenizer.from_pretrained(..., use_fast=True)`)
- [ ] Tokenize with `is_split_into_words=True`
- [ ] Write the label-alignment function: first subword of a word gets the real label, following subwords get `-100` (ignored in loss) — or a converted `I-` tag if you prefer sub-token supervision
- [ ] Apply identically across all languages via `.map()`
- [ ] Spot check: decode a few aligned examples back to tokens + labels and manually verify they line up correctly

**⚠️ Watch out for:**
- Off-by-one or misaligned labels — the single most common bug in this assignment.
  - **Fix:** after alignment, manually print `(subword_token, aligned_label)` pairs for a handful of multi-subword words and visually confirm correctness before training anything.
- Forgetting `-100` on subword continuations — model ends up "trained" on garbage repeated labels, degrading it silently rather than crashing.
  - **Fix:** assert that `word_ids()` is used correctly and continuation tokens are explicitly set to `-100`.
- This bug won't always crash your code — it'll just quietly produce a worse model with a normal-looking loss curve.
  - **Fix:** treat unexpectedly bad entity-F1 despite low training loss as a signal to re-check alignment, not just "the language is hard."

---

## 6. Pick & Configure the Model

- [ ] Choose base checkpoint: `xlm-roberta-base` as a baseline, and/or `Davlan/afro-xlmr-base` for better African-language performance
- [ ] Set up `AutoModelForTokenClassification` with correct `num_labels`, `id2label`, `label2id`
- [ ] Confirm label mappings match exactly what was used during preprocessing (same order, same string names)

**⚠️ Watch out for:**
- Mismatched `id2label`/`label2id` ordering vs. the actual label list used in tokenization — causes silently wrong evaluation even if training "works."
  - **Fix:** derive the label list programmatically from the dataset's own `ClassLabel` feature, don't hand-type it.

---

## 7. Train

- [ ] Start with a CoNLL-only baseline run to confirm the pipeline works end to end
- [ ] Then train on the full combined multilingual set
- [ ] Optionally run a second variant (single-language fine-tune vs. joint) for 2-3 languages as a comparison
- [ ] Save checkpoints; log training loss and eval metrics per epoch

**⚠️ Watch out for:**
- Only tracking loss, not entity-F1, during training — loss can look fine while the model is stuck predicting all-O.
  - **Fix:** run seqeval evaluation at each checkpoint/epoch, not just at the very end.
- Overfitting on high-resource languages while underfitting low-resource ones in a joint run.
  - **Fix:** check per-language eval metrics during training, not just the aggregate.

---

## 8. Evaluate at Entity Level (seqeval)

- [ ] Convert predicted label IDs back to BIO string tags (skip `-100` positions)
- [ ] Run `seqeval.metrics.classification_report` **per language**, not just pooled
- [ ] Report both **micro F1** (pooled) and **macro F1** (average across languages/classes)
- [ ] Report per-entity-type breakdown (PER/ORG/LOC/DATE) per language

**⚠️ Watch out for — this is the core "gotcha" of the whole assignment:**
- **The all-O trap:** a model predicting `O` for every token can show high token-level accuracy while having 0 entity-F1.
  - **Fix:** never report token accuracy as your headline metric. Always lead with seqeval entity-F1. If token accuracy is high but seqeval F1 is near zero, check the per-class report immediately — that's the collapse.
- **Partial-credit leakage:** naive scoring might give credit for getting the entity type right but the span wrong.
  - **Fix:** confirm you're using seqeval's exact-match scheme (full span + type), not a custom token-level comparison mislabeled as "entity F1."
- **Micro F1 masking weak languages/classes:** pooled F1 is dominated by whichever language/class has the most examples.
  - **Fix:** always also report macro F1 and per-language breakdowns — this is what actually shows whether Luo or Amharic is struggling.
- **Validation/test conflation:** tuning against validation and then reporting that same number as "final" overstates generalization.
  - **Fix:** only touch the test set once, at the very end, after all decisions are locked in.
- **Non-comparable postprocessing across languages:** if `-100` positions leak into scoring for one language but not another, F1 numbers become incomparable.
  - **Fix:** use one shared postprocessing/decoding function applied identically to every language's predictions.

---

## 9. Error Analysis

- [ ] Identify boundary errors (partial entity span matches)
- [ ] Identify entity-type confusion (e.g. ORG predicted as LOC)
- [ ] Compare performance gap between high-resource (English, Yoruba, Hausa) and low-resource (Luo) languages
- [ ] Manually read ~10-15 real predictions vs. gold labels per language — don't rely on aggregate numbers alone
- [ ] Write up a short explanation of *why* certain languages underperform (data volume, script, morphology, subword fragmentation)

**⚠️ Watch out for:**
- Assuming a low score = "bad model" without checking whether it's actually a data or alignment bug.
  - **Fix:** for your worst-performing language, re-verify steps 5 (label alignment) and 3 (no leakage/schema mismatch) before concluding it's a genuine data-scarcity effect.

---

## 10. Package & Deploy

- [ ] Save final model + tokenizer (`save_pretrained`)
- [ ] Write a clean `predict(text, language=None)` function wrapping tokenization → inference → label decoding → entity span extraction
- [ ] Test the inference function on a few raw, unseen sentences per language
- [ ] Wrap in a simple API (FastAPI) or Hugging Face `pipeline` object
- [ ] Optionally push to Hugging Face Hub for a shareable demo

**⚠️ Watch out for:**
- Inference-time preprocessing not matching training-time preprocessing (different tokenizer settings, truncation, etc.)
  - **Fix:** reuse the exact same tokenizer config and label maps saved from training — don't hand-rewrite them for the deployment script.
- No language detection/handling — if deploying multilingually, decide whether the user specifies the language or the model infers language-agnostically.
  - **Fix:** since you trained one joint multilingual model, this may be a non-issue — but document the assumption either way.

---

## Final Sanity Checklist Before Submission

- [ ] Report both micro and macro F1, clearly labeled
- [ ] Report per-language, per-entity-type breakdown, not just one aggregate number
- [ ] Include at least one worked example of the all-O trap or a similar caught pitfall in your writeup — shows you understood *why* entity-level eval matters, not just that you ran it
- [ ] Confirm test sets were never touched during training/tuning
- [ ] Document any dataset merging/augmentation choices and why you made them
