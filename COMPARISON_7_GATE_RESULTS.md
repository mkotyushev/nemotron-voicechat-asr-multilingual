# Comparison 7 gating checks: results

The four blocking checks of `REGMEAN_INTERFACE_DESIGN.md` §11 have been run.
This file indexes the frozen artifacts, the commands, the hashes and the
numbers, in the same way `COMPARISON_3_RESULTS.md` does for comparison 3.

**Verdict in one line.** Condition B — "reply in the input language" — **has a
teacher**, so comparison 7 is no longer blocked. It has one only through the
checkpoint's own text chat format; the deployment runtime's perception-channel
text path cannot produce it, and on Russian it fails almost completely. Nothing
here is evidence about the encoder, the merge, or a deployable model.

Output directory (ignored): `.cache/experiments/lm-gating-v1/`.
Result digest `fc814a46346beed1b054e818ca9014f97b9a63a45737e07d1a529d467d23c814`.

---

## Check 2 — `n_ff` is `intermediate_size`, and Dataset A is bigger than it looks

Confirmed, and it changes nothing in §5 except to make the number exact.

| | value |
|---|---|
| `n_embd` | 1024 |
| `n_ff` = `intermediate_size` | 4096 (= 4 × `n_embd`, in both `PT_ML` and `PT_EN`) |
| widest linear input | **`subsampling.linear`, d_in = 4352**, not `feed_forward*.linear2` |
| Dataset A at ≥4 rows per input dimension | **17,408 encoder frames = 1392.6 s = 23.2 minutes per candidate** |

`subsampling.linear` is the binding dimension, not the FFN: 128 mel bins survive
three stride-2 convolutions as 17 bins, and 256 channels × 17 = 4352. The
paper's literal 256 samples would give roughly 9.6k frames, solving that layer
at a ratio of 2.2 rather than 4.

Artifact: `dimensions.json`. Command:

```bash
.venv-align/bin/python lm_gating_check.py dimensions \
  --config PT_ML=.../asr-multilingual/config.json \
  --config PT_EN=.../asr-en/config.json \
  --output .cache/experiments/lm-gating-v1
```

---

## What "text-only" turned out to mean

§11 asks whether the frozen language model reads fr/de/ru text. VoiceChat has
no text input channel, so the question has two answers and they disagree, which
is why both were measured.

**`perception`** — the pinned deployment binary (`llama-voicechat --serve`,
image `nemotron-voicechat:f45001fc3d8013c72beb6753d3eb0b976b6a9fff`), with the
utterance carried in the system prompt. This is the channel VoiceChat was
trained to read text on, one token per 12.5 Hz frame; a 1 s silent clip
triggers the turn so no audio content reaches the model.

**`text`** — the Nemotron-H chat format the checkpoint inherits, rendered
explicitly and completed by stock llama.cpp `b10819-6a1a922d2`.

Two facts discovered while building this, both of which constrain comparison 7:

1. **The served bridge cannot carry a foreign-language prompt at all.**
   `render_system_prompt` in the pinned runtime ends with
   `text.encode("ascii", "ignore").decode("ascii")`, which drops every accent
   and erases Cyrillic entirely. Teacher generation therefore has to run
   through the CLI or a text server, never through `/v1/realtime`.
2. **The GGUF's declared end-of-turn token is wrong for text use.** It declares
   `</s>` (id 2), which is the *audio* text-channel turn boundary, while the
   chat format ends an assistant turn with `<SPECIAL_12>` (id 12). Without
   `--override-kv tokenizer.ggml.eos_token_id=int:12` every reply runs to the
   token cap. The text pass here was served by:

   ```bash
   .cache/tools/llama.cpp/llama-b10819/llama-server \
     -m /srv/bulk/ai/models/NemotronLabs-VoiceChat-11B-gguf/nemotron_voicechat_11b-stt-llm-Q8_0.gguf \
     -ngl 36 -c 8192 -np 4 --jinja --host 127.0.0.1 --port 9099 --no-warmup \
     --override-kv tokenizer.ggml.eos_token_id=int:12
   ```

---

## Checks 1 and 4 — does the teacher obey each condition?

Frozen sample: 100 MASSIVE 1.1 `dev` utterances present in all four locales,
seed 0, digest `026fe1b9fc514dd022a5926350e8c9d9812a3f7fe0a119e60fae0f196f858da8`.
The `text` path ran all 100 per language; the `perception` path ran the first
30 of the same ids. English is the control row of §8, not a graded language.

Rates are per cell. **expected** is the rate of replies in the language the
condition asks for; **usable** additionally requires that the reply is not
empty, not a repeating loop and not a parrot of the request — the §11.4 gate.

| path | prompt | input | n | expected | English out | in-language out | echo | usable |
|---|---|---|---:|---:|---:|---:|---:|---:|
| text | A english-only | de | 100 | 0.85 | 0.85 | 0.15 | 0.03 | **0.84** |
| text | A english-only | fr | 100 | 0.91 | 0.91 | 0.09 | 0.04 | **0.89** |
| text | A english-only | ru | 100 | 1.00 | 1.00 | 0.00 | 0.00 | **1.00** |
| text | A english-only | en | 100 | 1.00 | 1.00 | 1.00 | 0.07 | 0.92 |
| text | B in-language | de | 100 | 0.90 | 0.07 | 0.90 | 0.06 | **0.85** |
| text | B in-language | fr | 100 | 0.95 | 0.05 | 0.95 | 0.06 | **0.90** |
| text | B in-language | ru | 100 | 0.94 | 0.05 | 0.94 | 0.12 | **0.82** |
| text | B in-language | en | 100 | 0.99 | 0.99 | 0.99 | 0.07 | 0.92 |
| perception | A english-only | de | 30 | 0.90 | 0.90 | 0.07 | 0.03 | **0.90** |
| perception | A english-only | fr | 30 | 0.93 | 0.93 | 0.07 | 0.03 | **0.90** |
| perception | A english-only | ru | 30 | 1.00 | 1.00 | 0.00 | 0.00 | **1.00** |
| perception | A english-only | en | 30 | 1.00 | 1.00 | 1.00 | 0.07 | 0.93 |
| perception | B in-language | de | 30 | 0.67 | 0.27 | 0.67 | 0.10 | **0.60** |
| perception | B in-language | fr | 30 | 0.87 | 0.13 | 0.87 | 0.10 | **0.77** |
| perception | B in-language | ru | 30 | 0.07 | 0.73 | 0.07 | 0.00 | **0.07** |
| perception | B in-language | en | 30 | 1.00 | 1.00 | 1.00 | 0.03 | 0.97 |

Verdicts against the thresholds declared before the sample was drawn
(A ≥ 0.90, B ≥ 0.80, graded on the worst foreign language):

| | condition A | condition B |
|---|---|---|
| `text` | fail (de 0.84) | **pass** (worst ru 0.82) |
| `perception` | pass (worst 0.90) | fail (ru 0.07) |

### What that means

**Condition B is not blocked.** On the text path the model answers French,
German and Russian in the language it was addressed in, with replies that read
as an assistant's:

> `сделай в комнате темнее` → "Конечно, я могу включить выключатель. Комната станет темнее."
> `wie groß ist die zeitverschiebung...` → "Die Zeitverschiebung beträgt zwölf Stunden. …"

**Condition A's failure is a filtering cost, not a missing capability.** It
fails because the model answers *in German* on 15% of German inputs and in
French on 9% of French inputs — it disobeys "English only" rather than
producing nothing. §11.4 already requires gating targets on output-language
identification, and after that gate condition A retains 84–100% of its
utterances. Russian is the one language it never leaks on.

**The perception path cannot be the teacher.** On Russian it answers in English
73% of the time and produces transliterated nonsense when it does not:

> `сделай в комнате темнее` → "Sobremy! What is it?"
> `поставь будильник на шесть утра` → '"Poez" means "to poach" … and "budilnik" means "tooth."'

The same model reading the same sentence through the chat format answers it
correctly, so this is the input mode, not the model. Comparison 7's targets
must therefore be generated offline through the text format. The student is
unaffected: it is conditioned on audio, not on text.

### One measurement caveat, stated rather than tuned away

The echo rule is conservative in a command domain. "Будильник установлен на
шесть утра" is a correct confirmation of "поставь будильник на шесть утра", and
it is counted as an echo. So every **usable** column is a lower bound, and the
`expected` column is the same statistic without that penalty. The rule was not
retuned after seeing the data. It does not change the two verdicts that matter:
condition B passes on the text path even on the lower bound, and condition A's
German cell fails on the language rate alone (0.85 < 0.90). It does decide
condition A's French cell, which is 0.91 before the penalty and 0.89 after.

Output-language identification is a character-trigram naive Bayes fitted on
MASSIVE `train` and reported on the untouched `test` partition: **98.8%**
overall (de 0.980, en 0.984, fr 0.990, ru 0.999). It carries no external model
and no unpinned dependency.

---

## Check 3 — the text-path and audio-path targets are not on one scale

24 English BFCL v3 clips at revision `bce0c5dd23971bacd49112427ae4ab90d0a02ab0`,
seed 0, disjoint from the six frozen speech-to-action pilot cases, under
condition A. Each clip was answered three ways: **audio**, the original
VoiceChat perception encoder hearing the clip, which is B1's target by
definition; and the same model reading the transcript on each of the two text
paths, which is B2's procedure applied to English.

| text procedure | n | median token F1 vs audio | median words, audio → text | length ratio | same language |
|---|---:|---:|---:|---:|---:|
| `text_chat` | 24 | **0.41** | 29 → 39 | 1.34 | 0.96 |
| `text_perception` | 24 | **0.34** | 29 → 31 | 1.07 | 0.96 |

One of 24 audio clips produced no assistant turn at all (4.2%).

The two procedures answer the same request, usually correctly, and still share
under half their words. They differ in content, not only in wording:

> audio: "I cannot complete the booking for the Hilton Hotel in Chicago because I do not have access to hotel booking systems…"
> text: "I can book a single room for two nights at the Hilton Hotel in Chicago starting from the tenth of December…"

and in length, systematically: the text target is a third longer than what
VoiceChat itself says. **A 1:1 weighting of the B1 and B2 loss terms is
therefore not defensible as a default.** The design record's concern is
confirmed and now has a number: B1 is a self-distillation term with a zero
floor by construction, and B2's targets are drawn from a visibly different
distribution of the same model's own outputs.

---

## Provenance

| item | digest |
|---|---|
| frozen sample `sample.json` | `026fe1b9fc514dd022a5926350e8c9d9812a3f7fe0a119e60fae0f196f858da8` |
| `result.json` | `fc814a46346beed1b054e818ca9014f97b9a63a45737e07d1a529d467d23c814` |
| `nemotron_voicechat_11b-stt-llm-Q8_0.gguf` | `9b7ef9b6f30d179ff40f26eca613157e04b0b8e4989f34da4f355b8e18e20272` |
| `mmproj-voicechat-perception-Q8_0.gguf` (`FT_EN`) | `482ba5a6f92b483cdc816c8ed87798897cf9196a8976350fdc4b916d39609aa3` |
| `/app/llama-voicechat` in the pinned image | `99d0ec8d573de0a67be920ebd5c8fa95e084dcc233301a6e6a316e4491dc0c71` |
| `llama-b10819-bin-ubuntu-vulkan-x64.tar.gz` | `2175737ab85506e7639fc7f8c84b5247fd607cc6e7030825c6dcadd4279f62e3` |
| `amazon-massive-dataset-1.1.tar.gz` | `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577` |

Everything ran at the deployment precision, Q8_0, on both paths. Invariant 3's
extension is about the precision a projection is *fitted* against; no
projection was fitted here.

The runtime container was never stopped, restarted or reconfigured: both paths
attach to the running pinned image, and the served `asr_model` is unchanged.

### Commands

```bash
OUT=.cache/experiments/lm-gating-v1
PY=.venv-align/bin/python

$PY lm_gating_check.py sample --massive .cache/datasets/MASSIVE/1.1/data \
  --count 100 --partition dev --seed 0 --output $OUT

$PY lm_gating_check.py generate --sample $OUT/sample.json --path text \
  --language en --language fr --language de --language ru \
  --endpoint http://127.0.0.1:9099 --output $OUT \
  --llama-binary .cache/tools/llama.cpp/llama-b10819/llama-server

$PY lm_gating_check.py generate --sample $OUT/sample.json --path perception \
  --language en --language fr --language de --language ru \
  --limit 30 --n-gpu-layers 24 --output $OUT

$PY lm_gating_check.py gap --bfcl-audio .cache/datasets/BFCL_v3_audio \
  --bfcl-audio-revision bce0c5dd23971bacd49112427ae4ab90d0a02ab0 \
  --count 24 --seed 0 --prompt A_english_only \
  --exclude simple_1 --exclude simple_3 --exclude simple_19 \
  --exclude multiple_1 --exclude multiple_4 --exclude multiple_6 \
  --source text_chat --endpoint http://127.0.0.1:9099 --output $OUT
# ... and again with --source audio --source text_perception

$PY lm_gating_check.py score \
  --generations $OUT/generations-text.jsonl \
  --generations $OUT/generations-perception.jsonl \
  --gap $OUT/generations-gap-text_chat.jsonl \
  --gap $OUT/generations-gap-audio-text_perception.jsonl \
  --dimensions $OUT/dimensions.json \
  --massive .cache/datasets/MASSIVE/1.1/data --output $OUT
```

---

## What comparison 7 must carry forward

1. Generate B2 targets through the **text chat format**, not the perception
   channel, and record that path in the candidate's provenance beside the
   fitting precision invariant 3 requires.
2. Gate every target on output-language identification, as §11.4 requires. The
   cost is now known: 0–16% of condition A's foreign targets and 10–18% of
   condition B's.
3. Do not weight B1 and B2 equally without evidence. Their targets differ in
   length by a third and share about 40% of their tokens.
4. Exclude the 100 sampled `dev` ids from Dataset B2, or record the overlap;
   they are listed in `sample.json`.
5. Condition B is a real condition with a real teacher. §8's argument for
   keeping it — that condition A alone rewards discarding language identity —
   stands, and is no longer conditional on an untested assumption.
