# E4-B1a — post-hoc discordant analysis

**AUTOMATED PROXY, POST-HOC.** E4-B1 decision remains `E4_B1_PROMPT_ROUTE_REJECTED`; thresholds and E4-B1 artifacts were not changed. All denominators below are exact counts, with no p-values or confidence intervals.

## Methods and denominators

The unit is an image–finding cell. CheXpert output states remain Positive, Negative, Uncertain, Not mentioned, or Extraction failure. CLEAR direction is threshold-free `favors present`→Positive and `favors absent`→Negative. Pneumothorax has no CLEAR score and is excluded from direction-discordant sets. O–S discordance requires a Positive/Negative reference-report oracle and shuffled CLEAR direction opposite to it. The F1-O categorical table differs from the F1-M/S numeric table, limiting causal interpretation. `Unchanged` below means unchanged extracted state; exact text equality is separately counted.

There are **69** Positive/Negative oracle cells across six findings, **52** across the five CLEAR-scored findings, **27** oracle-Positive cells across six findings, and **19/24** images with at least one matched/shuffled CLEAR direction difference. These sets are distinct.

## 1–2. O–S discordant cells

| Format | Cells | O follows oracle | S follows shuffled | O=S state | O=S text | O Not mentioned | S Not mentioned |
|---|---|---|---|---|---|---|---|
| F1 | 34 | 6/34 | 2/34 | 23/34 | 0/34 | 24/34 | 25/34 |
| F2 | 34 | 7/34 | 3/34 | 15/34 | 0/34 | 26/34 | 14/34 |
| F3 | 34 | 12/34 | 5/34 | 20/34 | 0/34 | 19/34 | 13/34 |

By oracle state:

| Format/state | Cells | O follows oracle | S follows shuffled | O=S state | O=S text | O Not mentioned | S Not mentioned |
|---|---|---|---|---|---|---|---|
| F1/Positive | 20 | 6/20 | 2/20 | 12/20 | 0/20 | 13/20 | 11/20 |
| F1/Negative | 14 | 0/14 | 0/14 | 11/14 | 0/14 | 11/14 | 14/14 |
| F2/Positive | 20 | 7/20 | 3/20 | 11/20 | 0/20 | 13/20 | 10/20 |
| F2/Negative | 14 | 0/14 | 0/14 | 4/14 | 0/14 | 13/14 | 4/14 |
| F3/Positive | 20 | 4/20 | 4/20 | 11/20 | 0/20 | 14/20 | 8/20 |
| F3/Negative | 14 | 8/14 | 1/14 | 9/14 | 0/14 | 5/14 | 5/14 |

By finding (only findings with CLEAR scores):

| Format/finding | Cells | O follows oracle | S follows shuffled | O=S state | O=S text | O Not mentioned | S Not mentioned |
|---|---|---|---|---|---|---|---|
| F1/Atelectasis | 4 | 2/4 | 0/4 | 3/4 | 0/4 | 2/4 | 3/4 |
| F1/Cardiomegaly | 6 | 2/6 | 0/6 | 3/6 | 0/6 | 3/6 | 4/6 |
| F1/Consolidation | 9 | 0/9 | 0/9 | 8/9 | 0/9 | 9/9 | 8/9 |
| F1/Edema | 3 | 0/3 | 1/3 | 0/3 | 0/3 | 2/3 | 1/3 |
| F1/Pleural Effusion | 12 | 2/12 | 1/12 | 9/12 | 0/12 | 8/12 | 9/12 |
| F2/Atelectasis | 4 | 1/4 | 0/4 | 3/4 | 0/4 | 3/4 | 4/4 |
| F2/Cardiomegaly | 6 | 0/6 | 1/6 | 2/6 | 0/6 | 6/6 | 2/6 |
| F2/Consolidation | 9 | 3/9 | 0/9 | 4/9 | 0/9 | 6/9 | 5/9 |
| F2/Edema | 3 | 1/3 | 0/3 | 3/3 | 0/3 | 2/3 | 2/3 |
| F2/Pleural Effusion | 12 | 2/12 | 2/12 | 3/12 | 0/12 | 9/12 | 1/12 |
| F3/Atelectasis | 4 | 2/4 | 0/4 | 4/4 | 0/4 | 2/4 | 2/4 |
| F3/Cardiomegaly | 6 | 1/6 | 2/6 | 2/6 | 0/6 | 4/6 | 1/6 |
| F3/Consolidation | 9 | 3/9 | 1/9 | 4/9 | 0/9 | 6/9 | 7/9 |
| F3/Edema | 3 | 0/3 | 1/3 | 1/3 | 0/3 | 2/3 | 1/3 |
| F3/Pleural Effusion | 12 | 6/12 | 1/12 | 9/12 | 0/12 | 5/12 | 2/12 |

By finding and oracle state (retains sparse/zero denominators):

| Format/finding/state | Cells | O follows oracle | S follows shuffled | O=S state | O=S text | O Not mentioned | S Not mentioned |
|---|---|---|---|---|---|---|---|
| F1/Atelectasis/Positive | 4 | 2/4 | 0/4 | 3/4 | 0/4 | 2/4 | 3/4 |
| F1/Atelectasis/Negative | 0 | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) |
| F1/Cardiomegaly/Positive | 4 | 2/4 | 0/4 | 2/4 | 0/4 | 2/4 | 2/4 |
| F1/Cardiomegaly/Negative | 2 | 0/2 | 0/2 | 1/2 | 0/2 | 1/2 | 2/2 |
| F1/Consolidation/Positive | 3 | 0/3 | 0/3 | 2/3 | 0/3 | 3/3 | 2/3 |
| F1/Consolidation/Negative | 6 | 0/6 | 0/6 | 6/6 | 0/6 | 6/6 | 6/6 |
| F1/Edema/Positive | 3 | 0/3 | 1/3 | 0/3 | 0/3 | 2/3 | 1/3 |
| F1/Edema/Negative | 0 | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) |
| F1/Pleural Effusion/Positive | 6 | 2/6 | 1/6 | 5/6 | 0/6 | 4/6 | 3/6 |
| F1/Pleural Effusion/Negative | 6 | 0/6 | 0/6 | 4/6 | 0/6 | 4/6 | 6/6 |
| F2/Atelectasis/Positive | 4 | 1/4 | 0/4 | 3/4 | 0/4 | 3/4 | 4/4 |
| F2/Atelectasis/Negative | 0 | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) |
| F2/Cardiomegaly/Positive | 4 | 0/4 | 1/4 | 1/4 | 0/4 | 4/4 | 1/4 |
| F2/Cardiomegaly/Negative | 2 | 0/2 | 0/2 | 1/2 | 0/2 | 2/2 | 1/2 |
| F2/Consolidation/Positive | 3 | 3/3 | 0/3 | 1/3 | 0/3 | 0/3 | 2/3 |
| F2/Consolidation/Negative | 6 | 0/6 | 0/6 | 3/6 | 0/6 | 6/6 | 3/6 |
| F2/Edema/Positive | 3 | 1/3 | 0/3 | 3/3 | 0/3 | 2/3 | 2/3 |
| F2/Edema/Negative | 0 | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) |
| F2/Pleural Effusion/Positive | 6 | 2/6 | 2/6 | 3/6 | 0/6 | 4/6 | 1/6 |
| F2/Pleural Effusion/Negative | 6 | 0/6 | 0/6 | 0/6 | 0/6 | 5/6 | 0/6 |
| F3/Atelectasis/Positive | 4 | 2/4 | 0/4 | 4/4 | 0/4 | 2/4 | 2/4 |
| F3/Atelectasis/Negative | 0 | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) |
| F3/Cardiomegaly/Positive | 4 | 0/4 | 2/4 | 1/4 | 0/4 | 4/4 | 1/4 |
| F3/Cardiomegaly/Negative | 2 | 1/2 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 |
| F3/Consolidation/Positive | 3 | 1/3 | 0/3 | 2/3 | 0/3 | 2/3 | 3/3 |
| F3/Consolidation/Negative | 6 | 2/6 | 1/6 | 2/6 | 0/6 | 4/6 | 4/6 |
| F3/Edema/Positive | 3 | 0/3 | 1/3 | 1/3 | 0/3 | 2/3 | 1/3 |
| F3/Edema/Negative | 0 | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) | 0/0 (N/A) |
| F3/Pleural Effusion/Positive | 6 | 1/6 | 1/6 | 3/6 | 0/6 | 4/6 | 1/6 |
| F3/Pleural Effusion/Negative | 6 | 5/6 | 0/6 | 6/6 | 0/6 | 1/6 | 1/6 |

## 3. M–S direction of change

For each direction-discordant five-finding cell: `follows_evidence` requires M and S both to output their respective evidence directions; `opposes_evidence` requires the reverse pair. `text_changed_target_state_same` means the exact report text differs while this finding's extracted state does not; it does **not** prove the changed words are semantically unrelated. `unchanged_text_and_target_state` requires exact report equality. The extra `other_target_state_change` category preserves one-sided, Uncertain, Not mentioned and extraction-failure transitions without mislabeling them as directional following.

| Format | Direction-discordant cells | follows_evidence | opposes_evidence | text_changed_target_state_same | unchanged_text_and_target_state | other_target_state_change |
|---|---|---|---|---|---|---|
| F1 | 62 | 0/62 | 0/62 | 12/62 | 45/62 | 5/62 |
| F2 | 62 | 2/62 | 0/62 | 39/62 | 2/62 | 19/62 |
| F3 | 62 | 0/62 | 2/62 | 29/62 | 11/62 | 20/62 |

## 4. Arm-level collapse, mention and length

| Arm | Collapse | Mentioned cells | Mean findings/report | Chars mean/median | Tokens mean/median |
|---|---|---|---|---|---|
| C0 | 0/24 | 71/144 | 2.96/6 | 220.1/218 | 59.1/60 |
| N1 | 6/24 | 38/144 | 1.58/6 | 145.8/140.5 | 40.5/41 |
| F1-M | 17/24 | 14/144 | 0.58/6 | 74.5/34 | 21.7/12 |
| F1-S | 16/24 | 19/144 | 0.79/6 | 79.1/34 | 23.0/12 |
| F1-O | 10/24 | 38/144 | 1.58/6 | 159.5/129 | 41.7/34.5 |
| F2-M | 0/24 | 58/144 | 2.42/6 | 198.8/171.5 | 53.3/49 |
| F2-S | 0/24 | 57/144 | 2.38/6 | 205.1/187 | 55.2/50 |
| F2-O | 7/24 | 29/144 | 1.21/6 | 140.7/97.5 | 37.4/27 |
| F3-M | 0/24 | 64/144 | 2.67/6 | 244.2/235 | 62.6/61 |
| F3-S | 0/24 | 73/144 | 3.04/6 | 236.5/221.5 | 62.0/61 |
| F3-O | 0/24 | 69/144 | 2.88/6 | 249.0/258.5 | 65.2/68 |

## 5. No-evidence C0 baseline on identical cells

| Format | All definite cells | C0 matches oracle | S matches oracle | O matches oracle |
|---|---|---|---|---|
| F1 | 69 | 36/69 | 8/69 | 15/69 |
| F2 | 69 | 36/69 | 31/69 | 10/69 |
| F3 | 69 | 36/69 | 36/69 | 33/69 |

On the O–S discordant subset:

| Format | Discordant cells | C0 matches oracle | S matches oracle | O matches oracle |
|---|---|---|---|---|
| F1 | 34 | 17/34 | 6/34 | 6/34 |
| F2 | 34 | 17/34 | 17/34 | 7/34 |
| F3 | 34 | 17/34 | 15/34 | 12/34 |

On all 34 discordant cells, O does not exceed S in any format: F1 6/34 versus 6/34, F2 7/34 versus 17/34, F3 12/34 versus 15/34. F3's C0/S/O counts are 17/34, 15/34 and 12/34. Thus these proxy counts provide **no evidence that oracle text was used in the intended direction**. This does not imply the generated text is invariant to prompts. There is no post-hoc equivalence threshold; the counts are agreement with reference-report labels, not image truth.

## 6. Oracle-Positive cells

| Arm | Output Positive / oracle-Positive cells |
|---|---|
| C0 | 13/27 |
| N1 | 8/27 |
| F1-M | 6/27 |
| F1-S | 6/27 |
| F1-O | 8/27 |
| F2-M | 13/27 |
| F2-S | 9/27 |
| F2-O | 8/27 |
| F3-M | 5/27 |
| F3-S | 8/27 |
| F3-O | 5/27 |

## 7. Oracle block inventory

F2-O and F3-O blocks were checked sentence-by-sentence against all six saved oracle states for every image. Each block has six reference-report sentences; `Not mentioned` and `Uncertain` are explicit sentences rather than Negative evidence.

| Group | Images | Oracle Positive sentences | Negative | Uncertain | Not mentioned | F2-O output mentions |
|---|---|---|---|---|---|---|
| F2-O collapsed | 7 | 13/42 | 8/42 | 3/42 | 18/42 | 2/42 |
| F2-O not collapsed | 17 | 14/102 | 34/102 | 4/102 | 50/102 | 27/102 |

Per-image counts, F1/F2/F3-O mentions and F2-O collapse status are in `E4_B1a_oracle_block_inventory.csv`.

## 8. Neutral-text background versus M–S sensitivity

All image-level comparisons below use the **same 19 images** with at least one CLEAR direction difference. A finding-state change means any of the six extracted states changes; exact text change is counted separately. Target-cell changes use the same five CLEAR-scored findings (19×5=95 cells); N1/C0 has no evidence direction and cannot be classified as following or opposing it.

| Comparison | Exact text changed/images | Any six-state change/images | Five-finding state changes/cells |
|---|---|---|---|
| N1 vs C0 | 19/19 | 19/19 | 36/95 |
| F1 M vs S | 4/19 | 3/19 | 6/95 |
| F2 M vs S | 17/19 | 11/19 | 24/95 |
| F3 M vs S | 15/19 | 13/19 | 32/95 |

N1 versus C0 changes at least one extracted state on 19/19 eligible images, exceeding each M–S comparison's 3/19, 11/19 and 13/19. Arm-level collapse, mention and length for C0/N1/M/S/O are shown in section 4. The neutral-text comparison is a background change rate, not a causal subtraction or a calibrated noise correction.

## 9. CLEAR direction versus reference report

Among **52** Positive/Negative reference-report cells for the five scored findings, CLEAR-matched direction agrees with the report in **42/52**, and shuffled direction in **18/52**. This is an **AUTOMATED PROXY, POST-HOC** comparison, not correctness relative to the radiograph. The cell-level source is `E4_B1a_clear_reference_proxy.csv`.

| Finding | Definite cells | Matched agrees report | Shuffled agrees report |
|---|---|---|---|
| Atelectasis | 4 | 2/4 | 0/4 |
| Cardiomegaly | 13 | 12/13 | 7/13 |
| Consolidation | 13 | 9/13 | 4/13 |
| Edema | 4 | 4/4 | 1/4 |
| Pleural Effusion | 18 | 15/18 | 6/18 |

Split by reference-report state:

| Oracle state | Definite cells | Matched agrees report | Shuffled agrees report |
|---|---|---|---|
| Positive | 23 | 18/23 | 3/23 |
| Negative | 29 | 24/29 | 15/29 |

## Design limitations and two separate questions

- **Original criterion 2:** the ≥4/6 mention requirement is above C0's 71/24 = 2.96/6 baseline and was unmet by every E4-B1 arm. This is a strict development gate, not a model quality estimate.
- **Original criterion 3:** oracle following used all 69 definite cells, including cells where O and S evidence agreed and Pneumothorax with no CLEAR score. Such cells dilute a contrast focused on conflicting evidence. This analysis isolates conflicts post hoc; it cannot retroactively change the frozen gate.
- **Original criterion 4:** any extracted-state change measures sensitivity, not whether the change follows the evidence direction. The directional categories above retain that distinction.
- Reference reports may describe views absent from the frontal input; rule-based labels can omit nuance. No radiologist adjudication or image-grounded truth was added.

**(a) Does the model follow text evidence content?** VERIFIED: prompts/evidence differ, and exact output text changes are directly counted. AUTOMATED PROXY, POST-HOC: full directional following is rare on M–S discordant cells (F1 0/62, F2 2/62, F3 0/62), while N1/C0 changes extracted states on 19/19 eligible images. O does not beat S on the 34 O–S discordant cells in any format. Thus there is no evidence here of *reliable intended-direction following*, although the model's text is prompt-sensitive. UNVERIFIED: semantic reliance on evidence or its causal mechanism.

**(b) Is CLEAR evidence correct for the image?** AUTOMATED PROXY, POST-HOC: matched/shuffled CLEAR directions can be compared with reference-report labels in the definite cells above. UNVERIFIED: radiograph-grounded correctness, unsupported findings, clinical utility or generalization. No E4-B2 proposal or run is made.
