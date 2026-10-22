# Dry Eye Disease Detection

A multi-source screening system for dry eye disease (DED) that combines three
independent evidence streams behind a single dashboard:

| Track | Input | Model | Task |
| --- | --- | --- | --- |
| Symptom | 12-item OSDI questionnaire, 230 respondents | Regularised MLP + classical baselines | Normal / Mild / Moderate / Severe |
| Eye image | Clinical infrared photographs, 999 images across 338 subjects | MobileNet-style depthwise separable CNN | Affected / Not affected |
| Blink & gaze | Eye crops, 10,750 images | VGG-19-style five-stage CNN | Gaze direction and open/closed state |

The project runs fully offline on CPU. No Kaggle credentials or downloads are
needed, because the three datasets are already present under `data/raw/`.

---

## Read this first

Two findings dominate every number in this repository, and both are reported
here rather than buried:

1. **The severity labels are not a strict function of the OSDI index.** Eleven
   records share an identical OSDI of 22.917 yet are split between *Mild* and
   *Moderate*. The recorded `Results` column is therefore used as ground truth,
   and OSDI is treated as an input feature rather than a label generator.
2. **The eye-image label tracks eye laterality.** Capture filenames encode the
   eye as `OD` (right) or `OS` (left): 533 of 549 *Affected* images are `OD`,
   while all 450 *Not* images are `OS`. A model can score well by learning
   which eye it is looking at, so a high accuracy on this track is *not*
   evidence that the model detects dry eye pathology.

Full detail, including three further constraints, is in
[`docs/eda_report.md`](docs/eda_report.md).

---

## Results

| Task | Model | Metric | Value | Caveat |
| | --- | --- | --- | --- |
| Severity, 4-class | Random forest | macro F1 | 0.9485 ± 0.0298 | Bounded by label inconsistency |
| Dry eye vs healthy | Histogram gradient boosting | macro F1 | 0.9861 | OSDI is the dominant feature |
| Eye image, binary | Linear baseline | test accuracy | 0.9000 | Shipped over the CNN; laterality confound |
| Blink and gaze, 5-class | VGG-19-style CNN | test accuracy | 0.9488 | macro F1 0.8447; `Open` recall 0.41 |

Full severity leaderboard over six models (5-fold, 6 repeats):

| Model | Macro F1 | Accuracy |
| --- | --- | --- |
| Random forest | 0.9485 | 0.9609 |
| Histogram gradient boosting | 0.9465 | 0.9587 |
| RBF SVM | 0.8456 | 0.8761 |
| Logistic regression | 0.8246 | 0.8638 |
| Keras MLP | 0.7886 | 0.8413 |
| scikit-learn MLP | 0.5253 | 0.6688 |

Permutation importance puts `osdi` first (+0.045 macro F1 when shuffled),
followed by `osdi_standard` and `symptom_burden`, which is the expected result
given how the severity labels were constructed.

Two model-selection decisions turned out to matter far more than architecture:

- **Per-image standardisation and a subject-grouped split** moved the eye CNN
  from 0.45 to 0.9356 test accuracy on the full 999-image corpus. The first
  attempt memorised the training split at 96% and generalised at chance.
- **Capped square-root class weights** moved the blink CNN from about 0.25 to
  0.9488. Textbook inverse-frequency weighting demanded the full 30:1 ratio
  and the network simply predicted the two 109-image classes.

A third decision matters for deployment. The project caps every class at 250
images (`scripts/reduce_dataset.py`) so the whole pipeline reproduces on a
laptop without the 235 MB corpus. On that capped subset the eye CNN scores
0.50 while the logistic-regression baseline holds 0.90, so the dashboard ships
the baseline and `EyeImagePredictor` picks the winner by held-out accuracy
(`selected_model` in `artifacts/eye_metrics.json`). The blink track is the
reverse: the same VGG-19 collapses to 0.26 on 954 images and needs the full
10,750-crop corpus, so it is trained with `--full`.

Both are reported rather than tuned away. The eye track is better described as
a brightness threshold than as learned texture, and 533 of 549 "affected"
images are right eyes, so even a perfect score is not evidence of pathology
detection.

---

## Setup

```bash
pip install -r requirements.txt
```

The datasets live behind a symlink so the 235 MB corpus stays out of version
control:

```
data/raw -> ../../Source Code/Eye DIsease
├── Data.csv        OSDI-12 survey (an XLSX workbook despite the suffix)
├── Dataset/        Affected/ and Not/ infrared eye photographs
└── Blink/          gaze-direction and eye-state crops
```

## Reproduce everything

```bash
./scripts/run_all.sh
```

That runs the analysis, trains all three model families, regenerates the
report and presentation, and executes the test suite. To run the dashboard
afterwards:

```bash
python3 -m streamlit run app/app.py
```

If `python3` on your `PATH` is not the environment holding TensorFlow, pass it
explicitly: `PYTHON=/path/to/python ./scripts/run_all.sh`.

## Individual steps

```bash
python3 scripts/eda.py                        # figures + data-quality report
python3 scripts/train_symptom.py              # survey models, 5-fold x 6 repeats
python3 scripts/train_eye.py --subject-split  # eye CNN, no subject leakage
python3 scripts/train_blink.py                # blink CNN, the slow step
python3 scripts/make_report.py                # .docx and .pptx
python3 -m pytest                             # test suite
ruff check .                                   # lint
```

## Layout

```
src/ded/
  config.py         paths, label maps, severity bands, class definitions
  survey.py         survey loading, OSDI computation, feature engineering
  images.py         image loading, per-image standardisation, splits
  cnn_pipeline.py   shared CNN training loop and scoring
  inference.py      predictors consumed by the dashboard
  metrics.py        cross-validation harness and JSON serialisation
  models/
    symptom.py      classical baselines plus the regularised MLP
    eye.py          MobileNet-style depthwise separable CNN
    blink.py        VGG-19-style five-stage CNN
scripts/            eda, training, report generation, run_all
app/app.py          Streamlit dashboard
tests/              pytest suite
docs/               eda_report.md, figures, generated .docx and .pptx
artifacts/          trained models and metrics JSON
```

Regenerating the documents produces `docs/Dry_Eye_Disease_Report.docx` (12
chapters, 6 tables, 6 figures) and `docs/Dry_Eye_Disease_Presentation.pptx`
(9 slides), both populated from the artifacts rather than hard-coded.

## How the OSDI index was verified

The workbook ships a precomputed `OSDI` column but does not define it. Rather
than assume the published definition, the weighting was recovered by exhaustive
search over per-item weights and normalisers. The result: **all twelve items
carry a weight of 2, and the raw sum is divided by 96.**

`survey.osdi_index()` reproduces the shipped column to within `3.3e-09` on every
complete row, and `tests/test_survey.py` asserts it. `survey.osdi_standard()`
also provides the published variant (symptoms weighted 2, environment weighted
1, normalised by the maximum raw score of 84) as a second feature.

## Model notes

**Symptom models.** Five scikit-learn baselines plus a Keras MLP share one
median-imputer preprocessor. Evaluation is repeated stratified cross-validation,
and the harness is asserted in the test suite to agree with scikit-learn's own
`cross_val_score`, so a regression in the out-of-fold bookkeeping fails the
build rather than silently flattering the results.

**Eye CNN.** Trained from scratch on per-image standardised grayscale crops.
Per-image standardisation matters because global brightness across these
infrared frames varies widely. The network is deliberately narrow: the first
version used 519k parameters, memorised the training split to 96% accuracy and
generalised at chance. Reducing to 135k parameters with stronger dropout, plus
splitting by subject rather than by image, moved test accuracy from 0.45 to
0.94. A logistic-regression baseline on the same pixels is fitted alongside so
the CNN can be judged against a model with far less capacity.

**Blink CNN.** Keeps the full VGG-19 topology of five max-pooled stages and
sixteen convolutions but halves every stage's channel count, giving roughly
5.1M parameters against reference VGG-19's 138M. Inverse-frequency class
weights counter the 30:1 imbalance between the gaze classes and the
open/closed classes.

## Limitations

- The cohort is 91% female with a mean age of 20.4 and a maximum of 27, so
  results are calibrated for a young screen-using population and must not be
  read as prevalence estimates.
- 18 of 230 respondents left frequency items unanswered and receive median
  imputation.
- The blink corpus is heavily imbalanced; `Partial` holds only 10 images and is
  excluded because it cannot be split stratified.
- The severity score is bounded above by label inconsistency, not model
  capacity.
- Neither CNN is a medical device. This is decision support, and it must not be
  used to diagnose.

## References

- Wolff J, Jones D. Dry eye and the ocular surface. *Ocular Surface* 2007;5(2):78-83.
- Craig JP et al. TFOS DEWS II Definition and Classification Report. *Ocular Surface* 2017;15(3):276-283.
- Howard AG et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861.
- Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv:1409.1556.
- Pedregosa F et al. Scikit-learn: Machine Learning in Python. *JMLR* 2011;12:2825-2830.