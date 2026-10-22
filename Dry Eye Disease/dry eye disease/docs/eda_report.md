# Exploratory Data Analysis

## Datasets

| Task | Source | Size |
| --- | --- | --- |
| Symptom survey (OSDI-12) | `data/raw/Data.csv` | 230 respondents x 21 raw columns |
| Eye disease images | `data/raw/Dataset/` | 999 images |
| Blink / gaze crops | `data/raw/Blink/` | 10,760 images |

## Severity distribution

| Severity | Respondents | Share |
| --- | --- | --- |
| Normal | 105 | 45.7% |
| Mild | 47 | 20.4% |
| Moderate | 34 | 14.8% |
| Severe | 44 | 19.1% |

## Eye label versus laterality

| class    |   OD (right) |   OS (left) |
|:---------|-------------:|------------:|
| Affected |          533 |          16 |
| Not      |            0 |         450 |

## Key relationships

- Screen time (midpoint hours) vs OSDI: Pearson r = 0.263
- OSDI summary: mean 19.40, sd 15.32, range 0.00-79.17
- Mean age 20.4, 210 female / 20 male

## Data quality findings

1. **18 of 230 respondents (8%) have incomplete frequency answers**, so their OSDI score cannot be computed. The models impute these values with the training-set median.

2. **4 OSDI band(s) contain more than one severity label.** The band 13-23 holds both 42 *Mild* and 10 *Moderate* records at an identical OSDI of 22.917. The recorded `Results` column is therefore not a strict function of the OSDI column, so `Results` is used as ground truth and OSDI is treated as a feature rather than a label generator.

3. **Class imbalance in the blink corpus.** The three gaze classes hold about 3,500 images each while *Closed* and *Open* hold 109 each, and *Partial* holds only 10. *Partial* is dropped because ten samples cannot be split stratified, and the remaining classes are reweighted during training.

4. **Demographic skew.** The sample is 91% female with a mean age of 20.4 and a maximum of 27, so the models are calibrated for a young screen-using cohort and should not be read as population prevalence estimates.

5. **The eye-disease label tracks eye laterality.** Capture filenames encode the eye as `OD` (right) or `OS` (left): 533 of 549 *Affected* images are `OD` while all 450 *Not* images are `OS`. A model can therefore reach high accuracy by learning which eye it is looking at, so the reported score is not evidence that the model detects dry eye pathology.

6. **Several captures per subject.** The 999 images cover only 338 subjects, roughly three captures each. A naive random split leaks subject identity, so the eye model is trained on a subject-grouped split.

7. **Reported `Consent` column carries a single value.** Every row records the same consent sentence, so it has no predictive signal and is excluded.
