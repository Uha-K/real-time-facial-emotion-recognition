# Dataset Setup

The datasets used for this research are not redistributed in this repository.

## Primary Dataset

The primary experiments use a processed AffectNet subset containing 30,626
images across eight emotion classes.

The experimental dataset contained:

- Original training folder: 16,108 images
- Untouched test folder: 14,518 images
- Total: 30,626 images

This should not be confused with the complete original AffectNet dataset.

## External Evaluation

FERPlus annotations were used together with the corresponding FER2013 images.

Only the FERPlus PrivateTest partition was used for the final zero-adaptation
external evaluation.

After majority-vote filtering, the evaluation set contained 2,729 images.

## Emotion Mapping

The fixed class order used throughout the project is:

0 - anger
1 - contempt
2 - disgust
3 - fear
4 - happy
5 - neutral
6 - sad
7 - surprise

Users should obtain the datasets from their respective original sources and
comply with their licensing and usage requirements.
