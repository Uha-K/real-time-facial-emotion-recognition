# Model Checkpoints

Large trained model files are not stored directly in this repository.

The final selected model is a class-weighted, fine-tuned ResNet50 trained on
the processed AffectNet subset.

## Final Model

Architecture: ResNet50  
Output classes: 8  
Parameters: approximately 23.52 million  
Checkpoint size: approximately 90.05 MB

AffectNet test performance:

- Accuracy: 59.42%
- Macro-F1: 55.17%
- Weighted-F1: 58.46%

FERPlus zero-adaptation performance:

- Accuracy: 9.23%
- Macro-F1: 7.25%

The final deployment model was also exported using TorchScript.
