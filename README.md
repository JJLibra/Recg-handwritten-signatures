# 🤖 Recg-handwritten-signatures

## Current optimal training results

### Model 1.0

[Wavelet coefficient average power spectrum power spectrum] + gradient boosting algorithm

- Accuracy: 0.93
- Recall: 0.93
- F1 score: 0.93

Overfitting, deprecated.

### Model 2.0

[MFCC, zero-crossing rate, spectrum centroid, spectrum contrast, spectrum roll-off point] + support vector machine

- Accuracy: 0.88
- Recall: 0.88
- F1 score: 0.88

Although the basic indicators are reduced, there is no overfitting phenomenon.

## Analyze robustness

### Handwriting distance robustness

|   Places   |  Chence  | Lijunjie  | Wusiyuan  | Xuzhaoqi  | Average  |
|:----------:|:--------:|:---------:|:---------:|:---------:|:--------:|
|      1     |  86.67%  |  93.33%   |  56.67%   |  83.33%   |  80.00%  |
|      2     |  86.67%  |  93.94%   |  78.79%   |  93.55%   |  88.24%  |
|      3     |  63.33%  |  83.33%   |  66.67%   |  86.67%   |  75.00%  |

### Handwriting speed robustness

| Speed | Chence | Lijunjie | Wusiyuan | Xuzhaoqi | Average |
|:----------:|:--------:|:---------:|:---------:|:---------:|:--------:|
| 0.5 | 80.00% | 100% | 100% | 100% | 95.00% |
| 2 | 57.14% | 100% | 100% | 93.33% | 87.62% |
| 3 | 60.00% | 100% | 100% | 100% | 90.00% |