# Lung Sound Classification PCEN - Architecture Upgrades

This document outlines the systematic architectural changes and training loop enhancements applied to the model, which drove the validation/test accuracy from a baseline of ~78% up to **85.19%**, while achieving an OVR AUC of **0.9840**.

## 1. Conv2D Subnetwork Capacity (Scaling Up)
**Problem:** The original model had a very small bottleneck, peaking at 32 convolutional filters. For a complex 10-class problem processing 128 Mel-frequency bins over 6 seconds of audio, the model lacked the parameter precision to separate overlapping disease signatures.
**Change:** The convolutional architecture was significantly expanded:
*   `Layer 1`: 8 $\rightarrow$ **16** filters
*   `Layer 2`: 16 $\rightarrow$ **32** filters
*   `Layer 3`: 16 $\rightarrow$ **64** filters
*   `Layer 4`: 32 $\rightarrow$ **128** filters
*   `Layer 5`: 32 $\rightarrow$ **128** filters
*   `Dense Head`: 64 $\rightarrow$ **128** units.

## 2. Training Stability (Batch Normalization)
**Problem:** Deep CNNs natively suffer from Internal Covariate Shift, where the distribution of inputs to hidden layers changes as the weights below them update, causing slow and sometimes unstable training. The original script initialized a single `BatchNorm2d(1)` module but never called it in the `forward()` pass.
**Change:** `nn.BatchNorm2d` block layers were injected immediately following every `Conv2d` layer (prior to the `ReLU`/`Tanh` activations). A `nn.BatchNorm1d` layer was also added before the final classification head. This smoothed the loss landscape, enabling the deeper 128-filter layers to learn without exploding gradients.

## 3. Regularization & Data Augmentation (SpecAugment)
**Problem:** Models analyzing spectrograms easily overfit to highly specific, fixed noise patterns in the training set (such as the specific hum of a microphone or a patient's breathing cadence—leading to the "Patient Leakage" phenomenon we discovered).
**Change:** Natively inside the PyTorch `forward()` function (active only when `self.training == True`), two SpecAugment transforms were implemented:
*   **`FrequencyMasking(15)`**: Randomly zeros out up to 15 contiguous frequency bands in the Mel-spectrogram.
*   **`TimeMasking(35)`**: Randomly zeros out up to 35 contiguous time frames.
This forces the model to learn broader context features (like the shape of a crackle) rather than relying on an isolated pixel. Because the model capacity was increased, Dropout was also raised from `0.2` to `0.4`.

## 4. Preprocessing Precision (Multi-Rate PCEN)
**Change:** Integrated the updated `TVaryingPCEN` algorithm (or properly parameterized `speechbrain` PCEN) as the primary dynamic range compression layer directly following Mel-spectrogram extraction. The parameters were mathematically tuned (e.g., floor raised from $1e-12$ to $1e-6$, delta and root reduced to $1.0$ and $0.5$) to properly squash loud background noise without clipping the subtle acoustic anomalies found in class-imbalanced diseases (like Fibrosis).

## 5. Optimization & Fine-tuning (Learning Rate Scheduler)
**Problem:** A static learning rate of `0.0001` over 30 epochs leads to a high early loss drop but prevents the optimizer from settling into the absolute deepest minimum of the loss valley (it "bounces" around the optimal weights).
**Change:** Implemented PyTorch's `ReduceLROnPlateau`. The scheduler monitors `val_acc` at the end of every epoch. If accuracy stagnates for 4 consecutive epochs, it slices the learning rate exactly in half ($factor=0.5$). In our target 85% run, this allowed the model to rapidly reach 80% with large steps, and then take incredibly small steps during the final 5 epochs to perfect its precision.
