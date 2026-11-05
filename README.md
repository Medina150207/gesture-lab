# Gesture Lab

Hand gesture recognition for **peace, stop and thumbs up** by Sapabekova Medina and Alshynova Aiganym, AITU. Streamlit application with uploaded photos, camera snapshots, Grad-CAM explanations, model comparison and a session-only field-test log.

## Run locally

Use Python 3.12. Download **Code → Download ZIP** and extract it first.

On Windows, run `SETUP_WINDOWS.bat` once (internet required), then `START_WINDOWS.bat`. Keep the terminal open while using the app. Stop it with Ctrl+C.

Or run in a terminal from the project directory:

```sh
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Weights and examples are included; recognition does not download weights or retrain. A camera snapshot requires browser permission (localhost or HTTPS). GitHub hosts source code, not the running app. The assignment permits a local run guide.

## What to try

1. Recognize → Examples → choose an image → Recognize gesture. Repeat with each model.
2. Open Grad-CAM and download the explanation. Scores are not calibrated probabilities.
3. Try Upload photo and Camera with new real images, with permission.
4. Run brightness/blur stress tests. These are simulated transformations, not ArtAug.
5. On a new real image, open Record a real-world test, select the true gesture and capture condition, confirm permission and record it. Field tests shows accuracy and macro-F1 by model/condition and exports CSV. Test all three gestures per condition, avoiding duplicates.
6. Model comparison shows reference results, synthetic ablation, controlled robustness, training plots and marked synthetic examples. How it works describes the pipeline, dataset and limitations.

## Evidence and limits

761 real images: train 535, validation 114, test 112. The synthetic variant adds 36 training frames (12/class; 6.3% before augmentation). Real validation/test images stay fixed.

| Experiment | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| A2 ResNet18 reference | 90.18% | 0.9023 |
| A2 MobileNetV2 real-only | 91.96% | 0.9204 |
| MobileNetV2 real + synthetic | 91.07% | 0.9116 |

These are A2 frozen-backbone reference runs, **not an exact reproduction of A1**. Overall improvement from synthetic data was not established. The generator, prompts and seeds remain unverified; no Step-Video-T2V or ArtAug attribution is claimed. See `PARTNER_INPUTS_RU.md` for evidence still needed. HTML reports and the notebook document earlier work and are not a final 6–8-page report.

Live latency is the mean of 20 forward passes on the current image after one warm-up; preprocessing, softmax and Grad-CAM are excluded. Historical CSV timings include score conversion and use a different environment. Do not compare small timing differences across protocols/devices. The sidebar shows host/runtime and approximate client browser/OS; no cookies or IP addresses are displayed or logged.

The model always selects one of three gestures, including for unsupported inputs. The confidence threshold only warns. New people, backgrounds, multiple hands and empty frames need manual evaluation. Field-test records are user-labelled, exploratory and separate from the fixed test set; the app cannot verify their source. Participant-independent splits and publication consent are not established by the code.

Photos are processed on the host (your computer locally, a server if hosted). Uploaded photos are not intentionally saved to disk. Session memory retains results; field-test CSVs contain labels, scores, timings, coarse environment and image fingerprints, not photos. Download the log before closing the session.

## Verify and reproduce

```sh
python test_app.py
```

Tests cover the three included models, example recognition, finite Grad-CAM, mean timing, threshold/state behaviour, stress-test reset, evidence pages, field-test scoring, duplicate replacement and CSV export. They do not verify the physical webcam, consent or accuracy on new users.

Training additionally requires `Gesture_A2_Data` beside the project. The real dataset is not included in this public repository. Do not publish participant photos without consent. `data_manifests/` fixes the documented split.

```sh
python train.py --data ../Gesture_A2_Data
python check_and_stress.py
```

Synthetic ablation (Windows cmd; other shells need the equivalent environment variable):

```bat
set ADD_SYNTHETIC=0
python train_ablation.py --data ../Gesture_A2_Data --output control_results
set ADD_SYNTHETIC=1
python train_ablation.py --data ../Gesture_A2_Data --output synthetic_results
python finish_ablation.py
```

Training downloads ImageNet weights on first run and overwrites results; preserve supplied outputs before rerunning. One fixed augmented view per training image is cached, not fresh augmentation each epoch. Both ablation branches use 40 head-training epochs and seed 42, but more data means more optimizer steps.

Implemented references: [ResNet](https://arxiv.org/abs/1512.03385), [MobileNetV2](https://arxiv.org/abs/1801.04381), [Grad-CAM](https://arxiv.org/abs/1610.02391). A confirmed source and description for the new SOTA component are still required.
