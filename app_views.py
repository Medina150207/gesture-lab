"""Evidence and user-labelled evaluation views for Gesture Lab."""
import json
import platform
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps, ImageDraw
from evaluation import CLASSES, CONDITIONS, add_observation, client_environment, export_csv, summarize


def show_environment():
    try:
        agent = st.context.headers.get('User-Agent', '')
    except (AttributeError, RuntimeError):
        agent = ''
    browser, client_os = client_environment(agent)
    env = {'device': 'CPU', 'server_os': platform.system(),
           'python': platform.python_version(), 'torch': torch.__version__,
           'threads': torch.get_num_threads(), 'browser': browser, 'client_os': client_os}
    st.caption(f'Inference host: CPU · {env["server_os"]} · {env["threads"]} threads')
    st.caption(f'Browser: {browser} · Client OS: {client_os}')
    with st.expander('Runtime details'):
        st.json(env)
        st.caption('Browser details are approximate, based on User-Agent. Unavailable in automated tests or if the browser does not provide them. No IP address or cookies are displayed or logged.')
    return env


def record_field_test(image_id, model, result, env, source):
    with st.expander('Record a real-world test'):
        st.caption('Use a new real photo, not a training/validation image or generated media. Labels are supplied by you; the app cannot verify the source. Select one primary capture condition.')
        with st.form('field_observation'):
            actual = st.selectbox('Actual gesture', CLASSES, key='actual_gesture')
            condition = st.selectbox('Capture condition', CONDITIONS, key='capture_condition')
            confirmed = st.checkbox('This is a new real photo and I have permission to use it.', key='field_permission')
            save = st.form_submit_button('Record result')
        if save:
            if not confirmed:
                st.warning('Confirm the photo is new, real and used with permission before recording.')
            else:
                probs, ms, *_ = result
                record = {'image_id': image_id, 'model': model, 'actual': actual,
                          'predicted': CLASSES[int(probs.argmax())], 'condition': condition,
                          'score': float(probs.max()), 'mean_ms': float(ms), 'timing_runs': 20,
                          'source': source, 'recorded_utc': datetime.now(timezone.utc).isoformat(), **env}
                st.session_state.field_records = add_observation(st.session_state.get('field_records', []), record)
                st.success('Recorded. Open Field tests to view or download results. Re-recording this photo/model replaces its previous entry.')


def show_field_tests():
    st.subheader('Real-world field tests')
    st.write('On Recognize, upload a new photo or take a camera snapshot, recognize the gesture, then open “Record a real-world test”. Test all three gestures under each condition, using new people and sessions where possible.')
    st.caption('This session-only log is separate from the fixed 112-image test set and from simulated brightness/blur tests. Examples cannot be recorded. Duplicate photo/model entries are replaced. Different crops or copies can still be duplicates; avoid recording them.')
    records = st.session_state.get('field_records', [])
    if not records:
        st.info('No real-world test results yet. No accuracy is claimed until you record labelled photos.')
        return
    st.dataframe(pd.DataFrame(summarize(records)), hide_index=True, width='stretch')
    st.caption('Macro-F1 averages all three classes; absent classes count as zero. Include every gesture before interpreting it. Small, self-selected samples are exploratory, not proof of generalisation.')
    st.download_button('Download field-test CSV', export_csv(records).encode('utf-8-sig'),
                       'gesture_lab_field_tests.csv', 'text/csv', key='export_field_tests')
    with st.expander('Individual observations'):
        st.dataframe(pd.DataFrame(records), hide_index=True, width='stretch')
    st.caption('Only labels, scores, timings, coarse environment and an image fingerprint are logged. No photo files are included in the CSV. Download before closing the session; nothing is saved automatically.')
    if st.button('Clear test log', key='clear_field_tests'):
        st.session_state.field_records = []
        st.rerun()


def metric_table(frame):
    table = frame.copy()
    if 'model' in table:
        table['model'] = table['model'].replace({'resnet18': 'ResNet18', 'mobilenet_v2': 'MobileNetV2'})
    if 'training' in table:
        table['training'] = table['training'].replace({'real_only': 'Real only', 'real_plus_36_synthetic': 'Real + 36 synthetic'})
    if 'variant' in table:
        table['variant'] = table['variant'].replace({'control_results': 'Real only', 'synthetic_results': 'Real + synthetic'})
    for col in ['accuracy', 'macro_precision', 'macro_recall']:
        if col in table:
            table[col] = (table[col]*100).round(2)
    return table.rename(columns={'model': 'Model', 'accuracy': 'Accuracy (%)',
        'macro_precision': 'Macro precision (%)', 'macro_recall': 'Macro recall (%)',
        'macro_f1': 'Macro-F1', 'best_epoch': 'Best epoch',
        'mean_forward_ms': 'Saved mean inference (ms)', 'test_images': 'Test images',
        'training': 'Training data', 'variant': 'Variant', 'condition': 'Condition', 'n': 'Images'})


def show_comparison(root):
    st.subheader('A2 reference models')
    reference = pd.read_csv(root/'results/comparison.csv')
    st.dataframe(metric_table(reference)[['Model', 'Accuracy (%)', 'Macro-F1', 'Saved mean inference (ms)', 'Test images']], hide_index=True, width='stretch')
    st.caption('Same 112 real test images, one split and seed. Both ImageNet backbones are frozen; only classification heads were trained. Saved timings are historical benchmarks, not your current device speed.')
    st.info('A1 comparison pending: these are standardised A2 reference runs. Original A1 checkpoints, results and test protocol are needed before claiming A1 → A2 improvement.')
    st.download_button('Download reference metrics', (root/'results/comparison.csv').read_bytes(), 'a2_reference_metrics.csv', 'text/csv')
    with st.expander('Saved benchmark environment'):
        st.json(json.loads((root/'results/environment.json').read_text(encoding='utf-8')))
        st.caption('Historical benchmark code timed forward pass plus conversion to class scores. The current live mean measures forward passes only; do not compare small timing differences across protocols/devices.')
    with st.expander('Reference training curves and confusion matrices'):
        for suffix in ['curves', 'confusion']:
            for col, model in zip(st.columns(2), ['resnet18', 'mobilenet_v2']):
                col.image(str(root/'results'/f'{model}_{suffix}.png'), caption=f'{model} · {suffix}', width='stretch')
    st.subheader('Ablation: adding synthetic frames')
    ablation = pd.read_csv(root/'ablation.csv')
    st.dataframe(metric_table(ablation)[['Training data', 'Accuracy (%)', 'Macro-F1', 'Saved mean inference (ms)', 'Test images']], hide_index=True, width='stretch')
    real = ablation.loc[ablation.training == 'real_only'].iloc[0]
    synthetic = ablation.loc[ablation.training == 'real_plus_36_synthetic'].iloc[0]
    delta = (synthetic.accuracy-real.accuracy)*100
    st.write(f'Adding 36 synthetic frames changes test accuracy by {delta:+.2f} percentage points and macro-F1 by {synthetic.macro_f1-real.macro_f1:+.4f}. Overall improvement is not established.')
    st.caption('535 real training photos vs 535 real + 36 synthetic frames (12/class; 6.3% synthetic before augmentation). Same real validation/test sets; 40 epochs and seed 42. More data means more optimizer updates; only one seed was tested.')
    st.download_button('Download ablation metrics', (root/'ablation.csv').read_bytes(), 'synthetic_ablation.csv', 'text/csv')
    with st.expander('Synthetic-variant learning curves and confusion matrix'):
        for col, suffix in zip(st.columns(2), ['curves', 'confusion']):
            col.image(str(root/'synthetic_results'/f'mobilenet_v2_{suffix}.png'), width='stretch')
    st.subheader('Controlled robustness checks')
    stress = pd.read_csv(root/'ablation_stress.csv')
    st.dataframe(metric_table(stress), hide_index=True, width='stretch')
    st.caption('original = unchanged test set; dark_50pct = brightness ×0.5; blur_radius2 = Gaussian blur radius 2. These are simulated changes, not measured environments or ArtAug. The darkening gain is one image out of 112; it does not establish a general benefit.')
    st.download_button('Download robustness metrics', (root/'ablation_stress.csv').read_bytes(), 'controlled_robustness.csv', 'text/csv')
    show_synthetic_gallery(root)


def show_synthetic_gallery(root):
    with st.expander('Synthetic training examples and provenance'):
        st.warning('SYNTHETIC MEDIA — generator/model, prompts and seeds were not supplied. Do not attribute these frames to Step-Video-T2V or ArtAug. No original generated video is available in this repository.')
        for col, gesture in zip(st.columns(3), CLASSES):
            images = sorted((root/'synthetic_train'/gesture).glob('*.jpg'))
            if images:
                with Image.open(images[0]) as source:
                    thumbnail = ImageOps.contain(source.convert('RGB'), (480, 360))
                marked = Image.new('RGB', (480, 400), '#fff8fc')
                marked.paste(thumbnail, ((480-thumbnail.width)//2, (360-thumbnail.height)//2))
                draw = ImageDraw.Draw(marked)
                draw.rectangle((0, 360, 480, 400), fill='#502347')
                draw.text((12, 372), 'SYNTHETIC - generator unverified', fill='white')
                col.image(marked, caption=f'{gesture} · synthetic training frame', width='stretch')
        st.caption('36 selected original frames from seven sequences inferred from filenames. These are training examples, not independent evaluation images. Source provenance and permission to publish still need confirmation.')


def show_method(root):
    st.subheader('How it works')
    st.markdown('**Photo / camera snapshot → RGB + aspect-preserving padding → 160 × 160 → ImageNet normalisation → classifier → class scores + Grad-CAM**')
    st.write('Supported gestures: peace, stop and thumbs up. The camera captures a still image, not a video stream. ResNet18 and MobileNetV2 use frozen ImageNet features and trained classification heads.')
    st.markdown('**Training experiment:** real images → fixed train/validation/test split → real-only control **or** real + 36 synthetic training frames → same real test set.')
    st.caption('A confirmed new SOTA method is still pending. The synthetic-data experiment is available, but its generator cannot yet be identified.')
    st.subheader('Dataset and evaluation')
    summary = json.loads((root/'data_manifests/summary.json').read_text(encoding='utf-8'))
    st.dataframe(pd.DataFrame(summary['split_counts']).T.rename_axis('Split'), width='stretch')
    st.write('761 real images: 535 train, 114 validation, 112 test. Synthetic frames enter training only. Similar-image groups were kept together, but participant identities are incomplete; independence by person is not guaranteed.')
    st.subheader('What the explanation means')
    st.write('Grad-CAM weights the last convolutional feature maps using gradients for the predicted class. Warm areas contribute positively to that score. The heatmap is not a hand segmentation mask and does not prove that the model learned the right concept.')
    st.subheader('Limitations, consent and privacy')
    st.write('Unknown gestures, empty frames, multiple hands, new participants or busy backgrounds may produce confident errors. Scores are not calibrated. The confidence threshold only displays a warning; it does not reject unsupported inputs.')
    st.write('Use images with permission and confirm publication rights before sharing people or datasets. Label generated media. This is an educational prototype, not a validated sign-language translator or a safety-critical control system.')
    st.write('Processing runs on the host computer: your computer for local launch, a server if hosted. The app does not intentionally write uploaded photos to disk. Recognition results and field-test records remain in session memory; model weights are cached. Download field-test CSVs yourself and close the session when finished.')
    st.subheader('Method references')
    st.markdown('- [ResNet paper](https://arxiv.org/abs/1512.03385)\n- [MobileNetV2 paper](https://arxiv.org/abs/1801.04381)\n- [Grad-CAM paper](https://arxiv.org/abs/1610.02391)')
    st.caption('These describe the implemented reference models and explanation. A source and explanation for the new SOTA component must be added once its identity is confirmed.')
