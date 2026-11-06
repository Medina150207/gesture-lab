from pathlib import Path
import io, threading, hashlib
import pandas as pd
import torch
import streamlit as st
from PIL import Image, ImageOps
from core import load, predict, CLASSES
from stress_editor import render_editor, tested_photo
from app_views import show_environment, record_field_test, show_field_tests, show_comparison, show_method

ROOT = Path(__file__).parent
torch.set_num_threads(2)
st.set_page_config(page_title='Gesture Lab', page_icon='✋', layout='wide')
st.markdown('<style>' + (ROOT/'style.css').read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)
LABELS = {'resnet18':'ResNet18', 'mobilenet_v2':'MobileNetV2', 'mobilenet_v2_synthetic':'MobileNetV2 + synthetic'}
GESTURES = {'peace':'Peace', 'stop':'Stop', 'thumbs_up':'Thumbs up'}
ICONS = {'peace':'✌', 'stop':'✋', 'thumbs_up':'👍'}

@st.cache_resource
def get_model(name):
    path = ROOT/'synthetic_results'/'mobilenet_v2.pt' if name == 'mobilenet_v2_synthetic' else ROOT/'results'/f'{name}.pt'
    net, metadata = load(path)
    return net, threading.Lock()

def infer(image):
    net, lock = get_model(name)
    with lock:
        return predict(net, image, repeats=20)

def show_result(result, key):
    probs, ms, overlay, cam, cam_ms = result
    gesture = CLASSES[int(probs.argmax())]
    st.markdown(f'<div class="prediction"><div class="gesture-icon">{ICONS[gesture]}</div><div><div class="result-label">{GESTURES[gesture]}</div><div class="muted">Model score</div><div class="score">{probs.max():.1%}</div></div></div>', unsafe_allow_html=True)
    for label, prob in zip(CLASSES, probs):
        st.markdown(f'<div class="prob-row"><span>{GESTURES[label]}</span><div class="track"><div style="width:{float(prob)*100:.3f}%"></div></div><span>{prob:.1%}</span></div>', unsafe_allow_html=True)
    st.metric('Mean inference · 20 runs', f'{ms:.1f} ms')
    st.caption(f'{LABELS[name]} · CPU · one warm-up excluded. Mean of 20 forward passes on this image; excludes preprocessing, softmax and Grad-CAM.')
    st.caption('Scores are not calibrated probabilities of correctness. Only one hand and the three supported gestures are evaluated.')
    if name == 'mobilenet_v2_synthetic':
        st.info('Training includes 36 synthetic frames. Generator identity is unverified; this is not evidence of a confirmed SOTA method.')
    if probs.max() < threshold:
        st.warning('Low confidence. Try better light or a clearer hand position.')
    with st.expander('View explanation · Grad-CAM'):
        st.image(overlay, width=300, caption='Areas influencing the predicted class')
        if cam.max() == 0:
            st.info('No positive activation map for this input.')
        st.caption('Warm colours show positive influence on the class score. This is not a hand mask or a guarantee of correctness.')
        st.caption(f'Grad-CAM: {cam_ms:.1f} ms. Inference excludes image preparation, warm-up and explanation.')
        buf = io.BytesIO(); overlay.save(buf, format='PNG')
        st.download_button('Download explanation', buf.getvalue(), 'gradcam.png', 'image/png', key=key)

with st.sidebar:
    st.markdown('<div class="brand"><span>✧</span> Gesture Lab</div>', unsafe_allow_html=True)
    page = st.radio('Navigation', ['Recognize','Model comparison','Field tests','How it works'], label_visibility='collapsed', key='page')
    st.divider()
    st.subheader('Recognition settings')
    name = st.radio('Model', list(LABELS), format_func=LABELS.get, key='model')
    threshold = st.slider('Confidence warning', 0., 1., .6, .05, key='threshold')
    st.caption('Warn when the model score falls below this value.')
    environment = show_environment()
    st.markdown('<div class="sidebar-footer">AITU · Computer Vision<br>Medina Sapabekova<br>Aiganym Alshynova</div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">HAND GESTURE RECOGNITION</div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div><h1>A little gesture.<br>A clear result.</h1><p>Upload a photo and explore how your model sees it.</p><div class="hero-tags">PEACE &nbsp; / &nbsp; STOP &nbsp; / &nbsp; THUMBS UP</div></div><div class="hero-hand">✌</div></div>', unsafe_allow_html=True)

if page == 'Recognize':
    left, right = st.columns([1.06, 1], gap='medium')
    image = None
    identity = None
    with left, st.container(border=True, key='input_card'):
        st.subheader('Your image')
        choices = sorted((ROOT/'examples').glob('*.jpg'))
        modes = ['Examples', 'Upload photo', 'Camera'] if choices else ['Upload photo', 'Camera']
        mode = st.radio('Input', modes, horizontal=True, label_visibility='collapsed', key='input_mode')
        st.caption('Use photos with permission. Images are processed on the computer hosting this app and are not intentionally saved to disk by the app.')
        file = None
        if mode == 'Upload photo':
            file = st.file_uploader('Choose a photo of one hand', type=['jpg','jpeg','png'], label_visibility='collapsed')
        elif mode == 'Camera':
            file = st.camera_input('Take a snapshot of one hand')
            st.caption('Camera takes a still photo. Allow browser permission; a hosted app needs HTTPS, local use works on localhost.')
        else:
            file = st.radio('Example photo', choices, format_func=lambda p:p.stem.replace('_',' '), key='example', horizontal=True)
            st.caption('Real validation examples for demonstration; these are not new independent test images.')
        if file is not None:
            try:
                raw = file.read_bytes() if isinstance(file, Path) else file.getvalue()
                identity = hashlib.sha256(raw + name.encode()).hexdigest()
                source = Image.open(io.BytesIO(raw))
                if source.width * source.height > 30_000_000:
                    raise ValueError('Please use an image smaller than 30 megapixels.')
                image = ImageOps.exif_transpose(source).convert('RGB')
            except Exception as exc:
                st.error(f'Could not open this image: {exc}')
        else:
            st.markdown('<div class="empty-upload"><div>✋</div><h3>Your next gesture starts here</h3><p>Add a photo, take a snapshot<br>or explore an example.</p></div>', unsafe_allow_html=True)
        if image is not None:
            st.subheader('Stress-test')
            editor = render_editor(image, identity)
            if editor.reset:
                st.session_state.pop('stress_result', None)
            if editor.test:
                with st.spinner('Testing the adjusted photo…'):
                    try:
                        adjusted, settings = tested_photo(editor.test)
                        st.session_state.stress_result = (identity, settings, infer(adjusted))
                    except Exception as exc:
                        st.error(f'Stress test failed: {exc}')
        elif choices:
            def use_example():
                st.session_state.input_mode = 'Examples'
            st.button('Try an example photo', on_click=use_example)
        if st.button('Recognize original photo', type='primary', width="stretch", disabled=image is None, key='recognize'):
            st.session_state.pop('recognition', None)
            with st.spinner('Recognizing your gesture…'):
                try:
                    st.session_state.recognition = (identity, infer(image))
                except Exception as exc:
                    st.error(f'Recognition failed: {exc}')
    with right, st.container(border=True, key='result_card'):
        st.subheader('Recognition result')
        stress = st.session_state.get('stress_result')
        if image is not None and stress and stress[:2] == (identity, editor.settings):
            st.subheader('Adjusted image result')
            show_result(stress[2], 'download_stress')
        stored = st.session_state.get('recognition')
        if stored and image is not None and stored[0] == identity:
            st.subheader('Original photo result')
            show_result(stored[1], 'download_main')
            if mode != 'Examples':
                record_field_test(hashlib.sha256(raw).hexdigest(), name, stored[1], environment, mode)
        elif image is None or not stress or stress[:2] != (identity, editor.settings):
            st.markdown('<div class="empty-result"><div class="waiting-icon">✧</div><h3>A little moment of discovery</h3><p>Your gesture, confidence score and<br>visual explanation will appear here.</p><div class="class-pills"><span>✌ Peace</span><span>✋ Stop</span><span>👍 Thumbs up</span></div></div>', unsafe_allow_html=True)
elif page == 'Model comparison':
    show_comparison(ROOT)
elif page == 'Field tests':
    show_field_tests()
else:
    show_method(ROOT)
st.markdown('<div class="bottom-note">GESTURE LAB &nbsp; · &nbsp; AITU / COMPUTER VISION / ASSIGNMENT 2</div>', unsafe_allow_html=True)
