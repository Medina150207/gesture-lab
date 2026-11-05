from pathlib import Path
import io, threading, hashlib
import pandas as pd
import torch
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from core import load, predict, CLASSES

ROOT = Path(__file__).parent
torch.set_num_threads(2)
st.set_page_config(page_title='Gesture Lab', page_icon='✋', layout='wide')
st.markdown('<style>' + (ROOT/'style.css').read_text() + '</style>', unsafe_allow_html=True)
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
        return predict(net, image)

def show_result(result, key):
    probs, ms, overlay, cam, cam_ms = result
    gesture = CLASSES[int(probs.argmax())]
    st.markdown(f'<div class="prediction"><div class="gesture-icon">{ICONS[gesture]}</div><div><div class="result-label">{GESTURES[gesture]}</div><div class="muted">Model confidence</div><div class="score">{probs.max():.1%}</div></div></div>', unsafe_allow_html=True)
    for label, prob in zip(CLASSES, probs):
        st.markdown(f'<div class="prob-row"><span>{GESTURES[label]}</span><div class="track"><div style="width:{float(prob)*100:.3f}%"></div></div><span>{prob:.1%}</span></div>', unsafe_allow_html=True)
    st.caption(f'{LABELS[name]} · Inference {ms:.1f} ms')
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
    page = st.radio('Navigation', ['Recognize','Model comparison','How it works'], label_visibility='collapsed')
    st.divider()
    st.subheader('Recognition settings')
    name = st.selectbox('Model', list(LABELS), format_func=LABELS.get)
    threshold = st.slider('Confidence warning', 0., 1., .6, .05)
    st.caption('Warn when the model score falls below this value.')
    st.markdown('<div class="sidebar-footer">AITU · Computer Vision<br>Medina Sapabekova<br>Aiganym Alshynova</div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">HAND GESTURE RECOGNITION</div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div><h1>A little gesture.<br>A clear result.</h1><p>Upload a photo and explore how your model sees it.</p><div class="hero-tags">PEACE &nbsp; / &nbsp; STOP &nbsp; / &nbsp; THUMBS UP</div></div><div class="hero-hand">✌</div></div>', unsafe_allow_html=True)

if page == 'Recognize':
    left, right = st.columns([1.06, 1], gap='medium')
    image = None
    identity = None
    with left, st.container(border=True, key='input_card'):
        st.subheader('Your image')
        mode = st.radio('Input', ['Upload photo','Camera','Examples'], horizontal=True, label_visibility='collapsed')
        file = None
        if mode == 'Upload photo':
            file = st.file_uploader('Choose a photo of one hand', type=['jpg','jpeg','png'], label_visibility='collapsed')
        elif mode == 'Camera':
            file = st.camera_input('Take a snapshot of one hand')
        else:
            choices = sorted((ROOT/'examples').glob('*.jpg'))
            if choices:
                file = st.selectbox('Example photo', choices, format_func=lambda p:p.stem.replace('_',' '))
        if file is not None:
            try:
                raw = file.read_bytes() if isinstance(file, Path) else file.getvalue()
                identity = hashlib.sha256(raw + name.encode()).hexdigest()
                source = Image.open(io.BytesIO(raw))
                if source.width * source.height > 30_000_000:
                    raise ValueError('Please use an image smaller than 30 megapixels.')
                image = ImageOps.exif_transpose(source).convert('RGB')
                preview = ImageOps.contain(image, (900, 520))
                canvas = Image.new('RGB', (900,520), '#f7eff7')
                canvas.paste(preview, ((900-preview.width)//2,(520-preview.height)//2))
                st.image(canvas, width='stretch')
            except Exception as exc:
                st.error(f'Could not open this image: {exc}')
        else:
            st.markdown('<div class="empty-upload"><div>✋</div><h3>Your next gesture starts here</h3><p>Add a photo, take a snapshot<br>or explore an example.</p></div>', unsafe_allow_html=True)
        if st.button('Recognize gesture', type='primary', width="stretch", disabled=image is None):
            with st.spinner('Recognizing your gesture…'):
                try:
                    st.session_state.recognition = (identity, infer(image))
                except Exception as exc:
                    st.error(f'Recognition failed: {exc}')
    with right, st.container(border=True, key='result_card'):
        st.subheader('Recognition result')
        stored = st.session_state.get('recognition')
        if stored and image is not None and stored[0] == identity:
            show_result(stored[1], 'download_main')
        else:
            st.markdown('<div class="empty-result"><div class="waiting-icon">✧</div><h3>A little moment of discovery</h3><p>Your gesture, confidence score and<br>visual explanation will appear here.</p><div class="class-pills"><span>✌ Peace</span><span>✋ Stop</span><span>👍 Thumbs up</span></div></div>', unsafe_allow_html=True)
    with st.container(border=True, key='stress_card'):
        title, controls, action = st.columns([1.15, 1.6, .6], gap='large')
        with title:
            st.subheader('Stress-test')
            st.caption('Explore changes in lighting and blur.')
        with controls:
            st.session_state.setdefault('brightness', 1.)
            st.session_state.setdefault('blur', 0.)
            brightness = st.slider('Brightness', .2, 1.5, step=.1, key='brightness')
            blur = st.slider('Blur', 0., 4., step=.5, key='blur')
        with action:
            def reset_stress():
                st.session_state.brightness = 1.
                st.session_state.blur = 0.
                st.session_state.pop('stress_result', None)
            st.button('Reset', on_click=reset_stress, width="stretch")
            run_stress = st.button('Run test', disabled=image is None, width="stretch")
        if run_stress:
            altered = ImageEnhance.Brightness(image).enhance(brightness).filter(ImageFilter.GaussianBlur(blur))
            with st.spinner('Testing the altered image…'):
                try:
                    st.session_state.stress_result = (identity, brightness, blur, altered, infer(altered))
                except Exception as exc:
                    st.error(f'Stress test failed: {exc}')
        stress = st.session_state.get('stress_result')
        if stress and image is not None and stress[:3] == (identity, brightness, blur):
            a,b = st.columns(2)
            with a: st.image(stress[3], width=330, caption='Altered input')
            with b: show_result(stress[4], 'download_stress')
        st.caption('Simulated lighting and blur. These adjustments are not ArtAug or video generation.')
elif page == 'Model comparison':
    with st.container(border=True):
        st.subheader('Model comparison')
        p = ROOT/'results'/'comparison.csv'
        if p.exists():
            st.dataframe(pd.read_csv(p), hide_index=True, width='stretch')
            st.caption('Same fixed 112-image real test split. Frozen ImageNet backbones with trained classification heads; A2 reference runs.')
            for suffix in ['curves','confusion']:
                cols = st.columns(2)
                for col, model in zip(cols, ['resnet18','mobilenet_v2']):
                    path = ROOT/'results'/f'{model}_{suffix}.png'
                    if path.exists(): col.image(str(path), width='stretch')
        st.subheader('Synthetic-data ablation')
        p = ROOT/'ablation.csv'
        if p.exists(): st.dataframe(pd.read_csv(p), hide_index=True, width='stretch')
        st.caption('36 selected synthetic frames, 12 per class. Same real validation/test split. Fixed 40 epochs; adding data also adds optimizer updates. Generator identity remains unconfirmed.')
else:
    with st.container(border=True):
        st.subheader('How it works')
        st.write('Photo → RGB and aspect-preserving padding → 160 × 160 → ImageNet normalization → model → class scores → Grad-CAM.')
        st.write('Supported gestures: peace, stop and thumbs up. The model always chooses one of these classes. Other gestures, empty frames or multiple hands may produce confident errors.')
        st.write('The camera takes a snapshot, not live video. The confidence warning is a display setting, not an unknown-gesture detector.')
        st.write('Reference models use real images. The synthetic variant adds 36 selected generated frames. New-user performance is not established because complete participant IDs were unavailable.')
        st.write('With local launch, your photos are processed on your computer. Use photos with permission.')
st.markdown('<div class="bottom-note">GESTURE LAB &nbsp; · &nbsp; AITU / COMPUTER VISION / ASSIGNMENT 2</div>', unsafe_allow_html=True)
