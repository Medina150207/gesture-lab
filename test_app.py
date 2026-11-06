"""Run with python test_app.py. Uses shipped weights, no training/downloads."""
from pathlib import Path
import csv
import io
import unittest
import numpy as np
from streamlit.testing.v1 import AppTest
from evaluation import CLASSES, CONDITIONS, add_observation, client_environment, export_csv, summarize

ROOT = Path(__file__).parent


class FieldEvaluationTests(unittest.TestCase):
    def record(self, image, actual, predicted, model='mobilenet_v2'):
        return dict(image_id=image, model=model, actual=actual, predicted=predicted,
                    condition=CONDITIONS[0], mean_ms=10.)

    def test_scoring_and_duplicate_replacement(self):
        rows = []
        for i, label in enumerate(CLASSES):
            rows = add_observation(rows, self.record(str(i), label, label))
        self.assertEqual(summarize(rows)[0]['Accuracy (%)'], 100.)
        self.assertEqual(summarize(rows)[0]['Macro-F1 (3 classes)'], 1.)
        rows = add_observation(rows, self.record('0', 'peace', 'stop'))
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(summarize(rows)[0]['Accuracy (%)'], 200/3)
        self.assertAlmostEqual(summarize(rows)[0]['Macro-F1 (3 classes)'], 5/9)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(export_csv(rows))))), 3)
        self.assertEqual(summarize([]), [])

    def test_invalid_labels_and_absent_classes(self):
        with self.assertRaises(ValueError):
            add_observation([], self.record('0', 'unknown', 'stop'))
        row = summarize([self.record('1', 'peace', 'peace')])[0]
        self.assertEqual(row['Classes represented'], 1)
        self.assertAlmostEqual(row['Macro-F1 (3 classes)'], 1/3)

    def test_browser_detection(self):
        self.assertEqual(client_environment('Mozilla Windows Chrome/140.0 Edg/140.0'), ('Edge 140', 'Windows'))
        self.assertEqual(client_environment(''), ('Unavailable', 'Unavailable'))


def assert_clean(at):
    assert not at.exception, list(at.exception)
    assert not at.error, list(at.error)


def test_record_form():
    script = '''
import streamlit as st
import numpy as np
from app_views import record_field_test
record_field_test('photo1', 'mobilenet_v2', (np.array([.8,.1,.1]), 5., None, None, 0.),
                  {'device': 'CPU', 'browser': 'Test'}, 'Upload photo')
'''
    at = AppTest.from_string(script, default_timeout=30).run()
    at.button[0].click().run()
    assert_clean(at)
    assert at.warning and 'field_records' not in at.session_state
    at.checkbox(key='field_permission').check()
    at.selectbox(key='actual_gesture').set_value('peace')
    at.button[0].click().run()
    assert_clean(at)
    assert len(at.session_state['field_records']) == 1
    at.selectbox(key='actual_gesture').set_value('stop')
    at.button[0].click().run()
    assert_clean(at)
    assert len(at.session_state['field_records']) == 1
    assert at.session_state['field_records'][0]['actual'] == 'stop'
    print('PASS: field-test form, consent gate and duplicate replacement', flush=True)


def test_recognition():
    at = AppTest.from_file(str(ROOT/'app.py'), default_timeout=120).run()
    assert_clean(at)
    assert at.button(key='recognize').disabled
    at.radio(key='input_mode').set_value('Examples').run()
    for model in ['resnet18', 'mobilenet_v2', 'mobilenet_v2_synthetic']:
        at.radio(key='model').set_value(model).run()
        if 'recognition' in at.session_state:
            assert not at.metric, 'Changing model must hide the previous prediction'
        for example in sorted((ROOT/'examples').glob('*.jpg')):
            at.radio(key='example').set_value(example).run()
            at.button(key='recognize').click().run()
            assert_clean(at)
            _, result = at.session_state['recognition']
            probs, ms, overlay, cam, cam_ms = result
            assert np.isfinite(probs).all() and abs(probs.sum()-1) < 1e-5
            assert ms > 0 and cam_ms > 0 and np.isfinite(cam).all()
            assert overlay is not None and cam.shape == (160, 160)
            assert any(m.label == 'Mean inference · 20 runs' for m in at.metric)
        identity = at.session_state['recognition'][0]
        at.slider(key='threshold').set_value(.95).run()
        assert at.session_state['recognition'][0] == identity and at.metric
        at.slider(key='brightness').set_value(.5).run()
        at.button(key='run_stress').click().run()
        assert_clean(at)
        assert at.session_state['stress_result'][1] == .5
        at.slider(key='blur').set_value(2.).run()
        assert len(at.metric) == 1, 'Changed stress settings must hide the previous stress result'
        at.button(key='reset_stress').click().run()
        assert at.slider(key='brightness').value == 1.
        assert at.slider(key='blur').value == 0.
        assert 'stress_result' not in at.session_state
        print('PASS recognition + Grad-CAM + stress:', model, flush=True)
    at.radio(key='input_mode').set_value('Upload photo').run()
    assert not at.metric and at.button(key='recognize').disabled
    for page in ['Model comparison', 'How it works', 'Field tests']:
        at.radio(key='page').set_value(page).run()
        assert_clean(at)
        print('PASS page:', page, flush=True)
    at.session_state['field_records'] = [dict(image_id='sample', model='mobilenet_v2',
        actual='peace', predicted='peace', condition=CONDITIONS[0], mean_ms=10.)]
    at.run()
    assert_clean(at)
    assert at.dataframe[0].value.iloc[0]['Accuracy (%)'] == 100.
    at.button(key='clear_field_tests').click().run()
    assert at.session_state['field_records'] == []


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FieldEvaluationTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    test_record_form()
    test_recognition()
    print('PASS: all automated checks. Physical webcam and new-user accuracy require manual testing.', flush=True)
