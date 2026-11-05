"""Session-only, user-labelled field tests; never mix with saved test-set metrics."""
import csv
import io
import re
from collections import defaultdict

CLASSES = ('peace', 'stop', 'thumbs_up')
CONDITIONS = ('Indoor / normal light', 'Indoor / low light', 'Backlight',
              'Outdoor / daylight', 'Busy background', 'Other')


def client_environment(agent):
    """Display only coarse browser/OS; never expose headers, IPs or cookies."""
    browser = 'Unavailable'
    for token, label in [('Edg/', 'Edge'), ('OPR/', 'Opera'), ('Firefox/', 'Firefox'),
                         ('Chrome/', 'Chrome'), ('Version/', 'Safari')]:
        match = re.search(re.escape(token) + r'(\d+)', agent)
        if match:
            browser = f'{label} {match.group(1)}'
            break
    os_name = next((name for token, name in [('Android', 'Android'), ('iPhone', 'iOS'),
                    ('iPad', 'iPadOS'), ('Windows', 'Windows'), ('Macintosh', 'macOS'),
                    ('Linux', 'Linux')] if token in agent), 'Unavailable')
    return browser, os_name


def add_observation(records, record):
    """Replace repeated labels of the same photo/model instead of inflating N."""
    if record['actual'] not in CLASSES or record['predicted'] not in CLASSES:
        raise ValueError('Only the three known gestures can be scored.')
    if record['condition'] not in CONDITIONS:
        raise ValueError('Choose a valid capture condition.')
    key = record['image_id'], record['model']
    return [r for r in records if (r['image_id'], r['model']) != key] + [record]


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r['model'], r['condition'])].append(r)
    rows = []
    for (model, condition), items in sorted(groups.items()):
        scores = []
        for label in CLASSES:
            tp = sum(r['actual'] == label and r['predicted'] == label for r in items)
            fp = sum(r['actual'] != label and r['predicted'] == label for r in items)
            fn = sum(r['actual'] == label and r['predicted'] != label for r in items)
            scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        rows.append({'Model': model, 'Condition': condition, 'Images': len(items),
                     'Accuracy (%)': 100*sum(r['actual'] == r['predicted'] for r in items)/len(items),
                     'Macro-F1 (3 classes)': sum(scores)/3,
                     'Classes represented': len({r['actual'] for r in items}),
                     'Mean inference (ms)': sum(r['mean_ms'] for r in items)/len(items)})
    return rows


def export_csv(records):
    if not records:
        return ''
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()
