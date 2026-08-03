import os
from pathlib import Path
rootdir = Path(__file__).parent

with open(os.path.join(rootdir, 'mix_marker.js'), 'r') as f:
    mix_marker_script = f.read()

with open(os.path.join(rootdir, 'get_data.js'), 'r') as f:
    get_rect_script = f.read()

# draw label on page
with open(os.path.join(rootdir, 'label_marker.js'), 'r') as f:
    label_marker_script = f.read()

# remove label draw on page
remove_label_mark_script = """
    () => {
        document.querySelectorAll(".our-dom-marker").forEach(item => {
            document.body.removeChild(item);
        });
    }
"""

remove_id_script = """
    () => {
        Array.from(document.getElementsByClassName('possible-clickable-element')).forEach((element) => {
            element.classList.remove('possible-clickable-element');
            element.removeAttribute('data-testid');
        });
    }
"""
