import json
import os
import glob

def fix_notebook(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return

    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            
            # Fix Plotly & ipywidgets rendering
            if "import plotly.graph_objects as go" in source and "pio.renderers.default" not in source:
                new_source = []
                for line in cell['source']:
                    new_source.append(line)
                    if "import plotly.graph_objects as go" in line:
                        new_source.append("import plotly.io as pio\n")
                        new_source.append("pio.renderers.default = 'colab'\n")
                        new_source.append("try:\n")
                        new_source.append("    from google.colab import output\n")
                        new_source.append("    output.enable_custom_widget_manager()\n")
                        new_source.append("except ImportError:\n")
                        new_source.append("    pass\n")
                cell['source'] = new_source
                modified = True
                
            # Ensure matplotlib displays inline if not already doing so
            if "import matplotlib.pyplot as plt" in source and "%matplotlib inline" not in source:
                # Add %matplotlib inline to the top of the cell
                cell['source'].insert(0, "%matplotlib inline\n")
                modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Fixed visualization rendering in {filepath}")

notebooks = glob.glob('**/*.ipynb', recursive=True)
for nb in notebooks:
    # Skip checkpoint files or environment dirs
    if '.ipynb_checkpoints' in nb or 'artifex_sim.egg-info' in nb:
        continue
    fix_notebook(nb)

