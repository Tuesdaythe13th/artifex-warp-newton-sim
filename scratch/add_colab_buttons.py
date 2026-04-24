import json
import os

repo_base = "tuesdaythe13th/artifex-warp-newton-sim"
branch = "main"

notebooks = [
    "./artifex_digital_twin_demo.ipynb",
    "./artifex_cooling_sim_plotly.ipynb",
    "./artifex_disc_cool_3d.ipynb",
    "./artifex_disc_final.ipynb",
    "./artifex_disc_cool.nbconvert.ipynb",
    "./artifex_advanced_notebook.ipynb",
    "./artifex_disc_cool.ipynb",
    "./artifex_colab_interactive.ipynb",
    "./notebooks/disc_cooling_sim.ipynb"
]

def add_colab_button(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}, not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return

    # Clean path for URL (remove leading ./)
    clean_path = file_path.lstrip('./')
    colab_url = f"https://colab.research.google.com/github/{repo_base}/blob/{branch}/{clean_path}"
    badge_md = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})\n"

    # Check if badge already exists
    if nb.get('cells') and len(nb['cells']) > 0:
        first_cell = nb['cells'][0]
        if first_cell['cell_type'] == 'markdown':
            content = "".join(first_cell['source'])
            if "colab-badge.svg" in content:
                print(f"Badge already exists in {file_path}")
                return
            # Prepend to existing markdown cell
            first_cell['source'].insert(0, badge_md)
            first_cell['source'].insert(1, "\n")
        else:
            # Insert new markdown cell at top
            new_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [badge_md]
            }
            nb['cells'].insert(0, new_cell)
    else:
        # Empty notebook
        nb['cells'] = [{
            "cell_type": "markdown",
            "metadata": {},
            "source": [badge_md]
        }]

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {file_path}")

for nb_file in notebooks:
    add_colab_button(nb_file)
