with open("artifex_advanced_notebook.ipynb", "r") as f:
    lines = f.readlines()

# Fix 1
lines[69] = '        "header_html = f\\"\\"\\"\\n",\n'
lines[70] = '        "        <div style=\'font-family: \\"Syne Mono\\", monospace;\\n                    font-size: 48px;\\n                    font-weight: bold;\\n                    color: #0ff;\\n                    background: #111;\\n                    padding: 20px;\\n                    text-align: center;\'>\\n            ARTIFEX LABS – {now}\\n        </div>\\n",\n        "        \\"\\"\\"\\n",\n'

# Fix 2
lines[96] = '        "method = widgets.RadioButtons(\\n",\n'
lines[97] = '        "    options=[\'Colab Secrets\', \'Mount Drive\', \'Upload Widget\'],\\n",\n        "    description=\'Data Source:\',\\n",\n        "    disabled=False\\n",\n'

# Fix 3
lines[328] = '        "html = f\\"\\"\\"\\n",\n'
lines[329] = '        "        <style>\\n",\n'
lines[330] = '        "  body {{ background:#111; color:#eee; font-family:\'Epilogue\',sans-serif; }}\\n",\n'
lines[331] = '        "  table {{ width:100%; border-collapse:collapse; }}\\n",\n'
lines[332] = '        "  th, td {{ border:1px solid #444; padding:8px; text-align:left; }}\\n",\n'
lines[333] = '        "  th {{ background:#222; color:#0ff; }}\\n",\n'
lines[334] = '        "  tr:nth-child(even) {{ background:#1a1a1a; }}\\n",\n'
lines[335] = '        "</style>\\n",\n'
lines[336] = '        "<h2 style=\'color:#0ff; text-align:center;\'>Cluster Summaries</h2>\\n",\n'
lines[337] = '        "<table>\\n",\n'
lines[338] = '        "  <tr><th>Cluster</th><th>Size</th><th>LLM Summary</th></tr>\\n",\n'
lines[339] = '        "  {\'\'.join([f\'<tr><td>{row.Cluster}</td><td>{row.Size}</td><td>{row.Summary}</td></tr>\' for _, row in summary_df.iterrows()])}\\n",\n'
lines[340] = '        "</table>\\n",\n'
lines[341] = '        "<p style=\'margin-top:20px; font-size:0.9em;\'>\\n",\n'
lines[342] = '        "  Generated on {now}. For deeper insight, see the whitepapers linked in the introductory markdown.\\n",\n'
lines[343] = '        "</p>\\n",\n'
lines[344] = '        "        \\"\\"\\"\\n",\n'
lines[345] = '        "display(HTML(html))"\n'

with open("artifex_advanced_notebook.ipynb", "w") as f:
    f.writelines(lines)
