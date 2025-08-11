import argparse
import html
import json

from yiutils.project_utils import find_project_root

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>OpenAI SP Results Visualization</title>
	<style>
		html, body {{
			height: 100%;
			margin: 0;
			padding: 0;
		}}
		body {{
			font-family: Arial, sans-serif;
			margin: 0;
			padding: 0;
			height: 100vh;
			width: 100vw;
			box-sizing: border-box;
		}}
		.container {{
			display: flex;
			flex-direction: row;
			height: 100vh;
			width: 100vw;
		}}
		.toc {{
			flex: 0 0 220px;
			background: #f5f5f5;
			padding: 1em;
			overflow-y: auto;
			border-right: 1px solid #ccc;
			height: 100vh;
			box-sizing: border-box;
		}}
		.content {{
			flex: 1 1 420px;
			width: 100vw;
			padding: 2em;
			overflow-y: auto;
			height: 100vh;
			box-sizing: border-box;
		}}
		.record {{
			display: flex;
			flex-direction: row;
			margin-bottom: 2em;
			border-bottom: 1px solid #eee;
			padding-bottom: 2em;
		}}
		.record-main {{
			width: 45%;
			min-width: 320px;
			max-width: 45vw;
			padding-right: 2em;
		}}
		.record-results {{
			width: 55%;
			min-width: 320px;
			max-width: 55vw;
			max-height: 1200px;
			overflow-y: auto;
		}}
		.pubmed-info h2 {{
			margin: 0 0 0.5em 0;
		}}
		.pubmed-info h3 {{
			margin: 0 0 1em 0;
			font-weight: normal;
			color: #333;
		}}
		.abstract {{
			background: #f9f9f9;
			padding: 1em;
			border-radius: 4px;
			font-size: 0.95em;
		}}
		/* Tabs */
		.tab {{
			overflow: hidden;
			border-bottom: 1px solid #ccc;
		}}
		.tab button {{
			background-color: inherit;
			float: left;
			border: none;
			outline: none;
			cursor: pointer;
			padding: 10px 16px;
			transition: 0.3s;
			font-size: 1em;
		}}
		.tab button:hover {{
			background-color: #ddd;
		}}
		.tab button.active {{
			background-color: #ccc;
		}}
		.tabcontent {{
			display: none;
			padding: 1em 0 0 0;
		}}
		.tabcontent pre {{
			background: #f4f4f4;
			padding: 1em;
			border-radius: 4px;
			font-size: 0.95em;
			overflow-x: auto;
			max-height: 1200px;
			overflow-y: auto;
			text-wrap: wrap;
		}}
		.stats {{
			background: #e8f4fd;
			padding: 1em;
			border-radius: 4px;
			margin-bottom: 1em;
			font-size: 0.9em;
		}}
		@media (max-width: 1200px) {{
			.container {{
				flex-direction: column;
			}}
			.toc, .content {{
				width: 100vw !important;
				max-width: 100vw !important;
				min-width: 0 !important;
				height: auto !important;
			}}
			.record {{
				flex-direction: column;
			}}
			.record-main, .record-results {{
				width: 100% !important;
				max-width: 100vw !important;
				min-width: 0 !important;
			}}
		}}
	</style>
</head>
<body>
	<div class="container">
		{toc_html}
		<div class="content">
			{stats_html}
			{records_html}
		</div>
	</div>
	<script>
	function openTab(evt, tabName) {{
		var i, tabcontent, tablinks;
		tabcontent = document.getElementsByClassName("tabcontent");
		for (i = 0; i < tabcontent.length; i++) {{
			tabcontent[i].style.display = "none";
		}}
		// Only get tablinks within the same tab group
		tablinks = evt.currentTarget.parentNode.getElementsByTagName("button");
		for (i = 0; i < tablinks.length; i++) {{
			tablinks[i].className = tablinks[i].className.replace(" active", "");
		}}
		document.getElementById(tabName).style.display = "block";
		evt.currentTarget.className += " active";
	}}
	// No auto-click: first tab is already visible and active by default
	</script>
	</body>
	</html>
"""


def escape_html(text):
    return html.escape(str(text))


def render_stats(records):
    # records is a list if multiple, or dict if single
    if isinstance(records, dict):
        records = [records]
    
    total_records = len(records)
    model_counts = {}
    
    for rec in records:
        for model_name in rec["model_results"].keys():
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
    
    stats_lines = [f"<strong>Total Records:</strong> {total_records}"]
    stats_lines.append("<strong>Model Coverage:</strong>")
    for model_name, count in sorted(model_counts.items()):
        percentage = (count / total_records) * 100 if total_records > 0 else 0
        stats_lines.append(f"• {model_name}: {count}/{total_records} ({percentage:.1f}%)")
    
    return f'<div class="stats">{"<br>".join(stats_lines)}</div>'


def render_table_of_contents(records):
    # records is a list if multiple, or dict if single
    if isinstance(records, dict):
        records = [records]
    toc = ['<div class="toc"><h3>PMIDs</h3><ul>']
    for rec in records:
        pmid = escape_html(rec["pubmed_data"]["pmid"])
        models = list(rec["model_results"].keys())
        model_list = ", ".join(models) if models else "No models"
        toc.append(f'<li><a href="#pmid-{pmid}">{pmid}</a><br><small>({model_list})</small></li>')
    toc.append("</ul></div>")
    return "\n".join(toc)


def render_pubmed_info(rec):
    pd = rec["pubmed_data"]
    pmid = escape_html(pd["pmid"])
    title = escape_html(pd["title"])
    ab = escape_html(pd["ab"])
    return f"""
	<div class="pubmed-info" id="pmid-{pmid}">
		<h2>{pmid}</h2>
		<h3>{title}</h3>
		<pre class="abstract" style="white-space: pre-wrap; word-break: break-word;">{ab}</pre>
	</div>
	"""


def render_model_results(rec):
    model_names = list(rec["model_results"].keys())
    if not model_names:
        return '<div class="model-tabs"><p>No model results available</p></div>'
    
    tabs = []
    tab_contents = []
    pmid = rec["pubmed_data"]["pmid"]
    for i, model_name in enumerate(model_names):
        tab_id = f"tab-{model_name}-{pmid}"
        tabs.append(
            f'<button class="tablinks{" active" if i == 0 else ""}" onclick="openTab(event, \'{tab_id}\')">{model_name}</button>'
        )
        model_result = rec["model_results"][model_name]
        tab_contents.append(f'''
		<div id="{tab_id}" class="tabcontent" style="display:{"block" if i == 0 else "none"}">
			<pre>{escape_html(json.dumps(model_result, indent=2))}</pre>
		</div>
		''')
    return f"""
	<div class="model-tabs">
		<div class="tab">
			{"".join(tabs)}
		</div>
		{"".join(tab_contents)}
	</div>
	"""


def render_record(rec):
    return f"""
	<div class="record">
		<div class="record-main">
			{render_pubmed_info(rec)}
		</div>
		<div class="record-results">
			{render_model_results(rec)}
		</div>
	</div>
	"""


def render_html(data):
    # If data is a dict, wrap in a list for uniformity
    records = data if isinstance(data, list) else [data]
    stats_html = render_stats(records)
    toc_html = render_table_of_contents(records)
    records_html = [render_record(rec) for rec in records]
    html = HTML_TEMPLATE.format(
        stats_html=stats_html,
        toc_html=toc_html, 
        records_html="".join(records_html)
    )
    return html


def main():
    # ==== init ====
    proj_root = find_project_root(anchor_file="justfile")
    data_dir = proj_root / "data" / "intermediate" / "analysis-sample-sp"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    parser = argparse.ArgumentParser(
        description="Render OpenAI SP analysis sample HTML from JSON data."
    )
    # ---- --file ----
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help=f"Path to input JSON file, relative to {data_dir}",
    )
    args = parser.parse_args()
    if args.file is None:
        parser.error("You must specify an input file with --file or -f")
    input_path = data_dir / args.file

    with open(input_path, "r") as f:
        data = json.load(f)

    # ==== render html ====
    html = render_html(data)

    # ==== output ====
    html_path = input_path.with_suffix("")  # Remove extension if present
    html_path = html_path.with_suffix(".html")  # Add .html extension
    with open(html_path, "w") as f:
        f.write(html)
    
    print(f"HTML rendered to: {html_path}")


if __name__ == "__main__":
    main()
