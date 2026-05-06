import sys

import json

from accelerate.commands.estimate import create_empty_model

from accelerate.utils import calculate_maximum_sizes



def main():

    """

    Calculates model memory needs and prints a self-contained, interactive

    HTML snippet to standard output.

    """

    if len(sys.argv) < 2:

        print_error_html("No model ID provided.")

        sys.exit(1)



    model_id = sys.argv[1]



    try:

        # 1. Perform the fast, one-time calculation

        model = create_empty_model(model_id, library_name="transformers", trust_remote_code=True)

        total_size_bytes, _ = calculate_maximum_sizes(model)

        total_params = sum(p.numel() for p in model.parameters())



        # 2. Prepare the data payload for JavaScript

        data_for_js = {

            "model_id": model_id,

            "total_params": total_params,

            "base_ram_gb": {

                "float32": (total_size_bytes / 1) / (1024**3),

                "float16": (total_size_bytes / 2) / (1024**3),

                "int8":    (total_size_bytes / 4) / (1024**3),

                "int4":    (total_size_bytes / 8) / (1024**3)

            }

        }

        

        param_billion = f"{total_params / 1e9:.2f}"



        # 3. Print the complete HTML/JS snippet to stdout

        print_interactive_html(model_id, param_billion, data_for_js)



    except Exception as e:

        print_error_html(f"Could not analyze model <code>{model_id}</code>. <br><small>{e}</small>")

        sys.exit(1)



def print_error_html(message):

    """Prints a styled error message in HTML format."""

    print(f"""

<div class="model-memory-container" style="color: red; border-color: red;">

    <b>Error:</b> {message}

</div>""")



def print_interactive_html(model_id, param_billion, data_for_js):

    """Prints the main interactive HTML snippet using an f-string."""

    # The json.dumps function is crucial for safely embedding the Python dict as a JavaScript object

    json_data_str = json.dumps(data_for_js)



    print(f'''

<div class="model-memory-container">

<style>

    .model-memory-container {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #212529; background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6; }}

    .model-memory-container h4 {{ margin-top: 0; margin-bottom: 5px; font-size: 16px; color: #495057; }}

    .model-memory-container .model-stats {{ font-size: 13px; color: #6c757d; margin-bottom: 15px; }}

    .model-memory-container .controls {{ display: flex; align-items: center; gap: 20px; background-color: #f8f9fa; padding: 10px 15px; border-radius: 6px; margin-bottom: 20px; }}

    .model-memory-container .controls label {{ font-weight: 500; font-size: 14px; }}

    .model-memory-container #loraSlider {{ width: 150px; }}

    .model-memory-container table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}

    .model-memory-container th, .model-memory-container td {{ padding: 12px 15px; text-align: right; border: 1px solid #e9ecef; }}

    .model-memory-container th {{ font-weight: 600; text-align: center; }}

    .model-memory-container td:first-child {{ text-align: left; font-weight: 500; }}

    .model-memory-container code {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; background-color: #e9ecef; padding: 3px 6px; border-radius: 4px; }}

</style>



<h4>Memory Estimation for <code>{model_id}</code></h4>

<div class="model-stats"><strong>Total Parameters:</strong> {param_billion} Billion</div>



<div class="controls">

    <label for="loraToggle">

        <input type="checkbox" id="loraToggle"> Enable LoRa Tuning

    </label>

    <div id="loraSliderContainer" style="display:none; align-items:center; gap: 10px;">

        <label for="loraSlider">Trainable:</label>

        <input type="range" id="loraSlider" min="0.1" max="5.0" value="1.0" step="0.1">

        <span id="loraValueLabel">1.0%</span>

    </div>

</div>



<table>

    <thead>

        <tr>

            <th style="text-align: left;">Precision</th>

            <th>Base Model RAM</th>

            <th>Total Required RAM</th>

        </tr>

    </thead>

    <tbody>

        <tr><td>float32</td><td id="base-ram-float32">...</td><td id="total-ram-float32">...</td></tr>

        <tr><td>float16</td><td id="base-ram-float16">...</td><td id="total-ram-float16">...</td></tr>

        <tr><td>int8</td>   <td id="base-ram-int8">...</td>   <td id="total-ram-int8">...</td></tr>

        <tr><td>int4</td>   <td id="base-ram-int4">...</td>   <td id="total-ram-int4">...</td></tr>

    </tbody>

</table>



<script>

    const modelData = {json_data_str};

    const loraToggle = document.getElementById('loraToggle');

    const loraSliderContainer = document.getElementById('loraSliderContainer');

    const loraSlider = document.getElementById('loraSlider');

    const loraValueLabel = document.getElementById('loraValueLabel');



    function updateTable() {{

        const isLoRaEnabled = loraToggle.checked;

        const loraPercentage = parseFloat(loraSlider.value);

        const loraOverheadBytes = (modelData.total_params * (loraPercentage / 100)) * 8;

        const loraOverheadGB = loraOverheadBytes / (1024**3);



        for (const precision in modelData.base_ram_gb) {{

            const baseRam = modelData.base_ram_gb[precision];

            let totalRam = baseRam;

            if (isLoRaEnabled && precision === 'float16') {{

                totalRam += loraOverheadGB;

            }}

            document.getElementById('base-ram-' + precision).textContent = baseRam.toFixed(2) + ' GB';

            document.getElementById('total-ram-' + precision).textContent = totalRam.toFixed(2) + ' GB';

        }}

    }}

    loraToggle.addEventListener('change', () => {{

        loraSliderContainer.style.display = loraToggle.checked ? 'flex' : 'none';

        updateTable();

    }});

    loraSlider.addEventListener('input', () => {{

        loraValueLabel.textContent = parseFloat(loraSlider.value).toFixed(1) + '%';

        updateTable();

    }});

    updateTable();

</script>

</div>

''')



if __name__ == "__main__":

    main()
