
## Input parameters
flares_related_params = ['flare flag', 'noaa-sep_flag', 'fl_goes_xray', 'fl_lon', 'fl_lat', 'fl_rise_time', 
             'fl_start_time ref1', 'fl_peak_time ref1', 'CME + flare flag']
cme_related_params = ['CME flag', 'lasco_cme_width', 'p_cme_width', 'lasco_linear_speed', 'p_cme_speed', 
             'cme_launch_time ref1', 'CME + flare flag']
flux_related_params = ['noaa_pf10MeV', 'gsep_pf_gt10MeV', 'ppf_gt10MeV', 'ppf_gt30MeV', 'ppf_gt60MeV', 'ppf_gt100MeV']
fluence_related_params = ['gsep_fluence_gt10MeV', 'fluence_gt10MeV', 'fluence_gt30MeV', 'fluence_gt100MeV']
radio_related_params = ['radio burst 1', 'radio burst 2']
AR_related_params = ['AR_location', 'AR_lo', 'AR_area', 'AR_z', 'AR_ll', 'AR_nn', 'AR_mag_type', 'AR_mag_int', 'AR_z_int', 
                'AR_z_int_ranked', 'group_configuration', 'largest_spot_type', 'spots_distribution' , 'group_configuration_int', 
                'group_configuration_int_ranked', 'largest_spot_type_int', 'largest_spot_type_int_ranked', 'spots_distribution_int', 'spots_distribution_int_ranked']
SN_related_params = ['daily_sn']
others_relate_params = ['cdaw_start_time ref1', 'slice range']

## Output parameters
sepstorm_related_params = ['>= S1', '>= S2', '>= S3', '= S1', '= S2', '= S3', '= S4', 'S_class']

#%%
import re
import tkinter as tk

# --- Edit these before launching ---
# Step 1 is now split into several named groups, each shown in its own box.

STEP1_GROUPS_INPUT = [
    ("Flares", flares_related_params),

    ("Flux", flux_related_params),
    ("Sunspots Number", SN_related_params),

    ("CME", cme_related_params),
    ("Fluence", fluence_related_params),
    ("Radio", radio_related_params),


    ("Others", others_relate_params)
    # ("ARs", AR_related_params),
]


STEP1_GROUPS_OUTPUT = [
    ("S Storm Categories", sepstorm_related_params ),
]

# Step 4: pick exactly one model, then fill in that model's own parameters.
# Each model can have a different number of parameters (or none at all).
STEP4_MODELS = {
    "RandomForestClassifier": [
        {"label": "n_estimators", "default": 100},
        {"label": "random_state", "default": None}
    ],
    "LogisticRegression": [
        {"label": "max_iter", "default": 100},
        {"label": "random_state", "default": None}, 
        {"label": "solver", "default": 'lbfgs'}
    ]
}

# Each step is a dict describing what kind of screen to show.
# type "multi_select"            -> classic list of toggle buttons
# type "multi_select_grouped"    -> several named groups of toggle buttons,
#                                    each shown in its own boxed frame, laid
#                                    out in a grid ("columns" controls width)
# type "multi_select_grouped_dual" -> same as above, but split into two
#                                    independent halves side by side (e.g.
#                                    inputs on the left, outputs on the right),
#                                    saved as {"inputs": [...], "outputs": [...]}
# type "reveal_integer"          -> one button that toggles a numeric picker,
#                                    plus a save button usable with or without it
# type "float_and_optional_int"  -> a float entry + a button that toggles an int
#                                    picker (saved as None if never revealed)
# type "single_select_with_params" -> pick exactly one of several buttons; the
#                                    others disappear and that button's own set
#                                    of numeric parameters appears to fill in
# type "recap"                    -> read-only summary of every previous step
STEPS = [
    {
        "type": "multi_select_grouped_dual",
        "title": "1 - Inputs & Outputs selection:",
        "input_groups": STEP1_GROUPS_INPUT,
        "input_columns": 3,
        "output_groups": STEP1_GROUPS_OUTPUT,
        "output_columns": 1,
    },
# TODO : add a label 'Principal Components'
    {
        "type": "reveal_integer",
        "title": "2 - PCA Options:",
        "button_label": "Yes ?",
        "default_value": 2,
        "min_value": 1,
        "max_value": 100,
    },
    {
        "type": "float_and_optional_int",
        "title": "3 - Train/Test Split Options:",
        "default_float": 0.2,
        "int_button_label": "random_state",
        "default_int": 42,
        "min_int": 0,
        "max_int": 1000,
    },
    {
        "type": "single_select_with_params",
        "title": "4 - Model Selection",
        "models": STEP4_MODELS,
    },
    {
        "type": "recap",
        "title": "Summary of your choices",
    },
]

# --- Colors ---
DEFAULT_BG = "#F0F0F0"
DEFAULT_FG = "black"
SELECTED_BG = "#4CAF50"
SELECTED_FG = "white"

# Collects the saved value from every step, in order
all_selections = []


def finish_step(step_index, value, window):
    """Common continuation logic shared by every step type."""
    all_selections.append(value)
    print(f"Step {step_index + 1} saved:", value)

    window.destroy()

    if step_index + 1 < len(STEPS):
        open_step(step_index + 1)
    else:
        print("All selections:", all_selections)


def build_multiselect(window, step, step_index, is_last_step):
    options = step["options"]
    selected = set()
    buttons = {}

    status = tk.Label(
        window,
        text="Nothing selected yet",
        font=("Segoe UI", 10, "italic"),
        bg=DEFAULT_BG,
        fg="#555555",
    )

    def toggle(name):
        if name in selected:
            selected.remove(name)
            buttons[name].configure(bg=DEFAULT_BG, fg=DEFAULT_FG, relief="raised")
        else:
            selected.add(name)
            buttons[name].configure(bg=SELECTED_BG, fg=SELECTED_FG, relief="sunken")

        if selected:
            ordered = [n for n in options if n in selected]
            status.configure(text="Selected: " + ", ".join(ordered))
        else:
            status.configure(text="Nothing selected yet")

    for name in options:
        btn = tk.Button(
            window,
            text=name,
            font=("Segoe UI", 11),
            bg=DEFAULT_BG,
            fg=DEFAULT_FG,
            relief="raised",
            bd=2,
            width=22,
            command=lambda n=name: toggle(n),
        )
        btn.pack(pady=5, padx=20)
        buttons[name] = btn

    def on_save():
        ordered = [n for n in options if n in selected]
        finish_step(step_index, ordered, window)

    tk.Button(
        window,
        text="Finish" if is_last_step else "Save selection",
        font=("Segoe UI", 10, "bold"),
        bg="#2196F3",
        fg="white",
        relief="raised",
        bd=2,
        width=22,
        command=on_save,
    ).pack(pady=(10, 5))

    status.pack(pady=(5, 15))


def build_multiselect_grouped(window, step, step_index, is_last_step):
    groups = step["groups"]  # list of (group_title, options_list)
    columns = step.get("columns", 1)  # how many boxes per row

    selected = set()   # holds (group_title, name) tuples, so identical names across groups don't collide
    buttons = {}        # (group_title, name) -> button widget

    status = tk.Label(
        window,
        text="Nothing selected yet",
        font=("Segoe UI", 10, "italic"),
        bg=DEFAULT_BG,
        fg="#555555",
        wraplength=700,
        justify="left",
    )

    def refresh_status():
        if selected:
            # Keep the display order consistent with the groups/options order
            ordered = [
                name
                for group_title, options in groups
                for name in options
                if (group_title, name) in selected
            ]
            status.configure(text="Selected: " + ", ".join(ordered))
        else:
            status.configure(text="Nothing selected yet")

    def toggle(key):
        if key in selected:
            selected.remove(key)
            buttons[key].configure(bg=DEFAULT_BG, fg=DEFAULT_FG, relief="raised")
        else:
            selected.add(key)
            buttons[key].configure(bg=SELECTED_BG, fg=SELECTED_FG, relief="sunken")
        refresh_status()

    # Grid container holding all the boxes, arranged in `columns` columns
    grid_container = tk.Frame(window, bg=DEFAULT_BG)
    grid_container.pack(pady=8, padx=10, fill="both", expand=True)
    for c in range(columns):
        grid_container.columnconfigure(c, weight=1)

    for idx, (group_title, options) in enumerate(groups):
        row = idx // columns
        col = idx % columns

        box = tk.LabelFrame(
            grid_container,
            text=group_title,
            font=("Segoe UI", 10, "bold"),
            bg=DEFAULT_BG,
            fg=DEFAULT_FG,
            bd=2,
            relief="groove",
            padx=10,
            pady=8,
        )
        box.grid(row=row, column=col, padx=10, pady=10, sticky="n")

        for name in options:
            key = (group_title, name)
            btn = tk.Button(
                box,
                text=name,
                font=("Segoe UI", 11),
                bg=DEFAULT_BG,
                fg=DEFAULT_FG,
                relief="raised",
                bd=2,
                command=lambda k=key: toggle(k),
            )
            btn.pack(pady=3, fill="x")
            buttons[key] = btn

    def on_save():
        ordered = [
            name
            for group_title, options in groups
            for name in options
            if (group_title, name) in selected
        ]
        finish_step(step_index, ordered, window)

    tk.Button(
        window,
        text="Finish" if is_last_step else "Save selection",
        font=("Segoe UI", 10, "bold"),
        bg="#2196F3",
        fg="white",
        relief="raised",
        bd=2,
        width=22,
        command=on_save,
    ).pack(pady=(10, 5))

    status.pack(pady=(5, 15))


def _build_grouped_side(parent, groups, columns, selected, buttons, status):
    """Builds one half (grid of boxed groups) of a dual grouped multi-select."""
    grid_container = tk.Frame(parent, bg=DEFAULT_BG)
    grid_container.pack(fill="both", expand=True)
    for c in range(columns):
        grid_container.columnconfigure(c, weight=1)

    def refresh_status():
        if selected:
            ordered = [
                name
                for group_title, options in groups
                for name in options
                if (group_title, name) in selected
            ]
            status.configure(text="Selected: " + ", ".join(ordered))
        else:
            status.configure(text="Nothing selected yet")

    def toggle(key):
        if key in selected:
            selected.remove(key)
            buttons[key].configure(bg=DEFAULT_BG, fg=DEFAULT_FG, relief="raised")
        else:
            selected.add(key)
            buttons[key].configure(bg=SELECTED_BG, fg=SELECTED_FG, relief="sunken")
        refresh_status()

    for idx, (group_title, options) in enumerate(groups):
        row = idx // columns
        col = idx % columns

        box = tk.LabelFrame(
            grid_container,
            text=group_title,
            font=("Segoe UI", 10, "bold"),
            bg=DEFAULT_BG,
            fg=DEFAULT_FG,
            bd=2,
            relief="groove",
            padx=10,
            pady=8,
        )
        box.grid(row=row, column=col, padx=8, pady=8, sticky="n")

        for name in options:
            key = (group_title, name)
            btn = tk.Button(
                box,
                text=name,
                font=("Segoe UI", 11),
                bg=DEFAULT_BG,
                fg=DEFAULT_FG,
                relief="raised",
                bd=2,
                command=lambda k=key: toggle(k),
            )
            btn.pack(pady=3, fill="x")
            buttons[key] = btn


def build_multiselect_grouped_dual(window, step, step_index, is_last_step):
    input_groups = step["input_groups"]
    output_groups = step["output_groups"]
    input_columns = step.get("input_columns", 1)
    output_columns = step.get("output_columns", 1)

    input_selected = set()
    output_selected = set()
    input_buttons = {}
    output_buttons = {}

    # Two halves, side by side, separated by a thin vertical divider
    halves = tk.Frame(window, bg=DEFAULT_BG)
    halves.pack(fill="both", expand=True, padx=10, pady=5)

    left_frame = tk.Frame(halves, bg=DEFAULT_BG)
    left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

    tk.Frame(halves, bg="#B0B0B0", width=2).pack(side="left", fill="y", padx=5)

    right_frame = tk.Frame(halves, bg=DEFAULT_BG)
    right_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))

    tk.Label(
        left_frame, text="Inputs", font=("Segoe UI", 12, "bold"), bg=DEFAULT_BG
    ).pack(pady=(0, 8))
    tk.Label(
        right_frame, text="Outputs", font=("Segoe UI", 12, "bold"), bg=DEFAULT_BG
    ).pack(pady=(0, 8))

    input_status = tk.Label(
        left_frame,
        text="Nothing selected yet",
        font=("Segoe UI", 9, "italic"),
        bg=DEFAULT_BG,
        fg="#555555",
        wraplength=320,
        justify="left",
    )
    output_status = tk.Label(
        right_frame,
        text="Nothing selected yet",
        font=("Segoe UI", 9, "italic"),
        bg=DEFAULT_BG,
        fg="#555555",
        wraplength=320,
        justify="left",
    )

    _build_grouped_side(left_frame, input_groups, input_columns, input_selected, input_buttons, input_status)
    input_status.pack(pady=(8, 0))

    _build_grouped_side(right_frame, output_groups, output_columns, output_selected, output_buttons, output_status)
    output_status.pack(pady=(8, 0))

    def on_save():
        ordered_inputs = [
            name
            for group_title, options in input_groups
            for name in options
            if (group_title, name) in input_selected
        ]
        ordered_outputs = [
            name
            for group_title, options in output_groups
            for name in options
            if (group_title, name) in output_selected
        ]
        finish_step(step_index, {"inputs": ordered_inputs, "outputs": ordered_outputs}, window)

    tk.Button(
        window,
        text="Finish" if is_last_step else "Save selection",
        font=("Segoe UI", 10, "bold"),
        bg="#2196F3",
        fg="white",
        relief="raised",
        bd=2,
        width=22,
        command=on_save,
    ).pack(pady=(10, 15))


def build_reveal_integer(window, step, step_index, is_last_step):
    default_value = step.get("default_value", 2)
    min_value = step.get("min_value", 0)
    max_value = step.get("max_value", 100)

    # Validation: only allow digits (or an empty field while typing)
    def validate_digits(new_value):
        return new_value == "" or new_value.isdigit()

    vcmd = (window.register(validate_digits), "%P")

    # Tracks whether the user has clicked the reveal button at all
    revealed = {"value": False}
    value_var = tk.StringVar(value=str(default_value))
    spinbox_ref = {"widget": None}

    def toggle_spinbox():
        if not revealed["value"]:
            # Reveal it
            revealed["value"] = True
            reveal_btn.configure(bg=SELECTED_BG, fg=SELECTED_FG, relief="sunken")

            spinbox_ref["widget"] = tk.Spinbox(
                window,
                from_=min_value,
                to=max_value,
                textvariable=value_var,
                font=("Segoe UI", 12),
                width=10,
                justify="center",
                validate="key",
                validatecommand=vcmd,
            )
            spinbox_ref["widget"].pack(pady=(10, 10), before=save_btn)
        else:
            # Retract it back to the original state
            revealed["value"] = False
            reveal_btn.configure(bg=DEFAULT_BG, fg=DEFAULT_FG, relief="raised")

            if spinbox_ref["widget"] is not None:
                spinbox_ref["widget"].destroy()
                spinbox_ref["widget"] = None

            value_var.set(str(default_value))  # reset back to the default for next time

    def on_confirm():
        if not revealed["value"]:
            # Button was never clicked (or was retracted) -> save 0 and move on
            value = 0
        else:
            # Fall back to the default if the field was left empty
            raw = value_var.get()
            value = int(raw) if raw else default_value
        finish_step(step_index, value, window)

    reveal_btn = tk.Button(
        window,
        text=step.get("button_label", "Click here"),
        font=("Segoe UI", 11),
        bg=DEFAULT_BG,
        fg=DEFAULT_FG,
        relief="raised",
        bd=2,
        width=22,
        command=toggle_spinbox,
    )
    reveal_btn.pack(pady=(10, 10))

    save_btn = tk.Button(
        window,
        text="Finish" if is_last_step else "Save selection",
        font=("Segoe UI", 10, "bold"),
        bg="#2196F3",
        fg="white",
        relief="raised",
        bd=2,
        width=22,
        command=on_confirm,
    )
    save_btn.pack(pady=(5, 15))


def build_float_and_optional_int(window, step, step_index, is_last_step):
    default_float = step.get("default_float", 0.2)
    default_int = step.get("default_int", 42)
    min_int = step.get("min_int", 0)
    max_int = step.get("max_int", 1000)

    # Allows partial input while typing (e.g. "-", "0.", "") in addition to full floats
    float_pattern = re.compile(r"^-?\d*\.?\d*$")

    def validate_float(new_value):
        return new_value == "" or float_pattern.match(new_value) is not None

    def validate_digits(new_value):
        return new_value == "" or new_value.isdigit()

    float_vcmd = (window.register(validate_float), "%P")
    int_vcmd = (window.register(validate_digits), "%P")

    # --- Float value (always required, pre-filled with the default) ---
    tk.Label(
        window, text="Enter a float value:", font=("Segoe UI", 11), bg=DEFAULT_BG
    ).pack(pady=(5, 2))

    float_var = tk.StringVar(value=str(default_float))
    tk.Entry(
        window,
        textvariable=float_var,
        font=("Segoe UI", 12),
        width=12,
        justify="center",
        validate="key",
        validatecommand=float_vcmd,
    ).pack(pady=(0, 15))

    # --- Optional int value (None unless the button below is toggled on) ---
    int_revealed = {"value": False}
    int_var = tk.StringVar(value=str(default_int))
    spinbox_ref = {"widget": None}

    def toggle_int_spinbox():
        if not int_revealed["value"]:
            # Reveal it
            int_revealed["value"] = True
            int_btn.configure(bg=SELECTED_BG, fg=SELECTED_FG, relief="sunken")

            spinbox_ref["widget"] = tk.Spinbox(
                window,
                from_=min_int,
                to=max_int,
                textvariable=int_var,
                font=("Segoe UI", 12),
                width=10,
                justify="center",
                validate="key",
                validatecommand=int_vcmd,
            )
            spinbox_ref["widget"].pack(pady=(10, 10), before=save_btn)
        else:
            # Retract it back to the original state
            int_revealed["value"] = False
            int_btn.configure(bg=DEFAULT_BG, fg=DEFAULT_FG, relief="raised")

            if spinbox_ref["widget"] is not None:
                spinbox_ref["widget"].destroy()
                spinbox_ref["widget"] = None

            int_var.set(str(default_int))  # reset back to the default for next time

    int_btn = tk.Button(
        window,
        text=step.get("int_button_label", "Click here"),
        font=("Segoe UI", 11),
        bg=DEFAULT_BG,
        fg=DEFAULT_FG,
        relief="raised",
        bd=2,
        width=22,
        command=toggle_int_spinbox,
    )
    int_btn.pack(pady=(0, 10))

    def on_confirm():
        raw_float = float_var.get()
        # Fall back to the default for an empty or incomplete field (e.g. "-", ".")
        try:
            float_value = float(raw_float)
        except ValueError:
            float_value = default_float

        if not int_revealed["value"]:
            int_value = None
        else:
            raw_int = int_var.get()
            int_value = int(raw_int) if raw_int else default_int

        finish_step(step_index, {"float": float_value, "int": int_value}, window)

    save_btn = tk.Button(
        window,
        text="Finish" if is_last_step else "Save selection",
        font=("Segoe UI", 10, "bold"),
        bg="#2196F3",
        fg="white",
        relief="raised",
        bd=2,
        width=22,
        command=on_confirm,
    )
    save_btn.pack(pady=(5, 15))


def build_single_select_with_params(window, step, step_index, is_last_step):
    models = step["models"]
    model_names = list(models.keys())

    selected_model = {"name": None}
    param_vars = {}  # param label -> StringVar
    model_buttons = {}

    # Allows partial input while typing, same as the other numeric fields
    float_pattern = re.compile(r"^-?\d*\.?\d*$")

    def validate_float(new_value):
        return new_value == "" or float_pattern.match(new_value) is not None

    vcmd = (window.register(validate_float), "%P")

    # Frame that will hold the chosen model's parameter fields (empty for now)
    params_frame = tk.Frame(window, bg=DEFAULT_BG)
    params_frame.pack(pady=(5, 5))

    def go_back():
        selected_model["name"] = None

        for widget in params_frame.winfo_children():
            widget.destroy()
        param_vars.clear()

        # Bring the 4 model buttons back, in their original order
        for name in model_names:
            model_buttons[name].pack(pady=5, padx=20)

        back_btn.pack_forget()
        finish_btn.configure(state="disabled")

    back_btn = tk.Button(
        window,
        text="\u2190 Change model",
        font=("Segoe UI", 9, "underline"),
        bg=DEFAULT_BG,
        fg="#2196F3",
        relief="flat",
        bd=0,
        command=go_back,
    )

    def choose_model(name):
        selected_model["name"] = name

        # Hide the 4 model buttons, only the chosen model stays relevant
        for btn in model_buttons.values():
            btn.pack_forget()

        # Build this model's own parameter fields
        for widget in params_frame.winfo_children():
            widget.destroy()
        param_vars.clear()

        for param in models[name]:
            label = param["label"]
            default = param.get("default", 0)

            tk.Label(
                params_frame,
                text=label + ":",
                font=("Segoe UI", 10),
                bg=DEFAULT_BG,
            ).pack(pady=(8, 2))

            var = tk.StringVar(value=str(default))
            tk.Entry(
                params_frame,
                textvariable=var,
                font=("Segoe UI", 11),
                width=12,
                justify="center",
                validate="key",
                validatecommand=vcmd,
            ).pack()
            param_vars[label] = var

        back_btn.pack(pady=(5, 0), before=params_frame)
        finish_btn.configure(state="normal")  # can only finish once a model is chosen

    for name in model_names:
        btn = tk.Button(
            window,
            text=name,
            font=("Segoe UI", 11),
            bg=DEFAULT_BG,
            fg=DEFAULT_FG,
            relief="raised",
            bd=2,
            width=22,
            command=lambda n=name: choose_model(n),
        )
        btn.pack(pady=5, padx=20)
        model_buttons[name] = btn

    def on_confirm():
        name = selected_model["name"]
        if name is None:
            return  # button is disabled until a model is picked, this is just a safety net

        parameters = {}
        for param in models[name]:
            label = param["label"]
            default = param.get("default", 0)
            raw = param_vars[label].get()
            try:
                parameters[label] = float(raw)
            except ValueError:
                parameters[label] = default  # fall back if left empty/incomplete

        finish_step(step_index, {"model": name, "parameters": parameters}, window)

    finish_btn = tk.Button(
        window,
        text="Finish" if is_last_step else "Save selection",
        font=("Segoe UI", 10, "bold"),
        bg="#2196F3",
        fg="white",
        relief="raised",
        bd=2,
        width=22,
        command=on_confirm,
        state="disabled",  # can't finish before a model is selected
    )
    finish_btn.pack(pady=(10, 15))


def format_value(value):
    """Turns a saved step value into a readable one-line string for the recap."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "(none selected)"
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            if isinstance(val, dict):
                nested = ", ".join(f"{k}: {v}" for k, v in val.items())
                parts.append(f"{key}: {{{nested}}}")
            elif isinstance(val, list):
                inner = ", ".join(str(v) for v in val) if val else "(none selected)"
                parts.append(f"{key}: [{inner}]")
            else:
                parts.append(f"{key}: {val}")
        return ", ".join(parts)
    return str(value)


def build_recap(window, step, step_index, is_last_step):
    tk.Label(
        window,
        text=step["title"],
        font=("Segoe UI", 14, "bold"),
        bg=DEFAULT_BG,
    ).pack(pady=(5, 15))

    recap_frame = tk.Frame(window, bg=DEFAULT_BG)
    recap_frame.pack(padx=25)

    # Recap every step that came before this one
    for i, previous_step in enumerate(STEPS[:step_index]):
        line = f"{previous_step['title']}  {format_value(all_selections[i])}"
        tk.Label(
            recap_frame,
            text=line,
            font=("Segoe UI", 10),
            bg=DEFAULT_BG,
            justify="left",
            anchor="w",
            wraplength=380,
        ).pack(fill="x", pady=4)

    def on_close():
        print("All selections:", all_selections)
        window.destroy()

    tk.Button(
        window,
        text="Close",
        font=("Segoe UI", 10, "bold"),
        bg="#2196F3",
        fg="white",
        relief="raised",
        bd=2,
        width=22,
        command=on_close,
    ).pack(pady=(20, 15))


def open_step(step_index):
    """Opens a brand new window for STEPS[step_index]."""
    step = STEPS[step_index]
    is_last_step = step_index == len(STEPS) - 1

    window = tk.Tk()
    window.title(f"Step {step_index + 1} of {len(STEPS)}")
    window.configure(bg=DEFAULT_BG)
    window.resizable(True, True)

    tk.Label(
        window, text=step["title"], font=("Segoe UI", 14, "bold"), bg=DEFAULT_BG
    ).pack(pady=(15, 5))

    if step["type"] == "multi_select":
        build_multiselect(window, step, step_index, is_last_step)
    elif step["type"] == "multi_select_grouped":
        build_multiselect_grouped(window, step, step_index, is_last_step)
    elif step["type"] == "multi_select_grouped_dual":
        build_multiselect_grouped_dual(window, step, step_index, is_last_step)
    elif step["type"] == "reveal_integer":
        build_reveal_integer(window, step, step_index, is_last_step)
    elif step["type"] == "float_and_optional_int":
        build_float_and_optional_int(window, step, step_index, is_last_step)
    elif step["type"] == "single_select_with_params":
        build_single_select_with_params(window, step, step_index, is_last_step)
    elif step["type"] == "recap":
        build_recap(window, step, step_index, is_last_step)
    else:
        raise ValueError(f"Unknown step type: {step['type']}")

    window.mainloop()


if __name__ == "__main__":
    open_step(0)
    # After the last window closes, everything you saved is available here:
    print("Final results (one entry per step, in order):", all_selections)