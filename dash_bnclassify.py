import base64
import zlib
import pandas as pd
from io import StringIO

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Import the bnclassify wrapper
import bnclassify

# Initialize Dash app with Bootstrap theme for styling
# ---------- (1) CREATE DASH APP ---------- #
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    requests_pathname_prefix='/Model/LearningFromData/bnclassifyDash/',
    suppress_callback_exceptions=True
)

# Expose the server for Gunicorn
server = app.server

# Global variables to hold the current model and its algorithm type
current_model = None
current_algorithm = None

# Compressed sample dataset (Car Evaluation dataset) in base64 format

# Define layout components
app.layout = dbc.Container(fluid=True, children=[
    html.H1("Bayesian Network Classifier GUI", className="mb-4"),
    
    dcc.Store(id='dataset-store'),
    
    # Data upload and class selection card
    dbc.Card([
        dbc.CardHeader(html.H4("1. Upload Dataset")),
        dbc.CardBody([
            # Upload UI
            html.Div([
                html.Div([
                    html.Img(
                        src="https://img.icons8.com/ios-glyphs/40/cloud--v1.png",
                        className="upload-icon"
                    ),
                    html.Div("Drag and drop or select a CSV file", className="upload-text")
                ]),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([], style={'display': 'none'}),
                    className="upload-dropzone",
                    multiple=False
                ),
            ], className="upload-card"),

            # Use default checkbox
            html.Div([
                dcc.Checklist(
                    id='use-default-dataset',
                    options=[{'label': 'Use the default dataset', 'value': 'default'}],
                    value=[],
                    style={'display': 'inline-block', 'textAlign': 'center', 'marginTop': '10px'}
                ),
                dbc.Button(
                    html.I(className="fa fa-question-circle"),
                    id="help-button-default-dataset",
                    color="link",
                    style={"display": "inline-block", "marginLeft": "8px"}
                ),
            ], style={'textAlign': 'center'}),
            html.Div(id='data-preview'),      # preview table goes here

            html.Br(),

            dbc.Label("Target (class) column:"),
            dcc.Dropdown(id='class-dropdown',
                         placeholder="Select class column",
                         clearable=False),
        ])
    ]),
    
    # Actions and parameters card
    dbc.Card([
        dbc.CardHeader(html.H4("2. Choose Action and Parameters")),
        dbc.CardBody([
            # Action selection
            dbc.Label("Action:"),
            dcc.Dropdown(
                id='action-dropdown',
                options=[
                    {'label': 'Structure Learning Algorithms', 'value': 'structure'},
                    {'label': 'Parameter Learning Methods', 'value': 'param'},
                    {'label': 'Model Evaluating', 'value': 'evaluate'},
                    {'label': 'Prediction', 'value': 'predict'},
                    {'label': 'Inspecting Models', 'value': 'inspect'}
                ],
                placeholder="Select an action",
                clearable=False
            ),
            
            html.Div(id='structure-section', children=[
                html.Hr(),
                html.H5("Structure Learning", className="mb-3"),
                # Structure algorithm selection
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Algorithm:"),
                        dcc.Dropdown(
                            id='structure-method',
                            options=[
                                {'label': 'Naive Bayes (NB)', 'value': 'NB'},
                                {'label': 'TAN (Chow-Liu)', 'value': 'TAN_CL'},
                                {'label': 'TAN (Hill-Climbing)', 'value': 'TAN_HC'},
                                {'label': 'TAN (Hill-Climb + SP)', 'value': 'TAN_HCSP'},
                                {'label': 'Forward Sequential Selection & Joining (FSSJ)', 'value': 'FSSJ'},
                                {'label': 'Backward Sequential Elimination & Joining (BSEJ)', 'value': 'BSEJ'},
                                {'label': 'K-Dependence Bayesian (KDB)', 'value': 'KDB'},
                                {'label': 'Averaged One-Dependence Estimators (AODE)', 'value': 'AODE'}
                            ],
                            placeholder="Select structure learning algorithm",
                            clearable=False
                        )
                    ], width=6)
                ]),
                html.Br(),
                # Parameter fields for structure algorithms (hidden/shown dynamically)
                dbc.Row([
                    # Score metric
                    dbc.Col([
                        dbc.Label("Score Metric:", id='score-label'),
                        dcc.Dropdown(
                            id='score-dropdown',
                            options=[
                                {'label': 'AIC', 'value': 'AIC'},
                                {'label': 'BIC', 'value': 'BIC'},
                                {'label': 'Log-Likelihood', 'value': 'LL'}
                            ],
                            value='AIC', clearable=False
                        )
                    ], width=4, id='score-field'),
                    # Root feature for TAN-CL
                    dbc.Col([
                        dbc.Label("Root Feature:", id='root-label'),
                        dcc.Dropdown(id='root-dropdown', placeholder="Optional (Auto if blank)", clearable=True)
                    ], width=4, id='root-field')
                ], className="mb-3"),
                dbc.Row([
                    # Folds for FSSJ/BSEJ
                    dbc.Col([
                        dbc.Label("Folds:", id='struct-folds-label'),
                        dbc.Input(id='struct-folds-input', type='number', min=2, step=1, value=5)
                    ], width=3, id='struct-folds-field'),
                    # Epsilon for TAN-HC, TAN-HCSP, FSSJ, BSEJ
                    dbc.Col([
                        dbc.Label("Epsilon:", id='epsilon-label'),
                        dbc.Input(id='epsilon-input', type='number', step=0.001, value=0.0)
                    ], width=3, id='epsilon-field'),
                    # K for KDB
                    dbc.Col([
                        dbc.Label("Max Parents (K):", id='k-label'),
                        dbc.Input(id='k-input', type='number', min=1, step=1, value=1)
                    ], width=3, id='k-field')
                ])
            ], style={'display': 'none'}),  # hide structure section initially
            
            html.Div(id='param-section', children=[
                html.Hr(),
                html.H5("Parameter Learning", className="mb-3"),
                # Parameter learning method selection
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Method:"),
                        dcc.Dropdown(
                            id='param-method',
                            options=[
                                {'label': 'Maximum Likelihood (MLE)', 'value': 'MLE'},
                                {'label': 'Bayesian Estimation', 'value': 'Bayes'},
                                {'label': 'WANBIA (Feature Weighting)', 'value': 'WANBIA'},
                                {'label': 'AWNB (Attribute-Weighted NB)', 'value': 'AWNB'},
                                {'label': 'MANB (Model Averaged NB)', 'value': 'MANB'}
                            ],
                            placeholder="Select parameter learning method",
                            clearable=False
                        )
                    ], width=6)
                ]),
                html.Br(),
                # Parameter fields for parameter learning
                dbc.Row([
                    # Alpha for Bayesian
                    dbc.Col([
                        dbc.Label("Dirichlet Prior α:", id='alpha-label'),
                        dbc.Input(id='alpha-input', type='number', step=0.1, value=1.0)
                    ], width=3, id='alpha-field'),
                    # WANBIA has no parameters (no field needed, we will just show a note perhaps)
                    # AWNB fields
                    dbc.Col([
                        dbc.Label("No. of Trees (AWNB):", id='trees-label'),
                        dbc.Input(id='trees-input', type='number', min=1, step=1, value=10)
                    ], width=3, id='trees-field'),
                    dbc.Col([
                        dbc.Label("Subsample Size (AWNB):", id='subsample-label'),
                        dbc.Input(id='subsample-input', type='number', min=0.1, max=1.0, step=0.1, value=1.0)
                    ], width=3, id='subsample-field'),
                    # MANB prior
                    dbc.Col([
                        dbc.Label("MANB Prior:", id='prior-label'),
                        dbc.Input(id='prior-input', type='number', step=0.1, value=1.0)
                    ], width=3, id='prior-field')
                ])
            ], style={'display': 'none'}),
            
            html.Div(id='eval-section', children=[
                html.Hr(),
                html.H5("Model Evaluation", className="mb-3"),
                # Evaluation metric selection
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Evaluate:"),
                        dcc.Dropdown(
                            id='eval-metric',
                            options=[
                                {'label': 'Cross-Validation (CV)', 'value': 'CV'},
                                {'label': 'AIC (Akaike Information Criterion)', 'value': 'AIC'},
                                {'label': 'BIC (Bayesian Information Criterion)', 'value': 'BIC'},
                                {'label': 'Log-Likelihood', 'value': 'LL'}
                            ],
                            placeholder="Select evaluation method",
                            clearable=False
                        )
                    ], width=6)
                ]),
                html.Br(),
                # Fields for CV
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Folds:", id='eval-folds-label'),
                        dbc.Input(id='eval-folds-input', type='number', min=2, step=1, value=10)
                    ], width=3, id='eval-folds-field'),
                    dbc.Col([
                        dbc.Label("Fixed Structure:", id='dag-label'),
                        dbc.Checklist(
                            options=[{'label': ' Keep current structure (do not relearn each fold)', 'value': 'fixed'}],
                            value=['fixed'], id='dag-check', switch=True
                        )
                    ], width=6, id='dag-field')
                ])
            ], style={'display': 'none'}),
            
            html.Div(id='predict-section', children=[
                html.Hr(),
                html.H5("Prediction", className="mb-3"),
                # Upload prediction dataset
                dbc.Label("Upload dataset for prediction (features only):"),
                dcc.Upload(
                    id='upload-predict',
                    children=html.Div(['Drag and Drop or ', html.A('Select Prediction File')]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed',
                        'borderRadius': '5px', 'textAlign': 'center'
                    },
                    multiple=False
                ),
                html.Br(),
                # Checkbox for posterior probabilities
                dbc.Checklist(
                    options=[{'label': ' Return posterior probabilities for all classes', 'value': 'prob'}],
                    value=[], id='prob-check'
                )
            ], style={'display': 'none'}),
            
            html.Div(id='inspect-section', children=[
                html.Hr(),
                html.H5("Inspect Model", className="mb-3"),
                # Inspection option selection
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Inspect:"),
                        dcc.Dropdown(
                            id='inspect-option',
                            options=[
                                {'label': 'Plot Structure Graph', 'value': 'plot'},
                                {'label': 'Model Summary', 'value': 'summary'},
                                {'label': 'Conditional Probability Tables (CPTs)', 'value': 'cpts'},
                                {'label': 'Number of Free Parameters', 'value': 'nparams'}
                            ],
                            placeholder="Select what to inspect",
                            clearable=False
                        )
                    ], width=6)
                ]),
                html.Br(),
                # Additional options for plotting
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Layout:", id='layout-label'),
                        dcc.Dropdown(
                            id='layout-dropdown',
                            options=[
                                {'label': 'Circle', 'value': 'circle'},
                                {'label': 'Breadthfirst (tree)', 'value': 'breadthfirst'},
                                {'label': 'Grid', 'value': 'grid'},
                                {'label': 'Random', 'value': 'random'},
                                {'label': 'Concentric', 'value': 'concentric'}
                            ],
                            value='breadthfirst', clearable=False
                        )
                    ], width=3, id='layout-field'),
                    dbc.Col([
                        dbc.Label("Font Size:", id='font-label'),
                        dbc.Input(id='font-input', type='number', min=6, max=20, step=1, value=12)
                    ], width=2, id='font-field')
                ])
            ], style={'display': 'none'}),
            
            html.Br(),
            # Run action button
            dbc.Button("Run", id='run-button', color="primary", className="mt-2", disabled=True),
            # Spinner (loading indicator) and output will be shown in the output card below
        ])
    ], className="mb-4"),
    
    # Output card to display results
    dbc.Card([
        dbc.CardHeader(html.H4("3. Results")),
        dbc.CardBody(
            dcc.Loading(id='loading-output', type='circle', children=[
                html.Div(id='output-area')
            ])
        )
    ])
])

# Tooltips/Popovers for help text on various parameters
# (Attach to the labels of parameters)
help_texts = {
    'score-label': "Score metric for structure learning (AIC = Akaike Information Criterion, BIC = Bayesian Information Criterion, LL = log-likelihood). Used in TAN algorithms.",
    'root-label': "For TAN (Chow-Liu), optionally specify a root feature for the tree (if not set, one will be chosen automatically).",
    'struct-folds-label': "Number of folds for internal cross-validation used by FSSJ/BSEJ structure learning algorithms.",
    'epsilon-label': "Threshold for score improvement. Small positive value can stop hill-climbing or edge joining when improvement is below this value.",
    'k-label': "Maximum number of parent attributes (besides class) each attribute can have in the KDB (k-dependence Bayesian) structure.",
    'alpha-label': "Dirichlet prior equivalent sample size for Bayesian parameter learning (Laplace smoothing). α=0 corresponds to pure MLE.",
    'trees-label': "Number of bootstrap samples (decision trees) to generate for AWNB parameter averaging.",
    'subsample-label': "Size of each bootstrap subsample for AWNB, as a fraction of the training set (0 to 1). Default 1.0 = full bootstrap sample with replacement.",
    'prior-label': "Prior for model averaging in MANB. Higher values place more weight on including each feature in naive Bayes model averaging.",
    'dag-label': "If checked, keep the current learned structure fixed during cross-validation (only parameters are re-estimated each fold). If unchecked, re-learn structure in each CV fold.",
    'prob-check': "If checked, prediction will output posterior probabilities for each class for every instance, instead of only the predicted class.",
    'layout-label': "Layout algorithm for network graph visualization.",
    'font-label': "Font size for labels in the structure graph."
}
# Create a Tooltip for each help text entry
tooltips = []
for target_id, text in help_texts.items():
    tooltips.append(
        dbc.Tooltip(text, target=target_id, placement='right', style={"textAlign": "left", "maxWidth": "250px"})
    )
# Add tooltip components to the app layout (as they need to be in the layout)
app.layout.children.extend(tooltips)


@app.callback(
    Output('upload-data', 'contents'),
    Input('use-default-dataset', 'value'),
    prevent_initial_call=True
)
def use_default_dataset(value):
    if 'default' in value:
        try:
            default_file = '/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/bnclassify/carwithnames.data'
            with open(default_file, 'rb') as f:
                raw = f.read()
            b64 = base64.b64encode(raw).decode()
            return f"data:text/csv;base64,{b64}"
        except Exception as e:
            print(f"Error reading default dataset: {e}")
            return dash.no_update
    return dash.no_update


@app.callback(
    Output('data-preview',    'children'),
    Output('class-dropdown',  'options'),
    Output('class-dropdown',  'value'),
    Output('run-button',      'disabled', allow_duplicate=True),
    Output('dataset-store',   'data'),              # <-- JSON-serialised DF
    Input('upload-data',      'contents'),
    State('upload-data',      'filename'),
    prevent_initial_call=True
)
def load_dataset(upload_contents, upload_filename):
    """
    Parse the CSV that shows up in `upload-data.contents` (whether uploaded
    by the user or injected by the 'Use default dataset' checkbox),
    build a small preview, and fill the class selector.
    """
    import dash
    from dash import dash_table
    import base64, pandas as pd
    from io import StringIO

    if not upload_contents:
        raise dash.exceptions.PreventUpdate

    # ------------------------------------------------
    # 1. Decode the base64 payload -> pandas DataFrame
    # ------------------------------------------------
    try:
        _, content_string = upload_contents.split(',', 1)
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(StringIO(decoded.decode('utf-8')))
    except Exception as err:
        alert = dbc.Alert(f"Error reading dataset: {err}", color="danger")
        return alert, [], None, True, dash.no_update

    # ------------------------------------------------
    # 2. Build a 5-row preview table
    # ------------------------------------------------
    preview_df = df.head(5)
    preview_table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in preview_df.columns],
        data=preview_df.to_dict("records"),
        page_size=5,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'}
    )

    # ------------------------------------------------
    # 3. Populate class selector (default: last column)
    # ------------------------------------------------
    class_options = [{'label': c, 'value': c} for c in df.columns]
    class_value   = df.columns[-1] if len(df.columns) else None

    # ------------------------------------------------
    # 4. Serialise full DF for later use
    # ------------------------------------------------
    df_json = df.to_json(date_format='iso', orient='split')

    # Enable the Run button (disabled=False)
    return preview_table, class_options, class_value, False, df_json

# Callback: Enable/disable run button based on required inputs (class selection and action selection)
@app.callback(
    Output('run-button', 'disabled'),
    Input('class-dropdown', 'value'),
    Input('action-dropdown', 'value'),
    State('run-button', 'disabled')
)
def toggle_run_button(class_var, action, current_disabled):
    """
    Enable the Run button only if a class variable and an action are selected (and data is loaded).
    """
    if class_var and action:
        return False
    else:
        return True

# Callback: Show/hide the main sections based on selected Action
@app.callback(
    Output('structure-section', 'style'),
    Output('param-section', 'style'),
    Output('eval-section', 'style'),
    Output('predict-section', 'style'),
    Output('inspect-section', 'style'),
    Input('action-dropdown', 'value')
)
def display_action_sections(action):
    """
    Controls the visibility of each action section (structure, param, eval, predict, inspect) 
    according to the chosen action.
    """
    # Default hidden style
    hide = {'display': 'none'}
    show = {'display': 'block'}
    if action == 'structure':
        return show, hide, hide, hide, hide
    elif action == 'param':
        return hide, show, hide, hide, hide
    elif action == 'evaluate':
        return hide, hide, show, hide, hide
    elif action == 'predict':
        return hide, hide, hide, show, hide
    elif action == 'inspect':
        return hide, hide, hide, hide, show
    else:
        return hide, hide, hide, hide, hide

# Callback: Show/hide specific parameter fields based on selected structure learning algorithm
@app.callback(
    Output('score-field', 'style'),
    Output('root-field', 'style'),
    Output('struct-folds-field', 'style'),
    Output('epsilon-field', 'style'),
    Output('k-field', 'style'),
    Input('structure-method', 'value')
)
def toggle_structure_fields(struct_method):
    """
    Toggle visibility of structure algorithm parameter inputs depending on the algorithm chosen.
    """
    hide = {'display': 'none'}
    show = {'display': 'block'}
    if struct_method is None:
        # No algorithm selected yet
        return hide, hide, hide, hide, hide
    # Determine which fields to show for each algorithm
    score_style = show if struct_method in ['TAN_CL', 'TAN_HC', 'TAN_HCSP'] else hide
    root_style = show if struct_method == 'TAN_CL' else hide
    folds_style = show if struct_method in ['FSSJ', 'BSEJ'] else hide
    epsilon_style = show if struct_method in ['TAN_HC', 'TAN_HCSP', 'FSSJ', 'BSEJ'] else hide
    k_style = show if struct_method == 'KDB' else hide
    return score_style, root_style, folds_style, epsilon_style, k_style

# Callback: Show/hide specific parameter fields based on selected parameter learning method
@app.callback(
    Output('alpha-field', 'style'),
    Output('trees-field', 'style'),
    Output('subsample-field', 'style'),
    Output('prior-field', 'style'),
    Input('param-method', 'value')
)
def toggle_param_fields(param_method):
    """
    Toggle visibility of parameter learning inputs (alpha, trees, subsample, prior) based on method.
    """
    hide = {'display': 'none'}
    show = {'display': 'block'}
    if param_method is None:
        return hide, hide, hide, hide
    alpha_style = show if param_method == 'Bayes' else hide
    # WANBIA has no additional parameter inputs
    awnb_style = show if param_method == 'AWNB' else hide
    prior_style = show if param_method == 'MANB' else hide
    # trees and subsample fields are both for AWNB
    return alpha_style, awnb_style, awnb_style, prior_style

# Callback: Show/hide evaluation fields based on selected evaluation metric
@app.callback(
    Output('eval-folds-field', 'style'),
    Output('dag-field', 'style'),
    Input('eval-metric', 'value')
)
def toggle_eval_fields(eval_metric):
    """
    Toggle visibility of CV-specific inputs for evaluation.
    """
    hide = {'display': 'none'}
    show = {'display': 'block'}
    if eval_metric == 'CV':
        return show, show
    else:
        return hide, hide

# Callback: Show/hide inspect fields (layout, font) when "Plot structure" is selected
@app.callback(
    Output('layout-field', 'style'),
    Output('font-field', 'style'),
    Input('inspect-option', 'value')
)
def toggle_inspect_fields(inspect_option):
    """
    Toggle visibility of layout and font options for model plotting.
    """
    hide = {'display': 'none'}
    show = {'display': 'block'}
    if inspect_option == 'plot':
        return show, show
    else:
        return hide, hide

# Main Callback: Execute the selected action when Run button is clicked
# ---------------- MAIN CALLBACK ----------------
@app.callback(
    Output('output-area', 'children'),

    # ------------- trigger -------------
    Input('run-button', 'n_clicks'),

    # ------------- shared state -------------
    State('dataset-store',   'data'),       
    State('action-dropdown', 'value'),
    State('class-dropdown',  'value'),

    # ------------- structure-learning params -------------
    State('structure-method',   'value'),
    State('score-dropdown',     'value'),
    State('root-dropdown',      'value'),
    State('struct-folds-input', 'value'),
    State('epsilon-input',      'value'),
    State('k-input',            'value'),

    # ------------- parameter-learning params -------------
    State('param-method',    'value'),
    State('alpha-input',     'value'),
    State('trees-input',     'value'),
    State('subsample-input', 'value'),
    State('prior-input',     'value'),

    # ------------- evaluation params -------------
    State('eval-metric',     'value'),
    State('eval-folds-input','value'),
    State('dag-check',       'value'),

    # ------------- prediction params -------------
    State('upload-predict',  'contents'),
    State('prob-check',      'value'),

    # ------------- inspect params -------------
    State('inspect-option',  'value'),
    State('layout-dropdown', 'value'),
    State('font-input',      'value'),

    prevent_initial_call=True
)
def run_action(n_clicks, df_json, action, class_var,
               struct_method, score_metric, root_feat, struct_folds, epsilon, k,
               param_method, alpha, trees, subsample, prior,
               eval_metric, eval_folds, dag_check,
               predict_contents, prob_choice,
               inspect_option, layout_type, font_size):
    """
    Execute the requested action (structure learning, parameter learning,
    evaluation, prediction, or inspection) and return the UI element that
    should appear in 'output-area'.
    """
    import pandas as pd, base64, numpy as np
    from io import StringIO

    global current_model, current_algorithm

    # ---------------- sanity checks ----------------
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    if not df_json or not action or not class_var:
        return dbc.Alert("Dataset, class variable, and action are required.", color="warning")

    # ---------------- reconstruct DataFrame ----------------
    df = pd.read_json(df_json, orient='split')

    # ========================================================
    # A) STRUCTURE LEARNING
    # ========================================================
    if action == 'structure':
        if class_var not in df.columns:
            return dbc.Alert(f"Column '{class_var}' not found.", color="danger")

        try:
            if   struct_method == 'NB':      model = bnclassify.nb(class_var, df)
            elif struct_method == 'TAN_CL':  model = bnclassify.tan_cl (class_var, df,
                                                                         score=score_metric.lower(),
                                                                         root=root_feat or None)
            elif struct_method == 'TAN_HC':  model = bnclassify.tan_hc (class_var, df,
                                                                         score=score_metric.lower(),
                                                                         epsilon=epsilon)
            elif struct_method == 'TAN_HCSP':model = bnclassify.tan_hcsp(class_var, df,
                                                                         score=score_metric.lower(),
                                                                         epsilon=epsilon)
            elif struct_method == 'FSSJ':    model = bnclassify.fssj  (class_var, df,
                                                                         k=struct_folds,
                                                                         epsilon=epsilon)
            elif struct_method == 'BSEJ':    model = bnclassify.bsej  (class_var, df,
                                                                         k=struct_folds,
                                                                         epsilon=epsilon)
            elif struct_method == 'KDB':     model = bnclassify.kdb   (class_var, df, k=k)
            elif struct_method == 'AODE':    model = bnclassify.aode  (class_var, df)
            else:
                return dbc.Alert("Select a structure-learning algorithm.", color="warning")
        except Exception as err:
            return dbc.Alert(f"Structure learning failed: {err}", color="danger")

        current_model, current_algorithm = model, struct_method
        return dbc.Alert(f"Structure learned with {struct_method}.", color="success")

    # ========================================================
    # B) PARAMETER LEARNING
    # ========================================================
    if action == 'param':
        if current_model is None:
            return dbc.Alert("Learn a structure first.", color="warning")
        try:
            if   param_method == 'MLE':   current_model = bnclassify.lp(current_model, df, smooth=0)
            elif param_method == 'Bayes': current_model = bnclassify.lp(current_model, df, smooth=alpha)
            elif param_method == 'WANBIA':current_model = bnclassify.lp(current_model, df, wanbia=True)
            elif param_method == 'AWNB':  current_model = bnclassify.lp(current_model, df,
                                                                        awnb_trees=trees,
                                                                        awnb_bootstrap=subsample)
            elif param_method == 'MANB':  current_model = bnclassify.lp(current_model, df,
                                                                        manb_prior=prior)
            else:
                return dbc.Alert("Select a parameter-learning method.", color="warning")
        except Exception as err:
            return dbc.Alert(f"Parameter learning failed: {err}", color="danger")

        return dbc.Alert(f"Parameters learned using {param_method}.", color="success")

    # ========================================================
    # C) MODEL EVALUATION
    # ========================================================
    if action == 'evaluate':
        if current_model is None:
            return dbc.Alert("No model to evaluate.", color="warning")
        try:
            if eval_metric == 'CV':
                dag_fixed = ('fixed' in dag_check)
                acc = bnclassify.cv(current_model, df, k=eval_folds, dag=dag_fixed)
                acc_val = acc if isinstance(acc, float) else float(acc.get('accuracy', acc))
                return dbc.Alert(f"{eval_folds}-fold CV accuracy: {acc_val:.4f}", color="info")
            elif eval_metric == 'AIC':
                return dbc.Alert(f"AIC = {bnclassify.aic(current_model, df)}", color="info")
            elif eval_metric == 'BIC':
                return dbc.Alert(f"BIC = {bnclassify.bic(current_model, df)}", color="info")
            elif eval_metric == 'LL':
                return dbc.Alert(f"Log-Likelihood = {bnclassify.log_lik(current_model, df)}", color="info")
        except Exception as err:
            return dbc.Alert(f"Evaluation failed: {err}", color="danger")

    # ========================================================
    # D) PREDICTION
    # ========================================================
    if action == 'predict':
        if current_model is None:
            return dbc.Alert("Train a model before predicting.", color="warning")
        if not predict_contents:
            return dbc.Alert("Upload a prediction file.", color="warning")

        try:
            _, content_string = predict_contents.split(',', 1)
            test_df = pd.read_csv(StringIO(base64.b64decode(content_string).decode('utf-8')))
        except Exception as err:
            return dbc.Alert(f"Could not read prediction file: {err}", color="danger")

        X = test_df.drop(columns=[class_var], errors='ignore')
        post = ('prob' in (prob_choice or []))
        try:
            y_pred = bnclassify.predict(current_model, X, prob=post)
        except Exception as err:
            return dbc.Alert(f"Prediction failed: {err}", color="danger")

        if post:                                 # probabilities
            if isinstance(y_pred, tuple):        # (labels, probs)
                y_lbl, probs = y_pred
            else:                                # DataFrame of probabilities
                probs = y_pred.values
                y_lbl = y_pred.idxmax(axis=1).values
            out = pd.DataFrame(probs, columns=[f"P({c})" for c in sorted(df[class_var].unique())])
            out.insert(0, "Class", y_lbl)
        else:                                    # just labels
            out = pd.DataFrame({"Class": y_pred})

        preview = out.head(10)
        table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in preview.columns],
            data=preview.to_dict('records'),
            page_size=10, style_table={'overflowX': 'auto'}
        )
        return html.Div([html.P(f"First {len(preview)} predictions:"), table])

    # ========================================================
    # E) INSPECTION
    # ========================================================
    if action == 'inspect':
        if current_model is None:
            return dbc.Alert("Train a model first.", color="warning")
        if inspect_option == 'summary':
            # Display model summary (structure and parameters info)
            try:
                info = str(current_model)
            except Exception:
                info = "Model summary not available."
            # Display as preformatted text
            return html.Pre(info)
        if inspect_option == 'cpts':
            # Display Conditional Probability Tables for each node
            try:
                # Assume current_model has a property like get_cpds() if it's a pgmpy model
                cpds = current_model.get_cpds() if hasattr(current_model, 'get_cpds') else current_model.cpds
            except Exception as e:
                return dbc.Alert(f"Unable to retrieve CPTs: {e}", color="danger")
            if not cpds:
                return dbc.Alert("No CPTs available (model parameters may not be learned).", color="warning")
            cpt_texts = []
            for cpd in cpds:
                cpt_texts.append(str(cpd))
            # Join all CPTs text
            return html.Pre("\n\n".join(cpt_texts))
        if inspect_option == 'nparams':
            # Compute number of free parameters in the model
            try:
                # If model has a method or attribute for this
                nparam = current_model.nparams if hasattr(current_model, 'nparams') else None
            except:
                nparam = None
            if nparam is None:
                # Calculate manually from CPDs
                try:
                    cpds = current_model.get_cpds()
                except:
                    cpds = current_model.cpds if hasattr(current_model, 'cpds') else []
                nparam = 0
                for cpd in cpds:
                    # Each CPD: degrees of freedom = (cardinality(child)-1) * product(cardinalities(parents))
                    try:
                        child_card = cpd.cardinality[0]
                        parent_combo = int(cpd.values.size / child_card)
                        nparam += (child_card - 1) * parent_combo
                    except Exception:
                        continue
            return dbc.Alert(f"Number of free parameters: {nparam}", color="info")
        if inspect_option == 'plot':
            # Plot the structure as a network graph
            try:
                # Prepare nodes and edges for cytoscape
                if hasattr(current_model, 'edges'):
                    edges = list(current_model.edges())
                elif hasattr(current_model, 'structure_edges'):
                    edges = current_model.structure_edges
                else:
                    return dbc.Alert("Model edges could not be retrieved for plotting.", color="danger")
            except Exception as e:
                return dbc.Alert(f"Error retrieving model edges: {e}", color="danger")
            # Prepare cytoscape elements
            nodes = set()
            for u, v in edges:
                nodes.add(u); nodes.add(v)
            elements = [{"data": {"id": node, "label": node}} for node in nodes]
            for u, v in edges:
                elements.append({"data": {"source": u, "target": v}})
            # Set layout options
            layout_opts = {'name': layout_type}
            if layout_type == 'breadthfirst':
                # For breadthfirst layout, specify the class node as root if possible
                layout_opts['roots'] = f"[id = \"{class_var}\"]"
            # Create cytoscape component for graph
            cyto = dash.dependencies.no_update
            try:
                import dash_cytoscape as cy
                cyto = cy.Cytoscape(
                    id='structure-cytoscape',
                    elements=elements,
                    layout=layout_opts,
                    style={'width': '100%', 'height': '500px'},
                    stylesheet=[{'selector': 'node', 'style': {'label': 'data(label)', 'font-size': font_size}}]
                )
            except ImportError:
                # dash_cytoscape might not be installed
                return dbc.Alert("dash_cytoscape is not installed. Please install it to view the graph.", color="warning")
            return cyto
    
    # Default: if none of the above, return nothing
    return ""

# ---------- (5) RUN THE SERVER ---------- #
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8056)