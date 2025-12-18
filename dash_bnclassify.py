import base64
import zlib
import pandas as pd
import numpy as np
from io import StringIO
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Import the bnclassify wrapper
import bnclassify

# ---------- (1) CREATE DASH APP ---------- #
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    requests_pathname_prefix='/Model/LearningFromData/bnclassifyDash/',
    suppress_callback_exceptions=True
)

# Expose the server for Gunicorn
server = app.server

# Global variables to hold the current model and its algorithm type
current_model = None
current_algorithm = None

# =============================================================================
#                                   STYLES
# =============================================================================
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "backgroundColor": "#f8f9fa",
}

CONTENT_STYLE = {
    "marginLeft": "18.5rem",
    "marginRight": "2rem",
    "padding": "2rem 1rem",
}

# =============================================================================
#                                COMPONENTS
# =============================================================================

sidebar = html.Div(
    [
        html.H3("bnclassify", className="display-6"),
        html.Hr(),
        html.P(
            "Bayesian Network Classifiers", className="lead", style={"fontSize": "1rem"}
        ),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="fas fa-database me-2"), "Data & Connect"], href="#", id="nav-data", active=True),
                dbc.NavLink([html.I(className="fas fa-project-diagram me-2"), "Structure Learning"], href="#", id="nav-structure"),
                dbc.NavLink([html.I(className="fas fa-sliders-h me-2"), "Parameter Learning"], href="#", id="nav-params"),
                dbc.NavLink([html.I(className="fas fa-check-circle me-2"), "Evaluation"], href="#", id="nav-eval"),
                dbc.NavLink([html.I(className="fas fa-magic me-2"), "Prediction"], href="#", id="nav-predict"),
                dbc.NavLink([html.I(className="fas fa-search me-2"), "Inspect Model"], href="#", id="nav-inspect"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        dbc.Button("Reset / Clear", id="reset-button", color="danger", outline=True, size="sm", className="w-100"),
    ],
    style=SIDEBAR_STYLE,
)

# Function to helper create icon with tooltip
def help_icon(id, text):
    return html.Span([
        html.I(className="fas fa-question-circle ms-1 text-info", id=id, style={"cursor": "pointer"}),
        dbc.Tooltip(text, target=id)
    ])

# --- 1. DATA SECTION ---
data_section = html.Div(id="section-data", children=[
    html.H2("Data & Connectivity"),
    html.P("Upload your dataset or use the default one to get started."),
    html.Hr(),
    
    dbc.Card(
        dbc.CardBody([
            html.H5("Upload Dataset (CSV)", className="card-title"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select File')
                ]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            ),
            dbc.Checkbox(
                id='use-default-dataset',
                label=" Use default 'car' dataset (features & class column)",
                value=False,
                className="mb-3"
            ),
            
            html.Div(id='data-status-msg', className="text-success fw-bold"),
            html.Br(),
            
            html.Label(["Target (Class) Variable", help_icon("tt-class", "The variable you want to predict.")], className="fw-bold"),
            dcc.Dropdown(id='class-dropdown', placeholder="Select class column", clearable=False),
        ])
    ),
    html.Br(),
    html.H5("Preview"),
    html.Div(id='data-preview')
])

# --- 2. STRUCTURE LEARNING SECTION ---
structure_section = html.Div(id="section-structure", style={"display": "none"}, children=[
    html.H2("Structure Learning"),
    html.P("Learn the network structure from data."),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            dbc.Label(["Algorithm Family", help_icon("tt-algo", "Select the structure learning algorithm. NB is simplest. TAN allows one parent. KDB allows k parents.")]),
            dcc.Dropdown(
                id='structure-method',
                options=[
                    {'label': 'Naive Bayes (NB)', 'value': 'NB'},
                    {'label': 'TAN (Chow-Liu) [Standard]', 'value': 'TAN_CL'},
                    {'label': 'TAN (Hill-Climbing)', 'value': 'TAN_HC'},
                    {'label': 'AODE (Ensemble)', 'value': 'AODE'},
                    {'label': 'KDB (k-Dependence)', 'value': 'KDB'},
                    {'label': 'FSSJ (Forward Selection)', 'value': 'FSSJ'},
                    {'label': 'BSEJ (Backward Elimination)', 'value': 'BSEJ'},
                ],
                value='NB',
                clearable=False
            ),
        ], width=6),
    ]),
    html.Br(),
    
    # Dynamic Parameters Card
    dbc.Card([
        dbc.CardHeader("Algorithm Parameters"),
        dbc.CardBody([
            # TAN - Score
            dbc.Row([
                dbc.Col([
                    dbc.Label(["Score Metric", help_icon("tt-score", "AIC/BIC include penalties for complexity. Log-Likelihood (LL) is raw fit.")], id="lbl-score"), 
                    dcc.Dropdown(
                        id='score-dropdown',
                        options=[
                            {'label': 'AIC', 'value': 'AIC'},
                            {'label': 'BIC', 'value': 'BIC'},
                            {'label': 'Log-Likelihood', 'value': 'LL'}
                        ],
                        value='AIC', clearable=False
                    )
                ], width=4, id="field-score", style={"display": "none"}),
                
                # TAN - Root
                dbc.Col([
                    dbc.Label(["Root Node", help_icon("tt-root", "Root for the Tree-Augmented structure. Auto-selected if blank.")], id="lbl-root"),
                    dcc.Dropdown(id='root-dropdown', placeholder="Auto", clearable=True)
                ], width=4, id="field-root", style={"display": "none"}),
            ]),
            
            # Epsilon & K & Folds
            dbc.Row([
                dbc.Col([
                    dbc.Label(["Epsilon", help_icon("tt-eps", "Min improvement threshold for Greedy Search (Hill Climbing).")], id="lbl-eps"),
                    dbc.Input(id='epsilon-input', type='number', value=0.01, step=0.001)
                ], width=4, id="field-epsilon", style={"display": "none"}),
                
                dbc.Col([
                    dbc.Label(["k (Max Parents)", help_icon("tt-k", "Max number of parent nodes per attribute in KDB.")], id="lbl-k"),
                    dbc.Input(id='k-input', type='number', value=2, min=1, step=1)
                ], width=4, id="field-k", style={"display": "none"}),
                
                dbc.Col([
                   dbc.Label(["CV Folds", help_icon("tt-folds", "Number of folds for internal cross-validation in wrapper algorithms.")], id="lbl-folds"),
                   dbc.Input(id='struct-folds-input', type='number', value=5, min=2)
                ], width=4, id="field-folds", style={"display": "none"}),
            ], className="mt-2"),
        ])
    ]),
    
    html.Br(),
    dbc.Button("Train Structure", id="btn-train-structure", color="primary", size="lg")
])

# --- 3. PARAMETER LEARNING SECTION ---
params_section = html.Div(id="section-params", style={"display": "none"}, children=[
    html.H2("Parameter Learning"),
    html.P("Estimate the parameters (CPTs) for the learned structure."),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            dbc.Label(["Method", help_icon("tt-param", "Bayesin uses Dirichlet priors (smoothing). MLE is raw counts.")]),
            dcc.Dropdown(
                id='param-method',
                options=[
                    {'label': 'Maximum Likelihood (MLE)', 'value': 'MLE'},
                    {'label': 'Bayesian (Dirichlet Prior)', 'value': 'Bayes'},
                    {'label': 'WANBIA (Feature Weighting)', 'value': 'WANBIA'},
                    {'label': 'AWNB (Attribute Weighted)', 'value': 'AWNB'},
                    {'label': 'MANB (Model Averaged)', 'value': 'MANB'}
                ],
                value='Bayes',
                clearable=False
            )
        ], width=6),
    ]),
    html.Br(),
    
    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Label(["Smoothing Alpha", help_icon("tt-alpha", "Laplace smoothing parameter. 0 = MLE.")], id="lbl-alpha"),
                dbc.Input(id='alpha-input', type='number', value=0.5, step=0.1)
            ], width=4, id="field-alpha"),
            
            dbc.Col([
                dbc.Label(["AWNB Trees", help_icon("tt-trees", "Number of bootstrap trees to average.")], id="lbl-trees"),
                dbc.Input(id='trees-input', type='number', value=10, min=1)
            ], width=4, id="field-trees", style={"display": "none"}),
             
            dbc.Col([
                dbc.Label(["MANB Prior", help_icon("tt-prior", "Prior probability for Model Averaging.")], id="lbl-prior"),
                dbc.Input(id='prior-input', type='number', value=0.5, step=0.1)
            ], width=4, id="field-prior", style={"display": "none"}),
        ])
    ])),
    
    html.Br(),
    dbc.Button("Learn Parameters", id="btn-learn-params", color="primary", size="lg")
])

# --- 4. EVALUATION SECTION ---
eval_section = html.Div(id="section-eval", style={"display": "none"}, children=[
    html.H2("Model Evaluation"),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            dbc.Label(["Metric", help_icon("tt-eval-metric", "Evaluation metric to assess model performance.")]),
            dcc.Dropdown(
                id='eval-metric',
                options=[
                    {'label': 'Cross-Validation Accuracy', 'value': 'CV'},
                    {'label': 'AIC Score', 'value': 'AIC'},
                    {'label': 'BIC Score', 'value': 'BIC'},
                    {'label': 'Log-Likelihood', 'value': 'LL'}
                ],
                value='CV', clearable=False
            )
        ], width=6)
    ]),
    html.Br(),
    
    dbc.Row([
        dbc.Col([
             dbc.Label(["Folds", help_icon("tt-cv-folds", "Number of folds for Cross-Validation.")], id='lbl-eval-folds'),
             dbc.Input(id='eval-folds-input', type='number', min=2, step=1, value=10)
        ], width=3, id='field-eval-folds'),
        dbc.Col([
            dbc.Label(["Structure Setting", help_icon("tt-dag", "If fixed, only parameters are relearned each fold. If not, structure is relearned (slower).")]),
            dbc.Checklist(
                options=[{'label': ' Fix Structure (only relearn params)', 'value': 'fixed'}],
                value=['fixed'], id='dag-check', switch=True
            )
        ], width=5, id='field-dag-check')
    ]),
    
    html.Br(),
    dbc.Button("Evaluate Model", id="btn-evaluate", color="info", size="lg")
])

# --- 5. PREDICTION SECTION ---
predict_section = html.Div(id="section-predict", style={"display": "none"}, children=[
    html.H2("Prediction"),
    html.P("Predict classes for new data using the trained model."),
    html.Hr(),
    
    dbc.Label("Upload Unlabeled Data (CSV)"),
    dcc.Upload(
        id='upload-predict',
        children=html.Div(['Drag and Drop or ', html.A('Select File')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center'
        },
        multiple=False
    ),
    html.Br(),
    dbc.Checkbox(id='prob-check', label=" Output Posterior Probabilities", value=False),
    html.Br(),
    dbc.Button("Run Prediction", id="btn-predict", color="success", size="lg"),
    
    html.Br(), html.Br(),
    html.Div(id="prediction-results")
])

# --- 6. INSPECT SECTION ---
inspect_section = html.Div(id="section-inspect", style={"display": "none"}, children=[
    html.H2("Inspect Model"),
    html.Hr(),
    
    dbc.Tabs([
        dbc.Tab(label="Network Graph", tab_id="tab-graph"),
        dbc.Tab(label="Model Summary", tab_id="tab-summary"),
        dbc.Tab(label="CPTs", tab_id="tab-cpts"),
    ], id="inspect-tabs", active_tab="tab-graph"),
    html.Br(),
    
    html.Div(id="inspect-content")
])


# --- LAYOUT CONSTRUCTION ---
content = html.Div(id="page-content", style=CONTENT_STYLE, children=[
    data_section,
    structure_section,
    params_section,
    eval_section,
    predict_section,
    inspect_section,
    
    html.Hr(),
    # Global Output Area for status messages/results
    dbc.Card(dbc.CardBody([
        html.H4("Output / Logs", className="card-title"),
        dcc.Loading(id='loading-output', type='circle', children=[
            html.Div(id='output-area')
        ])
    ]), className="mt-4 bg-light")
])

app.layout = html.Div([
    dcc.Store(id='dataset-store'), 
    dcc.Store(id='active-section-store', data='data'),
    sidebar,
    content
])

# =============================================================================
#                                CALLBACKS
# =============================================================================

# 1. Navigation Callback: Toggle Sections
@app.callback(
    [Output("section-data", "style"),
     Output("section-structure", "style"),
     Output("section-params", "style"),
     Output("section-eval", "style"),
     Output("section-predict", "style"),
     Output("section-inspect", "style"),
     Output("nav-data", "active"),
     Output("nav-structure", "active"),
     Output("nav-params", "active"),
     Output("nav-eval", "active"),
     Output("nav-predict", "active"),
     Output("nav-inspect", "active")],
    [Input("nav-data", "n_clicks"),
     Input("nav-structure", "n_clicks"),
     Input("nav-params", "n_clicks"),
     Input("nav-eval", "n_clicks"),
     Input("nav-predict", "n_clicks"),
     Input("nav-inspect", "n_clicks")],
)
def navigate(n_data, n_struc, n_param, n_eval, n_pred, n_insp):
    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'nav-data'
    
    # Styles
    hide = {"display": "none"}
    show = {"display": "block"}
    
    # Defaults
    styles = [hide] * 6
    actives = [False] * 6
    
    if button_id == "nav-structure":
        styles[1] = show; actives[1] = True
    elif button_id == "nav-params":
        styles[2] = show; actives[2] = True
    elif button_id == "nav-eval":
        styles[3] = show; actives[3] = True
    elif button_id == "nav-predict":
        styles[4] = show; actives[4] = True
    elif button_id == "nav-inspect":
        styles[5] = show; actives[5] = True
    else: # Default or nav-data
        styles[0] = show; actives[0] = True
        
    return (*styles, *actives)

# 2. Data Loading Callback
@app.callback(
    Output('data-preview',    'children'),
    Output('class-dropdown',  'options'),
    Output('class-dropdown',  'value'),
    Output('dataset-store',   'data'),
    Output('data-status-msg', 'children'),
    Input('upload-data',      'contents'),
    Input('use-default-dataset', 'value'),
    State('upload-data',      'filename'),
    prevent_initial_call=True
)
def load_and_preview_data(upload_contents, use_default, upload_filename):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    df = None
    msg = ""
    
    if trigger_id == 'use-default-dataset' and use_default:
        try:
            default_file = '/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/bnclassify/carwithnames.data'
            df = pd.read_csv(default_file)
            msg = "Loaded default 'car' dataset."
        except Exception as e:
            return dbc.Alert(f"Error loading default: {e}", color="danger"), [], None, None, ""
            
    elif trigger_id == 'upload-data' and upload_contents:
        try:
            _, content_string = upload_contents.split(',', 1)
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(StringIO(decoded.decode('utf-8')))
            msg = f"Loaded file: {upload_filename}"
        except Exception as e:
            return dbc.Alert(f"Error reading file: {e}", color="danger"), [], None, None, ""
    
    if df is not None:
        # Create preview
        preview_table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in df.columns],
            data=df.head().to_dict("records"),
            style_table={'overflowX': 'auto'},
            page_size=5
        )
        
        # Class options (last col default)
        class_cols = [{'label': c, 'value': c} for c in df.columns]
        default_class = df.columns[-1]
        
        return preview_table, class_cols, default_class, df.to_json(date_format='iso', orient='split'), msg
    
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, ""

# 3. Dynamic Parameter Visibility (Structure & Params)
@app.callback(
    [Output("field-score", "style"), Output("field-root", "style"), 
     Output("field-epsilon", "style"), Output("field-k", "style"), Output("field-folds", "style")],
    Input("structure-method", "value")
)
def update_structure_fields(method):
    hide = {"display": "none"}
    show = {"display": "block"}
    
    # Defaults
    s_score = hide; s_root = hide; s_eps = hide; s_k = hide; s_folds = hide
    
    if method in ['TAN_CL']:
        s_score = show; s_root = show
    if method in ['TAN_HC']:
        s_score = show; s_eps = show
    if method in ['KDB']:
        s_k = show
    if method in ['FSSJ', 'BSEJ']:
        s_folds = show; s_eps = show
        
    return s_score, s_root, s_eps, s_k, s_folds

@app.callback(
    [Output("field-alpha", "style"), Output("field-trees", "style"), Output("field-prior", "style")],
    Input("param-method", "value")
)
def update_param_fields(method):
    hide = {"display": "none"}
    show = {"display": "block"}
    
    s_alpha = show if method == 'Bayes' else hide
    s_trees = show if method == 'AWNB' else hide
    s_prior = show if method == 'MANB' else hide
    
    return s_alpha, s_trees, s_prior

@app.callback(
    [Output('field-eval-folds', 'style'), Output('field-dag-check', 'style')],
    Input('eval-metric', 'value')
)
def update_eval_fields(val):
    if val == 'CV': return {'display': 'block'}, {'display': 'block'}
    return {'display': 'none'}, {'display': 'none'}


# 4. MAIN EXECUTION CALLBACK (Handling Button Clicks)
@app.callback(
    Output('output-area', 'children'),
    [Input('btn-train-structure', 'n_clicks'),
     Input('btn-learn-params', 'n_clicks'),
     Input('btn-evaluate', 'n_clicks'),
     Input('btn-predict', 'n_clicks')],
    [State('dataset-store', 'data'),
     State('class-dropdown', 'value'),
     # Structure params
     State('structure-method', 'value'), State('score-dropdown', 'value'), State('root-dropdown', 'value'),
     State('epsilon-input', 'value'), State('k-input', 'value'), State('struct-folds-input', 'value'),
     # Param params
     State('param-method', 'value'), State('alpha-input', 'value'), State('trees-input', 'value'), State('prior-input', 'value'),
     # Eval params
     State('eval-metric', 'value'), State('eval-folds-input', 'value'), State('dag-check', 'value'),
     # Predict params
     State('upload-predict', 'contents'), State('prob-check', 'value')]
)
def execute_action(n_struc, n_param, n_eval, n_pred, 
                  df_json, class_var,
                  struct_method, score, root, eps, k, s_folds,
                  param_method, alpha, trees, prior,
                  eval_metric, eval_folds, dag_check,
                  pred_contents, prob_check):
    
    ctx = callback_context
    if not ctx.triggered: return ""
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not df_json: return dbc.Alert("No dataset loaded!", color="warning")
    
    global current_model
    df = pd.read_json(df_json, orient='split')
    
    # --- TRAIN STRUCTURE ---
    if trigger == 'btn-train-structure':
        try:
            if   struct_method == 'NB':      model = bnclassify.nb(class_var, df)
            elif struct_method == 'TAN_CL':  model = bnclassify.tan_cl(class_var, df, score=score.lower(), root=root)
            elif struct_method == 'TAN_HC':  model = bnclassify.tan_hc(class_var, df, score=score.lower(), epsilon=eps)
            elif struct_method == 'KDB':     model = bnclassify.kdb(class_var, df, k=k)
            elif struct_method == 'AODE':    model = bnclassify.aode(class_var, df)
            elif struct_method == 'FSSJ':    model = bnclassify.fssj(class_var, df, k=s_folds, epsilon=eps)
            elif struct_method == 'BSEJ':    model = bnclassify.bsej(class_var, df, k=s_folds, epsilon=eps)
            else: return dbc.Alert("Unknown Algorithm", color="danger")
            
            current_model = model
            return dbc.Alert(f"Structure learned successfully: {struct_method}", color="success")
        except Exception as e:
             return dbc.Alert(f"Structure Learning Failed: {e}", color="danger")

    # --- LEARN PARAMETERS ---
    if trigger == 'btn-learn-params':
        if current_model is None: return dbc.Alert("No structure! Learn structure first.", color="warning")
        try:
            if   param_method == 'MLE':    current_model = bnclassify.lp(current_model, df, smooth=0)
            elif param_method == 'Bayes':  current_model = bnclassify.lp(current_model, df, smooth=alpha)
            elif param_method == 'WANBIA': current_model = bnclassify.lp(current_model, df, wanbia=True)
            elif param_method == 'AWNB':   current_model = bnclassify.lp(current_model, df, awnb_trees=trees)
            elif param_method == 'MANB':   current_model = bnclassify.lp(current_model, df, manb_prior=prior)
            
            return dbc.Alert(f"Parameters learned using {param_method}", color="success")
        except Exception as e:
            return dbc.Alert(f"Parameter Learning Failed: {e}", color="danger")
            
    # --- EVALUATE ---
    if trigger == 'btn-evaluate':
        if current_model is None: return dbc.Alert("Model not ready for evaluation.", color="warning")
        try:
            if eval_metric == 'CV':
                fixed = ('fixed' in dag_check)
                acc = bnclassify.cv(current_model, df, k=eval_folds, dag=fixed)
                val = acc['accuracy'] if isinstance(acc, dict) else acc
                return dbc.Alert(f"Cross-Validation Accuracy: {float(val):.4f}", color="info")
            elif eval_metric == 'AIC':
                return dbc.Alert(f"AIC: {bnclassify.aic(current_model, df)}", color="info")
            elif eval_metric == 'BIC':
                return dbc.Alert(f"BIC: {bnclassify.bic(current_model, df)}", color="info")
            elif eval_metric == 'LL':
                return dbc.Alert(f"Log-Likelihood: {bnclassify.log_lik(current_model, df)}", color="info")
            
        except Exception as e:
            return dbc.Alert(f"Evaluation Failed: {e}", color="danger")

    # --- PREDICT ---
    if trigger == 'btn-predict':
        if current_model is None: return dbc.Alert("No model trained!", color="warning")
        if not pred_contents: return dbc.Alert("Upload prediction data first.", color="warning")
        
        try:
            _, content_string = pred_contents.split(',', 1)
            test_df = pd.read_csv(StringIO(base64.b64decode(content_string).decode('utf-8')))
            X = test_df.drop(columns=[class_var], errors='ignore')
            probs = prob_check
            
            preds = bnclassify.predict(current_model, X, prob=probs)
            
            # Simple result display
            if probs:
                res_df = pd.DataFrame(preds) # simplified, assuming matrix return
                msg = "Probabilities calculated."
            else:
                res_df = pd.DataFrame({'Class': preds})
                msg = "Classes predicted."
            
            return html.Div([
                dbc.Alert(msg, color="success"),
                dash_table.DataTable(
                    data=res_df.head(10).to_dict('records'), 
                    columns=[{'name': str(c), 'id': str(c)} for c in res_df.columns],
                    page_size=10
                )
            ])
            
        except Exception as e:
            return dbc.Alert(f"Prediction Failed: {e}", color="danger")
            
    return dash.no_update

# 5. INSPECTION CALLBACK
@app.callback(
    Output("inspect-content", "children"),
    [Input("inspect-tabs", "active_tab"), Input("nav-inspect", "n_clicks")],
)
def update_inspection(active_tab, n_nav):
    # Only update if we have a model
    if current_model is None:
        return dbc.Alert("Train a model first to inspect it.", color="warning")
    
    if active_tab == "tab-summary":
        return html.Pre(str(current_model))
    
    if active_tab == "tab-cpts":
        # Rough dump of CPT methods
        try:
            # Depending on wrapper implementation
            return html.Pre(str(current_model.params)) 
        except:
             return html.Div("CPTs not directly accessible via string.")

    if active_tab == "tab-graph":
        # Graph visualization
        try:
            import dash_cytoscape as cy
            # We need edges. Assuming current_model has an edges() method or property
            edges = []
            if hasattr(current_model, 'edges'):
                edges = current_model.edges # might be a list of lists or tuples
                if callable(edges): edges = edges()
            
            # If edges is None or empty, handle
            if not edges: return dbc.Alert("No edges to plot.", color="info")
            
            elements = []
            nodes = set()
            for u, v in edges:
                nodes.add(u); nodes.add(v)
                elements.append({'data': {'source': u, 'target': v}})
            for n in nodes:
                elements.append({'data': {'id': n, 'label': n}})
                
            return cy.Cytoscape(
                id='cytoscape',
                elements=elements,
                layout={'name': 'breadthfirst'},
                style={'width': '100%', 'height': '600px'},
                stylesheet=[
                    {'selector': 'node', 'style': {'label': 'data(label)', 'background-color': '#007bff'}},
                    {'selector': 'edge', 'style': {'line-color': '#ccc', 'target-arrow-shape': 'triangle'}}
                ]
            )
        except ImportError:
             return dbc.Alert("dash_cytoscape not installed", color="danger")
        except Exception as e:
             return dbc.Alert(f"Plotting error: {e}", color="danger")
    
    return html.Div("Select a tab")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8056)