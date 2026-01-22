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
current_structure_algorithm = None  # Track what structure algorithm was used
current_params_learned = False  # Track whether parameters have been learned

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

# Cytoscape stylesheet - CIG Corporate Style (from MBC)
cytoscape_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'content': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': '#E3F2FD',
            'border-color': '#90CAF9',
            'border-width': 2,
            'width': 50,
            'height': 50,
            'font-size': 11,
            'color': '#333',
            'text-wrap': 'wrap',
            'text-max-width': '80px'
        }
    },
    {
        'selector': 'node[classes="class_node"]',
        'style': {
            'background-color': '#00A2E1',
            'border-color': '#0077A8',
            'shape': 'ellipse',
            'color': '#FFFFFF',
            'font-weight': 'bold',
            'border-width': 3
        }
    },
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#666',
            'line-color': '#666',
            'width': 2,
            'opacity': 0.7
        }
    }
]

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
                dbc.NavLink([html.I(className="fas fa-cogs me-2"), "Structure & Parameters"], href="#", id="nav-structure"),
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

# Function to helper create icon for popover
def help_icon(id):
    return html.I(className="fas fa-question-circle ms-1 text-info", id=id, style={"cursor": "pointer"})

# --- 1. DATA SECTION ---
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
                label=" Use default 'Asia' (lung cancer) dataset",
                value=False,
                className="mb-3"
            ),
            html.Div(id='data-status-msg', className="text-success fw-bold"),
            html.Br(),
            
            html.Label(["Target (Class) Variable ", 
                dbc.Button(
                    html.I(className="fas fa-question-circle"),
                    id="help-button-class",
                    color="link",
                    style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                ),
            ], className="fw-bold"),
            dcc.Dropdown(id='class-dropdown', placeholder="Select class column", clearable=False, style={'position': 'relative', 'zIndex': '9999'}),
        ])
    ),

    html.Br(),
    html.H5("Preview"),
    html.Div(id='data-preview')
])

# --- 2. COMBINED STRUCTURE & PARAMETER LEARNING SECTION ---
structure_section = html.Div(id="section-structure", style={"display": "none"}, children=[
    html.H2("Structure & Parameter Learning"),
    html.P("Configure both structure and parameters for your Bayesian Network Classifier. Both are interconnected and can be trained together."),
    dbc.Alert([
        html.I(className="fas fa-lightbulb me-2"),
        html.Strong("Scientific Note: "),
        "Structure and parameter learning are coexistent processes. The structure defines the dependencies, while parameters quantify them. Configure both aspects below."
    ], color="info", className="small"),
    html.Hr(),
    
    # === STRUCTURE LEARNING SECTION ===
    html.H4([html.I(className="fas fa-project-diagram me-2"), "Structure Learning"]),
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Label("Algorithm Family", style={'display': 'inline-block', 'marginRight': '5px'}),
                dbc.Button(
                    html.I(className="fas fa-question-circle"),
                    id="help-button-algorithms",
                    color="link",
                    style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                ),
            ]),
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
    
    # Structure Algorithm Parameters Card
    dbc.Card([
        dbc.CardHeader("Structure Algorithm Parameters"),
        dbc.CardBody([
            # TAN_CL - Score & Root (first row)
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Label("Score Metric", style={'display': 'inline-block', 'marginRight': '5px'}),
                        dbc.Button(
                            html.I(className="fas fa-question-circle"),
                            id="help-button-score",
                            color="link",
                            style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                        ),
                    ], id="lbl-score"),
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
                
                # TAN_CL - Root
                dbc.Col([
                    html.Div([
                        dbc.Label("Root Node", style={'display': 'inline-block', 'marginRight': '5px'}),
                        dbc.Button(
                            html.I(className="fas fa-question-circle"),
                            id="help-button-root",
                            color="link",
                            style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                        ),
                    ], id="lbl-root"),
                    dcc.Dropdown(id='root-dropdown', placeholder="Auto", clearable=True)
                ], width=4, id="field-root", style={"display": "none"}),
            ]),
            
            # Wrapper Algorithms - CV Folds & Epsilon (second row)
            dbc.Row([
                dbc.Col([
                   html.Div([
                        dbc.Label("CV Folds", style={'display': 'inline-block', 'marginRight': '5px'}),
                        dbc.Button(
                            html.I(className="fas fa-question-circle"),
                            id="help-button-folds",
                            color="link",
                            style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                        ),
                    ], id="lbl-folds"),
                   dbc.Input(id='struct-folds-input', type='number', value=5, min=2, step=1)
                ], width=4, id="field-folds", style={"display": "none"}),
                
                dbc.Col([
                    html.Div([
                        dbc.Label("Epsilon", style={'display': 'inline-block', 'marginRight': '5px'}),
                        dbc.Button(
                            html.I(className="fas fa-question-circle"),
                            id="help-button-epsilon",
                            color="link",
                            style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                        ),
                    ], id="lbl-eps"),
                    dbc.Input(id='epsilon-input', type='number', value=0.01, step=0.001, min=0)
                ], width=4, id="field-epsilon", style={"display": "none"}),
                
                # KDB - kdbk (Max Parents per Feature)
                dbc.Col([
                    html.Div([
                        dbc.Label("k (Max Parents)", style={'display': 'inline-block', 'marginRight': '5px'}),
                        dbc.Button(
                            html.I(className="fas fa-question-circle"),
                            id="help-button-k",
                            color="link",
                            style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                        ),
                    ], id="lbl-kdbk"),
                    dbc.Input(id='kdbk-input', type='number', value=2, min=1, step=1)
                ], width=4, id="field-kdbk", style={"display": "none"}),
            ], className="mt-2"),
        ])
    ]),
    
    html.Br(),
    html.Hr(),
    
    # === PARAMETER LEARNING SECTION ===
    html.H4([html.I(className="fas fa-sliders-h me-2"), "Parameter Learning"]),
    html.Div(id="current-params-info", className="alert alert-info"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Label("Parameter Learning Method", style={'display': 'inline-block', 'marginRight': '5px'}),
                dbc.Button(
                    html.I(className="fas fa-question-circle"),
                    id="help-button-methods",
                    color="link",
                    style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                ),
            ]),
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
    
    dbc.Card([
        dbc.CardHeader("Parameter Estimation Options"),
        dbc.CardBody([
            dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Label("Smoothing Alpha", style={'display': 'inline-block', 'marginRight': '5px'}),
                    dbc.Button(
                        html.I(className="fas fa-question-circle"),
                        id="help-button-alpha",
                        color="link",
                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                    ),
                ], id="lbl-alpha"),
                dbc.Input(id='alpha-input', type='number', value=0.5, step=0.1)
            ], width=4, id="field-alpha"),
            
            dbc.Col([
                html.Div([
                    dbc.Label("AWNB Trees", style={'display': 'inline-block', 'marginRight': '5px'}),
                    dbc.Button(
                        html.I(className="fas fa-question-circle"),
                        id="help-button-trees",
                        color="link",
                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                    ),
                ], id="lbl-trees"),
                dbc.Input(id='trees-input', type='number', value=10, min=1)
            ], width=4, id="field-trees", style={"display": "none"}),
             
            dbc.Col([
                html.Div([
                    dbc.Label("MANB Prior", style={'display': 'inline-block', 'marginRight': '5px'}),
                    dbc.Button(
                        html.I(className="fas fa-question-circle"),
                        id="help-button-prior",
                        color="link",
                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                    ),
                ], id="lbl-prior"),
                dbc.Input(id='prior-input', type='number', value=0.5, step=0.1)
            ], width=4, id="field-prior", style={"display": "none"}),
            ])
        ])
    ]),
    
    html.Br(),
    html.Hr(),
    
    # === TRAINING BUTTONS ===
    html.H5("Training Options"),
    html.Div(id="training-restrictions-alert"),  # Dynamic alert for restrictions
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Button([html.I(className="fas fa-play-circle me-2"), "Train Both (Structure + Parameters)"], 
                       id="btn-train-both", color="success", size="lg", className="w-100", disabled=False)
        ], width=6),
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("Structure Only", id="btn-train-structure", color="primary", outline=True, disabled=False),
                dbc.Button("Parameters Only", id="btn-learn-params", color="primary", outline=True, disabled=True)
            ], className="w-100")
        ], width=6)
    ])
])

# --- 3. PARAMS SECTION (REMOVED - NOW COMBINED ABOVE) ---
params_section = html.Div(id="section-params", style={"display": "none"}, children=[])

# --- 4. EVALUATION SECTION ---
eval_section = html.Div(id="section-eval", style={"display": "none"}, children=[
    html.H2("Model Evaluation"),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Label("Evaluation Metric", style={'display': 'inline-block', 'marginRight': '5px'}),
                dbc.Button(
                    html.I(className="fas fa-question-circle"),
                    id="help-button-eval-metric",
                    color="link",
                    style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                ),
            ]),
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
             html.Div([
                dbc.Label("CV Folds", style={'display': 'inline-block', 'marginRight': '5px'}),
                dbc.Button(
                    html.I(className="fas fa-question-circle"),
                    id="help-button-cv-folds",
                    color="link",
                    style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                ),
             ], id='lbl-eval-folds'),
             dbc.Input(id='eval-folds-input', type='number', min=2, step=1, value=10)
        ], width=3, id='field-eval-folds'),
        dbc.Col([
            html.Div([
                dbc.Label("Structure Setting", style={'display': 'inline-block', 'marginRight': '5px'}),
                dbc.Button(
                    html.I(className="fas fa-question-circle"),
                    id="help-button-dag",
                    color="link",
                    style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                ),
            ]),
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
    html.Div([
        html.Small("Or use the default 'Asia' test set:", className="text-muted me-2"),
        dbc.Button("Load Test Data (Asia)", id="btn-load-test-asia", color="secondary", size="sm", outline=True)
    ], className="mt-2 mb-3"),
    
    dbc.Alert([
        html.I(className="fas fa-info-circle me-2"),
        "How Prediction Works: Upload a CSV file (or load default) containing features. ",
        "The model predicts the class for each row (batch prediction). ",
        "Note: Unlike single-case prediction where you select values, this tool processes entire datasets for validation."
    ], color="light", className="small mt-2"),
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
    dcc.Store(id='model-state-store'),  # Store model state
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0),  # Update every 2 seconds
    sidebar,
    content,
    
    # Popovers with detailed descriptions
    dbc.Popover(
        [
            dbc.PopoverHeader("Target (Class) Variable", style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}),
            dbc.PopoverBody([
                html.P("The target variable is the class you want to predict in your classification task."),
                html.P("This should be a categorical variable in your dataset that you want the Bayesian Network Classifier to learn and predict."),
            ]),
        ],
        id="help-popover-class",
        target="help-button-class",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
    
    dbc.Popover(
        [
            dbc.PopoverHeader(
                [
                    "Structure Learning Algorithms",
                    html.I(className="fas fa-project-diagram ms-2", style={"color": "#0d6efd"})
                ],
                style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
            ),
            dbc.PopoverBody(
                [
                    html.Ul([
                        html.Li([html.Strong("NB - Naive Bayes: "), "Simplest structure. Assumes all features are independent given the class. Fast and effective baseline."]),
                        html.Li([html.Strong("TAN - Tree-Augmented Naive Bayes: "), "Extends NB by allowing tree-structured dependencies between features."]),
                        html.Li([html.Strong("  • TAN (Chow-Liu): "), "Uses Chow-Liu algorithm to find optimal tree structure."]),
                        html.Li([html.Strong("  • TAN (Hill-Climbing): "), "Uses greedy search to optimize tree structure."]),
                        html.Li([html.Strong("AODE - Averaged One-Dependence Estimators: "), "Ensemble method that averages multiple one-dependence models."]),
                        html.Li([html.Strong("KDB - k-Dependence Bayesian Classifier: "), "Allows each feature to have up to k parent features."]),
                        html.Li([html.Strong("FSSJ - Forward Sequential Selection Join: "), "Wrapper method using forward greedy search."]),
                        html.Li([html.Strong("BSEJ - Backward Sequential Elimination Join: "), "Wrapper method using backward elimination."]),
                    ], style={"fontSize": "13px"}),
                ],
                style={"backgroundColor": "#ffffff", "borderRadius": "0 0 0.25rem 0.25rem", "maxWidth": "500px"}
            ),
        ],
        id="help-popover-algorithms",
        target="help-button-algorithms",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
    
    dbc.Popover(
        [
            dbc.PopoverHeader(
                [
                    "Parameter Learning Methods",
                    html.I(className="fas fa-sliders-h ms-2", style={"color": "#0d6efd"})
                ],
                style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
            ),
            dbc.PopoverBody(
                [
                    html.Ul([
                        html.Li([html.Strong("MLE - Maximum Likelihood Estimation: "), "Uses raw frequency counts from data. No smoothing applied."]),
                        html.Li([html.Strong("Bayes - Bayesian Estimation: "), "Uses Dirichlet prior for smoothing. Helps prevent zero probabilities."]),
                        html.Li([html.Strong("WANBIA - Weighted Average of Naive Bayes with Instances Averaging: "), "Feature weighting method designed for NB, but works with any structure."]),
                        html.Li([html.Strong("AWNB - Attribute Weighted Naive Bayes: "), "Bootstrap ensemble approach with weighted features. Designed for NB but applicable to other structures."]),
                        html.Li([html.Strong("MANB - Model Averaged Naive Bayes: "), "Averages over multiple NB models. Only compatible with Naive Bayes structure."]),
                    ], style={"fontSize": "13px"}),
                    html.Hr(),
                    html.Small([
                        html.I(className="fas fa-info-circle text-info me-1"),
                        "Note: MANB requires Naive Bayes structure. WANBIA & AWNB are optimized for NB but can be used with other structures."
                    ], className="text-muted")
                ],
                style={"backgroundColor": "#ffffff", "borderRadius": "0 0 0.25rem 0.25rem", "maxWidth": "500px"}
            ),
        ],
        id="help-popover-methods",
        target="help-button-methods",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
    
    # Structure Parameters Popovers
    dbc.Popover(
        [
            dbc.PopoverHeader("Score Metric", style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}),
            dbc.PopoverBody([
                html.P([html.Strong("Score metrics"), " are used to evaluate and compare different network structures during learning."]),
                html.Ul([
                    html.Li([html.Strong("AIC (Akaike Information Criterion): "), "Balances model fit with complexity. Lower values are better. Includes penalty for number of parameters."]),
                    html.Li([html.Strong("BIC (Bayesian Information Criterion): "), "Similar to AIC but with stronger penalty for complexity. Preferred for larger datasets."]),
                    html.Li([html.Strong("LL (Log-Likelihood): "), "Raw measure of how well the model fits the data without complexity penalty."]),
                ], style={"fontSize": "13px"}),
                html.Hr(),
                html.Small([
                    html.I(className="fas fa-info-circle text-info me-1"),
                    "Only used with TAN (Chow-Liu) algorithm."
                ], className="text-muted")
            ]),
        ],
        id="help-popover-score",
        target="help-button-score",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
    
    dbc.Popover(
        [
            dbc.PopoverHeader("Root Node", style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}),
            dbc.PopoverBody([
                html.P("The root node is the starting point for the tree structure in Tree-Augmented Naive Bayes (TAN)."),
                html.P("If left blank, the algorithm will automatically select the most informative feature as root."),
                html.P("Manually selecting a root can be useful if you have domain knowledge about feature dependencies."),
                html.Hr(),
                html.Small([
                    html.I(className="fas fa-info-circle text-info me-1"),
                    "Only used with TAN (Chow-Liu) algorithm."
                ], className="text-muted")
            ]),
        ],
        id="help-popover-root",
        target="help-button-root",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
    
    dbc.Popover(
        [
            dbc.PopoverHeader("Cross-Validation Folds", style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}),
            dbc.PopoverBody([
                html.P("Number of folds for internal cross-validation during structure learning."),
                html.P("The dataset is divided into k parts. The model is trained on k-1 parts and validated on the remaining part. This process repeats k times."),
                html.P("Higher values give more reliable estimates but increase computation time."),
                html.Hr(),
                html.Small([
                    html.I(className="fas fa-info-circle text-info me-1"),
                    "Used in wrapper algorithms: TAN-HC, FSSJ, BSEJ, and KDB."
                ], className="text-muted")
            ]),
        ],
        id="help-popover-folds",
        target="help-button-folds",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
    
    dbc.Popover(
        [
            dbc.PopoverHeader("Epsilon (ε)", style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}),
            dbc.PopoverBody([
                html.P("Minimum improvement threshold for greedy search algorithms."),
                html.P("The algorithm stops when the improvement in score between iterations falls below this threshold."),
                html.Ul([
                    html.Li([html.Strong("Smaller values: "), "More thorough search, potentially better results, but slower."]),
                    html.Li([html.Strong("Larger values: "), "Faster convergence, but may stop before finding optimal structure."]),
                ], style={"fontSize": "13px"}),
                html.Hr(),
                html.Small([
                    html.I(className="fas fa-info-circle text-info me-1"),
                    "Used in: TAN-HC, FSSJ, BSEJ, and KDB algorithms."
                ], className="text-muted")
            ]),
        ],
        id="help-popover-epsilon",
        target="help-button-epsilon",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
    
    dbc.Popover(
        [
            dbc.PopoverHeader("k (Max Parents)", style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}),
            dbc.PopoverBody([
                html.P([html.Strong("k-Dependence"), " refers to the maximum number of feature parents that each feature can have (in addition to the class)."]),
                html.Ul([
                    html.Li([html.Strong("k = 0: "), "Equivalent to Naive Bayes (no feature dependencies)."]),
                    html.Li([html.Strong("k = 1: "), "Each feature can have at most 1 other feature as parent (similar to TAN)."]),
                    html.Li([html.Strong("k = 2 or more: "), "Allows richer dependency structures, but increases complexity."]),
                ], style={"fontSize": "13px"}),
                html.Hr(),
                html.Small([
                    html.I(className="fas fa-info-circle text-info me-1"),
                    "Only used with KDB (k-Dependence Bayesian Classifier) algorithm."
                ], className="text-muted")
            ]),
        ],
        id="help-popover-k",
        target="help-button-k",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
    
    # Parameter Learning Popovers
    dbc.Popover(
        [
            dbc.PopoverHeader("Smoothing Alpha (α)", style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}),
            dbc.PopoverBody([
                html.P("Laplace smoothing parameter for Bayesian parameter estimation."),
                html.P("Adds a small pseudo-count to all probability estimates to avoid zero probabilities."),
                html.Ul([
                    html.Li([html.Strong("α = 0: "), "No smoothing (equivalent to MLE). May cause issues with unseen data."]),
                    html.Li([html.Strong("α = 0.5: "), "Moderate smoothing. Good default for most cases."]),
                    html.Li([html.Strong("α = 1: "), "Laplace smoothing. Adds uniform prior."]),
                ], style={"fontSize": "13px"}),
                html.Hr(),
                html.Small([
                    html.I(className="fas fa-info-circle text-info me-1"),
                    "Only applies when using Bayesian parameter learning method."
                ], className="text-muted")
            ]),
        ],
        id="help-popover-alpha",
        target="help-button-alpha",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
    
    dbc.Popover(
        [
            dbc.PopoverHeader("AWNB Trees", style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}),
            dbc.PopoverBody([
                html.P("Number of bootstrap trees to create and average in the AWNB (Attribute Weighted Naive Bayes) ensemble."),
                html.P("Each tree is trained on a bootstrap sample of the data with weighted features."),
                html.Ul([
                    html.Li([html.Strong("Fewer trees (5-10): "), "Faster computation, but potentially less stable."]),
                    html.Li([html.Strong("More trees (20-50): "), "More stable predictions, but slower."]),
                ], style={"fontSize": "13px"}),
                html.Hr(),
                html.Small([
                    html.I(className="fas fa-info-circle text-info me-1"),
                    "Only applies when using AWNB parameter learning method."
                ], className="text-muted")
            ]),
        ],
        id="help-popover-trees",
        target="help-button-trees",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
    
    dbc.Popover(
        [
            dbc.PopoverHeader("MANB Prior", style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}),
            dbc.PopoverBody([
                html.P("Prior probability used in Model Averaged Naive Bayes (MANB) for model weighting."),
                html.P("MANB averages predictions over multiple models weighted by their posterior probabilities."),
                html.P("The prior influences how much weight is given to simpler vs. more complex models."),
                html.Hr(),
                html.Small([
                    html.I(className="fas fa-exclamation-triangle text-warning me-1"),
                    "MANB only works with Naive Bayes structure."
                ], className="text-muted")
            ]),
        ],
        id="help-popover-prior",
        target="help-button-prior",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
    
    # Evaluation Popovers
    dbc.Popover(
        [
            dbc.PopoverHeader(
                [
                    "Evaluation Metrics",
                    html.I(className="fas fa-chart-line ms-2", style={"color": "#0d6efd"})
                ],
                style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
            ),
            dbc.PopoverBody([
                html.P("Choose a metric to assess your model's performance:"),
                html.Ul([
                    html.Li([html.Strong("Cross-Validation Accuracy: "), "Most reliable. Tests model on multiple data splits. Gives percentage of correct predictions."]),
                    html.Li([html.Strong("AIC (Akaike Information Criterion): "), "Balances fit and complexity. Lower is better."]),
                    html.Li([html.Strong("BIC (Bayesian Information Criterion): "), "Similar to AIC with stronger complexity penalty. Lower is better."]),
                    html.Li([html.Strong("Log-Likelihood: "), "Raw goodness of fit. Higher is better. Doesn't penalize complexity."]),
                ], style={"fontSize": "13px"}),
            ]),
        ],
        id="help-popover-eval-metric",
        target="help-button-eval-metric",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
    
    dbc.Popover(
        [
            dbc.PopoverHeader("CV Folds (Evaluation)", style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}),
            dbc.PopoverBody([
                html.P("Number of folds for cross-validation during model evaluation."),
                html.P("The dataset is split into k equal parts. The model is trained on k-1 parts and tested on the remaining part. This repeats k times."),
                html.Ul([
                    html.Li([html.Strong("5-10 folds: "), "Common choices. Good balance of reliability and speed."]),
                    html.Li([html.Strong("Leave-one-out (n folds): "), "Most thorough but very slow for large datasets."]),
                ], style={"fontSize": "13px"}),
                html.Hr(),
                html.Small([
                    html.I(className="fas fa-info-circle text-info me-1"),
                    "Only applies when using Cross-Validation Accuracy metric."
                ], className="text-muted")
            ]),
        ],
        id="help-popover-cv-folds",
        target="help-button-cv-folds",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
    
    dbc.Popover(
        [
            dbc.PopoverHeader("Structure Setting", style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}),
            dbc.PopoverBody([
                html.P("Controls whether the network structure is fixed during cross-validation:"),
                html.Ul([
                    html.Li([html.Strong("Fixed Structure (Checked): "), "Only parameters (CPTs) are re-learned in each fold. The structure remains the same. Faster and evaluates the specific structure you learned."]),
                    html.Li([html.Strong("Relearn Structure (Unchecked): "), "Both structure and parameters are learned fresh in each fold. Slower but gives a more realistic estimate of the entire learning algorithm's performance."]),
                ], style={"fontSize": "13px"}),
                html.Hr(),
                html.Small([
                    html.I(className="fas fa-lightbulb text-warning me-1"),
                    "Tip: Use fixed structure to evaluate a specific network. Uncheck to evaluate the full learning procedure."
                ], className="text-muted")
            ]),
        ],
        id="help-popover-dag",
        target="help-button-dag",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
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
     Output("nav-eval", "active"),
     Output("nav-predict", "active"),
     Output("nav-inspect", "active")],
    [Input("nav-data", "n_clicks"),
     Input("nav-structure", "n_clicks"),
     Input("nav-eval", "n_clicks"),
     Input("nav-predict", "n_clicks"),
     Input("nav-inspect", "n_clicks")],
)
def navigate(n_data, n_struc, n_eval, n_pred, n_insp):
    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'nav-data'
    
    # Styles
    hide = {"display": "none"}
    show = {"display": "block"}
    
    # Defaults - section-params is now always hidden as it's integrated into section-structure
    styles = [hide] * 6
    actives = [False] * 5  # Only 5 nav items now
    
    if button_id == "nav-structure":
        styles[1] = show; actives[1] = True
    elif button_id == "nav-eval":
        styles[3] = show; actives[2] = True
    elif button_id == "nav-predict":
        styles[4] = show; actives[3] = True
    elif button_id == "nav-inspect":
        styles[5] = show; actives[4] = True
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
    prevent_initial_call=False  # Allow initial load if default is checked by default (it's False currently)
)
def load_and_preview_data(upload_contents, use_default, upload_filename):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'init'
    
    df = None
    msg = ""
    
    # Priority Logic:
    # 1. If 'Use Default' is checked, it overrides.
    # 2. If Upload triggers, load upload.
    
    if use_default:
        try:
            default_file = '/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/bnclassify/asia.csv'
            df = pd.read_csv(default_file)
            msg = "Loaded default 'Asia' (lung cancer) dataset."
        except Exception as e:
            return dbc.Alert(f"Error loading asia data: {e}", color="danger"), [], None, None, ""
            
    elif upload_contents:
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

@app.callback(
    Output('upload-predict', 'contents'),
    [Input('btn-load-test-car', 'n_clicks'),
     Input('btn-load-test-asia', 'n_clicks')],
    prevent_initial_call=True
)
def load_test_data(n_car, n_asia):
    ctx = callback_context
    if not ctx.triggered: return dash.no_update
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    df = None
    if button_id == 'btn-load-test-car':
        try:
             df = pd.read_csv('/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/bnclassify/carwithnames.data')
             # Take last 20 rows
             df = df.tail(20).copy()
        except: return dash.no_update
        
    elif button_id == 'btn-load-test-asia':
        try:
             df = pd.read_csv('/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/bnclassify/asia.csv')
             # Take last 20 rows
             df = df.tail(20).copy()
        except: return dash.no_update
    
    if df is not None:
        csv_string = df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;base64," + base64.b64encode(csv_string.encode('utf-8')).decode('utf-8')
        return csv_string
        
    return dash.no_update

# 3. Dynamic Parameter Visibility (Structure & Params)
@app.callback(
    [Output("field-score", "style"), Output("field-root", "style"), 
     Output("field-folds", "style"), Output("field-epsilon", "style"), Output("field-kdbk", "style")],
    Input("structure-method", "value")
)
def update_structure_fields(method):
    hide = {"display": "none"}
    show = {"display": "block"}
    
    # Defaults
    s_score = hide; s_root = hide; s_folds = hide; s_eps = hide; s_kdbk = hide
    
    # TAN_CL: Only uses score and root (non-wrapper algorithm)
    if method == 'TAN_CL':
        s_score = show; s_root = show
    
    # TAN_HC: Wrapper algorithm using CV folds and epsilon
    elif method == 'TAN_HC':
        s_folds = show; s_eps = show
    
    # KDB: Wrapper algorithm using CV folds, epsilon, and kdbk (max parents)
    elif method == 'KDB':
        s_folds = show; s_eps = show; s_kdbk = show
    
    # FSSJ, BSEJ: Wrapper algorithms using CV folds and epsilon
    elif method in ['FSSJ', 'BSEJ']:
        s_folds = show; s_eps = show
        
    return s_score, s_root, s_folds, s_eps, s_kdbk

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
     Input('btn-train-both', 'n_clicks'),
     Input('btn-evaluate', 'n_clicks'),
     Input('btn-predict', 'n_clicks')],
    [State('dataset-store', 'data'),
     State('class-dropdown', 'value'),
     # Structure params
     State('structure-method', 'value'), State('score-dropdown', 'value'), State('root-dropdown', 'value'),
     State('epsilon-input', 'value'), State('kdbk-input', 'value'), State('struct-folds-input', 'value'),
     # Param params
     State('param-method', 'value'), State('alpha-input', 'value'), State('trees-input', 'value'), State('prior-input', 'value'),
     # Eval params
     State('eval-metric', 'value'), State('eval-folds-input', 'value'), State('dag-check', 'value'),
     # Predict params
     State('upload-predict', 'contents'), State('prob-check', 'value')]
)
def execute_action(n_struc, n_param, n_both, n_eval, n_pred, 
                  df_json, class_var,
                  struct_method, score, root, eps, kdbk, s_folds,
                  param_method, alpha, trees, prior,
                  eval_metric, eval_folds, dag_check,
                  pred_contents, prob_check):
    
    ctx = callback_context
    if not ctx.triggered: return ""
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not df_json: return dbc.Alert("No dataset loaded!", color="warning")
    
    global current_model, current_structure_algorithm, current_params_learned
    df = pd.read_json(df_json, orient='split')
    
    # --- TRAIN BOTH (STRUCTURE + PARAMETERS) ---
    if trigger == 'btn-train-both':
        # PRE-VALIDATION: Check MANB compatibility before starting
        if param_method == 'MANB' and struct_method != 'NB':
            return dbc.Alert([
                html.I(className="fas fa-ban me-2"),
                html.Strong("Cannot Train - Incompatible Configuration: "),
                f"MANB parameters require Naive Bayes structure, but you selected '{struct_method}'. ",
                html.Br(),
                "Please either:",
                html.Ul([
                    html.Li("Change structure algorithm to 'Naive Bayes (NB)', or"),
                    html.Li("Change parameter method to MLE, Bayes, WANBIA, or AWNB")
                ])
            ], color="danger")
        
        try:
            # First: Train Structure
            if   struct_method == 'NB':      
                model = bnclassify.nb(class_var, df)
            elif struct_method == 'TAN_CL':  
                model = bnclassify.tan_cl(class_var, df, score=score.lower(), root=root)
            elif struct_method == 'TAN_HC':  
                model = bnclassify.tan_hc(class_var, df, k=s_folds, epsilon=eps, smooth=0)
            elif struct_method == 'KDB':     
                model = bnclassify.kdb(class_var, df, k=s_folds, kdbk=kdbk, epsilon=eps, smooth=0)
            elif struct_method == 'AODE':    
                model = bnclassify.aode(class_var, df)
            elif struct_method == 'FSSJ':    
                model = bnclassify.fssj(class_var, df, k=s_folds, epsilon=eps, smooth=0)
            elif struct_method == 'BSEJ':    
                model = bnclassify.bsej(class_var, df, k=s_folds, epsilon=eps, smooth=0)
            else: 
                return dbc.Alert("Unknown Algorithm", color="danger")
            
            current_model = model
            current_structure_algorithm = struct_method
            
            # Second: Learn Parameters
            if   param_method == 'MLE':    
                current_model = bnclassify.lp(current_model, df, smooth=0)
            elif param_method == 'Bayes':  
                current_model = bnclassify.lp(current_model, df, smooth=alpha)
            elif param_method == 'WANBIA': 
                current_model = bnclassify.lp(current_model, df, wanbia=True)
            elif param_method == 'AWNB':   
                current_model = bnclassify.lp(current_model, df, awnb_trees=trees)
            elif param_method == 'MANB':   
                current_model = bnclassify.lp(current_model, df, manb_prior=prior)
            
            current_params_learned = True
            
            # Algorithm and parameter descriptions
            algo_info = {
                'NB': 'Naive Bayes - simplest structure with class as only parent',
                'TAN_CL': 'Tree-Augmented Network (Chow-Liu) - tree structure between features',
                'TAN_HC': f'Tree-Augmented Network (Hill-Climbing) - optimized using {s_folds}-fold CV',
                'AODE': 'Averaged One-Dependence Estimator - ensemble of structures',
                'KDB': f'k-Dependence Bayesian Classifier - features can have up to {kdbk} parents (using {s_folds}-fold CV)',
                'FSSJ': f'Forward Sequential Selection Join - greedy forward search ({s_folds}-fold CV)',
                'BSEJ': f'Backward Sequential Elimination Join - greedy backward search ({s_folds}-fold CV)'
            }
            
            param_info = {
                'MLE': 'Maximum Likelihood Estimation - raw frequency counts',
                'Bayes': f'Bayesian estimation with Dirichlet prior (alpha={alpha})',
                'WANBIA': 'Weighted Averaged Naive Bayes with Instances - feature weighting',
                'AWNB': f'Attribute Weighted Naive Bayes - bootstrap ensemble ({trees} trees)',
                'MANB': f'Model Averaged Naive Bayes - prior={prior}'
            }
            
            struct_desc = algo_info.get(struct_method, struct_method)
            param_desc = param_info.get(param_method, param_method)
            
            # Add compatibility note if needed
            extra_note = None
            if param_method in ['WANBIA', 'AWNB'] and current_structure_algorithm != 'NB':
                extra_note = html.Small([
                    html.I(className="fas fa-info-circle text-info me-1"),
                    f"Note: {param_method} is designed for Naive Bayes but has been applied to {struct_method}. Results may vary."
                ], className="text-muted d-block mt-2")
            
            return html.Div([
                dbc.Alert([
                    html.H4([html.I(className="fas fa-check-double me-2"), "Complete Model Trained Successfully!"], className="alert-heading mb-3"),
                    
                    html.Div([
                        html.H6([html.I(className="fas fa-project-diagram me-2"), "Structure"], className="mb-2"),
                        html.P([html.Strong(struct_method), " - ", struct_desc], className="mb-3 ms-3"),
                        
                        html.H6([html.I(className="fas fa-sliders-h me-2"), "Parameters"], className="mb-2"),
                        html.P([html.Strong(param_method), " - ", param_desc], className="mb-2 ms-3"),
                        extra_note if extra_note else "",
                    ]),
                    
                    html.Hr(),
                    html.P([
                        html.Strong("Next steps: "),
                        "Model is fully trained and ready! Go to 'Evaluation' to assess performance, 'Prediction' to classify new data, or 'Inspect Model' to examine the learned network."
                    ], className="mb-0 small")
                ], color="success")
            ])
            
        except Exception as e:
            return dbc.Alert([
                html.H5([html.I(className="fas fa-times-circle me-2"), "Training Failed"], className="alert-heading"),
                html.P(f"Error: {str(e)}", className="mb-0 font-monospace small")
            ], color="danger")
    
    # --- TRAIN STRUCTURE ---
    if trigger == 'btn-train-structure':
        try:
            if   struct_method == 'NB':      
                model = bnclassify.nb(class_var, df)
            elif struct_method == 'TAN_CL':  
                model = bnclassify.tan_cl(class_var, df, score=score.lower(), root=root)
            elif struct_method == 'TAN_HC':  
                model = bnclassify.tan_hc(class_var, df, k=s_folds, epsilon=eps, smooth=0)
            elif struct_method == 'KDB':     
                model = bnclassify.kdb(class_var, df, k=s_folds, kdbk=kdbk, epsilon=eps, smooth=0)
            elif struct_method == 'AODE':    
                model = bnclassify.aode(class_var, df)
            elif struct_method == 'FSSJ':    
                model = bnclassify.fssj(class_var, df, k=s_folds, epsilon=eps, smooth=0)
            elif struct_method == 'BSEJ':    
                model = bnclassify.bsej(class_var, df, k=s_folds, epsilon=eps, smooth=0)
            else: 
                return dbc.Alert("Unknown Algorithm", color="danger")
            
            current_model = model
            current_structure_algorithm = struct_method
            current_params_learned = False  # Reset parameters flag when new structure is learned
            
            # Algorithm descriptions for user feedback
            algo_info = {
                'NB': 'Naive Bayes - simplest structure with class as only parent',
                'TAN_CL': 'Tree-Augmented Network (Chow-Liu) - tree structure between features',
                'TAN_HC': f'Tree-Augmented Network (Hill-Climbing) - optimized using {s_folds}-fold CV',
                'AODE': 'Averaged One-Dependence Estimator - ensemble of structures',
                'KDB': f'k-Dependence Bayesian Classifier - features can have up to {kdbk} parents (using {s_folds}-fold CV)',
                'FSSJ': f'Forward Sequential Selection Join - greedy forward search ({s_folds}-fold CV)',
                'BSEJ': f'Backward Sequential Elimination Join - greedy backward search ({s_folds}-fold CV)'
            }
            
            description = algo_info.get(struct_method, struct_method)
            
            return html.Div([
                dbc.Alert([
                    html.H5([html.I(className="fas fa-check-circle me-2"), f"Structure Learned: {struct_method}"], className="alert-heading mb-2"),
                    html.P(description, className="mb-2"),
                    html.Hr(),
                    html.P([
                        html.Strong("Next steps: "),
                        "Go to 'Parameter Learning' to estimate the conditional probability tables (CPTs), or go to 'Inspect Model' to see the structure."
                    ], className="mb-0 small")
                ], color="success")
            ])
        except Exception as e:
             return dbc.Alert([
                 html.H5([html.I(className="fas fa-times-circle me-2"), f"Structure Learning Failed: {struct_method}"], className="alert-heading"),
                 html.P(f"Error: {str(e)}", className="mb-0 font-monospace small")
             ], color="danger")

    # --- LEARN PARAMETERS ---
    if trigger == 'btn-learn-params':
        if current_model is None: 
            return dbc.Alert("⚠ No structure! Learn structure first.", color="warning")
        
        # Validate MANB restriction: only works with Naive Bayes structures
        if param_method == 'MANB' and current_structure_algorithm != 'NB':
            return dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                html.Strong("MANB Restriction: "),
                f"MANB can only be applied to Naive Bayes structures. Current structure: {current_structure_algorithm}. ",
                "Please use a different parameter learning method or re-train with Naive Bayes structure."
            ], color="danger")
        
        try:
            if   param_method == 'MLE':    
                current_model = bnclassify.lp(current_model, df, smooth=0)
            elif param_method == 'Bayes':  
                current_model = bnclassify.lp(current_model, df, smooth=alpha)
            elif param_method == 'WANBIA': 
                current_model = bnclassify.lp(current_model, df, wanbia=True)
            elif param_method == 'AWNB':   
                current_model = bnclassify.lp(current_model, df, awnb_trees=trees)
            elif param_method == 'MANB':   
                current_model = bnclassify.lp(current_model, df, manb_prior=prior)
            
            # Mark parameters as learned
            current_params_learned = True
            
            # Parameter method descriptions
            param_info = {
                'MLE': 'Maximum Likelihood Estimation - raw frequency counts',
                'Bayes': f'Bayesian estimation with Dirichlet prior (alpha={alpha})',
                'WANBIA': 'Weighted Averaged Naive Bayes with Instances - feature weighting',
                'AWNB': f'Attribute Weighted Naive Bayes - bootstrap ensemble ({trees} trees)',
                'MANB': f'Model Averaged Naive Bayes - prior={prior}'
            }
            
            description = param_info.get(param_method, param_method)
            
            # Add note for WANBIA/AWNB when used with non-NB structures
            extra_note = None
            if param_method in ['WANBIA', 'AWNB'] and current_structure_algorithm != 'NB':
                extra_note = html.Div([
                    html.Hr(),
                    html.Small([
                        html.I(className="fas fa-info-circle text-info me-1"),
                        f"Note: {param_method} is designed for Naive Bayes but can be applied to other structures. Results may vary."
                    ], className="text-muted")
                ])
            
            return html.Div([
                dbc.Alert([
                    html.H5([html.I(className="fas fa-check-circle me-2"), f"Parameters Learned: {param_method}"], className="alert-heading mb-2"),
                    html.P([
                        html.Strong("Structure: "), f"{current_structure_algorithm}", html.Br(),
                        html.Strong("Method: "), description
                    ], className="mb-2"),
                    extra_note if extra_note else "",
                    html.Hr(),
                    html.P([
                        html.Strong("Next steps: "),
                        "Model is ready! Go to 'Evaluation' to assess performance, 'Prediction' to classify new data, or 'Inspect Model' to examine the learned network."
                    ], className="mb-0 small")
                ], color="success")
            ])
        except Exception as e:
            return dbc.Alert([
                html.H5([html.I(className="fas fa-times-circle me-2"), f"Parameter Learning Failed: {param_method}"], className="alert-heading"),
                html.P(f"Error: {str(e)}", className="mb-0 font-monospace small")
            ], color="danger")
            
    # --- EVALUATE ---
    if trigger == 'btn-evaluate':
        if current_model is None: return dbc.Alert("Model not ready for evaluation.", color="warning")
        try:
            if eval_metric == 'CV':
                fixed = ('fixed' in dag_check)
                val = bnclassify.cv(current_model, df, k=eval_folds, dag=fixed)
                return dbc.Alert(f"Cross-Validation Accuracy ({eval_folds} folds): {val:.4f}", color="info")
            
            elif eval_metric == 'AIC':
                # Wrapper already handles conversion to float
                aic_val = bnclassify.aic(current_model, df)
                return dbc.Alert(f"AIC Score: {aic_val:.4f}", color="info")
            
            elif eval_metric == 'BIC':
                # Wrapper already handles conversion to float
                bic_val = bnclassify.bic(current_model, df)
                return dbc.Alert(f"BIC Score: {bic_val:.4f}", color="info")
            
            elif eval_metric == 'LL':
                # Wrapper already handles conversion to float
                ll_val = bnclassify.log_lik(current_model, df)
                return dbc.Alert(f"Log-Likelihood: {ll_val:.4f}", color="info")
            
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
            
            # The wrapper now returns a DataFrame (if probs=True) or numpy array (if probs=False)
            preds = bnclassify.predict(current_model, X, prob=probs)
            
            # Result display
            if probs:
                # preds is a DataFrame of probabilities
                res_df = preds.round(4)
                # Add ID column if useful, or just index
                res_df.reset_index(inplace=True)
                msg = "Posterior probabilities calculated."
            else:
                # preds is a numpy array of classes
                res_df = pd.DataFrame({'Predicted_Class': preds})
                res_df.reset_index(inplace=True)
                msg = "Class labels predicted."
            
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
        try:
            # Gather summary stats
            n_p = bnclassify.nparams(current_model)
            n_a = bnclassify.narcs(current_model)
            c_v = bnclassify.class_var(current_model)
            feats = bnclassify.features(current_model)
            ms = bnclassify.modelstring(current_model)
            
            summary = html.Div([
                html.H5("Model Summary"),
                html.Ul([
                    html.Li([html.Strong("Class Variable: "), c_v]),
                    html.Li([html.Strong("Features (Nodes): "), ", ".join(feats)]),
                    html.Li([html.Strong("Number of Parameters: "), str(n_p)]),
                    html.Li([html.Strong("Number of Arcs: "), str(n_a)]),
                    html.Li([html.Strong("Model String: "), html.Code(ms)]),
                ])
            ])
            return summary
        except Exception as e:
            return dbc.Alert(f"Error generating summary: {e}", color="danger")
    
    if active_tab == "tab-cpts":
        try:
            # Fetch CPTs dict
            cpts_dict = bnclassify.params(current_model)
            if 'error' in cpts_dict:
                return dbc.Alert(f"Error fetching parameters: {cpts_dict['error']}", color="danger")
            
            tabs = []
            for var_name, df_cpt in cpts_dict.items():
                # Each CPT is a dataframe. Display properly.
                # Usually it has columns for parents and probabilities.
                cpt_table = dash_table.DataTable(
                    columns=[{"name": str(c), "id": str(c)} for c in df_cpt.columns],
                    data=df_cpt.to_dict('records'),
                    style_table={'overflowX': 'auto', 'maxWidth': '100%'},
                    page_size=10,
                    style_header={'fontWeight': 'bold'}
                )
                
                tabs.append(dbc.AccordionItem(
                    cpt_table,
                    title=f"CPT: {var_name}"
                ))
            
            return dbc.Accordion(tabs, start_collapsed=True, always_open=False)
            
        except Exception as e:
             return dbc.Alert(f"CPT Inspection Error: {e}", color="danger")

    if active_tab == "tab-graph":
        # Graph visualization
        try:
            import dash_cytoscape as cy
            
            edges = bnclassify.edges(current_model)
            class_v = bnclassify.class_var(current_model)
            
            if not edges: 
                # Even if no edges, show nodes (like Naive Bayes with independent features?? NB has edges from Class->Features)
                # If truly empty, it's just main nodes?
                # Actually bnclassify edges() returns all edges. For NB it should return edges.
                pass
            
            elements = []
            nodes = set()
            
            # Add edges
            if edges:
                for u, v in edges:
                    nodes.add(u); nodes.add(v)
                    elements.append({'data': {'source': u, 'target': v}})
            
            # If no edges, we might still have features? 
            # But edges() usually captures the structure.
            
            # Add nodes with class distinction
            for n in nodes:
                is_class = (n == class_v)
                elements.append({
                    'data': {'id': n, 'label': n},
                    'classes': 'class_node' if is_class else ''
                })
            
            return cy.Cytoscape(
                id='structure-graph',
                elements=elements,
                stylesheet=cytoscape_stylesheet,
                style={'width': '100%', 'height': '600px', 'border': '1px solid #ddd', 'backgroundColor': '#ffffff'},
                layout={
                    'name': 'cose',
                    'animate': True,
                    'animationDuration': 500,
                    'nodeRepulsion': 8000,
                    'idealEdgeLength': 100,
                    'edgeElasticity': 100,
                    'nestingFactor': 5,
                    'gravity': 80,
                    'numIter': 1000,
                    'randomize': False
                },
                responsive=True
            )
            return cy.Cytoscape(
                id='cytoscape',
                elements=elements,
                layout={'name': 'breadthfirst', 'roots': [bnclassify.class_var(current_model)]},
                style={'width': '100%', 'height': '600px', 'border': '1px solid #ddd'},
                stylesheet=[
                    {'selector': 'node', 'style': {'label': 'data(label)', 'background-color': '#0d6efd', 'color': 'white', 'text-valign': 'center'}},
                    {'selector': 'edge', 'style': {'line-color': '#ccc', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#ccc', 'curve-style': 'bezier'}}
                ]
            )
        except ImportError:
             return dbc.Alert("dash_cytoscape not installed", color="danger")
        except Exception as e:
             return dbc.Alert(f"Plotting error: {e}", color="danger")
    
    return html.Div("Select a tab")


# 6. MODEL STATE INFO CALLBACK
@app.callback(
    Output('current-params-info', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_params_info(n):
    global current_model, current_structure_algorithm, current_params_learned
    if current_structure_algorithm is None:
        return html.Div([
            html.I(className="fas fa-info-circle me-2"),
            "No structure learned yet. Please train a structure first."
        ])
    
    if current_model is None:
        return html.Div([
            html.I(className="fas fa-info-circle me-2"),
            "No structure learned yet. Please train a structure first."
        ])
    
    # Use the tracking flag instead of heuristic
    status_icon = "fas fa-check-circle text-success me-2" if current_params_learned else "fas fa-hourglass-half text-warning me-2"
    params_status = "Parameters learned" if current_params_learned else "Parameters not yet learned"
    
    # Add compatibility notes
    compatibility_notes = []
    if current_structure_algorithm != 'NB':
        compatibility_notes.append(
            html.Small([
                html.I(className="fas fa-info-circle text-info me-1"),
                "MANB requires Naive Bayes structure. WANBIA & AWNB are designed for NB but can be used with other structures."
            ], className="text-muted d-block mt-2")
        )
    
    return html.Div([
        html.I(className=status_icon),
        html.Strong(f"Current Structure: {current_structure_algorithm}"),
        html.Span(f" | {params_status}", className="ms-2"),
        *compatibility_notes
    ])

# 6b. TRAINING BUTTONS STATE & RESTRICTIONS CALLBACK
@app.callback(
    [Output('btn-learn-params', 'disabled'),
     Output('btn-train-both', 'disabled'),
     Output('training-restrictions-alert', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('param-method', 'value'),
     Input('structure-method', 'value')]
)
def update_training_restrictions(n, param_method, struct_method):
    global current_model, current_structure_algorithm, current_params_learned
    
    # Check if structure exists
    has_structure = (current_model is not None and current_structure_algorithm is not None)
    
    # Default: Parameters Only disabled if no structure
    params_only_disabled = not has_structure
    train_both_disabled = False
    alert = None
    
    # If structure exists, check MANB compatibility for "Parameters Only"
    if has_structure and param_method == 'MANB' and current_structure_algorithm != 'NB':
        params_only_disabled = True
        alert = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            html.Strong("Restriction: "),
            f"MANB parameters cannot be applied to current structure ({current_structure_algorithm}). ",
            "Select a different parameter method or retrain with Naive Bayes structure."
        ], color="warning", className="mt-2")
    
    # Check MANB compatibility for "Train Both"
    elif param_method == 'MANB' and struct_method != 'NB':
        train_both_disabled = True
        alert = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            html.Strong("Incompatible Selection: "),
            f"MANB parameters require Naive Bayes structure. Currently selected: {struct_method}. ",
            "Either change structure to 'Naive Bayes (NB)' or select a different parameter method."
        ], color="danger", className="mt-2")
    
    # If no structure, show info message
    elif not has_structure:
        alert = dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            "No structure trained yet. Use 'Train Both' or 'Structure Only' to begin."
        ], color="info", className="mt-2")
    
    return params_only_disabled, train_both_disabled, alert

# 7. RESET CALLBACK
@app.callback(
    [Output('dataset-store', 'data', allow_duplicate=True),
     Output('output-area', 'children', allow_duplicate=True)],
    Input('reset-button', 'n_clicks'),
    prevent_initial_call=True
)
def reset_application(n_clicks):
    if n_clicks:
        global current_model, current_structure_algorithm, current_params_learned
        current_model = None
        current_structure_algorithm = None
        current_params_learned = False
        return None, dbc.Alert("Application reset successfully!", color="success")
    return dash.no_update, dash.no_update

# Popover callbacks
@app.callback(
    Output("help-popover-class", "is_open"),
    Input("help-button-class", "n_clicks"),
    State("help-popover-class", "is_open")
)
def toggle_class_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-algorithms", "is_open"),
    Input("help-button-algorithms", "n_clicks"),
    State("help-popover-algorithms", "is_open")
)
def toggle_algorithms_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-methods", "is_open"),
    Input("help-button-methods", "n_clicks"),
    State("help-popover-methods", "is_open")
)
def toggle_methods_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-score", "is_open"),
    Input("help-button-score", "n_clicks"),
    State("help-popover-score", "is_open")
)
def toggle_score_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-root", "is_open"),
    Input("help-button-root", "n_clicks"),
    State("help-popover-root", "is_open")
)
def toggle_root_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-folds", "is_open"),
    Input("help-button-folds", "n_clicks"),
    State("help-popover-folds", "is_open")
)
def toggle_folds_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-epsilon", "is_open"),
    Input("help-button-epsilon", "n_clicks"),
    State("help-popover-epsilon", "is_open")
)
def toggle_epsilon_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-k", "is_open"),
    Input("help-button-k", "n_clicks"),
    State("help-popover-k", "is_open")
)
def toggle_k_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-alpha", "is_open"),
    Input("help-button-alpha", "n_clicks"),
    State("help-popover-alpha", "is_open")
)
def toggle_alpha_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-trees", "is_open"),
    Input("help-button-trees", "n_clicks"),
    State("help-popover-trees", "is_open")
)
def toggle_trees_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-prior", "is_open"),
    Input("help-button-prior", "n_clicks"),
    State("help-popover-prior", "is_open")
)
def toggle_prior_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-eval-metric", "is_open"),
    Input("help-button-eval-metric", "n_clicks"),
    State("help-popover-eval-metric", "is_open")
)
def toggle_eval_metric_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-cv-folds", "is_open"),
    Input("help-button-cv-folds", "n_clicks"),
    State("help-popover-cv-folds", "is_open")
)
def toggle_cv_folds_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-dag", "is_open"),
    Input("help-button-dag", "n_clicks"),
    State("help-popover-dag", "is_open")
)
def toggle_dag_popover(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8056)