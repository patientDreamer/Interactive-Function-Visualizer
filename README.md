# Interactive Function Illustrator

A mini Desmos-like mathematical visualization app built with **Python**, **Streamlit**, **Plotly**, **SymPy**, **NumPy**, and **Pandas**.

This project allows users to graph functions, visualize multivariable surfaces, analyze calculus characteristics, create/import data tables, perform polynomial regressions, graph inequalities and restrictions, and explore complex-valued functions through an interactive browser interface.

---

## Live Demo

```txt
https://fcnvis.streamlit.app
```
---

## Project Overview

The goal of this project is to create an interactive mathematical function illustrator that behaves similarly to a lightweight version of Desmos.

The app supports:

- Single-variable function graphing
- Multivariable 3D surface graphing
- Piecewise functions
- Domain and range restrictions
- Inequality graphing
- Calculus visualization
- Tables and data plotting
- Lines of best fit
- Polynomial regression
- Complex-valued function visualization
- Multiple 3D graph color themes

---

## Features

### 1. Function Graphing

The app supports standard mathematical function input.

Examples:

```txt
f(x) = sin(x)
f(x) = sinx
f(x) = x^2 + 3x + 1
f(x) = sqrt(x)
f(x) = abs(x)
f(x) = |x|
```

It also supports two-variable and multivariable functions:

```txt
f(x,y) = sin(x)cos(y)
f(x,y) = exp(-x^2 - y^2)
f(x,y,a) = a exp(-x^2-y^2) sin(xy)
```

For multivariable functions, the app plots two selected variables and turns extra variables into adjustable sliders.

---

### 2. Supported Mathematical Functions

The parser supports many common mathematical functions.

#### Trigonometric Functions

```txt
sin(x)
cos(x)
tan(x)
sec(x)
csc(x)
cot(x)
```

#### Inverse Trigonometric Functions

```txt
asin(x)
acos(x)
atan(x)
arcsin(x)
arccos(x)
arctan(x)
```

#### Hyperbolic Functions

```txt
sinh(x)
cosh(x)
tanh(x)
```

#### Inverse Hyperbolic Functions

```txt
asinh(x)
acosh(x)
atanh(x)
arcsinh(x)
arccosh(x)
arctanh(x)
```

#### Logarithmic and Exponential Functions

```txt
ln(x)
log(x)
log10(x)
exp(x)
```

#### Other Functions

```txt
sqrt(x)
abs(x)
floor(x)
ceil(x)
gamma(x)
```

#### Constants

```txt
pi
e
E
i
I
```

---

### 3. Desmos-Like Input Shortcuts

The app supports several user-friendly input shortcuts.

```txt
sinx
cosx
tanx
sqrtx
lnx
```

These are interpreted as:

```txt
sin(x)
cos(x)
tan(x)
sqrt(x)
ln(x)
```

The app also supports trigonometric powers:

```txt
f(x) = sin^2(x) + cos^2(x)
```

Factorial-style input is handled using the Gamma function:

```txt
f(x) = x!
f(x) = x! / x^x
f(x) = gamma(x + 1) / x^x
```

---

### 4. Piecewise Functions

The app supports both Desmos-like piecewise syntax and SymPy-style piecewise syntax.

#### Desmos-Like Syntax

```txt
f(x) = {x<0:-x, otherwise:x}
```

```txt
f(x) = {x<-1:-1, x<=1:x, otherwise:1}
```

#### SymPy-Style Syntax

```txt
f(x) = Piecewise((x^2,x<0),(sin(x),x>=0))
```

---

### 5. Restrictions

The app supports basic domain and range restrictions.

Examples:

```txt
f(x) = x^2 {1<x<5}
```

```txt
f(x) = sin(x) {x>0}
```

```txt
f(x) = x^2 {-2<x<2, y<3}
```

Restrictions allow users to graph only part of a function.

---

### 6. Inequality Graphing

The app supports basic one-variable inequality graphing.

Examples:

```txt
y < x^2
y <= x^2 {1<x<5}
y > sin(x)
y >= cos(x)
```

The app shades the corresponding region above or below the boundary curve.

---

### 7. Calculus Visualization

For one-variable real-valued functions, the app can visualize several calculus characteristics.

Supported calculus features include:

- Approximate local maxima
- Approximate local minima
- Tangent line at a selected point
- Approximate continuity check
- Approximate differentiability check
- Symbolic derivative
- Symbolic indefinite integral

Example:

```txt
f(x) = x^3 - 3x
```

The app can detect approximate local extrema and display a tangent line at a chosen point.

Another useful example:

```txt
f(x) = |x|
```

This can be used to explore differentiability at a sharp corner.

#### Important Note

The continuity and differentiability checks are numerical approximations. They are useful for visualization, but they should not be treated as formal mathematical proofs.

---

### 8. Tables and Regression

The app includes a table editor where users can manually create datasets or import CSV files.

Supported table features:

- Editable data table
- Dynamic row creation
- CSV import
- Scatter plot visualization
- Linear regression
- Quadratic regression
- Polynomial regression
- Regression equation display
- Approximate R² value display

Example dataset:

| x | y |
|---|---|
| 0 | 1 |
| 1 | 2.1 |
| 2 | 3.9 |
| 3 | 6.2 |
| 4 | 8.1 |
| 5 | 10.2 |

Regression types:

```txt
Linear
Quadratic
Polynomial
```

---

### 9. Complex-Valued Functions

The app includes a complex function viewer for one-variable complex-valued functions.

Examples:

```txt
f(x) = exp(i*x)
```

```txt
f(x) = sin(x) + i*cos(x)
```

The complex viewer can display:

- Real part
- Imaginary part
- Magnitude
- Phase / argument

---

### 10. 3D Surface Themes

The app includes several color themes for 3D surface plots.

Available themes:

```txt
Pink Purple Blue
Cyber Neon
Ocean
Sunset Glow
Aurora
Galaxy
Cotton Candy
Emerald Night
Fire Ice
Lavender Dream
```

These themes make the 3D function visualizations more visually appealing and easier to distinguish.

---

## Example Inputs

### Basic Functions

```txt
f(x) = x^2
```

```txt
f(x) = sin(x)
```

```txt
f(x) = sinx
```

```txt
f(x) = exp(-x^2)
```

### Advanced Functions

```txt
f(x) = sin^2(x) + cos^2(x)
```

```txt
f(x) = floor(sin(x))
```

```txt
f(x) = ceil(cos(x))
```

```txt
f(x) = x! / x^x
```

### Piecewise Functions

```txt
f(x) = {x<0:-x, otherwise:x}
```

```txt
f(x) = {x<-1:-1, x<=1:x, otherwise:1}
```

```txt
f(x) = Piecewise((x^2,x<0),(sin(x),x>=0))
```

### Restrictions

```txt
f(x) = x^2 {1<x<5}
```

```txt
f(x) = sin(x) {x>0}
```

### Inequalities

```txt
y < x^2
```

```txt
y <= x^2 {1<x<5}
```

```txt
y > sin(x)
```

### Multivariable Functions

```txt
f(x,y) = sin(x)cos(y)
```

```txt
f(x,y) = exp(-x^2-y^2)
```

```txt
f(x,y,a) = a exp(-x^2-y^2) sin(xy)
```

### Complex Functions

```txt
f(x) = exp(i*x)
```

```txt
f(x) = sin(x) + i*cos(x)
```

---

## Technology Stack

This project uses:

- **Python**: Main programming language
- **Streamlit**: Web app framework
- **Plotly**: Interactive graphing and visualization
- **SymPy**: Symbolic mathematics and expression parsing
- **NumPy**: Numerical computation
- **Pandas**: Table handling and CSV import

---

## Project Structure

Recommended repository structure:

```txt
interactive-function-illustrator/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

`patientDreamer` and `Interactive-Function-Visualizer` with your actual GitHub username and repository name.

---

### 2. Create a Virtual Environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you may need to run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements File

Your `requirements.txt` should contain:

```txt
streamlit
plotly
sympy
numpy
pandas
```

---

## Run Locally

If your app file is named `main.py`, run:

```bash
streamlit run main.py
```

Or use:

```bash
python -m streamlit run main.py
```

The second command is especially helpful if your terminal does not recognize the `streamlit` command directly.

After running the command, Streamlit should open the app in your browser.

The local URL usually looks like:

```txt
http://localhost:8501
```

---

## Deploy to Streamlit Community Cloud

You can deploy this project for free using Streamlit Community Cloud.

### Steps

1. Push your project to GitHub.
2. Go to Streamlit Community Cloud.
3. Sign in with your GitHub account.
4. Click **New app**.
5. Select your repository.
6. Select the correct branch, usually `main`.
7. Set the main file path to:

```txt
main.py
```

8. Click **Deploy**.

After deployment, Streamlit will create a public URL for your app.

---

## GitHub Upload Guide

If you are uploading this project to GitHub for the first time, use these commands:

```bash
git init
git add .
git commit -m "Initial commit: interactive function illustrator"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```


---
```gitignore
.venv/
__pycache__/
*.pyc
.DS_Store
.idea/
.vscode/
.streamlit/secrets.toml
```

---

## Known Limitations

This app is a practical mini graphing calculator, but it does not fully match every Desmos feature.

Current limitations include:

- Full implicit graphing is not implemented yet.
- Equations like `x^2 + y^2 = 1` are not fully supported.
- Parametric curves are not fully supported yet.
- Polar graphing is not implemented yet.
- True Desmos-style list syntax is not fully implemented.
- Deeply nested absolute value bars may not always parse correctly.
- Some complex-valued expressions should be viewed in the Complex Functions tab.
- Continuity and differentiability checks are numerical estimates, not formal proofs.
- Some discontinuities or highly oscillatory functions may be difficult to analyze numerically.
- Some inequality and restriction formats may require simple syntax.

---

## Future Improvements

Possible future upgrades include:

- Add implicit graphing
- Add parametric curves
- Add polar graphing
- Add true list syntax
- Add sliders for parameters
- Add multiple function overlays
- Add better symbolic domain detection
- Add downloadable graphs
- Add exportable tables
- Add exponential and logarithmic regression
- Add trigonometric regression
- Add implicit differentiation tools
- Add root, intercept, and asymptote detection
- Add user-saved graph presets
- Add shareable graph links

---

## Educational Purpose

This project is designed as an educational mathematical visualization tool.

It can be used to explore:

- Algebra
- Precalculus
- Calculus
- Multivariable calculus
- Data fitting
- Regression
- Complex numbers
- Function behavior
- Numerical approximation

---

## Disclaimer

This app uses symbolic and numerical methods to parse, evaluate, and visualize mathematical expressions. Some results are approximate, especially for calculus characteristics, regressions, discontinuities, and complex-valued functions.

For formal mathematical proofs or high-stakes numerical work, verify results using rigorous methods.

---

## Author

Created by Jack Jiang.

---

## Acknowledgments

Built with:

- Python
- Streamlit
- Plotly
- SymPy
- NumPy
- Pandas
