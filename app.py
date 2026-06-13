from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained linear regression model
try:
    with open('linear_pkl.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    print("Warning: 'linear_pkl.pkl' not found. Please ensure it is in the same directory.")

# Categorical mappings for the model
MAPPINGS = {
    'sex': {'male': 0, 'female': 1},
    'smoker': {'no': 0, 'yes': 1},
    'region': {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
}

# Embedded HTML/CSS UI using Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <style>
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center p-6 text-slate-100 font-sans">

    <div class="w-full max-w-2xl bg-slate-900/60 backdrop-blur-xl border border-slate-800 rounded-2xl shadow-2xl p-8 space-y-6">
        
        <div class="text-center">
            <h1 class="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                ML Model Predictor
            </h1>
            <p class="text-sm text-slate-400 mt-2">Enter the metrics below to generate an instant algorithmic prediction.</p>
        </div>

        <hr class="border-slate-800">

        <form action="/predict" method="POST" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Age</label>
                    <input type="number" name="age" required min="0" max="120" 
                           value="{{ form_data.age if form_data else '' }}"
                           class="w-full bg-slate-950/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors" placeholder="e.g. 28">
                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">BMI</label>
                    <input type="number" name="bmi" step="0.1" required min="10" max="60" 
                           value="{{ form_data.bmi if form_data else '' }}"
                           class="w-full bg-slate-950/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors" placeholder="e.g. 24.5">
                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Number of Children</label>
                    <input type="number" name="children" required min="0" max="10" 
                           value="{{ form_data.children if form_data else '' }}"
                           class="w-full bg-slate-950/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors" placeholder="e.g. 2">
                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Sex</label>
                    <select name="sex" required class="w-full bg-slate-950/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-cyan-500 transition-colors">
                        <option value="male" {% if form_data and form_data.sex == 'male' %}selected{% endif %}>Male</option>
                        <option value="female" {% if form_data and form_data.sex == 'female' %}selected{% endif %}>Female</option>
                    </select>
                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Smoker Status</label>
                    <select name="smoker" required class="w-full bg-slate-950/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-cyan-500 transition-colors">
                        <option value="no" {% if form_data and form_data.smoker == 'no' %}selected{% endif %}>Non-Smoker</option>
                        <option value="yes" {% if form_data and form_data.smoker == 'yes' %}selected{% endif %}>Smoker</option>
                    </select>
                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Geographic Region</label>
                    <select name="region" required class="w-full bg-slate-950/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-cyan-500 transition-colors">
                        <option value="northeast" {% if form_data and form_data.region == 'northeast' %}selected{% endif %}>Northeast</option>
                        <option value="northwest" {% if form_data and form_data.region == 'northwest' %}selected{% endif %}>Northwest</option>
                        <option value="southeast" {% if form_data and form_data.region == 'southeast' %}selected{% endif %}>Southeast</option>
                        <option value="southwest" {% if form_data and form_data.region == 'southwest' %}selected{% endif %}>Southwest</option>
                    </select>
                </div>

            </div>

            <button type="submit" class="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold py-3 px-4 rounded-xl shadow-lg shadow-cyan-500/20 transform active:scale-[0.99] transition-all cursor-pointer">
                Generate Prediction
            </button>
        </form>

        {% if prediction_text %}
        <div class="mt-6 p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-center">
            <p class="text-emerald-400 font-medium text-lg">{{ prediction_text }}</p>
        </div>
        {% endif %}

        {% if error_text %}
        <div class="mt-6 p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-center">
            <p class="text-rose-400 font-medium text-sm">{{ error_text }}</p>
        </div>
        {% endif %}

    </div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error_text="Model file missing on server.")
        
    try:
        # Extract features
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        
        # Mapping categorical values to numeric counterparts
        sex = MAPPINGS['sex'][request.form['sex']]
        smoker = MAPPINGS['smoker'][request.form['smoker']]
        region = MAPPINGS['region'][request.form['region']]
        
        # Structure into model input format
        features = np.array([[age, sex, bmi, children, smoker, region]])
        
        # Prediction
        prediction = model.predict(features)[0]
        formatted_prediction = f"${prediction:,.2f}"
        
        return render_template_string(HTML_TEMPLATE, 
                                      prediction_text=f'Estimated Cost: {formatted_prediction}',
                                      form_data=request.form)
                               
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error_text=f"Error in prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
