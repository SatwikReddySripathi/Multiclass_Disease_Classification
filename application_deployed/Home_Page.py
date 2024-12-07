import streamlit as st


#st.image("assets/logo.png", width=150)  # Add your product logo here

# Hero section with a sleek dark banner
st.markdown(
    """
    <style>
        .hero {
            background-color: #1E1E1E;  /* Dark theme color */
            padding: 40px;
            border-radius: 10px;
            text-align: center;
        }
        .hero h1 {
            color: #BB86FC;  /* Purple for headings */
            font-size: 3em;
            margin-bottom: 20px;
        }
        .hero p {
            color: #E0E0E0;  /* Subtle gray for text */
            font-size: 1.2em;
        }
        .hero button {
            background-color: #03DAC6;
            color: #121212;
            border: none;
            padding: 10px 20px;
            font-size: 1.1em;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 20px;
        }
        .hero button:hover {
            background-color: #BB86FC;
            color: #FFFFFF;
        }
    </style>
    <div class="hero">
        <h1>Welcome to ThorAIx</h1>
        <p>Revolutionizing thoracic disease detection with AI-powered X-ray analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Benefits section
st.markdown(
    """
    <style>
        .benefits {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 40px;
        }
        .benefit-card {
            background-color: #2E2E2E;  /* Slightly lighter than background */
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .benefit-card:hover {
            transform: translateY(-5px);
        }
        .benefit-card h3 {
            color: #BB86FC;
            margin-bottom: 10px;
        }
        .benefit-card p {
            color: #E0E0E0;
            font-size: 0.9em;
        }
    </style>
    <div class="benefits">
        <div class="benefit-card">
            <h3>Fast Predictions</h3>
            <p>Our AI model delivers results in seconds, reducing diagnostic times.</p>
        </div>
        <div class="benefit-card">
            <h3>Accurate Results</h3>
            <p>Built on cutting-edge machine learning models, ensuring high precision.</p>
        </div>
        <div class="benefit-card">
            <h3>User-Friendly</h3>
            <p>Intuitive interface designed for healthcare professionals.</p>
        </div>
        <div class="benefit-card">
            <h3>Scalable</h3>
            <p>Ideal for hospitals, clinics, and remote healthcare settings.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Call to Action
st.markdown(
    """
    <style>
        .cta {
            background-color: #BB86FC;
            color: #121212;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 40px;
        }
        .cta h2 {
            color: #FFFFFF;
        }
        .cta p {
            color: #FFFFFF;
            font-size: 1em;
        }
        .cta button {
            background-color: #03DAC6;
            color: #121212;
            border: none;
            padding: 10px 20px;
            font-size: 1em;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 20px;
        }
        .cta button:hover {
            background-color: #121212;
            color: #FFFFFF;
        }
    </style>
    <div class="cta">
        <h2>Ready to Transform Diagnostics?</h2>
        <p>Click the button below to start exploring ThorAIx's powerful features.</p>  
    </div>
    """,
    unsafe_allow_html=True,
)

