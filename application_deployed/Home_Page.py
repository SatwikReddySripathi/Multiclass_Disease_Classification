import streamlit as st
st.image("assets/logo.png", caption="")
st.markdown(
    """
    <style>
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #121212;
            padding: 20px 40px;
            border-radius: 10px;
        }
        .header img {
            width: 150px;
        }
        .header nav a {
            color: #E0E0E0;
            text-decoration: none;
            margin: 0 10px;
            font-size: 1.1em;
        }
        .header nav a:hover {
            color: #BB86FC;
        }
    </style>
    <div class="header">
        <nav>
            <a href="#about">About Us</a>
            <a href="#features">Features</a>
            <a href="#contact">Contact</a>
        </nav>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hero Section
st.markdown(
    """
    <style>
        .hero {
            background-color: #1E1E1E;
            padding: 40px;
            border-radius: 10px;
            text-align: center;
        }
        .hero h1 {
            color: #BB86FC;
            font-size: 3em;
            margin-bottom: 20px;
        }
        .hero p {
            color: #E0E0E0;
            font-size: 1.2em;
        }
    </style>
    <div class="hero">
        <h1>Welcome to ThorAIx</h1>
        <p>Revolutionizing thoracic disease detection with AI-powered X-ray analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# About Us Section
st.markdown(
    """
    <div id="about" style="margin-top: 60px; padding: 20px; background-color: #1E1E1E; border-radius: 10px;">
        <h2 style="color: #BB86FC; text-align: center;">About ThorAIx</h2>
        <p style="color: #E0E0E0; font-size: 1.1em; text-align: justify;">
            ThorAIx is at the forefront of healthcare innovation, leveraging cutting-edge AI to enhance thoracic disease detection. 
            Our mission is to empower healthcare professionals with tools that combine speed, precision, and accessibility, ensuring 
            timely diagnoses and better patient outcomes.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Benefits Section
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
            background-color: #2E2E2E;
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
            <h3>Data Privacy & Security</h3>
            <p>Robust protocols ensure patient data confidentiality and security.</p>
        </div>
        <div class="benefit-card">
            <h3>User-Friendly</h3>
            <p>Intuitive interface designed for healthcare professionals.</p>
        </div>
        <div class="benefit-card">
            <h3>Scalable</h3>
            <p>Ideal for hospitals, clinics, and remote healthcare settings.</p>
        </div>
        <div class="benefit-card">
            <h3>Real-Time Insights</h3>
            <p>Make data-driven decisions with instant X-ray analysis.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Contact Section
st.markdown(
    """
    <div id="contact" style="margin-top: 60px; padding: 20px; background-color: #1E1E1E; border-radius: 10px;">
        <h2 style="color: #BB86FC; text-align: center;">Contact Us</h2>
        <p style="color: #E0E0E0; text-align: center; font-size: 1.1em;">
            Have questions or need assistance? We’d love to hear from you!
        </p>
        <p style="color: #BB86FC; text-align: center; font-size: 1.2em;">
            Email us at: <a href="mailto:thoraix.mlops@gmail.com" style="color: #03DAC6; text-decoration: none;">thoraix.mlops@gmail.com</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown(
    """
    <footer style="margin-top: 40px; background-color: #121212; padding: 20px; text-align: center; color: #E0E0E0;">
        <p>© 2024 ThorAIx. All rights reserved.</p>
        <p>Follow us on:
            <a href="#" style="color: #BB86FC; text-decoration: none;">LinkedIn</a> |
            <a href="#" style="color: #BB86FC; text-decoration: none;">Twitter</a> |
            <a href="#" style="color: #BB86FC; text-decoration: none;">Facebook</a>
        </p>
    </footer>
    """,
    unsafe_allow_html=True,
)
