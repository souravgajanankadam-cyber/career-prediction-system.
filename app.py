import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# Set Page Config
st.set_page_config(
    page_title="FuturePath AI - Career Prediction Portal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load pickled model data
MODEL_PATH = "career_model_data.pkl"

@st.cache_resource
def load_model_data():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model data file '{MODEL_PATH}' not found! Please run 'train_model.py' first.")
        st.stop()
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

data = load_model_data()
rf_model = data['rf_model']
knn_model = data['knn_model']
encoder = data['encoder']
label_encoder = data['label_encoder']
categorical_mappings = data['categorical_mappings']
feature_names = data['feature_names']
binary_cols = data['binary_cols']
rating_cols = data['rating_cols']
nominal_cols = data['nominal_cols']
numeric_cols = data['numeric_cols']
X_processed = data['X_processed']
y_encoded = data['y_encoded']
df_raw = data['df_raw']

# Map class names
class_names = list(label_encoder.classes_)

# Detailed Career Descriptions dictionary
career_profiles = {
    'Applications Developer': {
        'description': "Designs, builds, and maintains software applications for desktop, web, or enterprise environments.",
        'responsibilities': [
            "Write clean, modular, and maintainable code.",
            "Collaborate with product managers and engineers to define feature specs.",
            "Write unit tests and debug application issues.",
            "Integrate third-party libraries and APIs."
        ],
        'salary': "$105,000",
        'skills': ["Java / C# / Python", "Software Design Patterns", "SQL Databases", "Git Version Control"],
        'resources': "Recommended Certifications: App Development, Python, Full Stack. Workshops: Game Development, Web Technologies."
    },
    'CRM Technical Developer': {
        'description': "Customizes, integrates, and develops custom workflows for Customer Relationship Management (CRM) platforms like Salesforce, SAP, or Microsoft Dynamics.",
        'responsibilities': [
            "Develop custom scripts, plugins, and components on the CRM platform.",
            "Configure workflow automations, business rules, and user roles.",
            "Integrate the CRM system with email, marketing, and database tools.",
            "Optimize CRM database performance and data synchronization."
        ],
        'salary': "$110,000",
        'skills': ["Salesforce Apex / JavaScript", "CRM System Architecture", "REST/SOAP APIs", "Data Migration"],
        'resources': "Recommended Certifications: Python, App Development. Workshops: Database Security, Web Technologies."
    },
    'Database Developer': {
        'description': "Designs, implements, and optimizes complex database architectures to ensure fast, safe, and efficient data operations.",
        'responsibilities': [
            "Write complex and optimized SQL queries, stored procedures, and triggers.",
            "Design relational database schemas and normalize datasets.",
            "Create ETL (Extract, Transform, Load) pipelines for data warehousing.",
            "Audit databases for security, access control, and indexing efficiency."
        ],
        'salary': "$98,000",
        'skills': ["PostgreSQL / Oracle / MySQL", "Query Optimization & Indexing", "NoSQL (MongoDB, Redis)", "ETL Tools"],
        'resources': "Recommended Certifications: Information Security, Hadoop. Workshops: Database Security, System Designing."
    },
    'Mobile Applications Developer': {
        'description': "Develops high-performance native or cross-platform mobile applications for smartphones and tablets.",
        'responsibilities': [
            "Implement responsive mobile user interfaces based on design mocks.",
            "Integrate native hardware capabilities like GPS, camera, and push notifications.",
            "Debug mobile-specific issues like memory leaks, battery optimization, and cache management.",
            "Publish and update apps on Google Play Store and Apple App Store."
        ],
        'salary': "$112,000",
        'skills': ["Kotlin / Swift", "Flutter / React Native", "Mobile API Integration", "UI/UX Best Practices"],
        'resources': "Recommended Certifications: App Development, Distro Making. Workshops: Game Development, Web Technologies."
    },
    'Network Security Engineer': {
        'description': "Maintains and protects network infrastructures, gateways, and communication systems from security threats, hacks, and unauthorized access.",
        'responsibilities': [
            "Configure and maintain firewalls, VPNs, and Intrusion Detection Systems (IDS).",
            "Monitor network traffic logs for suspicious activities or security threats.",
            "Perform vulnerability scanning, penetration testing, and risk assessments.",
            "Draft network security protocols and incident response plans."
        ],
        'salary': "$118,000",
        'skills': ["Network Protocols (TCP/IP, DNS)", "Firewall & VPN Management", "Wireshark & Pentesting tools", "CompTIA Security+ / CEH"],
        'resources': "Recommended Certifications: Information Security, Machine Learning. Workshops: Hacking, Cloud Computing."
    },
    'Software Developer': {
        'description': "Develops applications, software systems, and utility packages to solve functional problems and improve workflow efficiency.",
        'responsibilities': [
            "Implement backend logic, mathematical algorithms, and computational models.",
            "Participate in code reviews, design documentation, and peer programming.",
            "Debug software components and optimize runtime performance.",
            "Collaborate across multidisciplinary technical departments."
        ],
        'salary': "$102,000",
        'skills': ["Python / C++ / Go", "Data Structures & Algorithms", "Git and Docker", "Software Development Lifecycle"],
        'resources': "Recommended Certifications: Python, Full Stack. Workshops: Web Technologies, Testing."
    },
    'Software Engineer': {
        'description': "Applies formal software engineering paradigms to design, build, and deploy highly scalable, distributed enterprise services and web applications.",
        'responsibilities': [
            "Architect large-scale software systems and microservices.",
            "Design CI/CD deployment pipelines and cloud infrastructure configurations.",
            "Enforce strict quality guidelines, unit-testing suites, and code standards.",
            "Optimize memory usage, scaling speed, and multi-thread concurrency."
        ],
        'salary': "$115,000",
        'skills': ["System Design & Microservices", "Cloud Computing (AWS/Azure/GCP)", "CI/CD & Kubernetes", "Object-Oriented Design"],
        'resources': "Recommended Certifications: Full Stack, R Programming. Workshops: Cloud Computing, Web Technologies."
    },
    'Software Quality Assurance (QA) / Testing': {
        'description': "Evaluates software systems to ensure they meet all functional requirements, security policies, and performance criteria before release.",
        'responsibilities': [
            "Write and execute automated test suites (UI, API, Integration).",
            "Develop manual test plans, edge-case test matrices, and regression cycles.",
            "Locate, isolate, document, and track software bugs in issue trackers.",
            "Ensure software releases meet performance, load, and accessibility standards."
        ],
        'salary': "$85,000",
        'skills': ["Selenium / Playwright / Cypress", "Postman & API Testing", "Bug Tracking & Reporting", "QA Methodologies"],
        'resources': "Recommended Certifications: Shell Programming, Python. Workshops: Testing, System Designing."
    },
    'Systems Security Administrator': {
        'description': "Secures, maintains, and configures operating systems, server environments, and computer resources across an enterprise network.",
        'responsibilities': [
            "Manage user accounts, permissions, Active Directory, and access controls.",
            "Apply security updates, OS patches, and system updates regularly.",
            "Analyze server system logs for intrusion patterns or hardware failures.",
            "Implement automated backup plans and disaster recovery protocols."
        ],
        'salary': "$95,000",
        'skills': ["Linux/Windows Administration", "Active Directory & IAM", "Shell Scripting (Bash/PowerShell)", "System Auditing & Compliance"],
        'resources': "Recommended Certifications: Information Security, Distro Making. Workshops: Hacking, Database Security."
    },
    'Technical Support': {
        'description': "Provides operational troubleshooting, hardware/software technical assistance, and resolution for clients and internal IT systems.",
        'responsibilities': [
            "Diagnose technical issues over remote systems, calls, or tickets.",
            "Troubleshoot network connectivity, operating system, and software bugs.",
            "Maintain hardware assets, update office software configurations.",
            "Document technical troubleshooting solutions in internal knowledge bases."
        ],
        'salary': "$65,000",
        'skills': ["OS Troubleshooting", "Helpdesk ticketing tools", "Hardware Diagnostics", "Customer Communication"],
        'resources': "Recommended Certifications: Distro Making, Shell Programming. Workshops: Testing, Cloud Computing."
    },
    'UX Designer': {
        'description': "Researches, designs, and refines the interactive layout and visual flow of software systems to create intuitive user experiences.",
        'responsibilities': [
            "Conduct user research, interviews, and compile user personas.",
            "Create wireframes, interactive prototypes, and design mockups.",
            "Establish unified design systems, brand guidelines, and UI component libraries.",
            "Conduct A/B testing and usability feedback loops."
        ],
        'salary': "$92,000",
        'skills': ["Figma / Adobe XD", "UI Layout & Wireframing", "Cognitive Psychology & Interaction Design", "A/B Testing"],
        'resources': "Recommended Certifications: Distro Making, Python. Workshops: Game Development, Web Technologies."
    },
    'Web Developer': {
        'description': "Designs, constructs, and deploys responsive websites, landing pages, and interactive web application frontends/backends.",
        'responsibilities': [
            "Develop modern, responsive web pages using modern frameworks.",
            "Integrate front-end views with backend databases and API services.",
            "Optimize web asset rendering speed, SEO keywords, and mobile layouts.",
            "Maintain domain configurations, SSL, and web server deployments."
        ],
        'salary': "$88,000",
        'skills': ["HTML / CSS / JavaScript", "React / Vue / Next.js", "Node.js & API Integration", "SEO & Web Optimization"],
        'resources': "Recommended Certifications: Full Stack, R Programming. Workshops: Web Technologies, Cloud Computing."
    }
}

# Inject Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    
    .main {
        background-color: #f7f9fc;
    }
    
    h1, h2, h3, .stHeader {
        font-family: 'Outfit', sans-serif;
    }
    p, span, div, label, li {
        font-family: 'Inter', sans-serif;
    }
    
    .glass-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        margin-bottom: 20px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .banner {
        background: linear-gradient(135deg, #1e3a8a, #0f172a);
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .banner h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .banner p {
        margin-top: 10px;
        font-size: 1.1rem;
        color: #93c5fd;
    }
    
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .badge-blue { background-color: #dbeafe; color: #1e40af; }
    .badge-green { background-color: #dcfce7; color: #15803d; }
    .badge-purple { background-color: #f3e8ff; color: #6b21a8; }
    .badge-amber { background-color: #fef3c7; color: #92400e; }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding-left: 10px;
        margin-bottom: 15px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Application Banner (No Emojis)
st.markdown("""
<div class="banner">
    <h1>FuturePath AI</h1>
    <p>Predictive Student Career Guidance System & Multi-Dimensional Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Info (No Emojis)
with st.sidebar:
    if os.path.exists("PS2-Interns_WorkFlow.png"):
        st.image("PS2-Interns_WorkFlow.png", caption="System Work Flow Graph", use_container_width=True)
    st.markdown("""
    ---
    ### Engine Summary
    FuturePath AI leverages a recommendation engine to guide student placements:
    - **Random Forest Classifier:** Learns non-linear interactions across 19 profile attributes.
    - **Case-Based Match (CBR):** KNN algorithm finds identical or highly similar historical student files.
    - **Rule-Based Fallback:** Match scoring on key fields (Subject + Career Area).
    
    *Powered by scikit-learn & streamlit.*
    """)

st.markdown("### Assess Your Professional & Cognitive Profile")
st.write("Complete the profile questionnaire below. Our recommendation engine will process your cognitive, academic, and behavioral metrics to recommend suitable job roles.")

# Form to hold quiz elements
with st.form("quiz_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='section-title'>Cognitive & Core Skills</div>", unsafe_allow_html=True)
        logical_quotient = st.slider("Logical Quotient Rating", min_value=1, max_value=9, value=6, help="Analytical reasoning score (1 to 9)")
        coding_skills = st.slider("Coding Skills Rating", min_value=1, max_value=9, value=6, help="Programming capability rating (1 to 9)")
        public_speaking = st.slider("Public Speaking Points", min_value=1, max_value=9, value=5, help="Presentation and communication score (1 to 9)")
        hackathons = st.slider("Hackathons Participated", min_value=0, max_value=6, value=2, help="Number of hackathons attended")
        
        st.markdown("<div class='section-title'>Communication & Memory</div>", unsafe_allow_html=True)
        read_write_skills = st.selectbox("Reading and Writing Skills", options=categorical_mappings['ratings'], index=1)
        memory_cap = st.selectbox("Memory Capability Score", options=categorical_mappings['ratings'], index=1)

    with col2:
        st.markdown("<div class='section-title'>Academic & Experience</div>", unsafe_allow_html=True)
        certification = st.selectbox("Key Certification Earned", options=categorical_mappings['nominal']['certifications'], index=4)
        workshop = st.selectbox("Workshop Category Attended", options=categorical_mappings['nominal']['workshops'], index=3)
        interested_subject = st.selectbox("Most Interested Academic Subject", options=categorical_mappings['nominal']['Interested subjects'], index=0)
        interested_career = st.selectbox("Interested Career Domain", options=categorical_mappings['nominal']['interested career area'], index=4)
        book_type = st.selectbox("Preferred Book Genres", options=categorical_mappings['nominal']['Interested Type of Books'], index=2)

        st.markdown("<div class='section-title'>Company Preference</div>", unsafe_allow_html=True)
        company_type = st.selectbox("Preferred Target Company Type", options=categorical_mappings['nominal']['Type of company want to settle in?'], index=8)

    with col3:
        st.markdown("<div class='section-title'>Personality & Background</div>", unsafe_allow_html=True)
        self_learning = st.selectbox("Self-Learning Capability?", options=categorical_mappings['binary'], index=1)
        extra_courses = st.selectbox("Completed Extra-Courses?", options=categorical_mappings['binary'], index=0)
        senior_input = st.selectbox("Sought Inputs from Seniors/Elders?", options=categorical_mappings['binary'], index=1)
        teamwork = st.selectbox("Worked in Teams Before?", options=categorical_mappings['binary'], index=1)
        introvert = st.selectbox("Are you an Introvert?", options=categorical_mappings['binary'], index=0)
        
        st.markdown("<div class='section-title'>Professional Inclination</div>", unsafe_allow_html=True)
        mgmt_tech = st.selectbox("Management or Technical Focus?", options=categorical_mappings['nominal']['Management or Technical'], index=1)
        worker_type = st.selectbox("Hard or Smart Worker?", options=categorical_mappings['nominal']['hard/smart worker'], index=1)

    # Submit button (No Emojis)
    submit_btn = st.form_submit_button("Predict My Career Path", use_container_width=True)

# Output results
if submit_btn:
    # Preprocess input values for prediction
    user_input_dict = {
        'Logical quotient rating': logical_quotient,
        'hackathons': hackathons,
        'coding skills rating': coding_skills,
        'public speaking points': public_speaking,
        'self-learning capability?': self_learning.lower(),
        'Extra-courses did': extra_courses.lower(),
        'certifications': certification,
        'workshops': workshop,
        'reading and writing skills': read_write_skills.lower(),
        'memory capability score': memory_cap.lower(),
        'Interested subjects': interested_subject,
        'interested career area': interested_career,
        'Type of company want to settle in?': company_type,
        'Taken inputs from seniors or elders': senior_input.lower(),
        'Interested Type of Books': book_type,
        'Management or Technical': mgmt_tech,
        'hard/smart worker': worker_type,
        'worked in teams ever?': teamwork.lower(),
        'Introvert': introvert.lower()
    }
    
    # Prepare df for encoders
    user_df = pd.DataFrame([user_input_dict])
    
    # Build processed features
    user_processed = user_df.copy()
    
    # Binary encoding
    for col in binary_cols:
        user_processed[col] = user_processed[col].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
        
    # Rating encoding
    for col in rating_cols:
        user_processed[col] = user_processed[col].map({'poor': 0, 'medium': 1, 'excellent': 2}).fillna(1).astype(int)
        
    # Nominal ordinal encode
    encoded_nominal = encoder.transform(user_df[nominal_cols])
    for i, col in enumerate(nominal_cols):
        user_processed[col] = encoded_nominal[:, i]
        
    # Ensure correct column order
    user_processed = user_processed[feature_names]
    
    # Predict Class probabilities and final class
    rf_probs = rf_model.predict_proba(user_processed)[0]
    rf_pred_code = np.argmax(rf_probs)
    rf_predicted_role = label_encoder.inverse_transform([rf_pred_code])[0]
    rf_confidence = rf_probs[rf_pred_code] * 100
    
    # Retrieve CBR recommendation (KNN)
    distances, indices = knn_model.kneighbors(user_processed, n_neighbors=1)
    closest_index = indices[0][0]
    cbr_role = df_raw.iloc[closest_index]['Suggested Job Role']
    
    # Retrieve Rule-Based Recommendation
    career_area_lower = interested_career.lower()
    subject_lower = interested_subject.lower()
    
    if 'testing' in career_area_lower:
        rule_role = "Software Quality Assurance (QA) / Testing"
    elif 'developer' in career_area_lower:
        if 'mobile' in subject_lower:
            rule_role = "Mobile Applications Developer"
        elif 'cloud' in career_area_lower or 'web' in subject_lower:
            rule_role = "Web Developer"
        else:
            rule_role = "Software Developer"
    elif 'security' in career_area_lower or 'hacking' in subject_lower:
        rule_role = "Network Security Engineer"
    elif 'system' in career_area_lower:
        rule_role = "Systems Security Administrator"
    elif 'analyst' in career_area_lower or 'management' in subject_lower:
        rule_role = "CRM Technical Developer"
    else:
        rule_role = "Software Engineer"
        
    # Display Results (No Emojis)
    st.markdown("---")
    st.markdown("### Prediction Results")
    
    rcol1, rcol2, rcol3 = st.columns(3)
    with rcol1:
        st.markdown(f"""
        <div class="glass-card" style="border-top: 5px solid #3b82f6; text-align: center;">
            <div class="metric-label">ML RandomForest Predictor</div>
            <div class="metric-value" style="font-size: 1.6rem; color: #1e3a8a; margin: 15px 0;">{rf_predicted_role}</div>
            <div style="font-size: 0.9rem; color: #64748b;">Confidence Weight: <b>{rf_confidence:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with rcol2:
        st.markdown(f"""
        <div class="glass-card" style="border-top: 5px solid #8b5cf6; text-align: center;">
            <div class="metric-label">Case-Based Reasoning (CBR) Match</div>
            <div class="metric-value" style="font-size: 1.6rem; color: #6d28d9; margin: 15px 0;">{cbr_role}</div>
            <div style="font-size: 0.9rem; color: #64748b;">Matching distance: <b>{1 - distances[0][0]:.3f} (Cosine Similarity)</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with rcol3:
        st.markdown(f"""
        <div class="glass-card" style="border-top: 5px solid #10b981; text-align: center;">
            <div class="metric-label">Rule-Based Profile Match</div>
            <div class="metric-value" style="font-size: 1.6rem; color: #047857; margin: 15px 0;">{rule_role}</div>
            <div style="font-size: 0.9rem; color: #64748b;">Matching Rule: <b>Subject & Career Alignment</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    # Select role details to show
    selected_role = rf_predicted_role
    if selected_role in career_profiles:
        profile = career_profiles[selected_role]
        
        det_col1, det_col2 = st.columns([1.2, 1])
        with det_col1:
            st.markdown("<div class='section-title'>Recommended Role Overview</div>", unsafe_allow_html=True)
            st.markdown(f"**Description:** {profile['description']}")
            st.markdown(f"**Average Industry Starting Salary:** `{profile['salary']}/yr`")
            
            st.markdown("**Core Responsibilities:**")
            for resp in profile['responsibilities']:
                st.markdown(f"- {resp}")
                
            st.markdown("**Key Skills to Develop:**")
            badges_html = "".join([f"<span class='badge badge-blue'>{s}</span>" for s in profile['skills']])
            st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)
            
            st.markdown(f"**Learning Path Suggestion:** {profile['resources']}")
            
        with det_col2:
            st.markdown("<div class='section-title'>Skill Fit Visualization</div>", unsafe_allow_html=True)
            
            # Fetch averages for this role in the database to compare
            subset = df_raw[df_raw['Suggested Job Role'] == selected_role]
            avg_logical = subset['Logical quotient rating'].mean()
            avg_coding = subset['coding skills rating'].mean()
            avg_speaking = subset['public speaking points'].mean()
            avg_hackathons = subset['hackathons'].mean()
            
            # Plotly Radar Chart
            categories = ['Logical Quotient', 'Coding Skills', 'Public Speaking', 'Hackathons']
            
            fig = go.Figure()
            
            # User values (scale hackathons to 1-9 for radar visualization equivalence, say multiply by 1.5)
            user_radar_vals = [logical_quotient, coding_skills, public_speaking, hackathons * 1.5]
            role_radar_vals = [avg_logical, avg_coding, avg_speaking, avg_hackathons * 1.5]
            
            fig.add_trace(go.Scatterpolar(
                r=user_radar_vals,
                theta=categories,
                fill='toself',
                name='Your Profile',
                line_color='#3b82f6'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=role_radar_vals,
                theta=categories,
                fill='toself',
                name=f'Average {selected_role}',
                line_color='#64748b',
                opacity=0.6
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 9]
                    )
                ),
                showlegend=True,
                margin=dict(l=40, r=40, t=20, b=20),
                height=320
            )
            
            st.plotly_chart(fig, use_container_width=True)
