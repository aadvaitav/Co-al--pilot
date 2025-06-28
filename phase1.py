import streamlit as st
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import requests
import json
import os
from datetime import datetime, timedelta
import google.generativeai as genai
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
import hashlib
from openai import OpenAI
import time
import random
warnings.filterwarnings('ignore')

# Configuration

# Configuration
st.set_page_config(
    page_title="Coal-Mine Carbon Neutrality Co-Pilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Black Theme with White Accents
st.markdown("""
<style>
    :root {
        --primary: #0f0f0f;
        --secondary: #1a1a1a;
        --accent: #ffffff;
        --accent-light: #f0f0f0;
        --highlight: #4a8cff;
        --transition: all 0.3s ease;
    }
    
    body {
        background-color: var(--primary);
        color: var(--accent-light);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        transition: var(--transition);
    }
    
    .main-header {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 100%);
        color: var(--accent);
        padding: 1.5rem;
        border-radius: 0;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        border-bottom: 1px solid #333;
        transition: var(--transition);
    }
    
    .stApp {
        background: var(--primary);
    }
    
    .metric-card {
        background: var(--secondary);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border-left: 4px solid var(--highlight);
        transition: var(--transition);
        color: var(--accent-light);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    
    .st-bb {
        border-color: #333 !important;
    }
    
    .st-ax {
        color: var(--accent-light) !important;
    }
    
    .stTextInput>div>div>input, 
    .stTextInput>div>div>textarea, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div {
        background: var(--secondary) !important;
        color: var(--accent) !important;
        border: 1px solid #333 !important;
        border-radius: 8px;
        transition: var(--transition);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 100%);
        color: var(--accent) !important;
        border: 1px solid #333 !important;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: var(--transition);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #252525 0%, #1a1a1a 100%);
        border-color: var(--highlight) !important;
        transform: scale(1.03);
    }
    
    .stButton>button:focus {
        box-shadow: 0 0 0 0.2rem rgba(74, 140, 255, 0.3);
    }
    
    .stTab {
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--secondary) !important;
        color: var(--accent-light) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        transition: var(--transition) !important;
        border: 1px solid #333 !important;
        margin: 0 5px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 100%) !important;
        color: var(--accent) !important;
        border-color: var(--highlight) !important;
        box-shadow: 0 0 15px rgba(74, 140, 255, 0.2);
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"]:hover {
        transform: translateY(-3px);
    }
    
    .stDataFrame {
        background: var(--secondary) !important;
        border-radius: 10px;
        border: 1px solid #333;
    }
    
    .stDataFrame th {
        background: #1a1a1a !important;
        color: var(--accent) !important;
    }
    
    .stDataFrame td {
        background: var(--secondary) !important;
        color: var(--accent-light) !important;
        border-color: #333 !important;
    }
    
    .chat-message {
        background: var(--secondary);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 3px solid var(--highlight);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: var(--transition);
    }
    
    .chat-message:hover {
        transform: translateX(5px);
    }
    
    .success-message {
        background: rgba(25, 135, 84, 0.15);
        color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #28a745;
    }
    
    .user-badge {
        background: var(--highlight);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-right: 0.75rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(74, 140, 255, 0.3);
    }
    
    .admin-badge {
        background: #dc3545;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-right: 0.75rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3);
    }
    
    .realtime-data {
        background: rgba(25, 135, 84, 0.1);
        border: 1px solid #28a745;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stSidebar {
        background: var(--secondary) !important;
        border-right: 1px solid #333;
    }
    
    .st-expander {
        background: var(--secondary) !important;
        border: 1px solid #333 !important;
        border-radius: 10px !important;
    }
    
    .st-expanderHeader {
        color: var(--accent) !important;
        font-weight: 600 !important;
    }
    
    .plot-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        transition: var(--transition);
    }
    
    .plot-container:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    
    .stAlert {
        border-radius: 10px !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--accent) !important;
        border-bottom: 1px solid #333;
        padding-bottom: 0.5rem;
    }
    
    .transition-all {
        transition: var(--transition);
    }
    
    .card-hover:hover {
        transform: translateY(-7px);
        box-shadow: 0 12px 25px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)

class DatabaseManager:
    def __init__(self):
        self.init_databases()
    
    def init_databases(self):
        """Initialize user and global databases"""
        # Global database
        conn = sqlite3.connect('global_data.db')
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT,
                is_admin BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Global leaderboard
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS global_leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                mine_name TEXT,
                state TEXT,
                green_score REAL,
                emission_intensity REAL,
                total_emissions REAL,
                mine_type TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')
        
        # --- Schema migration for realtime_data ---
        # We check for a column from the old schema. If it exists, we drop the table
        # to force a recreation with the new schema. This is a destructive migration.
        try:
            # Check for a column that only exists in the OLD schema ('production')
            cursor.execute("SELECT production FROM realtime_data LIMIT 1")
            # If the above line doesn't fail, it means we have the old schema.
            cursor.execute("DROP TABLE realtime_data")
            conn.commit()
        except sqlite3.OperationalError:
            # This will fail if the table/column doesn't exist, which is expected
            # for new databases or for databases that have already been migrated.
            pass

        # Recreate the table with the new schema.
        # This will run if the table was just dropped or if it never existed.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                scope1 REAL,
                scope2 REAL,
                scope3 REAL,
                carbon_sink REAL,
                carbon_offset REAL,
                total_emissions REAL
            )
        ''')
        
        # Create default admin and users
        admin_hash = hashlib.sha256("admin".encode()).hexdigest()
        cursor.execute('INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?)', 
                      ('admin', admin_hash, True, datetime.now()))
        
        for i in range(1, 4):
            user_hash = hashlib.sha256(f"user{i}".encode()).hexdigest()
            cursor.execute('INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?)', 
                          (f'user{i}', user_hash, False, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def create_user_database(self, username):
        """Create individual user database"""
        conn = sqlite3.connect(f'user_{username}.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mine_name TEXT,
                state TEXT,
                production REAL,
                mine_type TEXT,
                base_emissions REAL,
                transport_emissions REAL,
                total_emissions REAL,
                emission_intensity REAL,
                green_score REAL,
                latitude REAL,
                longitude REAL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_public BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def authenticate_user(self, username, password):
        """Authenticate user"""
        conn = sqlite3.connect('global_data.db')
        cursor = conn.cursor()
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('SELECT username, is_admin FROM users WHERE username = ? AND password_hash = ?', 
                      (username, password_hash))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {'username': result[0], 'is_admin': bool(result[1])}
        return None

class OpenRouterClient:
    def __init__(self, api_key):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    
    def chat(self, messages, model="mistralai/mistral-7b-instruct:free"):
        """Chat with OpenRouter model"""
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://coal-mine-app.streamlit.app",
                    "X-Title": "Coal Mine Carbon Neutrality Co-Pilot",
                },
                model=model,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

class CoalMineDataProcessor:
    def __init__(self):
        self.emission_factors = {
            'OC': {'base': 0.95, 'diesel': 2.68, 'electricity': 0.85},
            'UG': {'base': 0.82, 'diesel': 2.1, 'electricity': 1.2, 'methane': 0.3},
            'Mixed': {'base': 0.88, 'diesel': 2.4, 'electricity': 1.0, 'methane': 0.15}
        }
        
        self.regional_factors = {
            'Odisha': 1.1, 'Jharkhand': 1.05, 'Chhattisgarh': 1.0,
            'West Bengal': 0.95, 'Madhya Pradesh': 1.0, 'Telangana': 0.9,
            'Andhra Pradesh': 0.9, 'Maharashtra': 0.85, 'Karnataka': 0.8
        }
        
        self.required_columns = ['Mine Name', 'State/UT Name', 'Type of Mine (OC/UG/Mixed)']
        self.optional_columns = ['Coal/ Lignite Production (MT) (2019-2020)', 'Transport Distance (km)', 
                               'Diesel Consumption (L)', 'Electricity Consumption (kWh)', 'Latitude', 'Longitude']
    
    def validate_and_process_data(self, df):
        """Validate and process data with flexible column handling"""
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check required columns
        missing_required = [col for col in self.required_columns if col not in df.columns]
        if missing_required:
            return None, f"Missing required columns: {missing_required}"
        
        # Handle optional columns
        missing_optional = []
        for col in self.optional_columns:
            if col not in df.columns:
                missing_optional.append(col)
        
        # Add calculated columns
        df = self._add_calculated_columns(df, missing_optional)
        
        return df, missing_optional
    
    def _add_calculated_columns(self, df, missing_columns):
        """Add calculated emission columns"""
        # Default values for missing columns
        defaults = {
            'Coal/ Lignite Production (MT) (2019-2020)': 1000,
            'Transport Distance (km)': 150,
            'Diesel Consumption (L)': 10000,
            'Electricity Consumption (kWh)': 50000,
            'Latitude': 20.0,
            'Longitude': 85.0
        }
        
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val
        
        # Calculate emissions
        df['Base_Emissions_MT'] = df.apply(self.calculate_base_emissions, axis=1)
        df['Transport_Emissions_MT'] = df.apply(self.calculate_transport_emissions, axis=1)
        df['Total_Emissions_MT'] = df['Base_Emissions_MT'] + df['Transport_Emissions_MT']
        df['Emission_Intensity'] = df['Total_Emissions_MT'] / (df['Coal/ Lignite Production (MT) (2019-2020)'] + 0.001)
        df['Green_Score'] = 100 - (df['Emission_Intensity'] * 100)
        df['Green_Score'] = df['Green_Score'].clip(0, 100)
        
        return df
    
    def calculate_base_emissions(self, row):
        """Calculate base emissions"""
        production = row['Coal/ Lignite Production (MT) (2019-2020)']
        mine_type = row['Type of Mine (OC/UG/Mixed)']
        state = row['State/UT Name']
        
        if pd.isna(production) or production == 0:
            return 0
        
        base_factor = self.emission_factors.get(mine_type, self.emission_factors['Mixed'])['base']
        regional_factor = self.regional_factors.get(state, 1.0)
        
        return production * base_factor * regional_factor
    
    def calculate_transport_emissions(self, row):
        """Calculate transport emissions"""
        production = row['Coal/ Lignite Production (MT) (2019-2020)']
        distance = row.get('Transport Distance (km)', 150)
        
        if pd.isna(production) or production == 0:
            return 0
        
        truck_capacity = 25
        emission_per_km_per_mt = 0.12
        
        transport_emissions = (production / truck_capacity) * distance * emission_per_km_per_mt
        return transport_emissions / 1000
    
    def generate_improvement_suggestions(self, df):
        """Generate improvement suggestions based on data analysis"""
        suggestions = []
        
        # High emission intensity mines
        high_intensity = df[df['Emission_Intensity'] > df['Emission_Intensity'].quantile(0.75)]
        if len(high_intensity) > 0:
            suggestions.append({
                'category': 'High Emission Intensity',
                'mines': high_intensity['Mine Name'].tolist()[:5],
                'recommendation': 'Focus on operational efficiency improvements and renewable energy adoption'
            })
        
        # Transport optimization
        if 'Transport Distance (km)' in df.columns:
            long_distance = df[df['Transport Distance (km)'] > 200]
            if len(long_distance) > 0:
                suggestions.append({
                    'category': 'Transport Optimization',
                    'mines': long_distance['Mine Name'].tolist()[:5],
                    'recommendation': 'Consider rail transport, electric vehicles, or local processing facilities'
                })
        
        # Mine type specific suggestions
        for mine_type in df['Type of Mine (OC/UG/Mixed)'].unique():
            type_data = df[df['Type of Mine (OC/UG/Mixed)'] == mine_type]
            avg_intensity = type_data['Emission_Intensity'].mean()
            
            if mine_type == 'UG' and avg_intensity > 1.0:
                suggestions.append({
                    'category': f'{mine_type} Mine Optimization',
                    'mines': type_data['Mine Name'].tolist()[:3],
                    'recommendation': 'Implement methane capture systems and improve ventilation efficiency'
                })
            elif mine_type == 'OC' and avg_intensity > 0.8:
                suggestions.append({
                    'category': f'{mine_type} Mine Optimization',
                    'mines': type_data['Mine Name'].tolist()[:3],
                    'recommendation': 'Electrify heavy machinery and optimize haul road management'
                })
        
        return suggestions

class GeminiClient:
    def __init__(self, api_key):
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
    
    def generate_report(self, data_summary, analysis_type="executive"):
        """Generate comprehensive reports using Gemini"""
        if not self.model:
            return "Gemini API key not configured"
        
        prompt = f"""
        Create a comprehensive {analysis_type} report for coal mine carbon emissions analysis.
        
        Data Summary: {data_summary}
        
        Include:
        1. Executive Summary
        2. Key Findings
        3. Recommendations
        4. Implementation Roadmap
        5. Cost-Benefit Analysis
        
        Format as a professional report with clear sections and actionable insights.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating report: {str(e)}"

    def generate_improvement_roadmap(self, df_summary, top_emitters_str):
        """Generate an improvement roadmap using Gemini"""
        if not self.model:
            return "Gemini API key not configured"

        prompt = f"""
        As an expert carbon reduction strategist for the mining industry, analyze the following data summary for a user's portfolio of coal mines.

        **Data Summary:**
        {df_summary}

        **Top 5 Emitting Mines:**
        {top_emitters_str}

        **Your Task:**
        Create a detailed, actionable "Carbon Reduction Roadmap". The roadmap should be structured, professional, and easy to understand.

        **Include the following sections:**
        1.  **Executive Summary:** A brief overview of the current emissions status and the potential for improvement.
        2.  **Key Insight & Priority Areas:** Identify the most critical areas for intervention based on the data (e.g., specific high-emission mines, common issues in a region, problems with a certain mine type).
        3.  **Short-Term Goals (3-6 Months):** List 3-4 specific, low-cost, high-impact actions. For each, describe the action, the expected outcome, and the mines it applies to.
        4.  **Medium-Term Goals (6-18 Months):** List 2-3 significant investment-based actions (e.g., equipment electrification, process optimization). Detail the action, potential challenges, and expected emission reduction percentage.
        5.  **Long-Term Vision (2-5 Years):** Describe 1-2 transformative goals, such as transitioning to renewable energy sources, or achieving a specific green score target.
        6.  **Conclusion:** A concluding paragraph to motivate the user.

        Format the output using Markdown for clear headings, lists, and bold text.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating improvement roadmap: {str(e)}"

class RealtimeDataGenerator:
    @staticmethod
    def generate_realtime_data(simulation_settings):
        """Generate simulated real-time data based on React reference"""
        now_millis = time.time() * 1000

        base_scope1 = 45 + np.sin(now_millis / 60000) * 8 + (random.random() - 0.5) * 5
        base_scope2 = 25 + np.cos(now_millis / 80000) * 4 + (random.random() - 0.5) * 3
        base_scope3 = 35 + np.sin(now_millis / 120000) * 6 + (random.random() - 0.5) * 4

        scope1 = base_scope1 * (simulation_settings['fuel_usage'] / 100) * (100 / simulation_settings['equipment_efficiency'])
        scope2 = base_scope2 * (simulation_settings['electricity_usage'] / 100)
        scope3 = base_scope3 * (simulation_settings['transport_activity'] / 100)

        carbon_sink = -12 + np.sin(now_millis / 90000) * 3 + (random.random() - 0.5) * 2
        carbon_offset = (-18 + np.cos(now_millis / 100000) * 4 + (random.random() - 0.5) * 2) * (simulation_settings['carbon_offset_programs'] / 100)
        
        total_emissions = scope1 + scope2 + scope3 + carbon_sink + carbon_offset

        return {
            'timestamp': datetime.now(),
            'scope1': scope1,
            'scope2': scope2,
            'scope3': scope3,
            'carbon_sink': carbon_sink,
            'carbon_offset': carbon_offset,
            'total_emissions': total_emissions
        }

    @staticmethod
    def save_realtime_data(data):
        """Save real-time data to database"""
        conn = sqlite3.connect('global_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO realtime_data
            (timestamp, scope1, scope2, scope3, carbon_sink, carbon_offset, total_emissions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (data['timestamp'], data['scope1'], data['scope2'], data['scope3'],
              data['carbon_sink'], data['carbon_offset'], data['total_emissions']))
        
        conn.commit()
        conn.close()

class SyntheticDataGenerator:
    @staticmethod
    def generate_synthetic_data(num_rows=20):
        """Generate a synthetic dataset for a single company ('Company X') across a few states."""
        company_x_portfolio = {
            'Jharkhand': ['Jharia OC Mine', 'North Karanpura UG', 'Rajmahal Mixed Mine'],
            'Odisha': ['Talcher OC Field', 'Ib Valley UG Mine'],
            'Chhattisgarh': ['Gevra OC Mine', 'Dipka Mixed Project', 'Kusmunda UG']
        }
        states = list(company_x_portfolio.keys())
        mine_types = ['OC', 'UG', 'Mixed']
        
        data = []
        for i in range(num_rows):
            state = random.choice(states)
            mine_name = random.choice(company_x_portfolio[state])
            # Ensure unique mine names for better visualization
            mine_name = f"{mine_name} #{i+1}"
            
            mine_type = random.choice(mine_types)
            production = random.randint(800, 6000) * 1000 # In MT
            distance = random.randint(80, 400)
            
            # Geographic coordinates relevant to the states
            if state == 'Jharkhand':
                lat, lon = random.uniform(23.6, 24.4), random.uniform(85.0, 86.5)
            elif state == 'Odisha':
                lat, lon = random.uniform(20.5, 22.0), random.uniform(83.5, 85.0)
            else: # Chhattisgarh
                lat, lon = random.uniform(21.2, 23.0), random.uniform(81.0, 83.0)

            data.append({
                'Mine Name': mine_name,
                'State/UT Name': state,
                'Type of Mine (OC/UG/Mixed)': mine_type,
                'Coal/ Lignite Production (MT) (2019-2020)': production,
                'Transport Distance (km)': distance,
                'Latitude': lat,
                'Longitude': lon
            })
        
        return pd.DataFrame(data)

def login_page():
    """Login page with modern design"""
    st.markdown(
        """
        <div class="main-header" style="text-align: center;">
            <h1 style="font-size: 2.8rem; margin-bottom: 0.5rem;">⚡ COAL MINE CARBON CO-PILOT</h1>
            <p style="opacity: 0.8; font-size: 1.1rem;">Secure Login Portal</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Create 3 columns with the middle column wider for the login form
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Card-style login form
        with st.container():
            st.markdown(
                """
                <div style="background: #1a1a1a; padding: 2rem; border-radius: 12px; 
                            border: 1px solid #333; box-shadow: 0 8px 25px rgba(0,0,0,0.4);">
                    <h2 style="text-align: center; margin-bottom: 1.5rem;">🔐 Login</h2>
                """,
                unsafe_allow_html=True
            )
            
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", use_container_width=True, key="login_btn"):
                db_manager = DatabaseManager()
                user = db_manager.authenticate_user(username, password)
                
                if user:
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            
            st.markdown(
                """
                </div>
                <div style="margin-top: 1.5rem; background: #1a1a1a; padding: 1.5rem; 
                            border-radius: 12px; border: 1px solid #333;">
                    <h4 style="margin-bottom: 0.5rem;">💡 Test Accounts</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                        <div><code>user1 / user1</code></div>
                        <div><code>user2 / user2</code></div>
                        <div><code>user3 / user3</code></div>
                        <div><code>admin / admin</code></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def admin_panel():
    """Admin panel for managing global leaderboard"""
    st.header("👑 Admin Panel")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Global Leaderboard", "User Management", "System Stats", "Add Sample Data"])
    
    with tab1:
        st.subheader("Global Leaderboard Management")
        
        # Display current leaderboard
        conn = sqlite3.connect('global_data.db')
        leaderboard_df = pd.read_sql_query('''
            SELECT username, mine_name, state, green_score, emission_intensity, 
                   total_emissions, mine_type, updated_at
            FROM global_leaderboard 
            ORDER BY green_score DESC
        ''', conn)
        conn.close()
        
        if not leaderboard_df.empty:
            st.dataframe(leaderboard_df, use_container_width=True)
            
            # Admin actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh Leaderboard"):
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear Leaderboard"):
                    conn = sqlite3.connect('global_data.db')
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM global_leaderboard')
                    conn.commit()
                    conn.close()
                    st.success("Leaderboard cleared!")
                    st.rerun()
        else:
            st.info("No data in global leaderboard yet.")
    
    with tab2:
        st.subheader("User Management")
        
        conn = sqlite3.connect('global_data.db')
        users_df = pd.read_sql_query('SELECT username, is_admin, created_at FROM users', conn)
        conn.close()
        
        st.dataframe(users_df, use_container_width=True)
    
    with tab3:
        st.subheader("System Statistics")
        
        # Get system stats
        conn = sqlite3.connect('global_data.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM global_leaderboard')
        total_entries = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM realtime_data')
        realtime_entries = cursor.fetchone()[0]
        
        conn.close()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", total_users)
        with col2:
            st.metric("Leaderboard Entries", total_entries)
        with col3:
            st.metric("Real-time Data Points", realtime_entries)
    
    with tab4:
        st.subheader("Add Sample Data")
        
        # Admin can add sample data to test the system
        with st.form("admin_add_data"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mine_name = st.text_input("Mine Name", value="Admin Test Mine")
                state = st.selectbox("State", ['Odisha', 'Jharkhand', 'Chhattisgarh', 
                                              'West Bengal', 'Madhya Pradesh', 'Other'])
            
            with col2:
                mine_type = st.selectbox("Mine Type", ['OC', 'UG', 'Mixed'])
                production = st.number_input("Production (MT)", min_value=0.0, value=1000.0)
            
            with col3:
                transport_distance = st.number_input("Transport Distance (km)", min_value=0.0, value=150.0)
                target_user = st.selectbox("Assign to User", ['admin'] + [f'user{i}' for i in range(1, 4)])
            
            if st.form_submit_button("Add Sample Data"):
                if mine_name and state and mine_type:
                    # Create sample data
                    sample_data = {
                        'Mine Name': [mine_name],
                        'State/UT Name': [state],
                        'Type of Mine (OC/UG/Mixed)': [mine_type],
                        'Coal/ Lignite Production (MT) (2019-2020)': [production],
                        'Transport Distance (km)': [transport_distance],
                        'Latitude': [20.0 + random.uniform(-2, 2)],
                        'Longitude': [85.0 + random.uniform(-2, 2)]
                    }
                    
                    sample_df = pd.DataFrame(sample_data)
                    processor = CoalMineDataProcessor()
                    processed_sample, _ = processor.validate_and_process_data(sample_df)
                    
                    # Create user database if it doesn't exist
                    db_manager = DatabaseManager()
                    db_manager.create_user_database(target_user)
                    
                    # Save to user database
                    conn = sqlite3.connect(f'user_{target_user}.db')
                    cursor = conn.cursor()
                    row = processed_sample.iloc[0]
                    cursor.execute('''
                        INSERT INTO user_data 
                        (mine_name, state, production, mine_type, base_emissions, 
                         transport_emissions, total_emissions, emission_intensity, 
                         green_score, latitude, longitude, is_public)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (row['Mine Name'], row['State/UT Name'], 
                          row['Coal/ Lignite Production (MT) (2019-2020)'],
                          row['Type of Mine (OC/UG/Mixed)'], row['Base_Emissions_MT'],
                          row['Transport_Emissions_MT'], row['Total_Emissions_MT'],
                          row['Emission_Intensity'], row['Green_Score'],
                          row['Latitude'], row['Longitude'], True))
                    conn.commit()
                    conn.close()
                    
                    # Add to global leaderboard
                    conn = sqlite3.connect('global_data.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO global_leaderboard 
                        (username, mine_name, state, green_score, emission_intensity, 
                         total_emissions, mine_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (target_user, row['Mine Name'], row['State/UT Name'],
                          row['Green_Score'], row['Emission_Intensity'],
                          row['Total_Emissions_MT'], row['Type of Mine (OC/UG/Mixed)']))
                    conn.commit()
                    conn.close()
                    
                    st.success(f"✅ Added {mine_name} to {target_user}'s database!")
                else:
                    st.error("Please fill in all required fields")

def user_dashboard(username):
    """User dashboard"""

    st.markdown(
        f"""
        <div class="main-header" style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="font-size: 2.2rem; margin-bottom: 0.25rem;">⚡ CARBON NEUTRALITY CO-PILOT</h1>
                <p style="opacity: 0.8;">Advanced Emissions Intelligence Platform</p>
            </div>
            <div style="display: flex; align-items: center;">
                <span class="user-badge">{username}</span>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    # Initialize simulation settings in session state
    if 'simulation_settings' not in st.session_state:
        st.session_state.simulation_settings = {
            'fuel_usage': 100,
            'electricity_usage': 100,
            'transport_activity': 100,
            'equipment_efficiency': 85,
            'carbon_offset_programs': 100
        }
    
    # Initialize user database
    db_manager = DatabaseManager()
    db_manager.create_user_database(username)
    
    # Check if user has existing data
    conn = sqlite3.connect(f'user_{username}.db')
    existing_data = pd.read_sql_query('SELECT * FROM user_data', conn)
    conn.close()
    
    # OpenRouter API configuration
    st.sidebar.title("🔧 Configuration")
    openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password", 
                                              help="Get your API key from openrouter.ai")
    # Gemini API Key input
    gemini_api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password")
    gemini_client = GeminiClient(gemini_api_key)

    if openrouter_api_key:
        ai_client = OpenRouterClient(openrouter_api_key)
    else:
        ai_client = None
    
    # File upload
    st.sidebar.title("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Coal Mine Dataset", type=['csv', 'xlsx'])

    if st.sidebar.button("Load Synthetic Data"):
        st.session_state.synthetic_data = SyntheticDataGenerator.generate_synthetic_data()
        st.success("Synthetic data loaded!")

    # Display existing data or process new upload
    if not existing_data.empty or uploaded_file is not None or 'synthetic_data' in st.session_state:
        processor = CoalMineDataProcessor()
        
        if uploaded_file is not None:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Validate and process data
                processed_df, missing_columns = processor.validate_and_process_data(df)
                
                if processed_df is None:
                    st.error(missing_columns)  # Error message
                    return
                
                # Show missing columns info
                if missing_columns:
                    st.sidebar.warning(f"Missing optional columns: {missing_columns}")
                    st.sidebar.info("Using default values for missing data")
                
                # Option to make data public
                make_public = st.sidebar.checkbox("Share data with global leaderboard", 
                                                help="Check to contribute to global rankings")
                
                if st.sidebar.button("💾 Save Data"):
                    # Save to user database
                    conn = sqlite3.connect(f'user_{username}.db')
                    for _, row in processed_df.iterrows():
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO user_data 
                            (mine_name, state, production, mine_type, base_emissions, 
                             transport_emissions, total_emissions, emission_intensity, 
                             green_score, latitude, longitude, is_public)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (row['Mine Name'], row['State/UT Name'], 
                              row['Coal/ Lignite Production (MT) (2019-2020)'],
                              row['Type of Mine (OC/UG/Mixed)'], row['Base_Emissions_MT'],
                              row['Transport_Emissions_MT'], row['Total_Emissions_MT'],
                              row['Emission_Intensity'], row['Green_Score'],
                              row.get('Latitude', 0), row.get('Longitude', 0), make_public))
                    conn.commit()
                    conn.close()
                    
                    # Save to global leaderboard if public
                    if make_public:
                        conn = sqlite3.connect('global_data.db')
                        cursor = conn.cursor()
                        for _, row in processed_df.iterrows():
                            cursor.execute('''
                                INSERT INTO global_leaderboard 
                                (username, mine_name, state, green_score, emission_intensity, 
                                 total_emissions, mine_type)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            ''', (username, row['Mine Name'], row['State/UT Name'],
                                  row['Green_Score'], row['Emission_Intensity'],
                                  row['Total_Emissions_MT'], row['Type of Mine (OC/UG/Mixed)']))
                        conn.commit()
                        conn.close()
                    
                    st.success("✅ Data saved successfully!")
                    st.rerun()
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.info("Please ensure your file has the correct format. Required columns: Mine Name, State/UT Name, Type of Mine")
                return

        elif 'synthetic_data' in st.session_state and st.session_state.synthetic_data is not None:
            df = st.session_state.synthetic_data
            processed_df, _ = processor.validate_and_process_data(df)
            st.session_state.synthetic_data = None  # Clear after processing
        else:
            # Use existing data
            # Convert existing data to match expected format
            processed_df = pd.DataFrame()
            processed_df['Mine Name'] = existing_data['mine_name']
            processed_df['State/UT Name'] = existing_data['state']
            processed_df['Type of Mine (OC/UG/Mixed)'] = existing_data['mine_type']
            processed_df['Coal/ Lignite Production (MT) (2019-2020)'] = existing_data['production']
            processed_df['Base_Emissions_MT'] = existing_data['base_emissions']
            processed_df['Transport_Emissions_MT'] = existing_data['transport_emissions']
            processed_df['Total_Emissions_MT'] = existing_data['total_emissions']
            processed_df['Emission_Intensity'] = existing_data['emission_intensity']
            processed_df['Green_Score'] = existing_data['green_score']
            processed_df['Latitude'] = existing_data['latitude']
            processed_df['Longitude'] = existing_data['longitude']
        
        # Main tabs - fixed tab labels and count
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Dashboard", "📋 Improvements", "🤖 AI Assistant", "🎯 Scenarios", 
            "📋 Insights", "🏆 Leaderboard", "📡 Real-time"
        ])
        
        with tab1:
            st.header("📊 Your Carbon Intelligence Dashboard")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_production = processed_df['Coal/ Lignite Production (MT) (2019-2020)'].sum()
            total_emissions = processed_df['Total_Emissions_MT'].sum()
            avg_intensity = processed_df['Emission_Intensity'].mean()
            avg_green_score = processed_df['Green_Score'].mean()
            
            with col1:
                st.metric("Total Production", f"{total_production:,.0f} MT")
            with col2:
                st.metric("Total Emissions", f"{total_emissions:,.0f} MT CO₂")
            with col3:
                st.metric("Avg Intensity", f"{avg_intensity:.3f}")
            with col4:
                st.metric("Avg Green Score", f"{avg_green_score:.1f}")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                if len(processed_df) > 0:
                    fig = px.bar(processed_df, x='Mine Name', y='Total_Emissions_MT',
                               title="Emissions by Mine", color='Green_Score',
                               color_continuous_scale='RdYlGn')
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(processed_df) > 0:
                    fig = px.scatter(processed_df, x='Emission_Intensity', y='Green_Score',
                                   color='Type of Mine (OC/UG/Mixed)', size='Total_Emissions_MT',
                                   title="Green Score vs Emission Intensity",
                                   hover_data=['Mine Name'])
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("📋 Improvement Insights & Roadmap")

            # --- Existing rule-based suggestions ---
            st.subheader("Quick Wins & Automated Suggestions")
            suggestions = processor.generate_improvement_suggestions(processed_df)
            if suggestions:
                for suggestion in suggestions:
                    with st.expander(f"🎯 {suggestion['category']}"):
                        st.write(f"**Affected Mines:** {', '.join(suggestion['mines'])}")
                        st.write(f"**Recommendation:** {suggestion['recommendation']}")
            else:
                st.info("No automated suggestions generated. Your portfolio looks efficient, or more data is needed.")

            st.markdown("---")

            # --- New Gemini-powered roadmap ---
            st.subheader("🤖 AI-Powered Carbon Reduction Roadmap")

            if gemini_api_key:
                if st.button("✨ Generate Detailed Improvement Roadmap with Gemini"):
                    # Prepare data for the prompt
                    summary_stats = {
                        'total_mines': len(processed_df),
                        'total_production_mt': processed_df['Coal/ Lignite Production (MT) (2019-2020)'].sum(),
                        'total_emissions_mt_co2': processed_df['Total_Emissions_MT'].sum(),
                        'avg_emission_intensity': processed_df['Emission_Intensity'].mean(),
                        'avg_green_score': processed_df['Green_Score'].mean(),
                        'mine_types_present': processed_df['Type of Mine (OC/UG/Mixed)'].unique().tolist(),
                        'states_present': processed_df['State/UT Name'].unique().tolist()
                    }
                    df_summary_str = json.dumps(summary_stats, indent=2)
                    top_emitters_str = processed_df.nlargest(5, 'Total_Emissions_MT')[['Mine Name', 'Total_Emissions_MT', 'Emission_Intensity']].to_string()

                    with st.spinner("🤖 Generating your strategic roadmap... This may take a moment."):
                        roadmap = gemini_client.generate_improvement_roadmap(df_summary_str, top_emitters_str)
                        st.session_state[f'generated_roadmap_{username}'] = roadmap

                if f'generated_roadmap_{username}' in st.session_state:
                    st.markdown(st.session_state[f'generated_roadmap_{username}'])
                    st.download_button(
                        label="📥 Download Roadmap",
                        data=st.session_state[f'generated_roadmap_{username}'],
                        file_name=f"carbon_reduction_roadmap_{username}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
            else:
                st.warning("Please provide a Gemini API key in the sidebar to generate an AI-powered roadmap.")
        
        with tab3:
            st.header("🤖 AI Carbon Assistant")
            
            if ai_client:
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                # Predefined questions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🎯 Top Emission Sources"):
                        question = "What are my top 3 emission sources and how can I reduce them?"
                        st.session_state.chat_history.append(("user", question))
                
                with col2:
                    if st.button("💡 Improvement Ideas"):
                        question = "Give me specific improvement recommendations for my worst performing mines"
                        st.session_state.chat_history.append(("user", question))
                
                with col3:
                    if st.button("📈 Benchmarking"):
                        question = "How do my mines compare to industry benchmarks?"
                        st.session_state.chat_history.append(("user", question))
                
                # Custom question
                user_question = st.text_input("Ask about your data:")
                if st.button("Ask") and user_question:
                    st.session_state.chat_history.append(("user", user_question))
                
                # Process questions
                if st.session_state.chat_history:
                    latest_question = st.session_state.chat_history[-1]
                    if latest_question[0] == "user":
                        # Prepare context
                        context = f"""
                        You are a professional carbon emissions analyst. Analyze this coal mine data:
                        
                        Dataset Summary:
                        - Total mines: {len(processed_df)}
                        - Total production: {total_production:,.0f} MT
                        - Total emissions: {total_emissions:,.0f} MT CO₂
                        - Average intensity: {avg_intensity:.3f} MT CO₂/MT
                        - Average green score: {avg_green_score:.1f}
                        
                        Mine Details:
                        {processed_df[['Mine Name', 'State/UT Name', 'Type of Mine (OC/UG/Mixed)', 
                                     'Total_Emissions_MT', 'Emission_Intensity', 'Green_Score']].to_string()}
                        
                        Question: {latest_question[1]}
                        
                        Provide specific, actionable recommendations based on the data.
                        """
                        
                        with st.spinner("🤖 AI analyzing your data..."):
                            response = ai_client.chat([{"role": "user", "content": context}])
                            st.session_state.chat_history.append(("assistant", response))
                
                # Display chat
                for role, message in st.session_state.chat_history[-6:]:
                    if role == "user":
                        st.markdown(f"**You:** {message}")
                    else:
                        st.markdown(f'<div class="chat-message"><strong>🤖 AI Assistant:</strong><br>{message}</div>', 
                                  unsafe_allow_html=True)
            else:
                st.info("Please provide OpenRouter API key to use AI assistant")
        
        with tab4:
            st.header("🎯 What-if Scenario Simulator")
            st.write("Use the simulator to model how different operational strategies impact the emissions trend.")

            # Fetch the latest 20 real-time data points as a baseline
            conn = sqlite3.connect('global_data.db')
            historical_df = pd.read_sql_query("SELECT * FROM realtime_data ORDER BY timestamp DESC LIMIT 20", conn)
            conn.close()

            if historical_df.empty:
                st.warning("No real-time data available. Please run the 'Real-time' tab first to generate data.")
            else:
                # Ensure data is sorted chronologically for plotting
                historical_df = historical_df.sort_values(by='timestamp').reset_index(drop=True)
                
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader("⚙️ Simulator Controls")
                    
                    # Use st.session_state to make sliders interactive
                    sim_settings = st.session_state.simulation_settings
                    sim_settings['fuel_usage'] = st.slider("Fuel Usage (%)", 50, 150, sim_settings['fuel_usage'])
                    sim_settings['electricity_usage'] = st.slider("Electricity Usage (%)", 50, 150, sim_settings['electricity_usage'])
                    sim_settings['transport_activity'] = st.slider("Transport Activity (%)", 50, 150, sim_settings['transport_activity'])
                    sim_settings['equipment_efficiency'] = st.slider("Equipment Efficiency (%)", 60, 95, sim_settings['equipment_efficiency'])
                    sim_settings['carbon_offset_programs'] = st.slider("Carbon Offset Programs (%)", 50, 200, sim_settings['carbon_offset_programs'])

                    # --- Calculations for metrics based on the LATEST point ---
                    latest_original = historical_df.iloc[-1]
                    
                    # This assumes the historical data was generated with default efficiency of 85
                    default_efficiency = 85
                    
                    # Projected scopes for the latest point
                    proj_scope1 = latest_original['scope1'] * (sim_settings['fuel_usage'] / 100) * (default_efficiency / sim_settings['equipment_efficiency'])
                    proj_scope2 = latest_original['scope2'] * (sim_settings['electricity_usage'] / 100)
                    proj_scope3 = latest_original['scope3'] * (sim_settings['transport_activity'] / 100)

                    original_total = latest_original['scope1'] + latest_original['scope2'] + latest_original['scope3']
                    new_total = proj_scope1 + proj_scope2 + proj_scope3
                    savings = original_total - new_total

                    st.subheader("Projected Impact (on latest data point)")
                    st.metric("Original Gross Emissions", f"{original_total:.1f} tCO₂")
                    st.metric("Projected New Emissions", f"{new_total:.1f} tCO₂", f"{-savings:.1f} tCO₂")
                    st.metric("Total Reduction", f"{savings:.1f} tCO₂ ({savings/original_total*100:.1f}%)" if original_total != 0 else "N/A")

                with col2:
                    st.subheader("Impact Visualization Over Time")

                    # --- Calculations for the line chart ---
                    scenario_df = historical_df.copy()
                    
                    # Apply the same projection logic to the entire historical dataframe
                    scenario_df['scope1'] = scenario_df['scope1'] * (sim_settings['fuel_usage'] / 100) * (default_efficiency / sim_settings['equipment_efficiency'])
                    scenario_df['scope2'] = scenario_df['scope2'] * (sim_settings['electricity_usage'] / 100)
                    scenario_df['scope3'] = scenario_df['scope3'] * (sim_settings['transport_activity'] / 100)

                    # Calculate gross emissions for both original and scenario
                    historical_df['Gross Emissions'] = historical_df['scope1'] + historical_df['scope2'] + historical_df['scope3']
                    scenario_df['Gross Emissions'] = scenario_df['scope1'] + scenario_df['scope2'] + scenario_df['scope3']

                    # Prepare data for plotting
                    plot_df_original = historical_df[['timestamp', 'Gross Emissions']].copy()
                    plot_df_original['Scenario'] = 'Actual Path'
                    
                    plot_df_scenario = scenario_df[['timestamp', 'Gross Emissions']].copy()
                    plot_df_scenario['Scenario'] = 'Projected Path'

                    plot_df = pd.concat([plot_df_original, plot_df_scenario])

                    fig = px.line(plot_df,
                                  x='timestamp', y='Gross Emissions', color='Scenario',
                                  title="Emissions Trend: Actual vs. Projected",
                                  labels={'Gross Emissions': 'Gross Emissions (tCO₂)', 'timestamp': 'Time'},
                                  color_discrete_map={'Actual Path': '#d62728', 'Projected Path': '#2ca02c'})
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.header("📋 AI-Powered Reports")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Report Configuration")
                
                report_type = st.selectbox(
                    "Report Type",
                    ["Executive Summary", "Technical Analysis", "Compliance Report", "Investment Proposal"]
                )
                
                selected_states = st.multiselect(
                    "Focus States",
                    processed_df['State/UT Name'].unique(),
                    default=processed_df['State/UT Name'].unique()[:3] if len(processed_df) > 0 else []
                )
                
                if st.button("Generate Report with Gemini"):
                    if gemini_api_key:
                        filtered_df = processed_df[processed_df['State/UT Name'].isin(selected_states)]
                        summary = {
                            'user': username,
                            'total_mines': len(filtered_df),
                            'total_production': filtered_df['Coal/ Lignite Production (MT) (2019-2020)'].sum(),
                            'total_emissions': filtered_df['Total_Emissions_MT'].sum(),
                            'top_emitters': filtered_df.nlargest(5, 'Total_Emissions_MT')['Mine Name'].tolist(),
                            'avg_intensity': filtered_df['Emission_Intensity'].mean(),
                            'states': selected_states
                        }
                        
                        with st.spinner("🤖 Generating comprehensive report..."):
                            report = gemini_client.generate_report(summary, report_type.lower())
                            st.session_state[f'generated_report_{username}'] = report
                    else:
                        st.error("Please provide Gemini API key to generate reports")
            
            with col2:
                st.subheader("Generated Report")
                
                if f'generated_report_{username}' in st.session_state:
                    st.markdown(st.session_state[f'generated_report_{username}'])
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Report",
                        data=st.session_state[f'generated_report_{username}'],
                        file_name=f"coal_mine_report_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("Generate a report to see it here")
        
        with tab6:
            st.header("🏆 Leaderboard")
            
            # Personal leaderboard
            st.subheader("Your Mines Performance")
            if len(processed_df) > 0:
                personal_board = processed_df.nlargest(10, 'Green_Score')[
                    ['Mine Name', 'State/UT Name', 'Green_Score', 'Emission_Intensity', 'Type of Mine (OC/UG/Mixed)']
                ]
                st.dataframe(personal_board, use_container_width=True)
            else:
                st.info("No personal data available")
            
            # Global leaderboard
            st.subheader("🌍 Global Leaderboard")
            conn = sqlite3.connect('global_data.db')
            global_board = pd.read_sql_query('''
                SELECT username, mine_name, state, green_score, emission_intensity, mine_type
                FROM global_leaderboard 
                ORDER BY green_score DESC 
                LIMIT 20
            ''', conn)
            conn.close()
            
            if not global_board.empty:
                st.dataframe(global_board, use_container_width=True)
            else:
                st.info("No global data available yet")
        
        with tab7:
            st.header("📡 Smart Carbon Monitor")
            st.write("Real-time emissions tracking for coal mining operations")

            # The What-if simulator controls have been moved to the 'Scenarios' tab.

            # --- Auto-refresh control ---
            auto_refresh = st.checkbox("Enable Automatic Real-time Tracking (every 2s)")

            if auto_refresh:
                data = RealtimeDataGenerator.generate_realtime_data(st.session_state.simulation_settings)
                RealtimeDataGenerator.save_realtime_data(data)

            # --- Display logic ---
            conn = sqlite3.connect('global_data.db')
            historical_df = pd.read_sql_query("SELECT * FROM realtime_data ORDER BY timestamp DESC LIMIT 20", conn)
            conn.close()

            if not historical_df.empty:
                latest_data = historical_df.iloc[0]
                
                # --- Key Metrics ---
                st.subheader("Key Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Emissions", f"{latest_data['total_emissions']:.1f} tCO₂")
                with col2:
                    status_val = latest_data['total_emissions']
                    status_text = "Critical" if status_val > 100 else "High" if status_val > 75 else "Normal"
                    st.metric("Status", status_text)
                with col3:
                    st.metric("Carbon Sink", f"{abs(latest_data['carbon_sink']):.1f} tCO₂", "Absorbed")
                with col4:
                    st.metric("Carbon Offset", f"{abs(latest_data['carbon_offset']):.1f} tCO₂", "Purchased")
                with col5:
                    st.metric("Efficiency", f"{st.session_state.simulation_settings['equipment_efficiency']}%")

                # --- Charts ---
                st.subheader("Visualizations")
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.markdown("###### Real-time Emissions Trend")
                    line_df = historical_df.sort_values(by='timestamp')
                    fig_line = px.line(line_df, x='timestamp', y=['scope1', 'scope2', 'scope3', 'carbon_sink', 'carbon_offset'],
                                       labels={"timestamp": "Time", "value": "tCO₂", "variable": "Metric"})
                    fig_line.update_layout(legend_title_text='Metrics')
                    st.plotly_chart(fig_line, use_container_width=True)

                with chart_col2:
                    st.markdown("###### Current Emissions Breakdown")
                    scope_data = {
                        'Scope 1': latest_data['scope1'],
                        'Scope 2': latest_data['scope2'],
                        'Scope 3': latest_data['scope3'],
                        'Carbon Sink': abs(latest_data['carbon_sink']),
                    }
                    pie_df = pd.DataFrame(list(scope_data.items()), columns=['Scope', 'Emissions'])
                    fig_pie = px.pie(pie_df, values='Emissions', names='Scope')
                    st.plotly_chart(fig_pie, use_container_width=True)

                # --- Reduction Suggestions ---
                st.subheader("Reduction Suggestions")
                suggestions = [
                    "Optimize equipment maintenance schedule (5-8% reduction)",
                    "Switch to renewable energy sources (15-20% reduction)",
                    "Implement electric vehicle fleet (10-12% reduction)",
                    "Expand reforestation program (8-10% offset)",
                    "Purchase verified carbon credits (12-15% offset)",
                    "Install carbon capture technology (20-25% reduction)"
                ]
                for s in suggestions:
                    st.info(f"💡 {s}")

            else:
                st.info("Enable automatic tracking to see real-time data. If you just started, it might take a moment for the first data point to appear.")

            if auto_refresh:
                time.sleep(2)
                st.rerun()
    
    else:
        st.info("👆 Please upload your Coal Mine Dataset to get started!")
        
        # Show expected format
        st.subheader("📋 Expected Data Format")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Required Columns:**")
            st.markdown("- Mine Name")
            st.markdown("- State/UT Name") 
            st.markdown("- Type of Mine (OC/UG/Mixed)")
        
        with col2:
            st.markdown("**Optional Columns:**")
            st.markdown("- Coal/ Lignite Production (MT) (2019-2020)")
            st.markdown("- Transport Distance (km)")
            st.markdown("- Diesel Consumption (L)")
            st.markdown("- Electricity Consumption (kWh)")
            st.markdown("- Latitude, Longitude")
        
        # Sample data
        sample_data = {
            'Mine Name': ['Green Valley Mine', 'Eco Mining Co.', 'Carbon Neutral Mine'],
            'State/UT Name': ['Odisha', 'Jharkhand', 'Chhattisgarh'],
            'Type of Mine (OC/UG/Mixed)': ['OC', 'UG', 'Mixed'],
            'Coal/ Lignite Production (MT) (2019-2020)': [1000, 1500, 800],
            'Transport Distance (km)': [120, 180, 150],
            'Latitude': [20.5, 23.3, 21.2],
            'Longitude': [85.1, 85.9, 81.6]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        
        # Manual input option
        st.subheader("🖊️ Manual Data Entry")
        with st.expander("Add mine data manually"):
            with st.form("manual_entry"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    mine_name = st.text_input("Mine Name")
                    state = st.selectbox("State", ['Odisha', 'Jharkhand', 'Chhattisgarh', 
                                                  'West Bengal', 'Madhya Pradesh', 'Other'])
                
                with col2:
                    mine_type = st.selectbox("Mine Type", ['OC', 'UG', 'Mixed'])
                    production = st.number_input("Production (MT)", min_value=0.0, value=1000.0)
                
                with col3:
                    transport_distance = st.number_input("Transport Distance (km)", min_value=0.0, value=150.0)
                    make_public_manual = st.checkbox("Share with global leaderboard")
                
                if st.form_submit_button("Add Mine Data"):
                    if mine_name and state and mine_type:
                        # Create manual data entry
                        manual_data = {
                            'Mine Name': [mine_name],
                            'State/UT Name': [state],
                            'Type of Mine (OC/UG/Mixed)': [mine_type],
                            'Coal/ Lignite Production (MT) (2019-2020)': [production],
                            'Transport Distance (km)': [transport_distance],
                            'Latitude': [20.0],
                            'Longitude': [85.0]
                        }
                        
                        manual_df = pd.DataFrame(manual_data)
                        processor = CoalMineDataProcessor()
                        processed_manual, _ = processor.validate_and_process_data(manual_df)
                        
                        # Save to database
                        conn = sqlite3.connect(f'user_{username}.db')
                        cursor = conn.cursor()
                        row = processed_manual.iloc[0]
                        cursor.execute('''
                            INSERT INTO user_data 
                            (mine_name, state, production, mine_type, base_emissions, 
                             transport_emissions, total_emissions, emission_intensity, 
                             green_score, latitude, longitude, is_public)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (row['Mine Name'], row['State/UT Name'], 
                              row['Coal/ Lignite Production (MT) (2019-2020)'],
                              row['Type of Mine (OC/UG/Mixed)'], row['Base_Emissions_MT'],
                              row['Transport_Emissions_MT'], row['Total_Emissions_MT'],
                              row['Emission_Intensity'], row['Green_Score'],
                              row['Latitude'], row['Longitude'], make_public_manual))
                        conn.commit()
                        conn.close()
                        
                        # Add to global leaderboard if public
                        if make_public_manual:
                            conn = sqlite3.connect('global_data.db')
                            cursor = conn.cursor()
                            cursor.execute('''
                                INSERT INTO global_leaderboard 
                                (username, mine_name, state, green_score, emission_intensity, 
                                 total_emissions, mine_type)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            ''', (username, row['Mine Name'], row['State/UT Name'],
                                  row['Green_Score'], row['Emission_Intensity'],
                                  row['Total_Emissions_MT'], row['Type of Mine (OC/UG/Mixed)']))
                            conn.commit()
                            conn.close()
                        
                        st.success(f"✅ Added {mine_name} to your database!")
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields")

def main():
    """Main application"""
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Initialize database
    DatabaseManager()
    
    # Show login or main app
    if not st.session_state.logged_in:
        login_page()
    else:
        user = st.session_state.user
        
        # Logout button
        col1, col2, col3 = st.columns([6, 1, 1])
        with col2:
            if st.button("🔄 Refresh"):
                st.rerun()
        with col3:
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.user = None
                st.rerun()
        
        # Show appropriate interface
        if user['is_admin']:
            st.markdown(f'<span class="admin-badge">ADMIN</span> Welcome, {user["username"]}!', 
                       unsafe_allow_html=True)
            admin_panel()
        else:
            user_dashboard(user['username'])

if __name__ == "__main__":
    main()
