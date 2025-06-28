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
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings('ignore')

# --- FastAPI Setup ---
app = FastAPI(
    title="Coal-Mine Carbon Neutrality API",
    description="API for Coal-Mine Carbon Neutrality Co-Pilot functionalities.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Security ---
SECRET_KEY = "a_very_secret_key"  # In production, use environment variables
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    username: str
    is_admin: bool

class UserInDB(User):
    hashed_password: str

# --- Database Manager ---
class DatabaseManager:
    def __init__(self):
        self.init_databases()
    
    def init_databases(self):
        """Initialize user and global databases"""
        conn = sqlite3.connect('global_data.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT,
                is_admin BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
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
        try:
            cursor.execute("SELECT production FROM realtime_data LIMIT 1")
            cursor.execute("DROP TABLE realtime_data")
            conn.commit()
        except sqlite3.OperationalError:
            pass
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
    
    def get_user(self, username: str):
        conn = sqlite3.connect('global_data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT username, password_hash, is_admin FROM users WHERE username = ?', (username,))
        user_data = cursor.fetchone()
        conn.close()
        if user_data:
            return {"username": user_data[0], "hashed_password": user_data[1], "is_admin": bool(user_data[2])}
        return None

# --- Authentication ---
def verify_password(plain_password, hashed_password):
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    db_manager = DatabaseManager()
    user = db_manager.get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return User(username=user['username'], is_admin=user['is_admin'])

# --- API Endpoints ---
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    db_manager = DatabaseManager()
    user = db_manager.get_user(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# Placeholder for other classes and endpoints
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
    
    def validate_and_process_data(self, df, settings=None):
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
        df = self._add_calculated_columns(df, missing_optional, settings)
        
        return df, missing_optional
    
    def _add_calculated_columns(self, df, missing_columns, settings=None):
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
        df['Base_Emissions_MT'] = df.apply(lambda row: self.calculate_base_emissions(row, settings), axis=1)
        df['Transport_Emissions_MT'] = df.apply(lambda row: self.calculate_transport_emissions(row, settings), axis=1)
        
        df['Total_Emissions_MT'] = df['Base_Emissions_MT'] + df['Transport_Emissions_MT']
        df['Emission_Intensity'] = df['Total_Emissions_MT'] / (df['Coal/ Lignite Production (MT) (2019-2020)'] + 0.001)
        df['Green_Score'] = 100 - (df['Emission_Intensity'] * 100)
        df['Green_Score'] = df['Green_Score'].clip(0, 100)
        
        return df
    
    def calculate_base_emissions(self, row, settings=None):
        """Calculate base emissions"""
        production = row['Coal/ Lignite Production (MT) (2019-2020)']
        mine_type = row['Type of Mine (OC/UG/Mixed)']
        state = row['State/UT Name']
        
        if pd.isna(production) or production == 0:
            return 0
        
        base_factor = self.emission_factors.get(mine_type, self.emission_factors['Mixed'])['base']
        regional_factor = self.regional_factors.get(state, 1.0)
        
        emissions = production * base_factor * regional_factor

        if settings:
            emissions *= (settings.get('fuel_usage', 100) / 100)
            emissions *= (100 / settings.get('equipment_efficiency', 85))

        return emissions
    
    def calculate_transport_emissions(self, row, settings=None):
        """Calculate transport emissions"""
        production = row['Coal/ Lignite Production (MT) (2019-2020)']
        distance = row.get('Transport Distance (km)', 150)
        
        if pd.isna(production) or production == 0:
            return 0
        
        truck_capacity = 25
        emission_per_km_per_mt = 0.12
        
        transport_emissions = (production / truck_capacity) * distance * emission_per_km_per_mt
        
        if settings:
            transport_emissions *= (settings.get('transport_activity', 100) / 100)

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
    def generate_synthetic_data():
        """Generate a synthetic dataset for a single company with 3-5 mines in India."""
        company_name = "Bharat Coal Enterprises"
        mines = [
            {'name': 'Jharia Coal Block', 'state': 'Jharkhand', 'type': 'UG'},
            {'name': 'Raniganj Coalfield', 'state': 'West Bengal', 'type': 'UG'},
            {'name': 'Talcher Coalfield', 'state': 'Odisha', 'type': 'OC'},
            {'name': 'Singareni Collieries', 'state': 'Telangana', 'type': 'Mixed'},
            {'name': 'Korba Coalfield', 'state': 'Chhattisgarh', 'type': 'OC'}
        ]
        
        num_mines = random.randint(3, 5)
        selected_mines = random.sample(mines, num_mines)
        
        data = []
        for mine in selected_mines:
            production = random.randint(1000, 8000) * 1000  # In MT
            distance = random.randint(50, 500)
            
            if mine['state'] == 'Jharkhand':
                lat, lon = random.uniform(23.6, 24.4), random.uniform(85.0, 86.5)
            elif mine['state'] == 'West Bengal':
                lat, lon = random.uniform(22.5, 23.5), random.uniform(86.5, 87.5)
            elif mine['state'] == 'Odisha':
                lat, lon = random.uniform(20.5, 22.0), random.uniform(83.5, 85.0)
            elif mine['state'] == 'Telangana':
                lat, lon = random.uniform(17.5, 18.5), random.uniform(79.0, 80.0)
            else:  # Chhattisgarh
                lat, lon = random.uniform(21.2, 23.0), random.uniform(81.0, 83.0)

            data.append({
                'Mine Name': f"{company_name} - {mine['name']}",
                'State/UT Name': mine['state'],
                'Type of Mine (OC/UG/Mixed)': mine['type'],
                'Coal/ Lignite Production (MT) (2019-2020)': production,
                'Transport Distance (km)': distance,
                'Latitude': lat,
                'Longitude': lon
            })
            
        return pd.DataFrame(data)

# TODO: Add other endpoints here
@app.post("/process-data/")
async def process_data(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV or XLSX file.")

        processor = CoalMineDataProcessor()
        processed_df, missing_columns = processor.validate_and_process_data(df)

        if processed_df is None:
            raise HTTPException(status_code=400, detail=missing_columns)

        # Save to user database
        db_manager = DatabaseManager()
        db_manager.create_user_database(current_user.username)
        conn = sqlite3.connect(f'user_{current_user.username}.db')
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
                  row.get('Latitude', 0), row.get('Longitude', 0), False))
        conn.commit()
        conn.close()

        return {"message": "Data processed and saved successfully!", "missing_columns": missing_columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate-synthetic-data/")
async def get_synthetic_data(current_user: User = Depends(get_current_user)):
    synthetic_data = SyntheticDataGenerator.generate_synthetic_data()
    return synthetic_data.to_dict(orient="records")

@app.get("/data/{username}")
async def get_user_data(username: str, current_user: User = Depends(get_current_user)):
    if current_user.username != username and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized to access this data")
    
    conn = sqlite3.connect(f'user_{username}.db')
    try:
        user_data = pd.read_sql_query('SELECT * FROM user_data', conn)
        return user_data.to_dict(orient="records")
    except pd.io.sql.DatabaseError:
        return []
    finally:
        conn.close()

@app.get("/leaderboard")
async def get_leaderboard():
    conn = sqlite3.connect('global_data.db')
    leaderboard_df = pd.read_sql_query('''
        SELECT username, mine_name, state, green_score, emission_intensity,
               total_emissions, mine_type, updated_at
        FROM global_leaderboard
        ORDER BY green_score DESC
    ''', conn)
    conn.close()
    return leaderboard_df.to_dict(orient="records")

class ChatRequest(BaseModel):
    question: str
    api_key: str

@app.post("/chat")
async def chat_with_ai(request: ChatRequest, current_user: User = Depends(get_current_user)):
    if not request.api_key:
        raise HTTPException(status_code=400, detail="OpenRouter API key is required.")
    
    ai_client = OpenRouterClient(api_key=request.api_key)
    
    # Fetch user data to provide context
    conn = sqlite3.connect(f'user_{current_user.username}.db')
    try:
        processed_df = pd.read_sql_query('SELECT * FROM user_data', conn)
    except pd.io.sql.DatabaseError:
        processed_df = pd.DataFrame()
    finally:
        conn.close()

    if processed_df.empty:
        context = f"The user has no data yet. Please ask them to upload some data. Question: {request.question}"
    else:
        total_production = processed_df['production'].sum()
        total_emissions = processed_df['total_emissions'].sum()
        avg_intensity = processed_df['emission_intensity'].mean()
        avg_green_score = processed_df['green_score'].mean()
        context = f"""
        You are a professional carbon emissions analyst. Your goal is to provide cost-effective and actionable advice. Analyze this coal mine data:
        
        Dataset Summary:
        - Total mines: {len(processed_df)}
        - Total production: {total_production:,.0f} MT
        - Total emissions: {total_emissions:,.0f} MT CO₂
        - Average intensity: {avg_intensity:.3f} MT CO₂/MT
        - Average green score: {avg_green_score:.1f}
        
        Mine Details:
        {processed_df[['mine_name', 'state', 'mine_type',
                     'total_emissions', 'emission_intensity', 'green_score']].to_string()}
        
        Question: {request.question}
        
        Provide specific, actionable, and cost-effective recommendations based on the data. For any suggestion, try to estimate the potential cost savings or return on investment.
        """
    
    response = ai_client.chat([{"role": "user", "content": context}])
    return {"response": response}

class ReportRequest(BaseModel):
    api_key: str
    report_type: str
    focus_states: list[str]

@app.post("/report")
async def generate_report(request: ReportRequest, current_user: User = Depends(get_current_user)):
    if not request.api_key:
        raise HTTPException(status_code=400, detail="Gemini API key is required.")

    gemini_client = GeminiClient(api_key=request.api_key)
    
    conn = sqlite3.connect(f'user_{current_user.username}.db')
    try:
        user_df = pd.read_sql_query('SELECT * FROM user_data', conn)
    except pd.io.sql.DatabaseError:
        user_df = pd.DataFrame()
    finally:
        conn.close()

    if user_df.empty:
        raise HTTPException(status_code=400, detail="No data available for this user to generate a report.")

    filtered_df = user_df[user_df['state'].isin(request.focus_states)]
    if filtered_df.empty:
        raise HTTPException(status_code=400, detail="No data for the selected states.")

    summary = {
        'user': current_user.username,
        'total_mines': len(filtered_df),
        'total_production': filtered_df['production'].sum(),
        'total_emissions': filtered_df['total_emissions'].sum(),
        'top_emitters': filtered_df.nlargest(5, 'total_emissions')['mine_name'].tolist(),
        'avg_intensity': filtered_df['emission_intensity'].mean(),
        'states': request.focus_states
    }
    
    report = gemini_client.generate_report(summary, request.report_type.lower())
    return {"report": report}

class SimulationSettings(BaseModel):
    fuel_usage: int = 100
    electricity_usage: int = 100
    transport_activity: int = 100
    equipment_efficiency: int = 85
    carbon_offset_programs: int = 100

@app.post("/realtime")
def get_realtime_data(settings: SimulationSettings):
    data = RealtimeDataGenerator.generate_realtime_data(settings.dict())
    RealtimeDataGenerator.save_realtime_data(data)
    return data

@app.post("/predict-scenario")
async def predict_scenario(settings: SimulationSettings, current_user: User = Depends(get_current_user)):
    conn = sqlite3.connect(f'user_{current_user.username}.db')
    try:
        user_df = pd.read_sql_query('SELECT * FROM user_data', conn)
    except pd.io.sql.DatabaseError:
        raise HTTPException(status_code=404, detail="No data found for this user. Please upload data first.")
    finally:
        conn.close()

    if user_df.empty:
        raise HTTPException(status_code=404, detail="No data found for this user. Please upload data first.")

    # Create a deep copy to avoid modifying the original dataframe in memory
    scenario_df = user_df.copy()

    # We need to rename columns to match what the processor expects
    scenario_df.rename(columns={
        'mine_name': 'Mine Name',
        'state': 'State/UT Name',
        'mine_type': 'Type of Mine (OC/UG/Mixed)',
        'production': 'Coal/ Lignite Production (MT) (2019-2020)',
    }, inplace=True)


    processor = CoalMineDataProcessor()
    # We pass the settings to the processor now
    processed_scenario_df, _ = processor.validate_and_process_data(scenario_df, settings.dict())
    
    original_total_emissions = user_df['total_emissions'].sum()
    predicted_total_emissions = processed_scenario_df['Total_Emissions_MT'].sum()

    return {
        "original_emissions": original_total_emissions,
        "predicted_emissions": predicted_total_emissions
    }


class RoadmapRequest(BaseModel):
    api_key: str

@app.post("/roadmap")
async def generate_roadmap(request: RoadmapRequest, current_user: User = Depends(get_current_user)):
    if not request.api_key:
        raise HTTPException(status_code=400, detail="Gemini API key is required.")

    gemini_client = GeminiClient(api_key=request.api_key)
    
    conn = sqlite3.connect(f'user_{current_user.username}.db')
    try:
        user_df = pd.read_sql_query('SELECT * FROM user_data', conn)
    except pd.io.sql.DatabaseError:
        user_df = pd.DataFrame()
    finally:
        conn.close()

    if user_df.empty:
        raise HTTPException(status_code=400, detail="No data available for this user to generate a roadmap.")

    df_summary = user_df.describe().to_string()
    top_emitters = user_df.nlargest(5, 'total_emissions')
    top_emitters_str = top_emitters[['mine_name', 'total_emissions', 'emission_intensity']].to_string()

    roadmap = gemini_client.generate_improvement_roadmap(df_summary, top_emitters_str)
    return {"roadmap": roadmap}

class ScenarioRangeRequest(BaseModel):
    settings: SimulationSettings
    vary_param: str

@app.post("/predict-scenario-range")
async def predict_scenario_range(request: ScenarioRangeRequest, current_user: User = Depends(get_current_user)):
    conn = sqlite3.connect(f'user_{current_user.username}.db')
    try:
        user_df = pd.read_sql_query('SELECT * FROM user_data', conn)
    except pd.io.sql.DatabaseError:
        raise HTTPException(status_code=404, detail="No data found for this user. Please upload data first.")
    finally:
        conn.close()

    if user_df.empty:
        raise HTTPException(status_code=404, detail="No data found for this user. Please upload data first.")

    processor = CoalMineDataProcessor()
    param_to_vary = request.vary_param
    settings_dict = request.settings.dict()
    results = []

    ranges = {
        "fuel_usage": np.linspace(50, 150, 20),
        "electricity_usage": np.linspace(50, 150, 20),
        "transport_activity": np.linspace(50, 150, 20),
        "equipment_efficiency": np.linspace(60, 95, 20),
        "carbon_offset_programs": np.linspace(50, 200, 20)
    }

    range_to_use = ranges.get(param_to_vary)
    if range_to_use is None:
        raise HTTPException(status_code=400, detail="Invalid parameter to vary.")

    for val in range_to_use:
        temp_settings = settings_dict.copy()
        temp_settings[param_to_vary] = val
        
        scenario_df = user_df.copy()
        scenario_df.rename(columns={
            'mine_name': 'Mine Name',
            'state': 'State/UT Name',
            'mine_type': 'Type of Mine (OC/UG/Mixed)',
            'production': 'Coal/ Lignite Production (MT) (2019-2020)',
        }, inplace=True)
        
        processed_df, _ = processor.validate_and_process_data(scenario_df, temp_settings)
        predicted_emissions = processed_df['Total_Emissions_MT'].sum()
        results.append({"x": val, "y": predicted_emissions})
        
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)