
# 🌍 Coal Mine Carbon Neutrality Co(al)-Pilot

## 1. 🏷️ Title of the Project  
**Coal Mine Carbon Neutrality CO(al)-Pilot** – An AI-Powered Dashboard for Sustainable Mining

## 2. 🧠 Abstract  
The **Coal Mine Carbon Neutrality Co-Pilot** is a comprehensive web-based application designed to assist mining companies in monitoring, analyzing, and reducing their carbon emissions. By integrating real-time data visualization 📊, scenario simulation ⚙️, and artificial intelligence 🤖, the platform empowers stakeholders with actionable insights to achieve carbon neutrality goals in line with national and global environmental commitments 🌱.

## 3. 📘 Introduction  
The mining industry plays a vital role in industrial development ⚒️ but also contributes significantly to greenhouse gas emissions 🌫️. With increasing environmental regulations and the push toward sustainability, it is imperative to implement intelligent systems that provide clarity and direction in reducing carbon footprints. This project presents a centralized AI-enabled system that bridges operational data and sustainability strategies.

## 4. ❗ Problem Statement  
Conventional emission tracking in mining is often reactive 🚫, lacking timely insights and holistic assessment. Existing systems are not equipped to offer proactive recommendations, comparative analysis, or real-time feedback. This leads to inefficient resource utilization and delayed policy interventions.

## 5. 🎯 Objectives  
- 🖥️ Develop a real-time dashboard to monitor emissions  
- 🔁 Enable stakeholders to simulate carbon reduction strategies  
- 💡 Use AI to recommend emission reduction pathways  
- 📄 Generate professional-grade reports for compliance and strategy

## 6. 🏗️ System Architecture  

### 🔧 Backend:
- **Framework**: FastAPI (Python) 🐍  
- **Database**: SQLite 🗄️  
- **Libraries**: Pandas, NumPy, Plotly, Scikit-learn, JWT, Passlib, OpenAI, Google Generative AI

### 🎨 Frontend:
- **Technologies**: HTML, TailwindCSS, JavaScript  
- **Visualization Tools**: Chart.js, Plotly.js

### 🚀 Hosting (Local Deployment):
- Backend runs on Uvicorn server at `http://127.0.0.1:8000`  
- Frontend runs in a browser 🌐 (static HTML file)

## 7. 🧩 Features  

### 🔐 7.1 Secure Login  
- Role-based authentication using JWT 🔑  
- Separate access for admin and users

### 📈 7.2 Interactive Dashboard  
- KPIs: production, emissions, green score, intensity  
- Dynamic graphs and charts via Plotly and Chart.js 📊

### 🤖 7.3 AI Chat Assistant  
- Powered by OpenRouter and Gemini APIs  
- Answers queries about mine data, strategies, and improvements 💬

### 📁 7.4 File Upload & Emission Calculation  
- Supports `.csv` and `.xlsx` uploads  
- Calculates emissions with:

```python
def calculate_base_emissions(row):
    production = row['Production']
    factor = emission_factors[row['Mine_Type']]
    return production * factor
```

### 🧮 7.5 Scenario Simulation  
- Sliders for adjusting fuel, electricity, transport & offsets  
- Live updates on emissions and cost forecasts 📉

### 📑 7.6 AI-Generated Reports  
- Creates summaries and improvement plans via Gemini AI 📝

### 🏆 7.7 Leaderboard  
- Ranks mines based on green scores  
- Shows user-wise and global performance charts 🥇

### 🔄 7.8 Real-time Monitoring  
- Simulated Scope 1, 2, 3 data  
- Color indicators, trends, and suggestions 🟢🟡🔴

## 8. 🔄 Data Flow and Processing  
1. User logs in 🔐  
2. Uploads file for processing 📤  
3. Backend calculates emissions & stores to SQLite 💾  
4. Dashboard displays visual analytics 📊  
5. AI and simulations offer enhanced feedback 🧠

## 9. 💻 Technologies Used

| 🧩 Component       | ⚙️ Technology          |
|-------------------|------------------------|
| Backend API       | FastAPI                |
| Frontend UI       | HTML, TailwindCSS      |
| Charts            | Plotly.js, Chart.js    |
| Authentication    | JWT, Passlib           |
| Database          | SQLite                 |
| AI Models         | Gemini, OpenRouter     |
| Data Handling     | Pandas, NumPy          |

## 10. 🛠️ Installation & Usage  

### 🔙 Backend Setup
```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

### 🖼️ Frontend
Open `index.html` in a browser 🌐

### 🔑 User Credentials (Demo)
- Admin: `admin/admin`  
- Users: `user1/user1`, `user2/user2`, `user3/user3`

## 11. 🧪 Sample Code Snippet – Token Generation  
```python
@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}
```

## 12. ✅ Results  
- Accurate emissions and green score calculations  
- Effective AI-generated reports 📄  
- Real-time updates and smooth simulations

## 13. ⚠️ Limitations  
- Local only, no production deployment  
- External API keys required for AI  
- No IoT sensor integration

## 14. 🔮 Future Work  
- Cloud deployment with live DB ☁️  
- IoT-based real-time emission capture 📡  
- Role permissions & audit logs 🧾  
- Support for other carbon-heavy sectors 🏭

## 15. 🏁 Conclusion  
The **Coal Mine Carbon Neutrality Co-Pilot** offers an intelligent, scalable solution for mining sustainability. By fusing real-time monitoring, AI insights, and intuitive design, it closes the gap between data and environmental decision-making 💚.

## 16. 📂 Repository  
All source code and documentation available at:  
🔗 [Insert GitHub Repository URL Here]
