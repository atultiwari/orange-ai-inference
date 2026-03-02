# Orange Data Mining Predictor App

This repository houses a modular Full-Stack application that lets you upload an Orange Data Mining classification model (`.pkcls`), auto-generates a dynamic input form based on the model's features, and provides real-time predictions with class probabilities.

## 👨‍💻 Developer Information

- **Author**: Dr. Atul Tiwari
- **Email**: atultiwari.in@gmail.com
- **WhatsApp**: +91-9636143787
- **Website**: [https://atultiwari.in](https://atultiwari.in)

## 📁 Project Structure

```text
.
├── backend/                  # FastAPI Python application
│   ├── main.py               # REST API endpoints (upload, predict)
│   ├── requirements.txt      # Python dependencies
│   └── uploaded_models/      # Temporary storage for dynamically uploaded models 
├── frontend/                 # Vite + React + Tailwind CSS application
│   ├── src/                  # React components and styling UI
│   ├── package.json          # Node.js dependencies
│   └── tailwind.config.js    # Tailwind v3 config
├── demo_models/              # Sample Orange models used for testing
│   ├── demo_01.pkcls         # Iris Dataset Model
│   └── demo_02_heart.pkcls   # Heart Disease Dataset Model
└── README.md
```

## 🚀 Setup & Installation

You can run this project locally using either **Docker** (recommended) or **Native Python/Node**.

### Option A: Running with Docker (Recommended)
This approach mimics the production environment using Nginx as a reverse proxy for the API.

1. Ensure Docker and Docker Compose are installed on your machine.
2. Open your terminal in the root of the project.
3. Run the following command:
```bash
docker-compose up --build
```
4. Once the containers are running, access the React frontend at: http://localhost
   * *Note: The backend API runs internally on port 8080. Nginx handles the `/api` routing automatically.*

### Option B: Running Natively (For Development)

#### 1. Start the Backend API
The backend relies on `Orange3` to parse the underlying model files accurately. 

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```
The backend API documentation (Swagger UI) will be available at: http://localhost:8080/docs

#### 2. Start the Frontend Application
```bash
cd frontend
npm install
npm run dev
```
The React frontend will be available at: http://localhost:5173

## ☁️ Deployment (Hostinger VPS via Coolify)

This repository is optimized for one-click deployment via **Coolify**.
1. Add this repository to your Coolify dashboard.
2. Select **Docker Compose** as the build pack.
3. Set the Docker Compose Location to `/docker-compose.prod.yml`. This production override file removes local host port bindings, allowing Coolify to safely route external traffic without conflicts.
4. Coolify will automatically build the multi-stage Nginx frontend, spin up the Python backend, and attach the necessary persistent volumes.

## 🧠 Usage

1. Open the frontend URL in your browser.
2. Drag and drop any `.pkcls` model (like the ones found in the `demo_models/` folder).
3. The app will parse the model metadata and render the corresponding feature input form.
4. Input your test data and click **Generate Prediction** for real-time model results. 

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
