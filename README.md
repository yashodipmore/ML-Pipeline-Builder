# ML-Pipeline-Builder

A no-code tool to visually build, run, and manage machine learning pipelines.  
Features both a React frontend and a Flask backend, allowing users to upload data, preprocess, train models, and view results—without writing code.

---

## Features

- **No-code interface**: Easily create machine learning pipelines via a user-friendly web app.
- **Data upload and exploration**: Supports CSV and Excel uploads for dataset management.
- **Preprocessing**: Apply common preprocessing techniques like scaling and encoding.
- **Model training**: Train and evaluate models such as Logistic Regression and Decision Trees.
- **Session management**: All actions are sandboxed in isolated sessions.
- **Visualizations**: Results and statistics can be visualized interactively (frontend-powered).

---

## Directory Structure

```
ML-Pipeline-Builder/
├── backend/     # Flask server and ML logic (Python)
│   ├── app.py
│   ├── requirements.txt
│   ├── sample_data.csv
│   └── ...
├── frontend/    # React web application (JavaScript)
│   └── package.json
├── Dockerfile
├── Procfile
├── render.yaml
├── requirements.txt
├── vercel.json
└── runtime.txt
```

---

## Backend

- **Framework:** Flask (Python)
- **Key Libraries:** pandas, numpy, scikit-learn, flask-cors, werkzeug
- **REST API**: Provides endpoints for pipeline session creation, file upload, and ML operations.
- **Requirements:** See [`backend/requirements.txt`](backend/requirements.txt)

To run the backend:
```sh
cd backend
pip install -r requirements.txt
python app.py
```

---

## Frontend

- **Framework:** React (JavaScript)
- **Key Libraries:** axios, lucide-react, recharts
- **Scripts:** Start with `npm start` (see [`frontend/package.json`](frontend/package.json))

To run the frontend:
```sh
cd frontend
npm install
npm start
```

---

## Example Workflow

1. Launch backend (`python backend/app.py`).
2. Launch frontend (`npm start` in the `frontend` directory).
3. Access the app in the browser.
4. Create a new session, upload your data, explore it, preprocess, and try different models.
5. View and visualize evaluation metrics (accuracy, precision, recall, etc).

---

## Deployment

- Has configuration for Docker, Render, and Vercel out-of-the-box.
- Customizable via `render.yaml`, `Dockerfile`, and other environment configs.

---

## License

[Specify your license here if applicable]

---

## Contributing

Pull requests and issues welcome!

---

**Repo:** [yashodipmore/ML-Pipeline-Builder](https://github.com/yashodipmore/ML-Pipeline-Builder)
