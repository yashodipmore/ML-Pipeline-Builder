import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Upload, Database, Settings, Scissors, Brain, BarChart3, 
  CheckCircle2, AlertCircle, Loader2, ArrowRight, RefreshCw,
  FileSpreadsheet, TrendingUp, Layers, Target, Play, Sparkles,
  Activity, Zap, CircleDot
} from 'lucide-react';

// Use relative URL for production, localhost for development
const API_BASE = process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:5001/api';

// Pipeline Steps Configuration
const PIPELINE_STEPS = [
  { id: 1, title: 'Upload Data', subtitle: 'CSV or Excel', icon: Upload },
  { id: 2, title: 'Select Target', subtitle: 'Target column', icon: Target },
  { id: 3, title: 'Preprocess', subtitle: 'Scale features', icon: Settings },
  { id: 4, title: 'Train-Test Split', subtitle: 'Split data', icon: Scissors },
  { id: 5, title: 'Select Model', subtitle: 'ML algorithm', icon: Brain },
  { id: 6, title: 'Results', subtitle: 'View metrics', icon: BarChart3 },
];

function App() {
  // Session state
  const [sessionId, setSessionId] = useState(null);
  const [currentStep, setCurrentStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState(null);

  // Data state
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [preprocessing, setPreprocessing] = useState([]);
  const [splitRatio, setSplitRatio] = useState(80);
  const [splitInfo, setSplitInfo] = useState(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelParams, setModelParams] = useState({});
  const [results, setResults] = useState(null);

  // Step completion tracking
  const [completedSteps, setCompletedSteps] = useState([]);

  // Initialize session - use localStorage to persist session
  useEffect(() => {
    const existingSession = localStorage.getItem('ml_pipeline_session');
    if (existingSession) {
      setSessionId(existingSession);
    } else {
      createSession();
    }
  }, []);

  const createSession = async () => {
    try {
      const response = await axios.post(`${API_BASE}/session/create`);
      const newSessionId = response.data.session_id;
      setSessionId(newSessionId);
      localStorage.setItem('ml_pipeline_session', newSessionId);
    } catch (error) {
      showToast('Failed to initialize session', 'error');
    }
  };

  const showToast = (message, type = 'success') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  const markStepCompleted = (step) => {
    if (!completedSteps.includes(step)) {
      setCompletedSteps([...completedSteps, step]);
    }
  };

  // File Upload Handler
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    try {
      // Ensure we have a session ID
      let currentSessionId = sessionId;
      if (!currentSessionId) {
        const sessionResponse = await axios.post(`${API_BASE}/session/create`);
        currentSessionId = sessionResponse.data.session_id;
        setSessionId(currentSessionId);
        localStorage.setItem('ml_pipeline_session', currentSessionId);
      }

      const formData = new FormData();
      formData.append('file', file);
      formData.append('session_id', currentSessionId);

      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      console.log('Upload response:', response.data);
      if (response.data && response.data.column_info) {
        setDatasetInfo(response.data);
        markStepCompleted(1);
        setCurrentStep(2);
        showToast('Dataset uploaded successfully!');
      } else {
        showToast('Invalid response from server', 'error');
        setLoading(false);
      }
    } catch (error) {
      console.error('Upload error:', error);
      showToast(error.response?.data?.error || 'Upload failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Set Target Column
  const handleSetTarget = async () => {
    if (!targetColumn) {
      showToast('Please select a target column', 'error');
      return;
    }

    setLoading(true);
    try {
      await axios.post(`${API_BASE}/set-target`, {
        session_id: sessionId,
        target_column: targetColumn
      });
      markStepCompleted(2);
      setCurrentStep(3);
      showToast('Target column set!');
    } catch (error) {
      showToast(error.response?.data?.error || 'Failed to set target', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Apply Preprocessing
  const handlePreprocess = async (method) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/preprocess`, {
        session_id: sessionId,
        method: method
      });
      setPreprocessing(response.data.preprocessing_steps);
      setDatasetInfo(prev => ({ ...prev, preview: response.data.preview }));
      showToast(`${method === 'standardize' ? 'Standardization' : 'Normalization'} applied!`);
    } catch (error) {
      showToast(error.response?.data?.error || 'Preprocessing failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleResetPreprocessing = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/reset-preprocessing`, {
        session_id: sessionId
      });
      setPreprocessing([]);
      setDatasetInfo(prev => ({ ...prev, preview: response.data.preview }));
      showToast('Preprocessing reset!');
    } catch (error) {
      showToast(error.response?.data?.error || 'Reset failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const confirmPreprocessing = () => {
    markStepCompleted(3);
    setCurrentStep(4);
  };

  // Train-Test Split
  const handleSplit = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/split`, {
        session_id: sessionId,
        test_size: (100 - splitRatio) / 100
      });
      setSplitInfo(response.data.split_info);
      markStepCompleted(4);
      setCurrentStep(5);
      showToast('Data split successfully!');
    } catch (error) {
      showToast(error.response?.data?.error || 'Split failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Train Model
  const handleTrain = async () => {
    if (!selectedModel) {
      showToast('Please select a model', 'error');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/train`, {
        session_id: sessionId,
        model_type: selectedModel,
        model_params: modelParams
      });
      setResults(response.data.results);
      markStepCompleted(5);
      markStepCompleted(6);
      setCurrentStep(6);
      showToast('Model trained successfully!');
    } catch (error) {
      showToast(error.response?.data?.error || 'Training failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Reset Pipeline
  const handleReset = async () => {
    setLoading(true);
    try {
      await axios.post(`${API_BASE}/reset`, { session_id: sessionId });
      // Clear localStorage and create new session
      localStorage.removeItem('ml_pipeline_session');
      setDatasetInfo(null);
      setTargetColumn('');
      setPreprocessing([]);
      setSplitRatio(80);
      setSplitInfo(null);
      setSelectedModel('');
      setModelParams({});
      setResults(null);
      setCompletedSteps([]);
      setCurrentStep(1);
      // Create a new session
      await createSession();
      showToast('Pipeline reset!');
    } catch (error) {
      showToast('Reset failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Render Step Content
  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return <UploadStep 
          onUpload={handleFileUpload} 
          datasetInfo={datasetInfo}
          loading={loading}
        />;
      case 2:
        return <TargetStep 
          datasetInfo={datasetInfo}
          targetColumn={targetColumn}
          setTargetColumn={setTargetColumn}
          onConfirm={handleSetTarget}
          loading={loading}
        />;
      case 3:
        return <PreprocessStep 
          preprocessing={preprocessing}
          onPreprocess={handlePreprocess}
          onReset={handleResetPreprocessing}
          onConfirm={confirmPreprocessing}
          datasetInfo={datasetInfo}
          loading={loading}
        />;
      case 4:
        return <SplitStep 
          splitRatio={splitRatio}
          setSplitRatio={setSplitRatio}
          onSplit={handleSplit}
          splitInfo={splitInfo}
          loading={loading}
        />;
      case 5:
        return <ModelStep 
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          modelParams={modelParams}
          setModelParams={setModelParams}
          onTrain={handleTrain}
          loading={loading}
        />;
      case 6:
        return <ResultsStep 
          results={results}
          onReset={handleReset}
        />;
      default:
        return null;
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="header-logo">
            <Sparkles size={26} />
          </div>
          <div>
            <h1>ML Pipeline Builder</h1>
            <p>Build machine learning workflows without code</p>
          </div>
        </div>
        <div className="header-right">
          <div className="header-badge">
            <CircleDot size={14} />
            {sessionId ? 'Session Active' : 'Connecting...'}
          </div>
        </div>
      </header>

      <div className="pipeline-container">
        {/* Sidebar - Pipeline Steps */}
        <aside className="sidebar">
          <div className="pipeline-steps">
            <h3>Pipeline Steps</h3>
            {PIPELINE_STEPS.map((step) => (
              <div
                key={step.id}
                className={`step-item ${currentStep === step.id ? 'active' : ''} ${completedSteps.includes(step.id) ? 'completed' : ''}`}
                onClick={() => {
                  if (completedSteps.includes(step.id) || step.id <= Math.max(...completedSteps, 0) + 1) {
                    setCurrentStep(step.id);
                  }
                }}
              >
                <div className="step-number">
                  {completedSteps.includes(step.id) ? (
                    <CheckCircle2 size={18} />
                  ) : (
                    step.id
                  )}
                </div>
                <div className="step-content">
                  <div className="step-title">{step.title}</div>
                  <div className="step-subtitle">{step.subtitle}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Reset Button */}
          {datasetInfo && (
            <button 
              className="btn btn-danger" 
              onClick={handleReset}
              style={{ width: '100%' }}
            >
              <RefreshCw size={16} />
              Reset Pipeline
            </button>
          )}
        </aside>

        {/* Main Content */}
        <main className="main-content">
          {renderStepContent()}
        </main>
      </div>

      {/* Loading Overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Processing your request...</p>
        </div>
      )}

      {/* Toast Notification */}
      {toast && (
        <div className={`toast ${toast.type}`}>
          {toast.type === 'success' ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
          {toast.message}
        </div>
      )}
    </div>
  );
}

// Step 1: Upload Data
function UploadStep({ onUpload, datasetInfo, loading }) {
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload({ target: { files: e.dataTransfer.files } });
    }
  }, [onUpload]);

  return (
    <div className="step-panel">
      <div className="panel-header">
        <h2>
          <span className="icon"><Upload size={22} /></span>
          Upload Your Dataset
        </h2>
        <p>Upload a CSV or Excel file to get started with your ML pipeline</p>
      </div>

      <label
        className={`upload-zone ${dragActive ? 'dragging' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          onChange={onUpload}
          style={{ display: 'none' }}
          disabled={loading}
        />
        <div className="upload-icon">
          <FileSpreadsheet size={32} />
        </div>
        <h3>Drag & drop your file here</h3>
        <p>or click to browse files</p>
        <p className="file-types">Supported: CSV, Excel (.xlsx, .xls)</p>
      </label>

      {datasetInfo && datasetInfo.column_info && (
        <div className="dataset-info">
          <h4 className="section-title">Dataset Overview</h4>
          <div className="dataset-stats">
            <div className="stat-card">
              <div className="value" style={{ fontSize: '1rem' }}>{datasetInfo.filename}</div>
              <div className="label">File Name</div>
            </div>
            <div className="stat-card">
              <div className="value">{datasetInfo.rows?.toLocaleString() || 0}</div>
              <div className="label">Total Rows</div>
            </div>
            <div className="stat-card">
              <div className="value">{datasetInfo.columns || 0}</div>
              <div className="label">Columns</div>
            </div>
          </div>

          <h4 className="section-title">Column Details</h4>
          <div className="columns-list">
            {datasetInfo.column_info.map((col, idx) => (
              <div key={idx} className="column-item">
                <span className="column-name">{col.name}</span>
                <span className={`column-type ${col.type}`}>{col.type}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Step 2: Select Target
function TargetStep({ datasetInfo, targetColumn, setTargetColumn, onConfirm, loading }) {
  // Show loading message if no data
  if (!datasetInfo || !datasetInfo.column_info) {
    return (
      <div className="step-panel">
        <div className="panel-header">
          <h2>
            <span className="icon"><Target size={22} /></span>
            Select Target Column
          </h2>
          <p>Please upload a dataset first to select target column</p>
        </div>
        <div className="info-box">
          <p><strong>Note:</strong> No dataset loaded. Go back to Step 1 and upload a CSV or Excel file.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="step-panel">
      <div className="panel-header">
        <h2>
          <span className="icon"><Target size={22} /></span>
          Select Target Column
        </h2>
        <p>Choose the column you want to predict (the output variable)</p>
      </div>

      <div className="target-section">
        <h4>What is a Target Column?</h4>
        <p>
          The target column is what your model will learn to predict. For example, 
          if you're predicting whether a customer will churn, select the "churn" column.
        </p>
      </div>

      <div className="form-group">
        <label>Select Target Column</label>
        <select 
          className="form-select"
          value={targetColumn}
          onChange={(e) => setTargetColumn(e.target.value)}
        >
          <option value="">-- Select a column --</option>
          {datasetInfo.column_info.map((col, idx) => (
            <option key={idx} value={col.name}>
              {col.name} ({col.type} - {col.unique_count} unique values)
            </option>
          ))}
        </select>
      </div>

      {targetColumn && (
        <div className="info-box">
          <p>
            <strong>Selected:</strong> "{targetColumn}" - This column will be used as the target for classification.
          </p>
        </div>
      )}

      <div className="action-buttons">
        <button 
          className="btn btn-primary"
          onClick={onConfirm}
          disabled={!targetColumn || loading}
        >
          Continue <ArrowRight size={18} />
        </button>
      </div>
    </div>
  );
}

// Step 3: Preprocessing
function PreprocessStep({ preprocessing, onPreprocess, onReset, onConfirm, datasetInfo, loading }) {
  return (
    <div className="step-panel">
      <div className="panel-header">
        <h2>
          <span className="icon"><Settings size={22} /></span>
          Data Preprocessing
        </h2>
        <p>Scale your numerical features for better model performance</p>
      </div>

      <div className="option-cards">
        <div 
          className={`option-card ${preprocessing.some(p => p.method === 'StandardScaler') ? 'selected' : ''}`}
          onClick={() => !loading && onPreprocess('standardize')}
        >
          <div className="option-card-icon">
            <TrendingUp size={26} />
          </div>
          <h4>Standardization</h4>
          <p>StandardScaler - Zero mean, unit variance</p>
        </div>

        <div 
          className={`option-card ${preprocessing.some(p => p.method === 'MinMaxScaler') ? 'selected' : ''}`}
          onClick={() => !loading && onPreprocess('normalize')}
        >
          <div className="option-card-icon">
            <Layers size={26} />
          </div>
          <h4>Normalization</h4>
          <p>MinMaxScaler - Scale to 0-1 range</p>
        </div>
      </div>

      {preprocessing.length > 0 && (
        <div style={{ marginBottom: '24px' }}>
          <h4 className="section-title">Applied Preprocessing</h4>
          <div className="preprocessing-tags">
            {preprocessing.map((step, idx) => (
              <span key={idx} className="preprocessing-tag">
                <CheckCircle2 size={14} />
                {step.method}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="info-box">
        <p>
          <strong>Tip:</strong> Preprocessing is optional but recommended. You can apply multiple 
          transformations or skip this step entirely.
        </p>
      </div>

      <div className="action-buttons">
        {preprocessing.length > 0 && (
          <button className="btn btn-secondary" onClick={onReset}>
            <RefreshCw size={16} />
            Reset
          </button>
        )}
        <button className="btn btn-primary" onClick={onConfirm}>
          {preprocessing.length > 0 ? 'Continue' : 'Skip & Continue'} <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}

// Step 4: Train-Test Split
function SplitStep({ splitRatio, setSplitRatio, onSplit, splitInfo, loading }) {
  return (
    <div className="step-panel">
      <div className="panel-header">
        <h2>
          <span className="icon"><Scissors size={22} /></span>
          Train-Test Split
        </h2>
        <p>Divide your data into training and testing sets</p>
      </div>

      <div className="slider-container">
        <div className="slider-header">
          <span className="slider-label">Split Ratio</span>
          <span className="slider-value">{splitRatio}% Train / {100 - splitRatio}% Test</span>
        </div>
        <input
          type="range"
          className="slider"
          min="50"
          max="90"
          step="5"
          value={splitRatio}
          onChange={(e) => setSplitRatio(Number(e.target.value))}
          style={{ '--value': `${splitRatio}%` }}
        />
      </div>

      <div className="split-visualization">
        <div className="split-train" style={{ width: `${splitRatio}%` }}>
          Training ({splitRatio}%)
        </div>
        <div className="split-test" style={{ width: `${100 - splitRatio}%` }}>
          Test ({100 - splitRatio}%)
        </div>
      </div>

      <div className="info-box">
        <p>
          <strong>How it works:</strong> Your data will be randomly shuffled and split. 
          The training set is used to train the model, and the test set evaluates its performance.
        </p>
      </div>

      <div className="action-buttons">
        <button 
          className="btn btn-primary"
          onClick={onSplit}
          disabled={loading}
        >
          <Scissors size={16} />
          Split Data
        </button>
      </div>
    </div>
  );
}

// Step 5: Model Selection
function ModelStep({ selectedModel, setSelectedModel, modelParams, setModelParams, onTrain, loading }) {
  return (
    <div className="step-panel">
      <div className="panel-header">
        <h2>
          <span className="icon"><Brain size={22} /></span>
          Select Model
        </h2>
        <p>Choose a machine learning algorithm for classification</p>
      </div>

      <div className="option-cards">
        <div 
          className={`option-card ${selectedModel === 'logistic_regression' ? 'selected' : ''}`}
          onClick={() => setSelectedModel('logistic_regression')}
        >
          <div className="option-card-icon">
            <TrendingUp size={26} />
          </div>
          <h4>Logistic Regression</h4>
          <p>Simple, interpretable, works well for linearly separable data</p>
        </div>

        <div 
          className={`option-card ${selectedModel === 'decision_tree' ? 'selected' : ''}`}
          onClick={() => setSelectedModel('decision_tree')}
        >
          <div className="option-card-icon">
            <Activity size={26} />
          </div>
          <h4>Decision Tree</h4>
          <p>Captures non-linear patterns, easy to visualize</p>
        </div>
      </div>

      {selectedModel === 'decision_tree' && (
        <div className="form-group" style={{ marginTop: '24px' }}>
          <label>Max Depth (0 = unlimited)</label>
          <input
            type="number"
            className="form-input"
            min="0"
            max="50"
            value={modelParams.max_depth || 0}
            onChange={(e) => setModelParams({ ...modelParams, max_depth: Number(e.target.value) })}
          />
        </div>
      )}

      {selectedModel === 'logistic_regression' && (
        <div className="form-group" style={{ marginTop: '24px' }}>
          <label>Max Iterations</label>
          <input
            type="number"
            className="form-input"
            min="100"
            max="10000"
            step="100"
            value={modelParams.max_iter || 1000}
            onChange={(e) => setModelParams({ ...modelParams, max_iter: Number(e.target.value) })}
          />
        </div>
      )}

      <div className="action-buttons">
        <button 
          className="btn btn-success"
          onClick={onTrain}
          disabled={!selectedModel || loading}
        >
          <Zap size={16} />
          Train Model
        </button>
      </div>
    </div>
  );
}

// Step 6: Results
function ResultsStep({ results, onReset }) {
  if (!results) return null;

  return (
    <div className="step-panel results-container">
      <div className="panel-header" style={{ textAlign: 'center', borderBottom: 'none' }}>
        <div className="success-checkmark">
          <CheckCircle2 size={44} color="white" />
        </div>
        <h2 style={{ justifyContent: 'center' }}>Model Training Complete!</h2>
        <p>Your {results.model_name} model has been trained successfully</p>
        {results.total_time && (
          <div style={{ marginTop: '12px' }}>
            <span className="status-badge success">
              <Zap size={12} />
              Total: {results.total_time}s | Training: {results.training_time}s
            </span>
          </div>
        )}
      </div>

      <div className="metrics-grid">
        <div className="metric-card primary">
          <div className="metric-value">{results.test_accuracy}%</div>
          <div className="metric-label">Test Accuracy</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{results.precision}%</div>
          <div className="metric-label">Precision</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{results.recall}%</div>
          <div className="metric-label">Recall</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{results.f1_score}%</div>
          <div className="metric-label">F1 Score</div>
        </div>
      </div>

      {/* Cross-Validation Results */}
      {results.cv_mean && (
        <div className="info-box" style={{ marginBottom: '24px', background: 'linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%)', borderColor: '#22c55e' }}>
          <p style={{ color: '#166534' }}>
            <strong>Cross-Validation Score:</strong> {results.cv_mean}% (± {results.cv_std}%) 
            <br />
            <span style={{ fontSize: '0.85rem', opacity: 0.9 }}>Fold scores: [{results.cv_scores?.join('%, ')}%]</span>
          </p>
        </div>
      )}

      <div className="grid-2">
        {/* Confusion Matrix */}
        <div className="confusion-matrix-container">
          <h4><BarChart3 size={18} /> Confusion Matrix</h4>
          <div className="confusion-matrix">
            {results.confusion_matrix.map((row, i) => (
              <div key={i} className="confusion-row">
                {row.map((cell, j) => (
                  <div 
                    key={j} 
                    className={`confusion-cell ${i === j ? 'correct' : 'incorrect'}`}
                  >
                    {cell}
                  </div>
                ))}
              </div>
            ))}
          </div>
          <p style={{ textAlign: 'center', marginTop: '14px', fontSize: '0.75rem', color: '#64748b' }}>
            Green = Correct | Red = Incorrect
          </p>
        </div>

        {/* Model Info */}
        <div className="confusion-matrix-container">
          <h4><Activity size={18} /> Training Summary</h4>
          <div style={{ marginTop: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px', padding: '14px 16px', background: 'white', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
              <span style={{ color: '#64748b', fontSize: '0.9rem' }}>Model</span>
              <strong style={{ color: '#1e293b' }}>{results.model_name}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px', padding: '14px 16px', background: 'white', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
              <span style={{ color: '#64748b', fontSize: '0.9rem' }}>Training Samples</span>
              <strong style={{ color: '#1e293b' }}>{results.train_samples?.toLocaleString()}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px', padding: '14px 16px', background: 'white', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
              <span style={{ color: '#64748b', fontSize: '0.9rem' }}>Test Samples</span>
              <strong style={{ color: '#1e293b' }}>{results.test_samples?.toLocaleString()}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px', padding: '14px 16px', background: 'white', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
              <span style={{ color: '#64748b', fontSize: '0.9rem' }}>Features Used</span>
              <strong style={{ color: '#1e293b' }}>{results.total_features || 'N/A'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '14px 16px', background: 'linear-gradient(135deg, #eef2ff 0%, #faf5ff 100%)', borderRadius: '8px', border: '1px solid #c7d2fe' }}>
              <span style={{ color: '#64748b', fontSize: '0.9rem' }}>Training Accuracy</span>
              <strong style={{ color: '#6366f1' }}>{results.train_accuracy}%</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Importance (for Decision Tree) */}
      {results.feature_importance && results.feature_importance.length > 0 && (
        <div className="feature-importance" style={{ marginTop: '24px' }}>
          <h4><TrendingUp size={18} /> Top Feature Importance</h4>
          {results.feature_importance.map((feat, idx) => (
            <div key={idx} className="feature-bar">
              <div className="feature-bar-header">
                <span className="feature-name">{feat.feature}</span>
                <span className="feature-value">{(feat.importance * 100).toFixed(1)}%</span>
              </div>
              <div className="feature-bar-track">
                <div 
                  className="feature-bar-fill" 
                  style={{ width: `${feat.importance * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="action-buttons" style={{ marginTop: '32px', justifyContent: 'center' }}>
        <button className="btn btn-primary" onClick={onReset}>
          <RefreshCw size={16} />
          Start New Pipeline
        </button>
      </div>
    </div>
  );
}

export default App;
