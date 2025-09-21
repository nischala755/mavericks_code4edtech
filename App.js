import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

// Main App Component
function App() {
  const [activeTab, setActiveTab] = useState('upload-job');
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });

  // Fetch jobs on component mount
  useEffect(() => {
    fetchJobs();
  }, []);

  const fetchJobs = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/job-descriptions`);
      const data = await response.json();
      if (data.success) {
        setJobs(data.data);
      }
    } catch (error) {
      console.error('Error fetching jobs:', error);
      setMessage({ type: 'error', text: 'Failed to fetch job descriptions' });
    }
  };

  const showMessage = (type, text) => {
    setMessage({ type, text });
    setTimeout(() => setMessage({ type: '', text: '' }), 5000);
  };

  return (
    <div className="App">
      <header className="header">
        <div className="container">
          <h1>Resume Relevance Check System</h1>
          <p>Innomatics Research Labs - Automated Resume Evaluation</p>
        </div>
      </header>

      {message.text && (
        <div className={`message ${message.type}`}>
          {message.text}
        </div>
      )}

      <div className="container">
        <nav className="tab-nav">
          <button 
            className={`tab ${activeTab === 'upload-job' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload-job')}
          >
            Upload Job Description
          </button>
          <button 
            className={`tab ${activeTab === 'evaluate-resume' ? 'active' : ''}`}
            onClick={() => setActiveTab('evaluate-resume')}
          >
            Evaluate Resume
          </button>
          <button 
            className={`tab ${activeTab === 'view-evaluations' ? 'active' : ''}`}
            onClick={() => setActiveTab('view-evaluations')}
          >
            View Evaluations
          </button>
          <button 
            className={`tab ${activeTab === 'job-list' ? 'active' : ''}`}
            onClick={() => setActiveTab('job-list')}
          >
            Job Descriptions
          </button>
        </nav>

        <div className="tab-content">
          {activeTab === 'upload-job' && (
            <JobUploadForm 
              showMessage={showMessage} 
              onJobUploaded={fetchJobs}
            />
          )}
          
          {activeTab === 'evaluate-resume' && (
            <ResumeEvaluationForm 
              jobs={jobs}
              showMessage={showMessage}
              selectedJobId={selectedJobId}
              setSelectedJobId={setSelectedJobId}
            />
          )}
          
          {activeTab === 'view-evaluations' && (
            <EvaluationsList 
              jobs={jobs}
              showMessage={showMessage}
            />
          )}
          
          {activeTab === 'job-list' && (
            <JobsList 
              jobs={jobs}
              onRefresh={fetchJobs}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// Job Upload Form Component
function JobUploadForm({ showMessage, onJobUploaded }) {
  const [jobData, setJobData] = useState({
    role_title: '',
    company_name: '',
    must_have_skills: '',
    good_to_have_skills: '',
    qualifications: '',
    experience_required: '',
    job_description: '',
    location: ''
  });
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e) => {
    setJobData({
      ...jobData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const payload = {
        ...jobData,
        must_have_skills: jobData.must_have_skills.split(',').map(s => s.trim()).filter(s => s),
        good_to_have_skills: jobData.good_to_have_skills.split(',').map(s => s.trim()).filter(s => s),
        qualifications: jobData.qualifications.split(',').map(s => s.trim()).filter(s => s)
      };

      const response = await fetch(`${API_BASE_URL}/upload-job-description`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      
      if (data.success) {
        showMessage('success', `Job description uploaded successfully! Job ID: ${data.job_id}`);
        setJobData({
          role_title: '',
          company_name: '',
          must_have_skills: '',
          good_to_have_skills: '',
          qualifications: '',
          experience_required: '',
          job_description: '',
          location: ''
        });
        onJobUploaded();
      } else {
        showMessage('error', 'Failed to upload job description');
      }
    } catch (error) {
      showMessage('error', `Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="form-container">
      <h2>Upload Job Description</h2>
      <form onSubmit={handleSubmit} className="job-form">
        <div className="form-group">
          <label>Role Title *</label>
          <input
            type="text"
            name="role_title"
            value={jobData.role_title}
            onChange={handleInputChange}
            required
          />
        </div>

        <div className="form-group">
          <label>Company Name *</label>
          <input
            type="text"
            name="company_name"
            value={jobData.company_name}
            onChange={handleInputChange}
            required
          />
        </div>

        <div className="form-group">
          <label>Must Have Skills * (comma-separated)</label>
          <input
            type="text"
            name="must_have_skills"
            value={jobData.must_have_skills}
            onChange={handleInputChange}
            placeholder="Python, FastAPI, SQL, Machine Learning"
            required
          />
        </div>

        <div className="form-group">
          <label>Good to Have Skills (comma-separated)</label>
          <input
            type="text"
            name="good_to_have_skills"
            value={jobData.good_to_have_skills}
            onChange={handleInputChange}
            placeholder="React, AWS, Docker"
          />
        </div>

        <div className="form-group">
          <label>Qualifications * (comma-separated)</label>
          <input
            type="text"
            name="qualifications"
            value={jobData.qualifications}
            onChange={handleInputChange}
            placeholder="Bachelor's in Computer Science, Master's preferred"
            required
          />
        </div>

        <div className="form-group">
          <label>Experience Required *</label>
          <input
            type="text"
            name="experience_required"
            value={jobData.experience_required}
            onChange={handleInputChange}
            placeholder="2-3 years"
            required
          />
        </div>

        <div className="form-group">
          <label>Location *</label>
          <input
            type="text"
            name="location"
            value={jobData.location}
            onChange={handleInputChange}
            placeholder="Hyderabad, Bangalore"
            required
          />
        </div>

        <div className="form-group">
          <label>Job Description *</label>
          <textarea
            name="job_description"
            value={jobData.job_description}
            onChange={handleInputChange}
            rows="5"
            placeholder="Detailed job description..."
            required
          ></textarea>
        </div>

        <button type="submit" disabled={loading} className="submit-btn">
          {loading ? 'Uploading...' : 'Upload Job Description'}
        </button>
      </form>
    </div>
  );
}

// Resume Evaluation Form Component
function ResumeEvaluationForm({ jobs, showMessage, selectedJobId, setSelectedJobId }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [evaluation, setEvaluation] = useState(null);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedJobId) {
      showMessage('error', 'Please select a job description');
      return;
    }
    
    if (!selectedFile) {
      showMessage('error', 'Please select a resume file');
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('resume', selectedFile);

      const response = await fetch(`${API_BASE_URL}/evaluate-resume/${selectedJobId}`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      
      if (data.success) {
        setEvaluation(data.data);
        showMessage('success', 'Resume evaluated successfully!');
      } else {
        showMessage('error', 'Failed to evaluate resume');
      }
    } catch (error) {
      showMessage('error', `Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="form-container">
      <h2>Evaluate Resume</h2>
      
      <form onSubmit={handleSubmit} className="evaluation-form">
        <div className="form-group">
          <label>Select Job Description *</label>
          <select
            value={selectedJobId}
            onChange={(e) => setSelectedJobId(e.target.value)}
            required
          >
            <option value="">Choose a job...</option>
            {jobs.map(job => (
              <option key={job.id} value={job.id}>
                {job.role_title} - {job.company_name} ({job.location})
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Upload Resume * (PDF or DOCX)</label>
          <input
            type="file"
            accept=".pdf,.docx"
            onChange={handleFileChange}
            required
          />
          {selectedFile && (
            <p className="file-info">Selected: {selectedFile.name}</p>
          )}
        </div>

        <button type="submit" disabled={loading || !selectedJobId || !selectedFile} className="submit-btn">
          {loading ? 'Evaluating...' : 'Evaluate Resume'}
        </button>
      </form>

      {evaluation && (
        <EvaluationResult evaluation={evaluation} />
      )}
    </div>
  );
}

// Evaluation Result Component
function EvaluationResult({ evaluation }) {
  const getVerdictColor = (verdict) => {
    switch(verdict) {
      case 'High': return '#4CAF50';
      case 'Medium': return '#FF9800';
      case 'Low': return '#F44336';
      default: return '#757575';
    }
  };

  return (
    <div className="evaluation-result">
      <h3>Evaluation Results</h3>
      
      <div className="result-header">
        <div className="score-card">
          <h4>Relevance Score</h4>
          <div className="score">{evaluation.relevance_score}/100</div>
        </div>
        
        <div className="verdict-card" style={{ backgroundColor: getVerdictColor(evaluation.verdict) }}>
          <h4>Verdict</h4>
          <div className="verdict">{evaluation.verdict}</div>
        </div>
      </div>

      <div className="result-details">
        <div className="detail-section">
          <h4>Score Breakdown</h4>
          <p>Hard Match Score: {evaluation.hard_match_score}/100</p>
          <p>Semantic Match Score: {evaluation.semantic_match_score}/100</p>
        </div>

        <div className="detail-section">
          <h4>Matched Skills</h4>
          <div className="skills-list">
            {evaluation.matched_skills.length > 0 ? (
              evaluation.matched_skills.map((skill, index) => (
                <span key={index} className="skill matched">{skill}</span>
              ))
            ) : (
              <p>No skills matched</p>
            )}
          </div>
        </div>

        <div className="detail-section">
          <h4>Missing Skills</h4>
          <div className="skills-list">
            {evaluation.missing_skills.length > 0 ? (
              evaluation.missing_skills.map((skill, index) => (
                <span key={index} className="skill missing">{skill}</span>
              ))
            ) : (
              <p>No missing skills</p>
            )}
          </div>
        </div>

        <div className="detail-section">
          <h4>Suggestions for Improvement</h4>
          <ul>
            {evaluation.suggestions.map((suggestion, index) => (
              <li key={index}>{suggestion}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

// Evaluations List Component
function EvaluationsList({ jobs, showMessage }) {
  const [selectedJobId, setSelectedJobId] = useState('');
  const [evaluations, setEvaluations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({
    verdict: '',
    min_score: ''
  });

  const fetchEvaluations = async () => {
    if (!selectedJobId) return;
    
    setLoading(true);
    try {
      let url = `${API_BASE_URL}/evaluations/${selectedJobId}`;
      const params = new URLSearchParams();
      
      if (filters.verdict) params.append('verdict', filters.verdict);
      if (filters.min_score) params.append('min_score', filters.min_score);
      
      if (params.toString()) {
        url += `?${params.toString()}`;
      }

      const response = await fetch(url);
      const data = await response.json();
      
      if (data.success) {
        setEvaluations(data.data);
      } else {
        showMessage('error', 'Failed to fetch evaluations');
      }
    } catch (error) {
      showMessage('error', `Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedJobId) {
      fetchEvaluations();
    }
  }, [selectedJobId, filters]);

  return (
    <div className="evaluations-container">
      <h2>View Evaluations</h2>
      
      <div className="filters-section">
        <div className="form-group">
          <label>Select Job Description</label>
          <select
            value={selectedJobId}
            onChange={(e) => setSelectedJobId(e.target.value)}
          >
            <option value="">Choose a job...</option>
            {jobs.map(job => (
              <option key={job.id} value={job.id}>
                {job.role_title} - {job.company_name}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Filter by Verdict</label>
          <select
            value={filters.verdict}
            onChange={(e) => setFilters({ ...filters, verdict: e.target.value })}
          >
            <option value="">All</option>
            <option value="High">High</option>
            <option value="Medium">Medium</option>
            <option value="Low">Low</option>
          </select>
        </div>

        <div className="form-group">
          <label>Minimum Score</label>
          <input
            type="number"
            min="0"
            max="100"
            value={filters.min_score}
            onChange={(e) => setFilters({ ...filters, min_score: e.target.value })}
            placeholder="0-100"
          />
        </div>
      </div>

      {loading && <p>Loading evaluations...</p>}

      {evaluations.length > 0 && (
        <div className="evaluations-list">
          <h3>Evaluations ({evaluations.length})</h3>
          {evaluations.map((evaluation, index) => (
            <div key={index} className="evaluation-card">
              <div className="card-header">
                <span className="score">Score: {evaluation.relevance_score}/100</span>
                <span className={`verdict ${evaluation.verdict.toLowerCase()}`}>
                  {evaluation.verdict}
                </span>
              </div>
              <div className="card-details">
                <p><strong>Hard Match:</strong> {evaluation.hard_match_score}/100</p>
                <p><strong>Semantic Match:</strong> {evaluation.semantic_match_score}/100</p>
                <p><strong>Matched Skills:</strong> {evaluation.matched_skills.join(', ') || 'None'}</p>
                <p><strong>Missing Skills:</strong> {evaluation.missing_skills.join(', ') || 'None'}</p>
                <p><strong>Timestamp:</strong> {new Date(evaluation.timestamp).toLocaleString()}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedJobId && !loading && evaluations.length === 0 && (
        <p>No evaluations found for the selected job.</p>
      )}
    </div>
  );
}

// Jobs List Component
function JobsList({ jobs, onRefresh }) {
  return (
    <div className="jobs-container">
      <div className="section-header">
        <h2>Job Descriptions ({jobs.length})</h2>
        <button onClick={onRefresh} className="refresh-btn">
          Refresh
        </button>
      </div>

      {jobs.length > 0 ? (
        <div className="jobs-grid">
          {jobs.map((job, index) => (
            <div key={index} className="job-card">
              <div className="job-header">
                <h3>{job.role_title}</h3>
                <p className="company">{job.company_name}</p>
              </div>
              
              <div className="job-details">
                <p><strong>Location:</strong> {job.location}</p>
                <p><strong>Experience:</strong> {job.experience_required}</p>
                
                <div className="skills-section">
                  <p><strong>Must Have Skills:</strong></p>
                  <div className="skills-list">
                    {job.must_have_skills.map((skill, i) => (
                      <span key={i} className="skill required">{skill}</span>
                    ))}
                  </div>
                </div>

                {job.good_to_have_skills.length > 0 && (
                  <div className="skills-section">
                    <p><strong>Good to Have:</strong></p>
                    <div className="skills-list">
                      {job.good_to_have_skills.map((skill, i) => (
                        <span key={i} className="skill optional">{skill}</span>
                      ))}
                    </div>
                  </div>
                )}

                <p><strong>Job ID:</strong> <code>{job.id}</code></p>
                <p><strong>Created:</strong> {new Date(job.created_at).toLocaleDateString()}</p>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p>No job descriptions available. Upload one to get started!</p>
      )}
    </div>
  );
}

export default App;