from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
import os
import tempfile
import json
from datetime import datetime
import uuid
import re

# Document processing imports
import pdfplumber
import docx2txt
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Google Gemini imports
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI(
    title="Resume Relevance Check API",
    description="Automated Resume Evaluation System for Innomatics Research Labs",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")
DATABASE_PATH = "resume_evaluation.db"

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load spaCy model (download with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Some features may not work properly.")
    nlp = None

# Pydantic models
class JobDescription(BaseModel):
    role_title: str
    company_name: str
    must_have_skills: List[str]
    good_to_have_skills: List[str]
    qualifications: List[str]
    experience_required: str
    job_description: str
    location: str

class ResumeEvaluation(BaseModel):
    resume_id: str
    job_id: str
    relevance_score: float
    verdict: str
    missing_skills: List[str]
    matched_skills: List[str]
    suggestions: List[str]
    hard_match_score: float
    semantic_match_score: float
    timestamp: datetime

class EvaluationResponse(BaseModel):
    success: bool
    data: Optional[ResumeEvaluation]
    message: str

# Database initialization
def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Job descriptions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id TEXT PRIMARY KEY,
            role_title TEXT NOT NULL,
            company_name TEXT NOT NULL,
            must_have_skills TEXT NOT NULL,
            good_to_have_skills TEXT NOT NULL,
            qualifications TEXT NOT NULL,
            experience_required TEXT NOT NULL,
            job_description TEXT NOT NULL,
            location TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Resume evaluations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            resume_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            relevance_score REAL NOT NULL,
            verdict TEXT NOT NULL,
            missing_skills TEXT NOT NULL,
            matched_skills TEXT NOT NULL,
            suggestions TEXT NOT NULL,
            hard_match_score REAL NOT NULL,
            semantic_match_score REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Document processing functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using pdfplumber"""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {str(e)}")


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX using docx2txt"""
    try:
        text = docx2txt.process(file_path)
        return text.strip()
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {str(e)}")

def clean_and_normalize_text(text: str) -> str:
    """Clean and normalize extracted text"""
    # Remove extra whitespaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep essential punctuation
    text = re.sub(r'[^\w\s.,;:()\-+@]', '', text)
    return text.strip()

def extract_skills_with_spacy(text: str) -> List[str]:
    """Extract potential skills using spaCy NER and pattern matching"""
    skills = []
    
    if nlp is None:
        return skills
    
    doc = nlp(text)
    
    # Common skill patterns
    skill_patterns = [
        r'\b(?:python|java|javascript|react|angular|vue|node\.?js|express)\b',
        r'\b(?:sql|mysql|postgresql|mongodb|redis|elasticsearch)\b',
        r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins)\b',
        r'\b(?:machine learning|deep learning|ai|nlp|computer vision)\b',
        r'\b(?:tensorflow|pytorch|scikit-learn|pandas|numpy)\b',
        r'\b(?:html|css|bootstrap|tailwind|sass|less)\b',
        r'\b(?:git|github|gitlab|bitbucket|svn)\b',
        r'\b(?:agile|scrum|kanban|devops|ci/cd)\b'
    ]
    
    text_lower = text.lower()
    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        skills.extend(matches)
    
    # Extract entities that might be skills
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"] and len(ent.text) > 2:
            skills.append(ent.text.lower())
    
    return list(set(skills))

class ResumeProcessor:
    """Main class for processing resumes and job descriptions"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    
    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """Parse resume text and extract structured information"""
        cleaned_text = clean_and_normalize_text(resume_text)
        skills = extract_skills_with_spacy(cleaned_text)
        
        # Use Gemini to extract structured information
        prompt = f"""
        Analyze the following resume text and extract structured information in JSON format:
        
        Resume Text:
        {cleaned_text[:3000]}  # Limit text for API efficiency
        
        Please extract and return a JSON object with the following fields:
        - name: candidate's name
        - email: email address
        - phone: phone number
        - education: list of educational qualifications
        - experience: years of experience (estimate)
        - skills: list of technical skills
        - projects: list of project names/descriptions
        - certifications: list of certifications
        
        Return only valid JSON format.
        """
        
        try:
            response = model.generate_content(prompt)
            # Try to parse JSON from response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
            else:
                parsed_data = {}
        except Exception as e:
            print(f"Error parsing resume with Gemini: {e}")
            parsed_data = {}
        
        # Fallback data
        return {
            "name": parsed_data.get("name", "Unknown"),
            "email": parsed_data.get("email", ""),
            "phone": parsed_data.get("phone", ""),
            "education": parsed_data.get("education", []),
            "experience": parsed_data.get("experience", 0),
            "skills": list(set(parsed_data.get("skills", []) + skills)),
            "projects": parsed_data.get("projects", []),
            "certifications": parsed_data.get("certifications", []),
            "raw_text": cleaned_text
        }
    
    def calculate_hard_match_score(self, resume_data: Dict, job_desc: JobDescription) -> Dict[str, Any]:
        """Calculate hard match score based on exact keyword matching"""
        resume_skills = [skill.lower() for skill in resume_data.get("skills", [])]
        resume_text_lower = resume_data.get("raw_text", "").lower()
        
        must_have_skills = [skill.lower() for skill in job_desc.must_have_skills]
        good_to_have_skills = [skill.lower() for skill in job_desc.good_to_have_skills]
        
        # Match must-have skills
        matched_must_have = []
        for skill in must_have_skills:
            if skill in resume_skills or skill in resume_text_lower:
                matched_must_have.append(skill)
        
        # Match good-to-have skills
        matched_good_to_have = []
        for skill in good_to_have_skills:
            if skill in resume_skills or skill in resume_text_lower:
                matched_good_to_have.append(skill)
        
        # Calculate score
        must_have_score = len(matched_must_have) / len(must_have_skills) if must_have_skills else 0
        good_to_have_score = len(matched_good_to_have) / len(good_to_have_skills) if good_to_have_skills else 0
        
        # Weighted scoring (must-have: 70%, good-to-have: 30%)
        hard_score = (must_have_score * 0.7) + (good_to_have_score * 0.3)
        
        return {
            "score": hard_score * 100,
            "matched_must_have": matched_must_have,
            "matched_good_to_have": matched_good_to_have,
            "missing_must_have": [skill for skill in must_have_skills if skill not in matched_must_have],
            "missing_good_to_have": [skill for skill in good_to_have_skills if skill not in matched_good_to_have]
        }
    
    def calculate_semantic_match_score(self, resume_data: Dict, job_desc: JobDescription) -> float:
        """Calculate semantic similarity using sentence transformers"""
        resume_text = resume_data.get("raw_text", "")
        job_text = f"{job_desc.role_title} {job_desc.job_description}"
        
        # Generate embeddings
        resume_embedding = embedding_model.encode([resume_text])
        job_embedding = embedding_model.encode([job_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        return float(similarity * 100)
    
    def generate_llm_feedback(self, resume_data: Dict, job_desc: JobDescription, hard_match: Dict) -> Dict[str, Any]:
        """Generate detailed feedback using Gemini"""
        prompt = f"""
        Analyze the resume against the job requirements and provide detailed feedback:
        
        Job Role: {job_desc.role_title}
        Required Skills: {', '.join(job_desc.must_have_skills)}
        Preferred Skills: {', '.join(job_desc.good_to_have_skills)}
        Experience Required: {job_desc.experience_required}
        
        Candidate Skills: {', '.join(resume_data.get('skills', []))}
        Candidate Experience: {resume_data.get('experience', 'Unknown')} years
        
        Missing Must-Have Skills: {', '.join(hard_match['missing_must_have'])}
        Missing Preferred Skills: {', '.join(hard_match['missing_good_to_have'])}
        
        Please provide:
        1. A verdict (High/Medium/Low suitability) with reasoning
        2. Top 3 suggestions for improvement
        3. Overall assessment summary
        
        Format as JSON with fields: verdict, reasoning, suggestions, summary
        """
        
        try:
            response = model.generate_content(prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                feedback = json.loads(json_match.group())
            else:
                feedback = {
                    "verdict": "Medium",
                    "reasoning": "Unable to generate detailed analysis",
                    "suggestions": ["Update skills", "Gain more experience", "Add relevant projects"],
                    "summary": "Resume requires improvement"
                }
        except Exception as e:
            print(f"Error generating feedback: {e}")
            feedback = {
                "verdict": "Medium",
                "reasoning": "Error in analysis",
                "suggestions": ["Update skills", "Gain more experience", "Add relevant projects"],
                "summary": "Resume requires review"
            }
        
        return feedback
    
    def evaluate_resume(self, resume_data: Dict, job_desc: JobDescription) -> ResumeEvaluation:
        """Main evaluation function"""
        # Calculate hard match score
        hard_match = self.calculate_hard_match_score(resume_data, job_desc)
        
        # Calculate semantic match score
        semantic_score = self.calculate_semantic_match_score(resume_data, job_desc)
        
        # Generate LLM feedback
        feedback = self.generate_llm_feedback(resume_data, job_desc, hard_match)
        
        # Calculate final relevance score (weighted combination)
        relevance_score = (hard_match["score"] * 0.6) + (semantic_score * 0.4)
        
        # Determine verdict based on score
        if relevance_score >= 75:
            verdict = "High"
        elif relevance_score >= 50:
            verdict = "Medium"
        else:
            verdict = "Low"
        
        # Use LLM verdict if available
        llm_verdict = feedback.get("verdict", "").title()
        if llm_verdict in ["High", "Medium", "Low"]:
            verdict = llm_verdict
        
        return ResumeEvaluation(
            resume_id=str(uuid.uuid4()),
            job_id="",  # Will be set by calling function
            relevance_score=round(relevance_score, 2),
            verdict=verdict,
            missing_skills=hard_match["missing_must_have"] + hard_match["missing_good_to_have"],
            matched_skills=hard_match["matched_must_have"] + hard_match["matched_good_to_have"],
            suggestions=feedback.get("suggestions", []),
            hard_match_score=round(hard_match["score"], 2),
            semantic_match_score=round(semantic_score, 2),
            timestamp=datetime.now()
        )

# Initialize processor
processor = ResumeProcessor()

# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_database()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Resume Relevance Check API is running", "version": "1.0.0"}

@app.post("/upload-job-description")
async def upload_job_description(job_desc: JobDescription):
    """Upload and store job description"""
    try:
        job_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO job_descriptions 
            (id, role_title, company_name, must_have_skills, good_to_have_skills, 
             qualifications, experience_required, job_description, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_id,
            job_desc.role_title,
            job_desc.company_name,
            json.dumps(job_desc.must_have_skills),
            json.dumps(job_desc.good_to_have_skills),
            json.dumps(job_desc.qualifications),
            job_desc.experience_required,
            job_desc.job_description,
            job_desc.location
        ))
        
        conn.commit()
        conn.close()
        
        return {"success": True, "job_id": job_id, "message": "Job description uploaded successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading job description: {str(e)}")

@app.post("/evaluate-resume/{job_id}")
async def evaluate_resume(job_id: str, resume: UploadFile = File(...)):
    """Evaluate resume against job description"""
    try:
        # Validate file type
        if not resume.filename.endswith(('.pdf', '.docx')):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
        
        # Get job description
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM job_descriptions WHERE id = ?", (job_id,))
        job_row = cursor.fetchone()
        
        if not job_row:
            raise HTTPException(status_code=404, detail="Job description not found")
        
        # Parse job description
        job_desc = JobDescription(
            role_title=job_row[1],
            company_name=job_row[2],
            must_have_skills=json.loads(job_row[3]),
            good_to_have_skills=json.loads(job_row[4]),
            qualifications=json.loads(job_row[5]),
            experience_required=job_row[6],
            job_description=job_row[7],
            location=job_row[8]
        )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resume.filename)[1]) as tmp_file:
            tmp_file.write(await resume.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Extract text based on file type
            if resume.filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(tmp_file_path)
            else:  # .docx
                resume_text = extract_text_from_docx(tmp_file_path)
            
            # Parse resume
            resume_data = processor.parse_resume(resume_text)
            
            # Evaluate resume
            evaluation = processor.evaluate_resume(resume_data, job_desc)
            evaluation.job_id = job_id
            
            # Store evaluation
            cursor.execute('''
                INSERT INTO evaluations 
                (id, resume_id, job_id, relevance_score, verdict, missing_skills, 
                 matched_skills, suggestions, hard_match_score, semantic_match_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                evaluation.resume_id,
                job_id,
                evaluation.relevance_score,
                evaluation.verdict,
                json.dumps(evaluation.missing_skills),
                json.dumps(evaluation.matched_skills),
                json.dumps(evaluation.suggestions),
                evaluation.hard_match_score,
                evaluation.semantic_match_score
            ))
            
            conn.commit()
            conn.close()
            
            return EvaluationResponse(
                success=True,
                data=evaluation,
                message="Resume evaluated successfully"
            )
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        raise HTTPException(status_code=500, detail=f"Error evaluating resume: {str(e)}")

@app.get("/evaluations/{job_id}")
async def get_evaluations(job_id: str, verdict: Optional[str] = None, min_score: Optional[float] = None):
    """Get all evaluations for a job with optional filtering"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        query = "SELECT * FROM evaluations WHERE job_id = ?"
        params = [job_id]
        
        if verdict:
            query += " AND verdict = ?"
            params.append(verdict)
        
        if min_score is not None:
            query += " AND relevance_score >= ?"
            params.append(min_score)
        
        query += " ORDER BY relevance_score DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        evaluations = []
        for row in rows:
            evaluation = ResumeEvaluation(
                resume_id=row[1],
                job_id=row[2],
                relevance_score=row[3],
                verdict=row[4],
                missing_skills=json.loads(row[5]),
                matched_skills=json.loads(row[6]),
                suggestions=json.loads(row[7]),
                hard_match_score=row[8],
                semantic_match_score=row[9],
                timestamp=datetime.fromisoformat(row[10])
            )
            evaluations.append(evaluation)
        
        return {"success": True, "data": evaluations, "count": len(evaluations)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching evaluations: {str(e)}")

@app.get("/job-descriptions")
async def get_job_descriptions():
    """Get all job descriptions"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM job_descriptions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        jobs = []
        for row in rows:
            job = {
                "id": row[0],
                "role_title": row[1],
                "company_name": row[2],
                "must_have_skills": json.loads(row[3]),
                "good_to_have_skills": json.loads(row[4]),
                "qualifications": json.loads(row[5]),
                "experience_required": row[6],
                "job_description": row[7],
                "location": row[8],
                "created_at": row[9]
            }
            jobs.append(job)
        
        return {"success": True, "data": jobs}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching job descriptions: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
