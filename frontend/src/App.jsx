import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  FileText, 
  Link2, 
  Type, 
  Send, 
  HelpCircle, 
  RefreshCw, 
  History, 
  Settings, 
  TrendingUp, 
  Compass, 
  Award, 
  AlertCircle, 
  Moon, 
  Sun,
  FileCheck,
  CheckCircle2
} from 'lucide-react';

const API_BASE = '/api';

export default function App() {
  // Theme state
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark' || 
      (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches);
  });

  // UI state
  const [activeTab, setActiveTab] = useState('paste'); // 'paste', 'url', 'file'
  const [inputText, setInputText] = useState('');
  const [inputUrl, setInputUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  
  // Loading & Error states
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [error, setError] = useState(null);

  // Analysis result
  const [result, setResult] = useState(null);

  // System history & settings
  const [history, setHistory] = useState([]);
  const [healthStatus, setHealthStatus] = useState(null);
  const [trainingTasks, setTrainingTasks] = useState([]);

  // Load theme
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [darkMode]);

  // Load initial history & health
  useEffect(() => {
    fetchHistory();
    fetchHealth();
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await axios.get(`${API_BASE}/history`);
      setHistory(response.data);
    } catch (err) {
      console.error('Failed to load history', err);
    }
  };

  const fetchHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE}/health`);
      setHealthStatus(response.data);
    } catch (err) {
      console.error('Health check failed', err);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const triggerTraining = async (type) => {
    try {
      const response = await axios.post(`${API_BASE}/train-${type}`);
      const newTask = {
        id: response.data.task_id,
        type: type,
        status: response.data.status,
        message: response.data.message
      };
      setTrainingTasks(prev => [newTask, ...prev]);
      pollTaskStatus(response.data.task_id);
    } catch (err) {
      alert(`Failed to start training: ${err.response?.data?.detail || err.message}`);
    }
  };

  const pollTaskStatus = (taskId) => {
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE}/train/status/${taskId}`);
        setTrainingTasks(prev => 
          prev.map(task => task.id === taskId ? { ...task, status: response.data.status, message: response.data.message } : task)
        );
        
        if (response.data.status === 'completed' || response.data.status === 'failed') {
          clearInterval(interval);
          fetchHealth(); // refresh models status
        }
      } catch (err) {
        clearInterval(interval);
      }
    }, 3000);
  };

  const handleAnalyze = async () => {
    setError(null);
    setResult(null);
    setLoading(true);
    setLoadingMessage('Initializing model pipeline...');

    try {
      let response;
      if (activeTab === 'paste') {
        if (!inputText.trim()) throw new Error('Please enter some article text.');
        setLoadingMessage('Tokenizing & running Transformer encoders...');
        response = await axios.post(`${API_BASE}/analyze`, { text: inputText });
      } else if (activeTab === 'url') {
        if (!inputUrl.trim()) throw new Error('Please enter a valid URL.');
        setLoadingMessage('Fetching and extracting content from URL...');
        response = await axios.post(`${API_BASE}/analyze`, { url: inputUrl });
      } else {
        if (!selectedFile) throw new Error('Please upload a PDF or TXT file.');
        setLoadingMessage('Parsing file contents...');
        const formData = new FormData();
        formData.append('file', selectedFile);
        response = await axios.post(`${API_BASE}/analyze-file`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
      }

      setLoadingMessage('Computing SHAP attributions & compiling explanation...');
      // Small artificial delay for premium dynamic experience feel
      await new Promise(r => setTimeout(r, 600));

      setResult(response.data);
      fetchHistory(); // refresh sidebar list
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred during analysis.');
    } finally {
      setLoading(false);
    }
  };

  const loadFromHistory = (item) => {
    setResult({
      title: item.title,
      category: item.category,
      category_confidence: item.category_confidence,
      category_distribution: item.category_distribution,
      bias: item.bias,
      bias_confidence: item.bias_confidence,
      bias_distribution: item.bias_distribution,
      bias_score: item.bias_score,
      bias_interpretation: item.bias_interpretation,
      neutral_vs_biased: {
        Neutral: item.neutral_percent,
        Biased: item.biased_percent
      },
      important_words: item.important_words,
      explanation: item.explanation
    });
    // Set text box to look like we are looking at this item
    setInputText(item.content);
    window.scrollTo({ top: 350, behavior: 'smooth' });
  };

  // Helper function to render article with highlighted words
  const renderHighlightedText = (text, importantWords, bias) => {
    if (!text || !importantWords || importantWords.length === 0) return text;
    
    // Create map of word -> score
    const wordScores = {};
    importantWords.forEach(w => {
      wordScores[w.word.toLowerCase()] = w.score;
    });

    const highlightClass = 
      bias === 'Left' ? 'highlight-left' : 
      bias === 'Right' ? 'highlight-right' : 'highlight-center';

    // Regex to split text by word boundaries
    const parts = text.split(/(\b\w+\b)/g);
    
    return parts.map((part, idx) => {
      const lowerPart = part.toLowerCase();
      if (wordScores.hasOwnProperty(lowerPart)) {
        return (
          <span 
            key={idx} 
            className={`${highlightClass} font-semibold cursor-pointer`}
            title={`Importance Score: ${wordScores[lowerPart].toFixed(4)}`}
          >
            {part}
          </span>
        );
      }
      return part;
    });
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-[#0b0f19] text-slate-800 dark:text-slate-100 transition-colors duration-200">
      
      {/* Header Banner */}
      <header className="sticky top-0 z-40 border-b border-slate-200/80 dark:border-slate-800/80 bg-white/70 dark:bg-[#0f172a]/70 backdrop-blur-xl transition-all duration-300">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-tr from-violet-600 to-indigo-600 text-white p-2 rounded-xl shadow-lg shadow-indigo-500/20">
              <Compass className="w-6 h-6 animate-pulse-slow" />
            </div>
            <div>
              <span className="font-extrabold text-xl tracking-tight bg-gradient-to-r from-violet-600 to-indigo-500 dark:from-violet-400 dark:to-indigo-400 bg-clip-text text-transparent">
                FACTTRACK
              </span>
              <span className="text-[10px] font-bold block text-slate-400 tracking-widest uppercase">AI News Analytics</span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {healthStatus && (
              <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full border border-slate-200 dark:border-slate-800 bg-slate-100/50 dark:bg-slate-900/50 text-[11px] font-semibold text-slate-500 dark:text-slate-400">
                <span className={`w-2 h-2 rounded-full ${healthStatus.status === 'healthy' ? 'bg-emerald-500 animate-ping' : 'bg-red-500'}`} />
                <span>Backend API: Online</span>
              </div>
            )}
            
            <button 
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 transition-all"
              title="Toggle Dark Mode"
            >
              {darkMode ? <Sun className="w-5 h-5 text-amber-400" /> : <Moon className="w-5 h-5 text-slate-600" />}
            </button>
          </div>
        </div>
      </header>

      {/* Main Body */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Hero Section */}
        <div className="text-center max-w-3xl mx-auto mb-12">
          <h1 className="text-4xl sm:text-5xl font-black tracking-tight mb-4 bg-gradient-to-b from-slate-900 via-slate-800 to-slate-700 dark:from-white dark:to-slate-400 bg-clip-text text-transparent">
            Uncover the Subtext of News
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400 font-medium">
            Analyze any news article to instantly discover its classification domain, identify editorial bias, and view machine learning saliency attributions.
          </p>
        </div>

        {/* Core Layout: Sidebar + Workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Left panel / Input Area (8 columns) */}
          <div className="lg:col-span-8 space-y-6">
            
            {/* Input card */}
            <div className="bg-white dark:bg-[#151b2a] border border-slate-200 dark:border-slate-800/80 rounded-2xl shadow-xl shadow-slate-100 dark:shadow-none overflow-hidden transition-all duration-300">
              
              {/* Tab navigation */}
              <div className="flex border-b border-slate-200 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-900/50 p-2">
                <button
                  onClick={() => setActiveTab('paste')}
                  className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-xl text-sm font-semibold transition-all ${
                    activeTab === 'paste' 
                    ? 'bg-white dark:bg-[#1e293b] text-indigo-600 dark:text-indigo-400 shadow-md shadow-slate-200/50 dark:shadow-none' 
                    : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                  }`}
                >
                  <Type className="w-4 h-4" />
                  Paste Text
                </button>
                <button
                  onClick={() => setActiveTab('url')}
                  className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-xl text-sm font-semibold transition-all ${
                    activeTab === 'url' 
                    ? 'bg-white dark:bg-[#1e293b] text-indigo-600 dark:text-indigo-400 shadow-md shadow-slate-200/50 dark:shadow-none' 
                    : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                  }`}
                >
                  <Link2 className="w-4 h-4" />
                  Analyze URL
                </button>
                <button
                  onClick={() => setActiveTab('file')}
                  className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-xl text-sm font-semibold transition-all ${
                    activeTab === 'file' 
                    ? 'bg-white dark:bg-[#1e293b] text-indigo-600 dark:text-indigo-400 shadow-md shadow-slate-200/50 dark:shadow-none' 
                    : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                  }`}
                >
                  <FileText className="w-4 h-4" />
                  Upload File
                </button>
              </div>

              {/* Tab contents */}
              <div className="p-6">
                {activeTab === 'paste' && (
                  <div className="space-y-2">
                    <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">News Article Text</label>
                    <textarea
                      rows={10}
                      className="w-full rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50 p-4 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500/20 focus:border-violet-500 text-slate-800 dark:text-slate-100 transition-all placeholder-slate-400"
                      placeholder="Paste the full article content here (at least 200 words recommended for accurate bias detection)..."
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                    />
                  </div>
                )}

                {activeTab === 'url' && (
                  <div className="space-y-3 py-6">
                    <label className="text-xs font-bold text-slate-400 uppercase tracking-wider block">News Article URL</label>
                    <div className="relative">
                      <Link2 className="absolute left-4 top-3.5 w-5 h-5 text-slate-400" />
                      <input
                        type="url"
                        className="w-full rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50 py-3.5 pl-12 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500/20 focus:border-violet-500 text-slate-800 dark:text-slate-100 transition-all placeholder-slate-400"
                        placeholder="https://www.nytimes.com/article-url-here..."
                        value={inputUrl}
                        onChange={(e) => setInputUrl(e.target.value)}
                      />
                    </div>
                    <span className="text-[11px] text-slate-400 font-medium block">
                      Note: Boilerplate components like headers, footers, and scripts will be automatically removed prior to analysis.
                    </span>
                  </div>
                )}

                {activeTab === 'file' && (
                  <div className="py-6">
                    <label className="text-xs font-bold text-slate-400 uppercase tracking-wider block mb-2">Upload Article File (PDF or TXT)</label>
                    <div className="border-2 border-dashed border-slate-200 dark:border-slate-800 hover:border-violet-500 dark:hover:border-violet-500/60 rounded-xl p-8 text-center bg-slate-50/30 dark:bg-slate-900/20 transition-all relative">
                      <input
                        type="file"
                        accept=".pdf,.txt"
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        onChange={handleFileChange}
                      />
                      <FileCheck className="w-10 h-10 text-slate-400 mx-auto mb-3" />
                      <p className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                        {selectedFile ? selectedFile.name : 'Drag and drop your file here, or click to browse'}
                      </p>
                      <p className="text-xs text-slate-400 mt-1">Supports PDF and Plain Text (.txt) formats</p>
                    </div>
                  </div>
                )}

                {error && (
                  <div className="mt-4 flex items-center gap-3 p-4 rounded-xl border border-red-200 dark:border-red-900/50 bg-red-50/50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm">
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <span>{error}</span>
                  </div>
                )}

                {/* Submit trigger */}
                <div className="mt-6 flex justify-end">
                  <button
                    onClick={handleAnalyze}
                    disabled={loading}
                    className="w-full sm:w-auto px-8 py-3.5 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-700 hover:to-indigo-700 text-white rounded-xl font-bold shadow-lg shadow-indigo-500/20 dark:shadow-none hover:shadow-xl transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? (
                      <>
                        <RefreshCw className="w-5 h-5 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4" />
                        Analyze Article
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Loading Indicator with Step list */}
            {loading && (
              <div className="bg-white dark:bg-[#151b2a] border border-slate-200 dark:border-slate-800/80 rounded-2xl p-6 text-center space-y-4">
                <div className="flex justify-center">
                  <div className="w-12 h-12 border-4 border-violet-500/25 border-t-violet-600 rounded-full animate-spin" />
                </div>
                <div className="space-y-1">
                  <h3 className="font-bold text-base text-slate-800 dark:text-slate-200">FactTrack Inference Pipeline</h3>
                  <p className="text-sm text-slate-500 dark:text-slate-400 font-medium animate-pulse">{loadingMessage}</p>
                </div>
              </div>
            )}

            {/* Results Component */}
            {result && (
              <div className="space-y-6">
                
                {/* Result Top Overview */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  
                  {/* Categorization Card */}
                  <div className="bg-white dark:bg-[#151b2a] border border-slate-200 dark:border-slate-800/80 rounded-2xl p-6 shadow-sm">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="p-2 bg-indigo-500/10 dark:bg-indigo-400/10 text-indigo-600 dark:text-indigo-400 rounded-xl">
                        <TrendingUp className="w-5 h-5" />
                      </div>
                      <h2 className="font-extrabold text-lg text-slate-800 dark:text-slate-200">News Categorization</h2>
                    </div>

                    <div className="space-y-4">
                      <div className="bg-slate-50 dark:bg-slate-900/60 rounded-xl p-4 flex justify-between items-center border border-slate-100 dark:border-slate-800/50">
                        <div>
                          <span className="text-xs font-bold text-slate-400 uppercase block tracking-wider">Predicted Domain</span>
                          <span className="text-2xl font-black text-slate-800 dark:text-white">{result.category}</span>
                        </div>
                        <div className="text-right">
                          <span className="text-xs font-bold text-slate-400 uppercase block tracking-wider">Confidence</span>
                          <span className="text-xl font-bold text-indigo-600 dark:text-indigo-400">{(result.category_confidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>

                      {/* Prob distribution bar charts */}
                      <div className="space-y-2.5">
                        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">Probability Distribution</span>
                        {Object.entries(result.category_distribution).map(([cat, prob]) => (
                          <div key={cat} className="space-y-1">
                            <div className="flex justify-between text-xs font-semibold">
                              <span>{cat}</span>
                              <span>{(prob * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-2 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-gradient-to-r from-violet-500 to-indigo-500 rounded-full" 
                                style={{ width: `${prob * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Bias Card */}
                  <div className="bg-white dark:bg-[#151b2a] border border-slate-200 dark:border-slate-800/80 rounded-2xl p-6 shadow-sm">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="p-2 bg-rose-500/10 dark:bg-rose-400/10 text-rose-600 dark:text-rose-400 rounded-xl">
                        <Award className="w-5 h-5" />
                      </div>
                      <h2 className="font-extrabold text-lg text-slate-800 dark:text-slate-200">Editorial Bias</h2>
                    </div>

                    <div className="space-y-4">
                      <div className="bg-slate-50 dark:bg-slate-900/60 rounded-xl p-4 flex justify-between items-center border border-slate-100 dark:border-slate-800/50">
                        <div>
                          <span className="text-xs font-bold text-slate-400 uppercase block tracking-wider">Bias Label</span>
                          <span className={`text-2xl font-black ${
                            result.bias === 'Left' ? 'text-blue-500' :
                            result.bias === 'Right' ? 'text-red-500' : 'text-emerald-500'
                          }`}>{result.bias}</span>
                        </div>
                        <div className="text-right">
                          <span className="text-xs font-bold text-slate-400 uppercase block tracking-wider">Confidence</span>
                          <span className="text-xl font-bold text-rose-600 dark:text-rose-400">{(result.bias_confidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>

                      {/* Bias score gauge */}
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Continuous Bias Score</span>
                          <span className="text-xs font-bold px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
                            {result.bias_score.toFixed(2)} / 1.00
                          </span>
                        </div>
                        <div className="h-3 w-full bg-slate-100 dark:bg-slate-800 rounded-full relative overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-emerald-500 via-yellow-500 to-red-500 rounded-full"
                            style={{ width: '100%' }}
                          />
                          {/* Indicator pointer */}
                          <div 
                            className="absolute top-0 bottom-0 w-1.5 bg-black dark:bg-white border border-slate-300 shadow-md"
                            style={{ left: `calc(${result.bias_score * 100}% - 3px)` }}
                          />
                        </div>
                        <div className="flex justify-between text-[10px] font-bold text-slate-400 uppercase">
                          <span>Neutral ({result.bias_interpretation})</span>
                          <span>Biased</span>
                        </div>
                      </div>

                      {/* Bias Probabilities */}
                      <div className="grid grid-cols-3 gap-2 text-center pt-2">
                        <div className="p-2 bg-blue-50 dark:bg-blue-950/20 border border-blue-100 dark:border-blue-900/40 rounded-xl">
                          <span className="text-[10px] font-bold text-blue-500 dark:text-blue-400 block uppercase">Left</span>
                          <span className="text-base font-bold text-blue-600 dark:text-blue-400">{(result.bias_distribution.Left * 100).toFixed(0)}%</span>
                        </div>
                        <div className="p-2 bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-100 dark:border-emerald-900/40 rounded-xl">
                          <span className="text-[10px] font-bold text-emerald-500 dark:text-emerald-400 block uppercase">Center</span>
                          <span className="text-base font-bold text-emerald-600 dark:text-emerald-400">{(result.bias_distribution.Center * 100).toFixed(0)}%</span>
                        </div>
                        <div className="p-2 bg-red-50 dark:bg-red-950/20 border border-red-100 dark:border-red-900/40 rounded-xl">
                          <span className="text-[10px] font-bold text-red-500 dark:text-red-400 block uppercase">Right</span>
                          <span className="text-base font-bold text-red-600 dark:text-red-400">{(result.bias_distribution.Right * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Neutral vs Biased Circular Chart representation */}
                <div className="bg-white dark:bg-[#151b2a] border border-slate-200 dark:border-slate-800/80 rounded-2xl p-6">
                  <h3 className="font-extrabold text-base text-slate-800 dark:text-slate-200 mb-4">Neutral vs Biased Synthesis</h3>
                  
                  <div className="flex flex-col sm:flex-row items-center gap-8 justify-around">
                    
                    {/* Ring representation */}
                    <div className="relative w-36 h-36 flex items-center justify-center">
                      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="40" stroke="currentColor" className="text-slate-100 dark:text-slate-800" strokeWidth="10" fill="transparent" />
                        <circle 
                          cx="50" 
                          cy="50" 
                          r="40" 
                          stroke="currentColor" 
                          className="text-violet-500" 
                          strokeWidth="10" 
                          fill="transparent" 
                          strokeDasharray={251.2}
                          strokeDashoffset={251.2 - (251.2 * (result.neutral_vs_biased.Biased)) }
                        />
                      </svg>
                      <div className="absolute flex flex-col items-center">
                        <span className="text-2xl font-black">{(result.neutral_vs_biased.Biased * 100).toFixed(0)}%</span>
                        <span className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">Biased Rating</span>
                      </div>
                    </div>

                    <div className="flex-1 space-y-4 max-w-md w-full">
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm font-semibold">
                          <span className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-slate-200 dark:bg-slate-700" /> Neutral Content</span>
                          <span className="font-bold">{(result.neutral_vs_biased.Neutral * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-3 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                          <div className="h-full bg-slate-300 dark:bg-slate-600 rounded-full" style={{ width: `${result.neutral_vs_biased.Neutral * 100}%` }} />
                        </div>
                      </div>

                      <div className="space-y-1">
                        <div className="flex justify-between text-sm font-semibold">
                          <span className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-violet-500" /> Biased / Editorialized Content</span>
                          <span className="font-bold">{(result.neutral_vs_biased.Biased * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-3 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                          <div className="h-full bg-violet-500 rounded-full" style={{ width: `${result.neutral_vs_biased.Biased * 100}%` }} />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Interactive article text viewer with highlighted words */}
                <div className="bg-white dark:bg-[#151b2a] border border-slate-200 dark:border-slate-800/80 rounded-2xl p-6 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800/60 pb-3">
                    <h3 className="font-extrabold text-base text-slate-800 dark:text-slate-200">Saliency & Attributions Highlight</h3>
                    <div className="flex items-center gap-2 text-xs">
                      <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-blue-500/20 border-b border-blue-500" /> Left-weight</span>
                      <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-red-500/20 border-b border-red-500" /> Right-weight</span>
                      <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-emerald-500/20 border-b border-emerald-500" /> Neutral-weight</span>
                    </div>
                  </div>

                  <div className="max-h-72 overflow-y-auto bg-slate-50/50 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-800/50 rounded-xl p-4 text-sm leading-relaxed whitespace-pre-line text-slate-700 dark:text-slate-300">
                    {renderHighlightedText(inputText || result.content, result.important_words, result.bias)}
                  </div>
                </div>

                {/* Explainability & Dynamic explanation card */}
                <div className="bg-white dark:bg-[#151b2a] border border-slate-200 dark:border-slate-800/80 rounded-2xl p-6 space-y-4">
                  <h3 className="font-extrabold text-base text-slate-800 dark:text-slate-200">Explainable AI (XAI) Reasoner</h3>
                  
                  {/* Dynamic Explanation */}
                  <div className="bg-violet-500/5 dark:bg-violet-400/5 border border-violet-500/20 rounded-xl p-4 text-sm font-semibold text-slate-700 dark:text-violet-300/90 leading-relaxed">
                    "{result.explanation}"
                  </div>

                  {/* Influential Words table */}
                  <div className="space-y-3 pt-2">
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Top Influential Words (Attributions)</h4>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      {result.important_words.map((item, idx) => (
                        <div key={idx} className="p-3 bg-slate-50 dark:bg-slate-900/60 rounded-xl border border-slate-100 dark:border-slate-800/50 flex justify-between items-center text-xs font-semibold">
                          <span className="text-slate-700 dark:text-slate-300">{item.word}</span>
                          <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                            item.score > 0 ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400' : 'bg-red-500/10 text-red-600 dark:text-red-400'
                          }`}>
                            {item.score > 0 ? '+' : ''}{item.score.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Terminal Report Box (as requested) */}
                <div className="bg-[#0b0f19] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
                  <div className="bg-[#151b2a] px-4 py-2 border-b border-slate-800 flex justify-between items-center">
                    <span className="text-xs font-mono font-bold text-slate-400 uppercase tracking-widest">FactTrack Analysis Report Output</span>
                    <button 
                      onClick={() => {
                        const raw_text = document.getElementById('ascii-report').innerText;
                        navigator.clipboard.writeText(raw_text);
                        alert('Report copied to clipboard!');
                      }}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-750 text-[10px] font-bold font-mono text-slate-300 transition-all border border-slate-700"
                    >
                      COPY
                    </button>
                  </div>
                  <pre id="ascii-report" className="p-5 font-mono text-xs text-slate-300 overflow-x-auto leading-relaxed">
{`========================================================
FACTTRACK BIAS ANALYSIS REPORT
========================================================

Overall Bias
---
${result.bias.toUpperCase()}

Confidence
---
${(result.bias_confidence * 100).toFixed(1)}%

Bias Distribution
---
Left   : ${(result.bias_distribution.Left * 100).toFixed(0)}%
Center : ${(result.bias_distribution.Center * 100).toFixed(0)}%
Right  : ${(result.bias_distribution.Right * 100).toFixed(0)}%

Neutral vs Biased
---
Neutral : ${(result.neutral_vs_biased.Neutral * 100).toFixed(0)}%
Biased  : ${(result.neutral_vs_biased.Biased * 100).toFixed(0)}%

Bias Score
---
${result.bias_score.toFixed(2)} / 1.00

Interpretation:
${result.bias_interpretation}

========================================================
Explainability
========================================================

Most Influential Words
${result.important_words.slice(0, 5).map(w => `- ${w.word}`).join('\n')}

Reason
${result.explanation}
========================================================`}
                  </pre>
                </div>

              </div>
            )}
            
          </div>

          {/* Right panel / Side Utilities & History (4 columns) */}
          <div className="lg:col-span-4 space-y-6">
            
            {/* System Info card */}
            <div className="bg-white dark:bg-[#151b2a] border border-slate-200 dark:border-slate-800/80 rounded-2xl p-6 shadow-sm space-y-4">
              <div className="flex items-center gap-2 border-b border-slate-100 dark:border-slate-800/60 pb-3">
                <Settings className="w-5 h-5 text-indigo-500" />
                <h3 className="font-extrabold text-base text-slate-800 dark:text-slate-200">System Model Status</h3>
              </div>

              {healthStatus && (
                <div className="space-y-3 text-xs font-semibold">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500 dark:text-slate-400">News Domain Classifier</span>
                    <span className={`px-2 py-0.5 rounded text-[10px] ${
                      healthStatus.models_status.category_model_fine_tuned 
                      ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20' 
                      : 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20'
                    }`}>
                      {healthStatus.models_status.category_model_fine_tuned ? 'Fine-tuned Loaded' : 'Fallback pipeline'}
                    </span>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-slate-500 dark:text-slate-400">DeBERTa-v3 Bias Classifier</span>
                    <span className={`px-2 py-0.5 rounded text-[10px] ${
                      healthStatus.models_status.bias_model_fine_tuned 
                      ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20' 
                      : 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20'
                    }`}>
                      {healthStatus.models_status.bias_model_fine_tuned ? 'Fine-tuned Loaded' : 'Fallback pipeline'}
                    </span>
                  </div>

                  {/* Training controls */}
                  <div className="pt-2 border-t border-slate-100 dark:border-slate-800/60 space-y-2">
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Incremental Retraining</span>
                    
                    <div className="grid grid-cols-2 gap-2">
                      <button 
                        onClick={() => triggerTraining('category')}
                        className="py-2 px-2 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:bg-indigo-500 hover:text-white dark:hover:bg-indigo-600 rounded-lg text-center transition-all text-[11px] font-bold"
                      >
                        Retrain Category
                      </button>
                      <button 
                        onClick={() => triggerTraining('bias')}
                        className="py-2 px-2 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:bg-rose-500 hover:text-white dark:hover:bg-rose-600 rounded-lg text-center transition-all text-[11px] font-bold"
                      >
                        Retrain Bias
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Running tasks status */}
              {trainingTasks.length > 0 && (
                <div className="space-y-2.5 pt-2 border-t border-slate-100 dark:border-slate-800/60">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Active Training Tasks</span>
                  {trainingTasks.map(task => (
                    <div key={task.id} className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-800 text-[11px] font-semibold space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="font-bold uppercase text-indigo-500 dark:text-indigo-400">{task.type} Model</span>
                        <span className={`px-1.5 py-0.5 rounded text-[9px] uppercase ${
                          task.status === 'running' ? 'bg-amber-500/10 text-amber-500 animate-pulse' :
                          task.status === 'completed' ? 'bg-emerald-500/10 text-emerald-500' :
                          task.status === 'failed' ? 'bg-red-500/10 text-red-500' : 'bg-slate-300 text-slate-600'
                        }`}>
                          {task.status}
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400 leading-tight">{task.message}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* History card */}
            <div className="bg-white dark:bg-[#151b2a] border border-slate-200 dark:border-slate-800/80 rounded-2xl p-6 shadow-sm space-y-4">
              <div className="flex items-center gap-2 border-b border-slate-100 dark:border-slate-800/60 pb-3">
                <History className="w-5 h-5 text-indigo-500" />
                <h3 className="font-extrabold text-base text-slate-800 dark:text-slate-200">Prediction History</h3>
              </div>

              {history.length === 0 ? (
                <div className="text-center py-6 text-slate-400 text-xs font-semibold">
                  No predictions analyzed yet
                </div>
              ) : (
                <div className="space-y-3 max-h-96 overflow-y-auto pr-1">
                  {history.map((item, idx) => (
                    <div 
                      key={item.id || idx} 
                      onClick={() => loadFromHistory(item)}
                      className="group p-3 rounded-xl border border-slate-100 dark:border-slate-850 bg-slate-50/50 dark:bg-slate-900/30 hover:bg-slate-100/60 dark:hover:bg-slate-800/50 cursor-pointer transition-all space-y-1.5"
                    >
                      <div className="flex justify-between items-start gap-2">
                        <span className="font-bold text-xs line-clamp-1 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                          {item.title}
                        </span>
                        <span className="text-[9px] font-bold text-slate-400 uppercase bg-slate-200/50 dark:bg-slate-800 px-1.5 py-0.5 rounded">
                          {item.source}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center text-[10px] font-semibold">
                        <span className="text-slate-500 dark:text-slate-400">Category: <strong className="text-slate-700 dark:text-slate-200">{item.category}</strong></span>
                        <span className={`px-1.5 py-0.5 rounded ${
                          item.bias === 'Left' ? 'bg-blue-500/10 text-blue-600 dark:text-blue-400' :
                          item.bias === 'Right' ? 'bg-red-500/10 text-red-600 dark:text-red-400' :
                          'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                        }`}>{item.bias}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

          </div>

        </div>

      </main>

      {/* Footer */}
      <footer className="mt-16 border-t border-slate-200 dark:border-slate-800 py-8 bg-white dark:bg-[#0f172a] text-center text-xs text-slate-400 font-semibold">
        <p>FactTrack © 2026. Fine-tuned Transformer NLP Engine with Explainable AI.</p>
      </footer>

    </div>
  );
}
