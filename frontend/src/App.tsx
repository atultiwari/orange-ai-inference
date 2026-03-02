import React, { useState, useRef } from 'react';
import axios from 'axios';
import { UploadCloud, FileType, CheckCircle, Activity, ChevronRight, BarChart } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Types
type Feature = {
  name: string;
  type: 'continuous' | 'discrete' | 'other';
  values?: string[];
};

type ModelMetadata = {
  filename: string;
  features: Feature[];
  class_variable: {
    name: string;
    values: string[];
  };
};

type PredictionResult = {
  prediction: string;
  probabilities: Record<string, number>;
};

const API_BASE = 'http://localhost:8080/api';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string>('');

  const [formValues, setFormValues] = useState<Record<string, string | number>>({});
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predictError, setPredictError] = useState<string>('');

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setMetadata(null);
      setPrediction(null);
      setFormValues({});
      setUploadError('');
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.pkcls')) {
        setFile(droppedFile);
        setMetadata(null);
        setPrediction(null);
        setFormValues({});
        setUploadError('');
      } else {
        setUploadError('Only .pkcls Orange models are supported.');
      }
    }
  };

  const uploadModel = async () => {
    if (!file) return;

    setIsUploading(true);
    setUploadError('');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setMetadata(response.data);
      
      // Initialize form values
      const initialValues: Record<string, string | number> = {};
      response.data.features.forEach((f: Feature) => {
          if (f.type === 'discrete' && f.values && f.values.length > 0) {
              initialValues[f.name] = f.values[0];
          } else {
              initialValues[f.name] = '';
          }
      });
      setFormValues(initialValues);
      
    } catch (err: any) {
      setUploadError(err.response?.data?.detail || 'Failed to analyze model.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleInputChange = (featureName: string, value: string) => {
    setFormValues(prev => ({
      ...prev,
      [featureName]: value
    }));
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!metadata) return;

    setIsPredicting(true);
    setPredictError('');
    
    // Ensure numeric values are numbers where appropriate
    const processedValues: Record<string, any> = {};
    for (const [key, val] of Object.entries(formValues)) {
        const feature = metadata.features.find(f => f.name === key);
        if (feature?.type === 'continuous') {
           processedValues[key] = val === '' ? 0 : Number(val);
        } else {
           processedValues[key] = val;
        }
    }

    const formData = new FormData();
    formData.append('filename', metadata.filename);
    formData.append('features_json', JSON.stringify(processedValues));

    try {
      const response = await axios.post(`${API_BASE}/predict`, formData);
      setPrediction(response.data);
    } catch (err: any) {
      setPredictError(err.response?.data?.detail || 'Prediction failed.');
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans p-6 text-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="flex items-center gap-4 border-b border-slate-200 pb-6 pt-4">
          <div className="bg-gradient-to-br from-teal-400 to-teal-600 p-3 rounded-xl text-white shadow-lg shadow-teal-500/30">
            <Activity className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-900 to-slate-600">
              Orange AI Inference
            </h1>
            <p className="text-slate-500 font-medium mt-1">
              Dynamic Model Analyzer & Predictor
            </p>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Upload & Form */}
          <div className="lg:col-span-7 space-y-6">
            
            {/* Uploader Card */}
            <motion.div 
               initial={{ opacity: 0, y: 20 }}
               animate={{ opacity: 1, y: 0 }}
               className="glass-panel p-8 rounded-2xl relative overflow-hidden"
            >
              <div className="absolute top-0 left-0 w-2 h-full bg-teal-500 rounded-l-2xl"></div>
              <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                <UploadCloud className="text-teal-600 w-6 h-6" />
                Step 1: Upload Model
              </h2>
              
              <div 
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300
                  ${file ? 'border-teal-400 bg-teal-50/50' : 'border-slate-300 hover:border-teal-400 hover:bg-slate-50'}
                `}
              >
                <input 
                  type="file" 
                  accept=".pkcls" 
                  className="hidden" 
                  ref={fileInputRef}
                  onChange={handleFileChange}
                />
                
                {file ? (
                  <div className="flex flex-col items-center gap-3">
                    <div className="p-4 bg-teal-100 text-teal-700 rounded-full shadow-sm">
                      <FileType className="w-8 h-8" />
                    </div>
                    <div>
                      <p className="font-semibold text-slate-800 text-lg">{file.name}</p>
                      <p className="text-sm text-slate-500">{(file.size / 1024).toFixed(2)} KB</p>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3 text-slate-500">
                     <UploadCloud className="w-12 h-12 text-slate-400" />
                     <p className="font-medium">Click to browse or drag and drop</p>
                     <p className="text-sm">Only Orange Data Mining .pkcls files</p>
                  </div>
                )}
              </div>

              {uploadError && (
                <div className="mt-4 p-3 bg-red-50 text-red-600 rounded-lg text-sm border border-red-100">
                  {uploadError}
                </div>
              )}

              {file && !metadata && (
                <div className="mt-6 flex justify-end">
                  <button 
                    onClick={uploadModel}
                    disabled={isUploading}
                    className="bg-slate-900 hover:bg-slate-800 text-white px-6 py-2.5 rounded-lg font-medium transition-colors shadow-md disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  >
                    {isUploading ? 'Analyzing...' : 'Analyze Model'}
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              )}
            </motion.div>

            {/* Dynamic Form Card */}
            <AnimatePresence>
              {metadata && (
                <motion.div
                  initial={{ opacity: 0, height: 0, marginTop: 0 }}
                  animate={{ opacity: 1, height: 'auto', marginTop: 24 }}
                  exit={{ opacity: 0, height: 0, marginTop: 0 }}
                  className="glass-panel p-8 rounded-2xl relative overflow-hidden"
                >
                  <div className="absolute top-0 left-0 w-2 h-full bg-blue-500 rounded-l-2xl"></div>
                  <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                    <CheckCircle className="text-blue-500 w-6 h-6" />
                    Step 2: Enter Input Features
                  </h2>

                  <form onSubmit={handlePredict} className="space-y-5">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {metadata.features.map((feature, idx) => (
                        <div key={idx} className="space-y-1.5">
                          <label className="block text-sm font-semibold text-slate-700">
                            {feature.name}
                            <span className="ml-2 text-xs font-normal text-slate-400 px-2 py-0.5 bg-slate-100 rounded-md">
                              {feature.type}
                            </span>
                          </label>
                          
                          {feature.type === 'discrete' && feature.values ? (
                            <select
                              value={formValues[feature.name] as string}
                              onChange={(e) => handleInputChange(feature.name, e.target.value)}
                              className="input-field"
                              required
                            >
                              <option value="" disabled>Select {feature.name}</option>
                              {feature.values.map(val => (
                                <option key={val} value={val}>{val}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="number"
                              step="any"
                              value={formValues[feature.name] || ''}
                              onChange={(e) => handleInputChange(feature.name, e.target.value)}
                              className="input-field"
                              placeholder={`Enter ${feature.name}`}
                              required
                            />
                          )}
                        </div>
                      ))}
                    </div>

                    {predictError && (
                      <div className="p-3 bg-red-50 text-red-600 rounded-lg text-sm border border-red-100">
                        {predictError}
                      </div>
                    )}

                    <div className="pt-4 flex justify-end">
                      <button 
                        type="submit"
                        disabled={isPredicting}
                        className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-xl font-semibold transition-all shadow-lg hover:shadow-blue-600/30 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transform hover:-translate-y-0.5"
                      >
                        {isPredicting ? 'Running Inference...' : 'Generate Prediction'}
                      </button>
                    </div>
                  </form>
                </motion.div>
              )}
            </AnimatePresence>

          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-5">
            <AnimatePresence mode="popLayout">
              {prediction ? (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="bg-white border border-slate-200 shadow-xl rounded-2xl p-8 sticky top-8"
                >
                  <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-indigo-100 text-indigo-600 rounded-lg">
                      <BarChart className="w-6 h-6" />
                    </div>
                    <h2 className="text-2xl font-bold text-slate-800">Results</h2>
                  </div>

                  <div className="mb-8">
                    <p className="text-sm text-slate-500 font-medium mb-2 uppercase tracking-wider">Predicted Class (target: {metadata?.class_variable.name})</p>
                    <div className="bg-gradient-to-r from-indigo-50 to-blue-50 border border-indigo-100 rounded-xl p-5 shadow-inner">
                      <span className="text-3xl font-extrabold text-indigo-700 break-words">
                        {prediction.prediction}
                      </span>
                    </div>
                  </div>

                  <div>
                    <p className="text-sm text-slate-500 font-medium mb-4 uppercase tracking-wider">Class Probabilities</p>
                    <div className="space-y-4">
                      {Object.entries(prediction.probabilities).sort((a,b) => b[1] - a[1]).map(([className, prob]) => (
                        <div key={className}>
                          <div className="flex justify-between text-sm font-medium mb-1.5">
                            <span className="text-slate-700 truncate mr-2" title={className}>{className}</span>
                            <span className={className === prediction.prediction ? 'text-indigo-600 font-bold' : 'text-slate-500'}>
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                            <motion.div 
                              initial={{ width: 0 }}
                              animate={{ width: `${prob * 100}%` }}
                              transition={{ duration: 0.8, ease: "easeOut" }}
                              className={`h-2.5 rounded-full ${className === prediction.prediction ? 'bg-indigo-500' : 'bg-slate-300'}`}
                            ></motion.div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              ) : (
                <div className="h-full min-h-[400px] border-2 border-dashed border-slate-200 rounded-2xl flex flex-col items-center justify-center text-slate-400 bg-slate-50/50 p-8 text-center">
                  <BarChart className="w-16 h-16 mb-4 text-slate-300" />
                  <p className="text-lg font-medium text-slate-500">Awaiting Prediction</p>
                  <p className="text-sm mt-2 max-w-xs">Upload a model and enter features to see the prediction results and confidence scores here.</p>
                </div>
              )}
            </AnimatePresence>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;
