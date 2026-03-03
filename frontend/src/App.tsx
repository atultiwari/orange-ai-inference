import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { UploadCloud, FileType, CheckCircle, Activity, ChevronRight, BarChart, FileSpreadsheet, Download } from 'lucide-react';
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

type CsvColumn = {
  name: string;
  type: 'continuous' | 'discrete';
  unique_values: string[];
};

type CsvMetadata = {
  filename: string;
  columns: CsvColumn[];
  total_rows: number;
};

const API_BASE = '/api';

function App() {
  const [mode, setMode] = useState<'single' | 'batch'>('single');

  // Shared - Step 1
  const [file, setFile] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string>('');

  // Single Entry - Step 2
  const [formValues, setFormValues] = useState<Record<string, string | number>>({});
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predictError, setPredictError] = useState<string>('');

  // Batch Mode - Step 2 (CSV Upload)
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [csvMetadata, setCsvMetadata] = useState<CsvMetadata | null>(null);
  const [isUploadingCsv, setIsUploadingCsv] = useState(false);
  const [uploadCsvError, setUploadCsvError] = useState<string>('');

  // Batch Mode - Step 3/4 (Mapping)
  const [colMap, setColMap] = useState<Record<string, string>>({}); // modelFeatureName -> csvColName
  const [valMap, setValMap] = useState<Record<string, Record<string, string>>>({}); // modelFeatureName -> { csvVal -> modelVal }
  const [isPredictingBatch, setIsPredictingBatch] = useState(false);
  const [batchError, setBatchError] = useState<string>('');
  const [batchResultUrl, setBatchResultUrl] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const csvInputRef = useRef<HTMLInputElement>(null);

  // Auto-map columns when csvMetadata and metadata are both available
  useEffect(() => {
    if (metadata && csvMetadata) {
      const initialColMap: Record<string, string> = {};
      metadata.features.forEach(feat => {
        const exactMatch = csvMetadata.columns.find(c => c.name === feat.name);
        if (exactMatch) {
          initialColMap[feat.name] = exactMatch.name;
        } else {
          initialColMap[feat.name] = ''; // requires mapping
        }
      });
      setColMap(initialColMap);
      setValMap({});
    }
  }, [metadata, csvMetadata]);

  // Model Upload Functions
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setMetadata(null);
      setPrediction(null);
      setFormValues({});
      setUploadError('');
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

  // Single Predict Functions
  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!metadata) return;
    setIsPredicting(true);
    setPredictError('');
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

  // CSV Upload Functions
  const handleCsvChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setCsvFile(e.target.files[0]);
      setCsvMetadata(null);
      setUploadCsvError('');
      setBatchResultUrl(null);
    }
  };

  const uploadCsv = async () => {
    if (!csvFile) return;
    setIsUploadingCsv(true);
    setUploadCsvError('');
    const formData = new FormData();
    formData.append('file', csvFile);
    try {
      const response = await axios.post(`${API_BASE}/upload_csv`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setCsvMetadata(response.data);
    } catch (err: any) {
      setUploadCsvError(err.response?.data?.detail || 'Failed to analyze CSV.');
    } finally {
      setIsUploadingCsv(false);
    }
  };

  // Batch Predict Functions
  const handleBatchPredict = async () => {
    if (!csvFile || !metadata || !csvMetadata) return;
    setIsPredictingBatch(true);
    setBatchError('');
    setBatchResultUrl(null);

    const formData = new FormData();
    formData.append('file', csvFile);
    formData.append('filename', metadata.filename);
    formData.append('column_mapping', JSON.stringify(colMap));
    formData.append('value_mapping', JSON.stringify(valMap));

    try {
      const response = await axios.post(`${API_BASE}/predict_batch`, formData, {
        responseType: 'blob', // Important: indicates that we expect a binary response
      });

      const url = window.URL.createObjectURL(new Blob([response.data], { type: 'text/csv' }));
      setBatchResultUrl(url);
    } catch (err: any) {
      let msg = 'Batch prediction failed.';
      if (err.response && err.response.data instanceof Blob) {
        const text = await err.response.data.text();
        try {
          msg = JSON.parse(text).detail || msg;
        } catch (e) { }
      }
      setBatchError(msg);
    } finally {
      setIsPredictingBatch(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans p-6 text-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">

        {/* Header */}
        <header className="flex items-center justify-between border-b border-slate-200 pb-6 pt-4">
          <div className="flex items-center gap-4">
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
          </div>

          {/* Mode Switcher */}
          <div className="flex bg-slate-200/60 p-1.5 rounded-xl">
            <button
              onClick={() => setMode('single')}
              className={`px-5 py-2 rounded-lg font-medium transition-all ${mode === 'single' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Single Entry
            </button>
            <button
              onClick={() => setMode('batch')}
              className={`px-5 py-2 rounded-lg font-medium transition-all ${mode === 'batch' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Batch Upload
            </button>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

          {/* Left Column: Upload & Form */}
          <div className={`space-y-6 ${mode === 'single' ? 'lg:col-span-7' : 'lg:col-span-12'}`}>

            {/* Step 1: Uploader Card (Shared) */}
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
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300
                  ${file ? 'border-teal-400 bg-teal-50/50' : 'border-slate-300 hover:border-teal-400 hover:bg-slate-50'}
                `}
              >
                <input type="file" accept=".pkcls" className="hidden" ref={fileInputRef} onChange={handleFileChange} />
                {file ? (
                  <div className="flex flex-col items-center gap-3">
                    <div className="p-4 bg-teal-100 text-teal-700 rounded-full shadow-sm">
                      <FileType className="w-8 h-8" />
                    </div>
                    <div>
                      <p className="font-semibold text-slate-800 text-lg">{file.name}</p>
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

              {uploadError && <div className="mt-4 p-3 bg-red-50 text-red-600 rounded-lg text-sm border border-red-100">{uploadError}</div>}

              {file && !metadata && (
                <div className="mt-6 flex justify-end">
                  <button onClick={uploadModel} disabled={isUploading} className="btn-primary">
                    {isUploading ? 'Analyzing...' : 'Analyze Model'} <ChevronRight className="w-4 h-4 ml-2 inline" />
                  </button>
                </div>
              )}
            </motion.div>

            {/* Step 2: SINGLE PREDICTION FORM */}
            <AnimatePresence>
              {metadata && mode === 'single' && (
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
                            {feature.name} <span className="ml-2 text-xs font-normal text-slate-400 px-2 py-0.5 bg-slate-100 rounded-md">{feature.type}</span>
                          </label>

                          {feature.type === 'discrete' && feature.values ? (
                            <select
                              value={formValues[feature.name] as string}
                              onChange={(e) => setFormValues(prev => ({ ...prev, [feature.name]: e.target.value }))}
                              className="input-field" required
                            >
                              <option value="" disabled>Select {feature.name}</option>
                              {feature.values.map(val => <option key={val} value={val}>{val}</option>)}
                            </select>
                          ) : (
                            <input
                              type="number" step="any"
                              value={formValues[feature.name] || ''}
                              onChange={(e) => setFormValues(prev => ({ ...prev, [feature.name]: e.target.value }))}
                              className="input-field" placeholder={`Enter ${feature.name}`} required
                            />
                          )}
                        </div>
                      ))}
                    </div>

                    {predictError && <div className="p-3 bg-red-50 text-red-600 rounded-lg text-sm border border-red-100">{predictError}</div>}

                    <div className="pt-4 flex justify-end">
                      <button type="submit" disabled={isPredicting} className="btn-primary bg-blue-600 hover:bg-blue-700 hover:shadow-blue-600/30">
                        {isPredicting ? 'Running Inference...' : 'Generate Prediction'}
                      </button>
                    </div>
                  </form>
                </motion.div>
              )}
            </AnimatePresence>

            {/* BATCH UPLOAD SECTIONS */}
            <AnimatePresence>
              {metadata && mode === 'batch' && (
                <motion.div
                  initial={{ opacity: 0, marginTop: 0 }}
                  animate={{ opacity: 1, marginTop: 24 }}
                  exit={{ opacity: 0, marginTop: 0 }}
                  className="space-y-6"
                >
                  {/* Step 2: Upload CSV */}
                  <div className="glass-panel p-8 rounded-2xl relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-2 h-full bg-purple-500 rounded-l-2xl"></div>
                    <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                      <FileSpreadsheet className="text-purple-600 w-6 h-6" /> Step 2: Upload Dataset (.csv)
                    </h2>

                    <div
                      onClick={() => csvInputRef.current?.click()}
                      className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300
                        ${csvFile ? 'border-purple-400 bg-purple-50/50' : 'border-slate-300 hover:border-purple-400 hover:bg-slate-50'}
                      `}
                    >
                      <input type="file" accept=".csv" className="hidden" ref={csvInputRef} onChange={handleCsvChange} />
                      {csvFile ? (
                        <div className="flex flex-col items-center gap-3">
                          <p className="font-semibold text-slate-800 text-lg">{csvFile.name}</p>
                        </div>
                      ) : (
                        <p className="font-medium text-slate-500">Click to browse for a CSV dataset</p>
                      )}
                    </div>
                    {uploadCsvError && <div className="mt-4 p-3 bg-red-50 text-red-600 rounded-lg text-sm border border-red-100">{uploadCsvError}</div>}

                    {csvFile && !csvMetadata && (
                      <div className="mt-6 flex justify-end">
                        <button onClick={uploadCsv} disabled={isUploadingCsv} className="btn-primary bg-purple-600 hover:bg-purple-700">
                          {isUploadingCsv ? 'Reading CSV...' : 'Analyze Dataset'} <ChevronRight className="w-4 h-4 ml-2 inline" />
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Step 3: Mapping */}
                  {csvMetadata && (
                    <div className="glass-panel p-8 rounded-2xl relative overflow-hidden">
                      <div className="absolute top-0 left-0 w-2 h-full bg-orange-500 rounded-l-2xl"></div>
                      <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                        <Activity className="text-orange-500 w-6 h-6" /> Step 3: Map Columns & Values
                      </h2>

                      <div className="space-y-8">
                        {metadata.features.map(feat => {
                          const selectedCsvColName = colMap[feat.name];
                          const csvCol = csvMetadata.columns.find(c => c.name === selectedCsvColName);
                          // Needs value resolution if discrete and map is set
                          let needsValMap = false;
                          if (feat.type === 'discrete' && feat.values && csvCol?.type === 'discrete') {
                            // simplistic check
                            needsValMap = true;
                          }

                          return (
                            <div key={feat.name} className="p-5 border border-slate-200 rounded-xl bg-white shadow-sm">
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                                <div>
                                  <label className="block text-sm font-semibold text-slate-700 mb-1">
                                    Model Focus: <span className="text-indigo-600">{feat.name}</span>
                                    <span className="ml-2 text-xs font-normal text-slate-400 px-2 py-0.5 bg-slate-100 rounded-md">{feat.type}</span>
                                  </label>
                                  {feat.type === 'discrete' && (
                                    <p className="text-xs text-slate-500 mt-1">Accepts: {feat.values?.join(', ')}</p>
                                  )}
                                </div>
                                <div>
                                  <select
                                    value={colMap[feat.name] || ''}
                                    onChange={e => setColMap(prev => ({ ...prev, [feat.name]: e.target.value }))}
                                    className="input-field"
                                  >
                                    <option value="">-- Ignore / Missing --</option>
                                    {csvMetadata.columns.map(c => (
                                      <option key={c.name} value={c.name}>{c.name} ({c.type})</option>
                                    ))}
                                  </select>
                                </div>
                              </div>

                              {/* Value Mapping UI if both are discrete */}
                              {needsValMap && csvCol?.unique_values && (
                                <div className="mt-4 pt-4 border-t border-slate-100 pl-4 border-l-4 border-l-slate-200">
                                  <p className="text-sm font-medium text-slate-600 mb-3">Map CSV values to Model expected values:</p>
                                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                                    {csvCol.unique_values.map(uVal => {
                                      const currentMappedObj = valMap[feat.name] || {};
                                      // Default: try to exact match
                                      const exactMatch = feat.values?.find(fv => fv === uVal);
                                      const mappedTo = currentMappedObj[uVal] !== undefined ? currentMappedObj[uVal] : (exactMatch || '');

                                      return (
                                        <div key={uVal} className="flex items-center gap-2 text-sm">
                                          <span className="truncate w-1/3 text-slate-500 font-mono text-xs">{uVal} &rarr;</span>
                                          <select
                                            value={mappedTo}
                                            onChange={e => setValMap(prev => ({
                                              ...prev,
                                              [feat.name]: { ...(prev[feat.name] || {}), [uVal]: e.target.value }
                                            }))}
                                            className="py-1 px-2 border border-slate-300 rounded text-xs flex-1"
                                          >
                                            <option value="">(Error/NaN)</option>
                                            {feat.values?.map(fv => <option key={fv} value={fv}>{fv}</option>)}
                                          </select>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>

                      {batchError && <div className="mt-6 p-4 bg-red-50 text-red-600 rounded-xl text-sm border border-red-200">{batchError}</div>}

                      <div className="mt-8 flex justify-end border-t border-slate-200 pt-6">
                        <button
                          onClick={handleBatchPredict}
                          disabled={isPredictingBatch}
                          className="btn-primary bg-slate-900 hover:bg-slate-800 hover:shadow-lg w-full md:w-auto"
                        >
                          {isPredictingBatch ? 'Processing Data...' : 'Run Batch Analysis'}
                        </button>
                      </div>

                      {batchResultUrl && (
                        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-6 p-6 bg-green-50 border border-green-200 rounded-xl flex items-center justify-between">
                          <div>
                            <h3 className="text-green-800 font-bold mb-1">Batch Processing Complete</h3>
                            <p className="text-green-700 text-sm">Your annotated CSV is ready for download.</p>
                          </div>
                          <a href={batchResultUrl} download={`Annotated_${metadata.filename}.csv`} className="bg-green-600 hover:bg-green-700 text-white px-5 py-2.5 rounded-lg text-sm font-semibold flex items-center gap-2 shadow-sm">
                            <Download className="w-4 h-4" /> Download Results
                          </a>
                        </motion.div>
                      )}

                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

          </div>

          {/* Right Column: Results (ONLY FOR SINGLE MODE) */}
          {mode === 'single' && (
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
                        <span className="text-3xl font-extrabold text-indigo-700 break-words">{prediction.prediction}</span>
                      </div>
                    </div>
                    <div>
                      <p className="text-sm text-slate-500 font-medium mb-4 uppercase tracking-wider">Class Probabilities</p>
                      <div className="space-y-4">
                        {Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1]).map(([className, prob]) => (
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
                  <div className="h-full min-h-[400px] border-2 border-dashed border-slate-200 rounded-2xl flex flex-col items-center justify-center text-slate-400 bg-slate-50/50 p-8 text-center sticky top-8">
                    <BarChart className="w-16 h-16 mb-4 text-slate-300" />
                    <p className="text-lg font-medium text-slate-500">Awaiting Prediction</p>
                    <p className="text-sm mt-2 max-w-xs">Upload a model and enter features to see the prediction results and confidence scores here.</p>
                  </div>
                )}
              </AnimatePresence>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}

export default App;
