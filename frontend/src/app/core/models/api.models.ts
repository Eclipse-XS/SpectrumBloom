export type LLMProvider = 'ollama' | 'llamacpp' | 'openai' | 'gemini';

export type SemanticType =
  | 'numeric_continuous'
  | 'numeric_discrete'
  | 'categorical'
  | 'boolean'
  | 'datetime'
  | 'text'
  | 'identifier'
  | 'empty';

export interface UploadedDatasetMeta {
  filename: string;
  size_bytes: number;
  sha256: string;
  rows: number;
  columns: number;
  storage_backend: string;
  storage_ref: string;
  uploaded_at: string;
}

export interface ToolTraceItem {
  tool: string;
  code: string;
  stdout: string;
  result: unknown;
  error: string | null;
  figure_count: number;
  duration_ms: number;
  timestamp: string;
}

export interface FigurePayload {
  id: string;
  title: string;
  chart_kind: string;
  figure_json: string;
}

export interface EvaluationSummary {
  passed?: boolean;
  tool_execution_ok?: boolean;
  execution_pass_rate?: number;
  tool_selection_rate?: number;
  figure_validity_rate?: number;
  numeric_grounding_passed?: boolean;
  numeric_grounding_ratio?: number;
  grounding_score?: number;
  judge_score?: number | null;
  judge_reason?: string | null;
  issues?: string[];
  [key: string]: unknown;
}

export interface ChatMessageRecord {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  figures: FigurePayload[];
  tool_traces: ToolTraceItem[];
  evaluation: EvaluationSummary | null;
  status: 'success' | 'failed';
  error: string | null;
  latency_ms: number | null;
  retries_used: number;
  client_turn_id?: string | null;
}

export interface CreateSessionRequest {
  provider?: LLMProvider;
  model?: string;
  api_key?: string | null;
}

export interface CreateSessionResponse {
  session_id: string;
  thread_id: string;
  provider: string;
  model: string;
  has_data: boolean;
}

export interface SessionStatusResponse {
  session_id: string;
  thread_id: string;
  provider: string;
  model: string;
  has_data: boolean;
  message_count: number;
  figure_count: number;
  dataset: UploadedDatasetMeta | null;
  created_at: string;
  updated_at: string;
}

export interface SessionHistoryResponse {
  session_id: string;
  messages: ChatMessageRecord[];
  last_tool_traces: ToolTraceItem[];
}

export interface UpdateProviderRequest {
  provider: LLMProvider;
  model: string;
  api_key?: string | null;
}

export interface NumericStats {
  mean: number | null;
  std: number | null;
  min: number | null;
  q25: number | null;
  median: number | null;
  q75: number | null;
  max: number | null;
  skew: number | null;
  kurtosis: number | null;
  zeros_pct: number | null;
}

export interface TopValueItem {
  value: string;
  count: number;
  pct: number;
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  semantic_type: SemanticType;
  non_null_count: number;
  null_count: number;
  null_pct: number;
  unique_count: number;
  unique_pct: number;
  sample_values: unknown[];
  numeric_stats: NumericStats | null;
  top_values: TopValueItem[] | null;
}

export interface DatasetProfile {
  filename: string;
  rows: number;
  columns: number;
  duplicate_rows: number;
  duplicate_pct: number;
  memory_mb: number;
  completeness_pct: number;
  column_profiles: ColumnProfile[];
  warnings: string[];
}

export interface MissingColumnInfo {
  column: string;
  missing_count: number;
  missing_pct: number;
}

export interface MissingCooccurrencePair {
  column_a: string;
  column_b: string;
  null_correlation: number;
}

export interface MissingnessReport {
  total_cells: number;
  total_missing_cells: number;
  overall_missing_pct: number;
  rows_with_any_missing: number;
  rows_with_any_missing_pct: number;
  columns_with_missing: MissingColumnInfo[];
  missing_cooccurrence_pairs: MissingCooccurrencePair[];
}

export interface CorrelationPair {
  column_a: string;
  column_b: string;
  pearson_r: number | null;
  spearman_rho: number | null;
}

export interface CategoricalAssociation {
  column_a: string;
  column_b: string;
  cramers_v: number;
}

export interface OutlierSummary {
  column: string;
  iqr_lower: number;
  iqr_upper: number;
  iqr_outlier_count: number;
  iqr_outlier_pct: number;
  zscore_outlier_count: number;
  mad_outlier_count: number;
}

export interface NormalityCheck {
  column: string;
  sample_size: number;
  skewness: number | null;
  kurtosis: number | null;
  test_name: string;
  statistic: number | null;
  p_value: number | null;
  is_normal_alpha_05: boolean | null;
}

export interface GroupTestResult {
  categorical_column: string;
  numeric_column: string;
  test: string;
  group_count: number;
  statistic: number | null;
  p_value: number;
}

export interface StatisticalReport {
  missingness: MissingnessReport;
  numeric_columns: string[];
  categorical_columns: string[];
  pearson_matrix: Record<string, Record<string, number | null>>;
  spearman_matrix: Record<string, Record<string, number | null>>;
  top_correlations: CorrelationPair[];
  categorical_associations: CategoricalAssociation[];
  outliers: OutlierSummary[];
  multivariate_anomaly_count: number;
  normality_checks: NormalityCheck[];
  group_tests: GroupTestResult[];
}

export interface DatasetPreviewResponse {
  filename: string;
  shape: [number, number];
  columns: string[];
  dtypes: Record<string, string>;
  preview: Record<string, unknown>[];
}

export interface FullEDAReportResponse {
  profile: DatasetProfile;
  statistics: StatisticalReport;
  figures: FigurePayload[];
}

export interface QueryRequest {
  query: string;
  client_turn_id?: string | null;
}

export interface QueryResponse {
  session_id: string;
  answer: string;
  status: 'success' | 'failed';
  error: string | null;
  timed_out: boolean;
  figures: FigurePayload[];
  tool_traces: ToolTraceItem[];
  retries_used: number;
  latency_ms: number;
  evaluation: EvaluationSummary | null;
}

export interface MLExperimentRequest {
  target_column: string;
  feature_columns?: string[] | null;
  task_type?: 'auto' | 'classification' | 'regression';
  test_size?: number;
  cv_folds?: number;
  random_state?: number;
}

export interface MLModelResult {
  model_name: string;
  cv_primary_metric_name: string;
  cv_primary_mean: number;
  cv_primary_std: number;
  test_metrics: Record<string, number | null>;
}

export interface FeatureImportanceItem {
  feature: string;
  importance_mean: number;
  importance_std: number;
}

export interface MLExperimentResponse {
  target_column: string;
  task_type: 'classification' | 'regression';
  train_rows: number;
  test_rows: number;
  feature_count: number;
  best_model: string;
  models: MLModelResult[];
  feature_importances: FeatureImportanceItem[];
  figures: FigurePayload[];
  notes: string[];
}

export interface HealthLLMStatus {
  provider: string;
  status: string;
  configured_model?: string;
  available_models?: string[];
  detail?: string;
}

export interface HealthResponse {
  status: 'ok' | 'degraded';
  version: string;
  session_repository: string;
  dataset_storage: {
    backend: string;
    status: string;
  };
  sandbox_backend: string;
  sandbox_dynamic_python_enabled: boolean;
  llm: HealthLLMStatus;
  observability: {
    opentelemetry: boolean;
    langfuse: boolean;
  };
  auth_required?: boolean;
  max_upload_mb?: number;
  default_provider?: LLMProvider;
  default_model?: string;
}

export interface ApiError {
  status: number;
  detail: string;
  kind: 'auth' | 'validation' | 'timeout' | 'sandbox' | 'network' | 'server' | 'not_found';
}
