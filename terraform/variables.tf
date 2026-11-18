variable "project_id" {
  description = "Google Cloud Project ID hosting the Travel Concierge Agent."
  type        = string
}

variable "region" {
  description = "Primary Google Cloud region for Cloud Run, Vertex AI, and Firestore."
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment label (dev, staging, prod)."
  type        = string
  default     = "prod"
}

variable "service_name" {
  description = "Cloud Run v2 service name for the ADK Travel Concierge Agent."
  type        = string
  default     = "travel-concierge-agent"
}

variable "container_image" {
  description = "Artifact Registry container image URI for the Travel Concierge Agent."
  type        = string
  default     = "us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/agents/travel-concierge-agent:latest"
}

variable "fast_model" {
  description = "Low-latency Gemini model tier for routine single-step travel lookups."
  type        = string
  default     = "gemini-2.5-flash"
}

variable "reasoning_model" {
  description = "Deep-reasoning Gemini model tier for multi-city synthesis and budget optimization."
  type        = string
  default     = "gemini-2.5-pro"
}

variable "high_value_payment_threshold_usd" {
  description = "USD threshold above which Human-in-the-Loop confirmation is strictly enforced."
  type        = number
  default     = 500
}

variable "min_instances" {
  description = "Minimum number of warm Cloud Run container instances."
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of autoscaled Cloud Run container instances."
  type        = number
  default     = 10
}
