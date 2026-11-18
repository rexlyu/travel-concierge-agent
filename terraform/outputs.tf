output "cloud_run_service_uri" {
  description = "HTTPS endpoint URI of the deployed Cloud Run v2 Travel Concierge Agent."
  value       = google_cloud_run_v2_service.agent_service.uri
}

output "agent_service_account_email" {
  description = "Email of the least-privilege runtime Service Account."
  value       = google_service_account.agent_runtime.email
}

output "payment_gateway_secret_id" {
  description = "Secret Manager resource ID storing the payment gateway token."
  value       = google_secret_manager_secret.payment_gateway_token.id
}

output "firestore_database_name" {
  description = "Firestore Native database name backing persistent traveler session state."
  value       = google_firestore_database.session_store.name
}

output "vertex_search_data_store_id" {
  description = "Vertex AI Search data store ID for grounded travel advisories and memory."
  value       = google_discovery_engine_data_store.travel_grounding_store.data_store_id
}

output "dlp_inspect_template_id" {
  description = "Sensitive Data Protection inspect template resource ID."
  value       = google_data_loss_prevention_inspect_template.pii_inspect.id
}

output "dlp_deidentify_template_id" {
  description = "Sensitive Data Protection de-identification template resource ID."
  value       = google_data_loss_prevention_deidentify_template.pii_deidentify.id
}
