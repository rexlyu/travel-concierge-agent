locals {
  required_apis = toset([
    "aiplatform.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "firestore.googleapis.com",
    "discoveryengine.googleapis.com",
    "dlp.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
  ])

  agent_iam_roles = toset([
    "roles/aiplatform.user",
    "roles/secretmanager.secretAccessor",
    "roles/datastore.user",
    "roles/discoveryengine.viewer",
    "roles/dlp.user",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
  ])

  common_labels = {
    application = "travel-concierge-agent"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# 1. Enable required Google Cloud Platform APIs
resource "google_project_service" "required_apis" {
  for_each = local.required_apis

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# 2. Dedicated least-privilege Service Account for the ADK Agent
resource "google_service_account" "agent_runtime" {
  account_id   = "${var.service_name}-sa"
  display_name = "Travel Concierge ADK Agent Runtime Service Account"
  description  = "Least-privilege identity for Vertex AI, Secret Manager, Firestore, DLP, and Cloud Trace."
  project      = var.project_id

  depends_on = [google_project_service.required_apis]
}

resource "google_project_iam_member" "agent_least_privilege" {
  for_each = local.agent_iam_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.agent_runtime.email}"
}

# 3. Secret Manager secret for external payment gateway credentials
resource "google_secret_manager_secret" "payment_gateway_token" {
  project   = var.project_id
  secret_id = "travel-payment-gateway-token"
  labels    = local.common_labels

  replication {
    auto {}
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret_iam_member" "payment_secret_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.payment_gateway_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent_runtime.email}"
}

# 4. Managed Firestore Native database for persistent multi-turn session state
resource "google_firestore_database" "session_store" {
  project                     = var.project_id
  name                        = "travel-concierge-sessions"
  location_id                 = var.region
  type                        = "FIRESTORE_NATIVE"
  concurrency_mode            = "OPTIMISTIC"
  app_engine_integration_mode = "DISABLED"
  deletion_policy             = "DELETE"

  depends_on = [google_project_service.required_apis]
}

# 5. Vertex AI Search (Discovery Engine) data store for grounded travel advisories & memory
resource "google_discovery_engine_data_store" "travel_grounding_store" {
  project                     = var.project_id
  location                    = "global"
  data_store_id               = "${var.service_name}-grounding-store"
  display_name                = "Travel Concierge Advisories and Grounding Store"
  industry_vertical           = "GENERIC"
  content_config              = "CONTENT_REQUIRED"
  solution_types              = ["SOLUTION_TYPE_SEARCH"]
  create_advanced_site_search = false

  depends_on = [google_project_service.required_apis]
}

# 6. Sensitive Data Protection (Cloud DLP) inspect & de-identification templates
resource "google_data_loss_prevention_inspect_template" "pii_inspect" {
  parent       = "projects/${var.project_id}/locations/${var.region}"
  display_name = "Travel Concierge PII Inspect Template"
  description  = "Detects traveler emails, phone numbers, payment cards, passports, and SSNs."

  inspect_config {
    min_likelihood = "POSSIBLE"

    info_types {
      name = "EMAIL_ADDRESS"
    }
    info_types {
      name = "PHONE_NUMBER"
    }
    info_types {
      name = "CREDIT_CARD_NUMBER"
    }
    info_types {
      name = "PASSPORT"
    }
    info_types {
      name = "US_SOCIAL_SECURITY_NUMBER"
    }
    info_types {
      name = "PERSON_NAME"
    }
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_data_loss_prevention_deidentify_template" "pii_deidentify" {
  parent       = "projects/${var.project_id}/locations/${var.region}"
  display_name = "Travel Concierge PII Redaction Template"
  description  = "Replaces detected traveler PII with info-type placeholders prior to logging and persistence."

  deidentify_config {
    info_type_transformations {
      transformations {
        primitive_transformation {
          replace_with_info_type_config {}
        }
      }
    }
  }

  depends_on = [google_project_service.required_apis]
}

# 7. Cloud Run v2 Service deploying the ADK Travel Concierge Agent container
resource "google_cloud_run_v2_service" "agent_service" {
  name     = var.service_name
  location = var.region
  project  = var.project_id
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  labels   = local.common_labels

  template {
    service_account = google_service_account.agent_runtime.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }

      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }

      env {
        name  = "TRAVEL_FAST_MODEL"
        value = var.fast_model
      }

      env {
        name  = "TRAVEL_REASONING_MODEL"
        value = var.reasoning_model
      }

      env {
        name  = "HIGH_VALUE_PAYMENT_THRESHOLD_USD"
        value = tostring(var.high_value_payment_threshold_usd)
      }

      env {
        name  = "FIRESTORE_DATABASE_ID"
        value = google_firestore_database.session_store.name
      }

      env {
        name  = "VERTEX_SEARCH_DATA_STORE_ID"
        value = google_discovery_engine_data_store.travel_grounding_store.data_store_id
      }

      env {
        name  = "DLP_INSPECT_TEMPLATE_NAME"
        value = google_data_loss_prevention_inspect_template.pii_inspect.id
      }

      env {
        name  = "DLP_DEIDENTIFY_TEMPLATE_NAME"
        value = google_data_loss_prevention_deidentify_template.pii_deidentify.id
      }

      env {
        name  = "ENABLE_CLOUD_TRACE"
        value = "true"
      }

      env {
        name  = "ENABLE_CLOUD_DLP"
        value = "true"
      }

      env {
        name = "TRAVEL_PAYMENT_GATEWAY_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.payment_gateway_token.secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [
    google_project_iam_member.agent_least_privilege,
    google_secret_manager_secret_iam_member.payment_secret_accessor,
  ]
}
