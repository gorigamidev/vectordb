use crate::dsl::{execute_line, DslOutput};
use crate::engine::TensorDb;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use toon_format::encode_default;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

struct AppState {
    db: Arc<Mutex<TensorDb>>,
}

const MAX_COMMAND_LENGTH: usize = 16 * 1024; // 16KB
const QUERY_TIMEOUT_SECS: u64 = 30;

#[derive(Deserialize, utoipa::IntoParams)]
struct ExecuteParams {
    /// Format of the output: 'toon' (default) or 'json'
    #[serde(default = "default_format")]
    format: String,
}

fn default_format() -> String {
    "toon".to_string()
}

#[derive(Deserialize, utoipa::ToSchema)]
pub struct ExecuteRequest {
    command: String,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct ExecuteResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<DslOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(OpenApi)]
#[openapi(
    paths(
        execute_command,
        health_check
    ),
    components(
        schemas(ExecuteRequest, ExecuteResponse)
    ),
    tags(
        (name = "VectorDB", description = "LINAL Analytical Engine API")
    )
)]
struct ApiDoc;

pub async fn start_server(db: Arc<Mutex<TensorDb>>, port: u16) {
    let state = Arc::new(AppState { db });

    let app = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/health", get(health_check))
        .route("/execute", post(execute_command))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    println!("Server running at http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Health check", body = String)
    )
)]
async fn health_check() -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::OK, Json(serde_json::json!({ "status": "ok" })))
}

#[utoipa::path(
    post,
    path = "/execute",
    request_body = String,
    params(
        ExecuteParams
    ),
    responses(
        (status = 200, description = "Execution result", body = ExecuteResponse)
    )
)]
async fn execute_command(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ExecuteParams>,
    headers: axum::http::HeaderMap,
    body: String,
) -> impl IntoResponse {
    // Determine if request is JSON (legacy) or plain text (preferred)
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("text/plain");

    let command = if content_type.contains("application/json") {
        // Legacy JSON format: {"command": "..."}
        // Log deprecation warning
        eprintln!("[DEPRECATED] JSON request format is deprecated. Use Content-Type: text/plain with raw DSL command instead.");

        match serde_json::from_str::<ExecuteRequest>(&body) {
            Ok(req) => req.command,
            Err(_) => {
                // If JSON parsing fails, treat as raw DSL
                body.trim().to_string()
            }
        }
    } else {
        // Preferred: raw DSL text
        body.trim().to_string()
    };

    if command.len() > MAX_COMMAND_LENGTH {
        return (
            StatusCode::BAD_REQUEST,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            serde_json::to_string(&ExecuteResponse {
                status: "error".to_string(),
                result: None,
                error: Some(format!(
                    "Command too long (max {} bytes)",
                    MAX_COMMAND_LENGTH
                )),
            })
            .unwrap(),
        )
            .into_response();
    }

    if command.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            serde_json::to_string(&ExecuteResponse {
                status: "error".to_string(),
                result: None,
                error: Some("Command cannot be empty".to_string()),
            })
            .unwrap(),
        )
            .into_response();
    }

    // Wrap execution in timeout and spawn_blocking to keep server responsive
    let db_arc = state.db.clone();
    let command_clone = command.clone();

    let exec_result = tokio::time::timeout(
        std::time::Duration::from_secs(QUERY_TIMEOUT_SECS),
        tokio::task::spawn_blocking(move || {
            let mut db = db_arc.lock().unwrap();
            execute_line(&mut db, &command_clone, 1)
        }),
    )
    .await;

    let response = match exec_result {
        Ok(Ok(Ok(output))) => {
            let result = match output {
                DslOutput::None => None,
                _ => Some(output),
            };
            ExecuteResponse {
                status: "ok".to_string(),
                result,
                error: None,
            }
        }
        Ok(Ok(Err(e))) => ExecuteResponse {
            status: "error".to_string(),
            result: None,
            error: Some(format!("{}", e)),
        },
        Ok(Err(e)) => ExecuteResponse {
            status: "error".to_string(),
            result: None,
            error: Some(format!("Execution task panicked: {}", e)),
        },
        Err(_) => ExecuteResponse {
            status: "error".to_string(),
            result: None,
            error: Some(format!("Query timed out after {}s", QUERY_TIMEOUT_SECS)),
        },
    };

    // Serialize based on requested format
    match params.format.as_str() {
        "json" => {
            // JSON format (opt-in)
            let body = serde_json::to_string(&response).unwrap_or_else(|e| {
                format!(
                    "{{\"status\": \"error\", \"error\": \"Serialization failed: {}\"}}",
                    e
                )
            });
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                body,
            )
                .into_response()
        }
        _ => {
            // TOON format (default)
            let body = encode_default(&response)
                .unwrap_or_else(|e| format!("status: error\nerror: Serialization failed: {}", e));
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "text/toon")],
                body,
            )
                .into_response()
        }
    }
}
