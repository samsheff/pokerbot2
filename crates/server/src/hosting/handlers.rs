use super::*;
use actix_web::HttpRequest;
use actix_web::HttpResponse;
use actix_web::Responder;
use actix_web::web;
use rbp_auth;
use rbp_core::ID;
use rbp_gameroom::Room;
use std::sync::Arc;

pub struct RoomHosting {
    casino: Option<Arc<Casino>>,
}

impl RoomHosting {
    pub fn enabled(casino: Arc<Casino>) -> Self {
        Self {
            casino: Some(casino),
        }
    }

    pub fn disabled() -> Self {
        Self { casino: None }
    }

    fn casino(&self) -> Result<&Arc<Casino>, HttpResponse> {
        self.casino.as_ref().ok_or_else(|| {
            HttpResponse::ServiceUnavailable()
                .body("room hosting requires a loaded 6-player blueprint")
        })
    }
}

pub async fn start(hosting: web::Data<RoomHosting>) -> impl Responder {
    match hosting.casino() {
        Ok(casino) => match casino.start().await {
            Ok(id) => HttpResponse::Ok().json(serde_json::json!({ "room_id": id.to_string() })),
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
        },
        Err(response) => response,
    }
}

pub async fn leave(hosting: web::Data<RoomHosting>, path: web::Path<uuid::Uuid>) -> impl Responder {
    match hosting.casino() {
        Ok(casino) => match casino.close(ID::from(path.into_inner())).await {
            Ok(()) => HttpResponse::Ok().json(serde_json::json!({ "status": "left" })),
            Err(e) => HttpResponse::NotFound().body(e.to_string()),
        },
        Err(response) => response,
    }
}

pub async fn enter(
    hosting: web::Data<RoomHosting>,
    tokens: web::Data<rbp_auth::Crypto>,
    path: web::Path<uuid::Uuid>,
    query: web::Query<std::collections::HashMap<String, String>>,
    body: web::Payload,
    req: HttpRequest,
) -> impl Responder {
    let casino = match hosting.casino() {
        Ok(casino) => casino,
        Err(response) => return response.map_into_right_body(),
    };
    let id: ID<Room> = ID::from(path.into_inner());
    query
        .get("token")
        .and_then(|t| tokens.decode(t).ok())
        .filter(|c| !c.expired())
        .inspect(|c| log::info!("authenticated user {} entering room {}", c.usr, id))
        .map(std::mem::drop)
        .unwrap_or_else(|| log::info!("anonymous user entering room {}", id));
    match actix_ws::handle(&req, body) {
        Ok((response, session, stream)) => match casino.bridge(id, session, stream).await {
            Ok(()) => response.map_into_left_body(),
            Err(e) => HttpResponse::NotFound()
                .body(e.to_string())
                .map_into_right_body(),
        },
        Err(e) => HttpResponse::InternalServerError()
            .body(e.to_string())
            .map_into_right_body(),
    }
}
