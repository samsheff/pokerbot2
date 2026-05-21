//! Unified Backend Server
//!
//! Combines analysis API routes and live game hosting routes
//! into a single actix-web server.
//!
//! ## Submodules
//!
//! - [`analysis`] — Training result analysis and query interface
//! - [`hosting`] — WebSocket game hosting infrastructure

pub mod analysis;
pub mod hosting;

// Re-export main types (not handlers or Client, which conflict between modules)
pub use analysis::API;
pub use analysis::CLI;
pub use analysis::Query;
pub use hosting::Casino;
pub use hosting::RoomHandle;

use actix_cors::Cors;
use actix_web::App;
use actix_web::HttpResponse;
use actix_web::HttpServer;
use actix_web::Responder;
use actix_web::middleware::Logger;
use actix_web::web;
use rbp_database::Hydrate;
use rbp_nlhe::*;
use std::sync::Arc;
use tokio_postgres::Client;

#[derive(Clone, Copy, Debug)]
pub struct ServerOptions {
    pub max_inference_players: usize,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            max_inference_players: rbp_core::MAX_PLAYERS,
        }
    }
}

pub struct BlueprintRegistry {
    p2: Option<&'static Flagship2>,
    p3: Option<&'static Flagship3>,
    p4: Option<&'static Flagship4>,
    p5: Option<&'static Flagship5>,
    p6: Option<&'static Flagship6>,
}

impl BlueprintRegistry {
    async fn load(client: Arc<Client>, max_inference_players: usize) -> Self {
        ensure_blueprint_players(client.clone()).await;
        Self {
            p2: load_blueprint::<2>(client.clone(), max_inference_players).await,
            p3: load_blueprint::<3>(client.clone(), max_inference_players).await,
            p4: load_blueprint::<4>(client.clone(), max_inference_players).await,
            p5: load_blueprint::<5>(client.clone(), max_inference_players).await,
            p6: load_blueprint::<6>(client.clone(), max_inference_players).await,
        }
    }
    pub fn p2(&self) -> Option<&'static Flagship2> {
        self.p2
    }
    pub fn p3(&self) -> Option<&'static Flagship3> {
        self.p3
    }
    pub fn p4(&self) -> Option<&'static Flagship4> {
        self.p4
    }
    pub fn p5(&self) -> Option<&'static Flagship5> {
        self.p5
    }
    pub fn p6(&self) -> Option<&'static Flagship6> {
        self.p6
    }
}

async fn ensure_blueprint_players(client: Arc<Client>) {
    client
        .batch_execute(const_format::concatcp!(
            "ALTER TABLE ",
            rbp_database::BLUEPRINT,
            " ADD COLUMN IF NOT EXISTS players SMALLINT NOT NULL DEFAULT 6;
             ALTER TABLE ",
            rbp_database::BLUEPRINT,
            " DROP CONSTRAINT IF EXISTS blueprint_past_present_choices_edge_key;
             DROP INDEX IF EXISTS idx_blueprint_upsert;
             DROP INDEX IF EXISTS idx_blueprint_bucket;
             CREATE UNIQUE INDEX IF NOT EXISTS idx_blueprint_upsert_players ON ",
            rbp_database::BLUEPRINT,
            " (players, present, past, choices, edge);"
        ))
        .await
        .expect("ensure blueprint players");
}

async fn load_blueprint<const P: usize>(
    client: Arc<Client>,
    max_inference_players: usize,
) -> Option<&'static FlagshipFor<P>> {
    if P > max_inference_players {
        log::info!(
            "skipping {}-player blueprint; max inference players is {}",
            P,
            max_inference_players
        );
        return None;
    }
    let rows = client
        .query_one(
            const_format::concatcp!(
                "SELECT COUNT(*) FROM ",
                rbp_database::BLUEPRINT,
                " WHERE players = $1"
            ),
            &[&(P as i16)],
        )
        .await
        .ok()?
        .get::<_, i64>(0);
    if rows > 0 {
        log::info!("loading {}-player blueprint for inference", P);
        Some(Box::leak(Box::new(FlagshipFor::<P>::hydrate(client).await)))
    } else {
        log::info!(
            "{}-player blueprint is not trained; skipping inference load",
            P
        );
        None
    }
}

async fn health(client: web::Data<Arc<Client>>) -> impl Responder {
    match client
        .execute("SELECT 1", &[])
        .await
        .inspect_err(|e| log::error!("health check failed: {}", e))
    {
        Ok(_) => HttpResponse::Ok().body("ok"),
        Err(_) => HttpResponse::ServiceUnavailable().body("database unavailable"),
    }
}

#[rustfmt::skip]
pub async fn run() -> Result<(), std::io::Error> {
    run_with_options(ServerOptions::default()).await
}

#[rustfmt::skip]
pub async fn run_with_options(options: ServerOptions) -> Result<(), std::io::Error> {
    let max_inference_players = rbp_core::validate_players(options.max_inference_players)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let client = rbp_database::db().await;
    let api = web::Data::new(analysis::API::new(client.clone()));
    let crypto = web::Data::new(rbp_auth::Crypto::from_env());
    log::info!(
        "loading blueprints for inference up to {} players",
        max_inference_players
    );
    let registry = web::Data::new(BlueprintRegistry::load(client.clone(), max_inference_players).await);
    let room_hosting = web::Data::new(match registry.p6() {
        Some(blueprint) => {
            log::info!("room hosting enabled with 6-player blueprint");
            hosting::handlers::RoomHosting::enabled(Arc::new(hosting::Casino::new(
                client.clone(),
                blueprint,
            )))
        }
        None => {
            log::warn!("room hosting disabled; 6-player blueprint is not loaded");
            hosting::handlers::RoomHosting::disabled()
        }
    });
    let client = web::Data::new(client);
    log::info!("starting unified server");
    HttpServer::new(move || {
        App::new()
            .wrap(Logger::new("%r %s %Ts"))
            .wrap(
                Cors::default()
                    .allow_any_origin()
                    .allow_any_method()
                    .allow_any_header(),
            )
            .app_data(api.clone())
            .app_data(room_hosting.clone())
            .app_data(crypto.clone())
            .app_data(registry.clone())
            .app_data(client.clone())
            .route("/health", web::get().to(health))
            .service(
                web::scope("/auth")
                    .route("/register", web::post().to(rbp_auth::register))
                    .route("/logout", web::post().to(rbp_auth::logout))
                    .route("/login", web::post().to(rbp_auth::login))
                    .route("/me", web::get().to(rbp_auth::me)),
            )
            .service(
                web::scope("/room")
                    .route("/start", web::post().to(hosting::handlers::start))
                    .route("/enter/{room_id}", web::get().to(hosting::handlers::enter))
                    .route("/leave/{room_id}", web::post().to(hosting::handlers::leave)),
            )
            .service(
                web::scope("/api")
                    .route("/replace-obs", web::post().to(analysis::handlers::replace_obs))
                    .route("/nbr-any-abs", web::post().to(analysis::handlers::nbr_any_wrt_abs))
                    .route("/nbr-obs-abs", web::post().to(analysis::handlers::nbr_obs_wrt_abs))
                    .route("/nbr-abs-abs", web::post().to(analysis::handlers::nbr_abs_wrt_abs))
                    .route("/nbr-kfn-abs", web::post().to(analysis::handlers::kfn_wrt_abs))
                    .route("/nbr-knn-abs", web::post().to(analysis::handlers::knn_wrt_abs))
                    .route("/nbr-kgn-abs", web::post().to(analysis::handlers::kgn_wrt_abs))
                    .route("/exp-wrt-str", web::post().to(analysis::handlers::exp_wrt_str))
                    .route("/exp-wrt-abs", web::post().to(analysis::handlers::exp_wrt_abs))
                    .route("/exp-wrt-obs", web::post().to(analysis::handlers::exp_wrt_obs))
                    .route("/hst-wrt-abs", web::post().to(analysis::handlers::hst_wrt_abs))
                    .route("/hst-wrt-obs", web::post().to(analysis::handlers::hst_wrt_obs))
                    .route("/blueprint", web::post().to(analysis::handlers::blueprint))
                    .route("/decide", web::post().to(analysis::handlers::decide)),
            )
    })
    .workers(6)
    .bind(std::env::var("BIND_ADDR").expect("BIND_ADDR must be set"))?
    .run()
    .await
}
