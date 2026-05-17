use super::API;
use rbp_cards::*;
use rbp_core::*;
use rbp_gameplay::*;
use rbp_mccfr::Profile;
use rbp_mccfr::Solver;
use rbp_nlhe::*;
use rbp_transport::Density;
use actix_web::HttpResponse;
use actix_web::Responder;
use actix_web::web;
use rand::prelude::*;
use rand::distr::weighted::WeightedIndex;

pub async fn replace_obs(api: web::Data<API>, req: web::Json<ReplaceObs>) -> impl Responder {
    match Observation::try_from(req.obs.as_str()) {
        Err(_) => HttpResponse::BadRequest().body("invalid observation format"),
        Ok(obs) => match api.replace_obs(obs).await {
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
            Ok(new) => HttpResponse::Ok().json(new.to_string()),
        },
    }
}
pub async fn exp_wrt_str(api: web::Data<API>, req: web::Json<SetStreets>) -> impl Responder {
    match Street::try_from(req.street.as_str()) {
        Err(_) => HttpResponse::BadRequest().body("invalid street format"),
        Ok(street) => match api.exp_wrt_str(street).await {
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
            Ok(row) => HttpResponse::Ok().json(row),
        },
    }
}
pub async fn exp_wrt_abs(api: web::Data<API>, req: web::Json<ReplaceAbs>) -> impl Responder {
    match Abstraction::try_from(req.wrt.as_str()) {
        Err(_) => HttpResponse::BadRequest().body("invalid abstraction format"),
        Ok(abs) => match api.exp_wrt_abs(abs).await {
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
            Ok(row) => HttpResponse::Ok().json(row),
        },
    }
}
pub async fn exp_wrt_obs(api: web::Data<API>, req: web::Json<RowWrtObs>) -> impl Responder {
    match Observation::try_from(req.obs.as_str()) {
        Err(_) => HttpResponse::BadRequest().body("invalid observation format"),
        Ok(obs) => match api.exp_wrt_obs(obs).await {
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
            Ok(row) => HttpResponse::Ok().json(row),
        },
    }
}
pub async fn nbr_any_wrt_abs(api: web::Data<API>, req: web::Json<ReplaceAbs>) -> impl Responder {
    match Abstraction::try_from(req.wrt.as_str()) {
        Err(_) => HttpResponse::BadRequest().body("invalid abstraction format"),
        Ok(abs) => match api.nbr_any_wrt_abs(abs).await {
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
            Ok(row) => HttpResponse::Ok().json(row),
        },
    }
}
pub async fn nbr_abs_wrt_abs(api: web::Data<API>, req: web::Json<ReplaceOne>) -> impl Responder {
    let wrt = Abstraction::try_from(req.wrt.as_str());
    let abs = Abstraction::try_from(req.abs.as_str());
    match (wrt, abs) {
        (Err(_), _) => HttpResponse::BadRequest().body("invalid abstraction format"),
        (_, Err(_)) => HttpResponse::BadRequest().body("invalid abstraction format"),
        (Ok(wrt), Ok(abs)) => match api.nbr_abs_wrt_abs(wrt, abs).await {
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
            Ok(row) => HttpResponse::Ok().json(row),
        },
    }
}
pub async fn nbr_obs_wrt_abs(api: web::Data<API>, req: web::Json<ReplaceRow>) -> impl Responder {
    let wrt = Abstraction::try_from(req.wrt.as_str());
    let obs = Observation::try_from(req.obs.as_str());
    match (wrt, obs) {
        (Err(_), _) => HttpResponse::BadRequest().body("invalid abstraction format"),
        (_, Err(_)) => HttpResponse::BadRequest().body("invalid observation format"),
        (Ok(abs), Ok(obs)) => match api.nbr_obs_wrt_abs(abs, obs).await {
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
            Ok(rows) => HttpResponse::Ok().json(rows),
        },
    }
}
pub async fn kfn_wrt_abs(api: web::Data<API>, req: web::Json<ReplaceAbs>) -> impl Responder {
    match Abstraction::try_from(req.wrt.as_str()) {
        Err(_) => HttpResponse::BadRequest().body("invalid abstraction format"),
        Ok(abs) => match api.kfn_wrt_abs(abs).await {
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
            Ok(rows) => HttpResponse::Ok().json(rows),
        },
    }
}
pub async fn knn_wrt_abs(api: web::Data<API>, req: web::Json<ReplaceAbs>) -> impl Responder {
    match Abstraction::try_from(req.wrt.as_str()) {
        Err(_) => HttpResponse::BadRequest().body("invalid abstraction format"),
        Ok(abs) => match api.knn_wrt_abs(abs).await {
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
            Ok(rows) => HttpResponse::Ok().json(rows),
        },
    }
}
pub async fn kgn_wrt_abs(api: web::Data<API>, req: web::Json<ReplaceAll>) -> impl Responder {
    match Abstraction::try_from(req.wrt.as_str()) {
        Err(_) => HttpResponse::BadRequest().body("invalid abstraction format"),
        Ok(wrt) => {
            let obs = req
                .neighbors
                .iter()
                .map(|string| string.as_str())
                .map(Observation::try_from)
                .filter_map(|result| result.ok())
                .filter(|o| o.street() == wrt.street())
                .chain((0..).map(|_| Observation::from(wrt.street())))
                .take(5)
                .collect::<Vec<_>>();
            match api.kgn_wrt_abs(wrt, obs).await {
                Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
                Ok(rows) => HttpResponse::Ok().json(rows),
            }
        }
    }
}
pub async fn hst_wrt_abs(api: web::Data<API>, req: web::Json<AbsHist>) -> impl Responder {
    match Abstraction::try_from(req.abs.as_str()) {
        Err(_) => HttpResponse::BadRequest().body("invalid abstraction format"),
        Ok(abs) => match api.hst_wrt_abs(abs).await {
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
            Ok(rows) => HttpResponse::Ok().json(rows),
        },
    }
}
pub async fn hst_wrt_obs(api: web::Data<API>, req: web::Json<ObsHist>) -> impl Responder {
    match Observation::try_from(req.obs.as_str()) {
        Err(_) => HttpResponse::BadRequest().body("invalid observation format"),
        Ok(obs) => match api.hst_wrt_obs(obs).await {
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
            Ok(rows) => HttpResponse::Ok().json(rows),
        },
    }
}
pub async fn decide(
    blueprint: web::Data<&'static Flagship>,
    req: web::Json<DecideRequest>,
) -> impl Responder {
    let obs_str = format!("{} ~ {}", req.hole.join(" "), req.board.join(" "));
    let seen = match Observation::try_from(obs_str.as_str()) {
        Ok(o) => o,
        Err(e) => return HttpResponse::BadRequest().body(format!("invalid cards: {}", e)),
    };
    let actions = match req
        .actions
        .iter()
        .map(|s| Action::try_from(s.as_str()))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(a) => a,
        Err(e) => return HttpResponse::BadRequest().body(format!("invalid action: {}", e)),
    };
    let partial = match Partial::try_build(Turn::from(req.pov), seen, actions) {
        Ok(p) => p,
        Err(e) => return HttpResponse::BadRequest().body(format!("invalid game state: {}", e)),
    };
    let game = partial.head();
    let bp: &Flagship = &**blueprint;
    let abstraction = bp.encoder().abstraction(&partial.seen());
    let info = NlheInfo::from((&partial, abstraction));
    let policy = bp.profile().averaged_distribution(&info);
    let edges: Vec<_> = policy.support().collect();
    let weights: Vec<f32> = edges.iter().map(|e| policy.density(e)).collect();
    let action = WeightedIndex::new(&weights)
        .ok()
        .map(|dist| edges[dist.sample(&mut rand::rng())])
        .map(|edge| game.actionize(Edge::from(edge)))
        .unwrap_or_else(|| *game.legal().choose(&mut rand::rng()).unwrap());
    let legal: Vec<String> = game.legal().iter().map(|a| a.to_string()).collect();
    HttpResponse::Ok().json(serde_json::json!({
        "action": action.to_string(),
        "legal": legal,
    }))
}

pub async fn blueprint(api: web::Data<API>, req: web::Json<GetPolicy>) -> impl Responder {
    let hero = Turn::try_from(req.turn.as_str());
    let seen = Observation::try_from(req.seen.as_str());
    let path = req
        .past
        .iter()
        .map(|string| string.as_str())
        .map(Action::try_from)
        .collect::<Result<Vec<_>, _>>();
    match (hero, seen, path) {
        (Ok(hero), Ok(seen), Ok(path)) => match Partial::try_build(hero, seen, path) {
            Err(e) => HttpResponse::BadRequest().body(format!("invalid action sequence: {}", e)),
            Ok(recall) => match api.policy(recall).await {
                Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
                Ok(Some(strategy)) => HttpResponse::Ok().json(strategy),
                Ok(None) => HttpResponse::Ok().json(serde_json::Value::Null),
            },
        },
        _ => HttpResponse::BadRequest().body("invalid recall format"),
    }
}
