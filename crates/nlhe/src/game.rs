//! NLHE game type: poker game state.
use super::*;
use rbp_cards::*;
use rbp_core::{self, Utility};
use rbp_gameplay::*;
use rbp_mccfr::*;

/// NLHE game state for CFR traversal.
///
/// Newtype wrapper around gameplay `Game` for NLHE-specific CFR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NlheGame<const P: usize = { rbp_core::N }>(Game<P>);

impl<const P: usize> NlheGame<P> {
    /// Current betting round (street).
    pub fn street(&self) -> Street {
        self.0.street()
    }
    /// Current observation (hole cards + board).
    pub fn sweat(&self) -> Observation {
        self.0.sweat()
    }
}

impl<const P: usize> CfrGame for NlheGame<P> {
    type E = NlheEdge;
    type T = NlheTurn;
    fn root() -> Self {
        Self(Game::<P>::root())
    }
    fn turn(&self) -> Self::T {
        NlheTurn::from(self.0.turn())
    }
    fn apply(&self, edge: Self::E) -> Self {
        let action = self.0.actionize(Edge::from(edge));
        let action = self.0.snap(action);
        Self(self.0.apply(action))
    }
    fn payoff(&self, turn: Self::T) -> Utility {
        self.0
            .settlements()
            .get(Turn::from(turn).position())
            .map(|settlement| settlement.won() as Utility)
            .expect("player index in bounds")
    }
}

impl<const P: usize> From<Game<P>> for NlheGame<P> {
    fn from(game: Game<P>) -> Self {
        Self(game)
    }
}
impl<const P: usize> From<NlheGame<P>> for Game<P> {
    fn from(game: NlheGame<P>) -> Self {
        game.0
    }
}
impl<const P: usize> AsRef<Game<P>> for NlheGame<P> {
    fn as_ref(&self) -> &Game<P> {
        &self.0
    }
}
