//! Complete-information game history for **training time**.
//!
//! # Information Boundary
//!
//! | Type | Perspective | Context |
//! |------|-------------|---------|
//! | `Partial` | Hero only | Inference (strategy lookup) |
//! | `Perfect` | Both hands | Training (CFR traversal) |
//!
//! During CFR training, we traverse the game tree knowing both players' cards
//! (god's view), but strategies are indexed only by `NlheInfo` (public edges +
//! private bucket). `Perfect` stores the complete root state needed for reach
//! probability computation and counterfactual value calculation.
//!
//! # Conversions
//!
//! ```text
//! Perfect::from((partial, hole))  ────►  Perfect     (add opponent info)
//!                                 ◄────
//! perfect.partial(hero)                              (erase opponent info)
//!
//! partial.histories() ─────►  Vec<(Obs, Perfect)>  (iterate all opponents)
//! ```
//!
//! # Blind Handling
//!
//! Like `Partial`, blinds are constant and NOT stored in `actions`.
//! The `root` field stores a POST-blind game state.
use super::*;
use rbp_cards::*;

/// Complete game history with both players' cards known.
///
/// Stores root game state (POST-blind, with all cards set) and action sequence
/// (excluding blinds). Game states are derived by applying actions to root.
#[derive(Debug, Clone)]
pub struct Perfect<const P: usize = { rbp_core::N }> {
    root: Game<P>,
    actions: Vec<Action>,
}

impl<const P: usize> From<(&Partial<P>, Hole)> for Perfect<P> {
    /// Creates history from partial with assumed opponent hole.
    ///
    /// Hero is derived from `partial.turn()`. The root game has:
    /// - Hero's cards from `partial.seen()`
    /// - Opponent's cards from `hole` parameter
    /// - Blinds already posted (POST-blind state)
    fn from((partial, hole): (&Partial<P>, Hole)) -> Self {
        let preblind = partial.base().assume(partial.turn(), hole);
        let root = Game::<P>::blinds()
            .into_iter()
            .fold(preblind, |mut g, a| g.consume(a));
        Self {
            root,
            actions: partial.actions().to_vec(),
        }
    }
}

impl<const P: usize> Recall<P> for Perfect<P> {
    fn root(&self) -> Game<P> {
        self.root
    }
    fn actions(&self) -> &[Action] {
        &self.actions
    }
}

#[allow(dead_code)]
impl<const P: usize> Perfect<P> {
    /// Erases opponent information, returning hero's perspective.
    ///
    /// Extracts hero's hole cards and board from root, discarding
    /// opponent's cards. Inverse of construction from `(&Partial, Hole)`.
    fn erase(&self, hero: Turn) -> Partial<P> {
        let hole = self.root.seats()[hero.position()].cards();
        let board = Hand::from(self.root.board());
        let observation = Observation::from((Hand::from(hole), board));
        let actions = self
            .actions
            .iter()
            .filter(|a| a.is_choice())
            .cloned()
            .collect();
        Partial::from((hero, observation, actions))
    }
}
