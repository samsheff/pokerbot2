use super::*;
use rbp_cards::*;
use rbp_core::*;
use std::ops::Not;

/// The memoryless state of a poker hand.
///
/// `Game` is the core state machine for No-Limit Texas Hold'em, encoding everything
/// needed to determine legal actions and compute payoffs. It manages player stacks,
/// the pot, community cards, and whose turn it is to act.
///
/// # Architecture
///
/// The design is deliberately memoryless: `Game` contains only the current state,
/// not the history of how we got here. This makes it suitable as a CFR node
/// representation where states can be reached via different action sequences.
///
/// State transitions are functional—[`apply`](Self::apply) returns a new `Game`
/// rather than mutating in place. This enables tree traversal without undo logic.
///
/// # Fields
///
/// - `pot` — Total chips in the center (including current street bets)
/// - `board` — Community cards (0–5 depending on street)
/// - `seats` — Per-player state (stack, stake, status, hole cards)
/// - `dealer` — Button position (0–N-1)
/// - `ticker` — Action counter for determining whose turn it is
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Game {
    pot: Chips,
    board: Board,
    seats: [Seat; N],
    dealer: Position,
    ticker: Position,
}

impl Default for Game {
    fn default() -> Self {
        let mut deck = Deck::new();
        Self {
            pot: 0,
            board: Board::empty(),
            seats: [(); N]
                .map(|_| deck.hole())
                .map(|h| (h, STACK))
                .map(Seat::from),
            dealer: 0usize,
            ticker: 1usize,
        }
    }
}

/// Game tree entry points.
impl Game {
    /// Creates the canonical starting state for MCCFR traversal.
    ///
    /// Returns a game with blinds posted and ready for the dealer's first
    /// decision. Default stack is 100bb with P0 on the button.
    pub fn root() -> Self {
        let mut game = Self::default();
        game.act(game.posts());
        game.act(game.posts());
        game
    }
    /// Replaces all players' hole cards with the given hand.
    ///
    /// Used for setting up counterfactual game states during analysis.
    pub fn wipe(mut self, hole: Hole) -> Self {
        for seat in self.seats.iter_mut() {
            seat.reset_cards(hole);
        }
        self
    }
    /// Replaces all players' hole cards EXCEPT the given seat.
    ///
    /// Used for computing opponent reach: sets all non-hero seats to the
    /// assumed opponent hole while preserving hero's cards.
    ///
    /// TODO
    /// this might be incorrect, i don't know if it takes into considration the relativity of
    /// dealer position in Turn.
    pub fn assume(mut self, hero: Turn, hole: Hole) -> Self {
        self.seats
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| Turn::Choice(*i) != hero)
            .for_each(|(_, seat)| seat.reset_cards(hole));
        self
    }
    /// Fast-forward to the given street by taking passive actions.
    ///
    /// From the root state, advances the game by repeatedly applying
    /// `passive()` (check if allowed, fold otherwise) until reaching
    /// the target street. This is useful for constructing subgame roots
    /// at arbitrary streets for exact subgame solving.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The target street has already passed
    /// - The game reaches a terminal state before the target street
    /// - An all-in situation occurs before reaching the target street
    pub fn ffwd(mut self, target: Street) -> Self {
        while self.street() < target {
            match self.turn() {
                Turn::Terminal => panic!("reached terminal before target street"),
                Turn::Chance => {
                    let action = self.reveal();
                    self.act(action);
                }
                Turn::Choice(_) => {
                    let action = self.passive();
                    self.act(action);
                }
            }
        }
        debug_assert_eq!(self.street(), target, "overshot target street");
        self
    }
}

/// Public state accessors.
impl Game {
    /// Number of players (constant for heads-up).
    pub fn n(&self) -> usize {
        self.seats.len()
    }
    /// Total chips in the pot.
    pub fn pot(&self) -> Chips {
        self.pot
    }
    /// All player seats.
    pub fn seats(&self) -> [Seat; N] {
        self.seats
    }
    /// Community cards on the board.
    pub fn board(&self) -> Board {
        self.board
    }
    /// Determines whether it's a player's turn, chance node, or terminal.
    pub fn turn(&self) -> Turn {
        if self.must_stop() {
            Turn::Terminal
        } else if self.must_deal() {
            Turn::Chance
        } else {
            Turn::Choice(self.actor_idx())
        }
    }
    /// The seat of the player to act.
    pub fn actor(&self) -> &Seat {
        self.actor_ref()
    }
    /// The current observation from the acting player's perspective.
    pub fn sweat(&self) -> Observation {
        Observation::from((
            Hand::from(self.actor().cards()), //
            Hand::from(self.board()),         //
        ))
    }
    /// The dealer position as a turn.
    pub fn dealer(&self) -> Turn {
        Turn::Choice(self.dealer)
    }
    /// Current street based on board cards.
    pub fn street(&self) -> Street {
        self.board.street()
    }
}

/// Action validation and application.
impl Game {
    /// Applies an action mutably and returns a clone of the new state.
    pub fn consume(&mut self, action: Action) -> Self {
        self.act(action);
        self.clone()
    }
    /// Returns a new game state with the action applied.
    ///
    /// Panics if the action is not legal in the current state.
    pub fn apply(&self, action: Action) -> Self {
        self.try_apply(action).expect("valid action")
    }
    /// Fallible version of [`apply`](Self::apply).
    ///
    /// Returns `Err` if the action is not legal in the current state,
    /// enabling graceful error handling instead of panicking.
    pub fn try_apply(&self, action: Action) -> anyhow::Result<Self> {
        if !self.is_allowed(&action) {
            return Err(anyhow::anyhow!(
                "illegal action {:?} in state {:?}",
                action,
                self.turn()
            ));
        }
        let mut child = self.clone();
        child.act(action);
        Ok(child)
    }
    /// Returns all legal actions in the current state.
    ///
    /// Empty at terminal nodes. Contains exactly one action at chance nodes.
    /// Contains multiple options at decision nodes.
    pub fn legal(&self) -> Vec<Action> {
        // action is determined if it's Turn::Chance
        if self.must_stop() {
            return vec![];
        }
        if self.must_deal() {
            return vec![self.reveal()];
        }
        if self.must_post() {
            return vec![self.posts()];
        }
        // now it's certainly a Turn::Choice
        let mut options = Vec::new();
        if self.may_raise() {
            options.push(self.raise());
        }
        if self.may_shove() {
            options.push(self.shove());
        }
        if self.may_call() {
            options.push(self.calls());
        }
        if self.may_fold() {
            options.push(self.folds());
        }
        if self.may_check() {
            options.push(self.check());
        }
        debug_assert!(options.len() > 0);
        options
    }
    /// Checks if a specific action is legal.
    ///
    /// Performs bounds checking for raises (min/max) and draws (correct cards).
    pub fn is_allowed(&self, action: &Action) -> bool {
        // do "bounds checking" on the two actions with degrees of freedom;
        // Action::Raise is constrained by min/max raise
        // Action::Draw is constrained by the deck and the number of cards
        match action {
            Action::Raise(raise) => {
                self.may_raise()
                    && self.must_stop().not()
                    && self.must_deal().not()
                    && raise.clone() >= self.to_raise()
                    && raise.clone() <= self.to_shove() - 1
            }
            Action::Draw(cards) => {
                self.must_deal()
                    && self.must_stop().not()
                    && cards.clone().all(|c| self.deck().contains(&c))
                    && cards.count() == self.board().street().next().n_revealed()
            }
            other => self.legal().contains(other),
        }
    }
}

/// Hand-to-hand transitions.
impl Game {
    /// Advances to the next hand if both players have sufficient stacks.
    ///
    /// Returns `None` if a player is busted (can't cover the big blind).
    /// Otherwise resets the board, deals new cards, posts blinds, and
    /// rotates the button.
    pub fn continuation(mut self) -> Option<Self> {
        debug_assert!(self.turn() == Turn::Terminal);
        self.settlements()
            .iter()
            .zip(self.seats())
            .all(|(s, seat)| seat.stack() + s.pnl().reward() >= Game::bblind())
            .then(|| {
                self.give_chips();
                self.wipe_board();
                self.wipe_seats();
                self.move_button();
                self.act(self.posts());
                self.act(self.posts());
                self
            })
    }

    fn give_chips(&mut self) {
        for (_, (settlement, seat)) in self
            .settlements()
            .iter()
            .zip(self.seats.iter_mut())
            .enumerate()
            .inspect(|(i, (x, s))| log::trace!("{} {} {:>7} {}", i, s.cards(), s.stack(), x.won()))
        {
            seat.win(settlement.pnl().reward());
        }
        self.pot = 0;
    }

    fn wipe_board(&mut self) {
        debug_assert!(self.pot() == 0);
        self.board.clear();
    }
    fn wipe_seats(&mut self) {
        debug_assert!(self.pot() == 0);
        debug_assert!(self.street() == Street::Pref);
        let mut deck = Deck::new();
        for seat in self.seats.iter_mut() {
            seat.reset_state(State::Betting);
            seat.reset_cards(deck.hole());
            seat.reset_stake();
            seat.reset_spent();
        }
    }

    fn move_button(&mut self) {
        debug_assert!(self.pot() == 0);
        debug_assert!(self.seats.len() == self.n());
        debug_assert!(self.street() == Street::Pref);
        self.dealer = self.dealer + 1;
        self.dealer = self.dealer % self.n();
        self.ticker = 1;
    }
}

/// Private mutation methods.
impl Game {
    /// Core state transition logic.
    fn act(&mut self, a: Action) {
        debug_assert!(self.is_allowed(&a));
        match a {
            Action::Check => {
                self.next_player();
            }
            Action::Fold => {
                self.fold();
                self.next_player();
            }
            Action::Call(chips)
            | Action::Blind(chips)
            | Action::Raise(chips)
            | Action::Shove(chips) => {
                self.bet(chips);
                self.next_player();
            }
            Action::Draw(cards) => {
                self.show(cards);
                self.next_player();
                self.next_street();
            }
        }
    }
    fn bet(&mut self, bet: Chips) {
        debug_assert!(self.actor_ref().stack() >= bet);
        self.pot += bet;
        self.actor_mut().bet(bet);
        if self.actor_ref().stack() == 0 {
            self.allin();
        }
    }
    fn allin(&mut self) {
        self.actor_mut().reset_state(State::Shoving);
    }
    fn fold(&mut self) {
        self.actor_mut().reset_state(State::Folding);
    }
    fn show(&mut self, hand: Hand) {
        self.ticker = 0;
        self.board.add(hand);
    }
}

/// Street and player advancement.
impl Game {
    /// Resets per-street stakes when a new street begins.
    fn next_street(&mut self) {
        for seat in self.seats.iter_mut() {
            seat.reset_stake();
        }
    }
    /// Advances to the next active player, skipping folded/all-in players.
    fn next_player(&mut self) {
        if !self.is_everyone_alright() {
            loop {
                self.ticker += 1;
                match self.actor_ref().state() {
                    State::Betting => break,
                    State::Folding => continue,
                    State::Shoving => continue,
                }
            }
        }
    }
}

/// Termination and continuation predicates.
impl Game {
    /// True if the hand is complete (showdown or everyone folded).
    pub fn must_stop(&self) -> bool {
        if self.street() == Street::Rive {
            self.is_everyone_alright()
        } else {
            self.is_everyone_folding()
        }
    }
    /// True if we need to deal the next street's cards.
    pub fn must_deal(&self) -> bool {
        self.street() != Street::Rive && self.is_everyone_alright()
    }
    /// True if blinds have not yet been posted.
    pub fn must_post(&self) -> bool {
        self.street() == Street::Pref && self.pot() < Self::sblind() + Self::bblind()
    }
    /// All players have acted and the pot is right.
    fn is_everyone_alright(&self) -> bool {
        self.is_everyone_calling() || self.is_everyone_folding() || self.is_everyone_shoving()
    }
    /// All betting players are in for the same amount.
    fn is_everyone_calling(&self) -> bool {
        self.is_everyone_touched() && self.is_everyone_matched()
    }
    /// All players have acted at least once this street.
    /// Preflop bonus is 2 because ticker initializes at 1 (ring game: BTN ≠ SB).
    fn is_everyone_touched(&self) -> bool {
        self.ticker > self.n() + if self.street() == Street::Pref { 2 } else { 0 }
    }
    /// All betting players are in for the effective stake.
    fn is_everyone_matched(&self) -> bool {
        let stake = self.stakes();
        self.seats
            .iter()
            .filter(|s| s.state() == State::Betting)
            .all(|s| s.stake() == stake)
    }
    /// All non-folded players are all-in.
    fn is_everyone_shoving(&self) -> bool {
        self.seats
            .iter()
            .filter(|s| s.state() != State::Folding)
            .all(|s| s.state() == State::Shoving)
    }
    /// Exactly one player remains (all others folded).
    fn is_everyone_folding(&self) -> bool {
        self.seats
            .iter()
            .filter(|s| s.state() != State::Folding)
            .count()
            == 1
    }
    /// True if folding is a legal option (facing a bet).
    pub fn may_fold(&self) -> bool {
        matches!(self.turn(), Turn::Choice(_)) && self.to_call() > 0
    }
    /// True if calling is legal (facing a bet we can cover).
    pub fn may_call(&self) -> bool {
        matches!(self.turn(), Turn::Choice(_))
            && self.may_fold()
            && self.to_call() < self.to_shove()
    }
    /// True if checking is legal (no bet to call).
    pub fn may_check(&self) -> bool {
        matches!(self.turn(), Turn::Choice(_)) && self.stakes() == self.actor_ref().stake()
    }
    /// True if raising is legal (have chips beyond the min-raise).
    pub fn may_raise(&self) -> bool {
        matches!(self.turn(), Turn::Choice(_)) && self.to_raise() < self.to_shove()
    }
    /// True if shoving (all-in) is legal.
    pub fn may_shove(&self) -> bool {
        matches!(self.turn(), Turn::Choice(_)) && self.to_shove() > 0
    }
}

/// Bet sizing constraints and action constructors.
impl Game {
    /// Chips needed to call the current bet.
    pub fn to_call(&self) -> Chips {
        self.stakes() - self.actor_ref().stake()
    }
    /// Blind amount to post (SB or BB depending on position).
    pub fn to_post(&self) -> Chips {
        debug_assert!(self.street() == Street::Pref);
        let sb_pos = (self.dealer + 1) % self.n();
        if self.actor_idx() == sb_pos {
            Self::sblind().min(self.actor_ref().stack())
        } else {
            Self::bblind().min(self.actor_ref().stack())
        }
    }
    /// All remaining chips (for all-in).
    pub fn to_shove(&self) -> Chips {
        self.actor_ref().stack()
    }
    /// Minimum legal raise size.
    ///
    /// Computed as: chips to call + max(last raise increment, big blind).
    pub fn to_raise(&self) -> Chips {
        let (most_large_stake, next_large_stake) = self
            .seats
            .iter()
            .filter(|s| s.state() != State::Folding)
            .map(|s| s.stake())
            .fold((0, 0), |(most, next), stake| {
                if stake > most {
                    (stake, most)
                } else if stake > next {
                    (most, stake)
                } else {
                    (most, next)
                }
            });
        let relative_raise = most_large_stake - self.actor().stake();
        let marginal_raise = most_large_stake - next_large_stake;
        let required_raise = std::cmp::max(marginal_raise, Self::bblind());
        relative_raise + required_raise
    }
    /// Constructs a minimum-raise action.
    pub fn raise(&self) -> Action {
        Action::Raise(self.to_raise())
    }
    /// Constructs an all-in action.
    pub fn shove(&self) -> Action {
        Action::Shove(self.to_shove())
    }
    /// Constructs a call action.
    pub fn calls(&self) -> Action {
        Action::Call(self.to_call())
    }
    /// Constructs a blind-posting action.
    pub fn posts(&self) -> Action {
        Action::Blind(self.to_post())
    }
    /// Constructs a fold action.
    pub fn folds(&self) -> Action {
        Action::Fold
    }
    /// Constructs a check action.
    pub fn check(&self) -> Action {
        Action::Check
    }
    /// Returns check if allowed, otherwise fold.
    pub fn passive(&self) -> Action {
        if self.may_check() {
            Action::Check
        } else {
            Action::Fold
        }
    }
    /// Deals the next street's cards from the deck.
    pub fn reveal(&self) -> Action {
        Action::Draw(self.deck().deal(self.street()))
    }
}

/// Showdown and payout logic.
impl Game {
    /// Computes final chip distributions at a terminal node.
    pub fn settlements(&self) -> Vec<Settlement> {
        debug_assert!(self.must_stop(), "non terminal game state:\n{}", self);
        Showdown::from(self.ledger()).settle()
    }
    /// Returns true if this is a showdown (multiple players remain).
    pub fn is_showdown(&self) -> bool {
        self.seats.iter().filter(|s| s.state().is_active()).count() > 1
    }
    fn ledger(&self) -> Vec<Settlement> {
        self.seats
            .iter()
            .enumerate()
            .map(|(position, _)| self.settlement(position))
            .collect()
    }
    fn settlement(&self, position: usize) -> Settlement {
        let seat = &self.seats[position];
        let strength = Strength::from(Hand::add(
            Hand::from(seat.cards()),
            Hand::from(self.board()),
        ));
        Settlement::from((seat.spent(), seat.state(), strength))
    }
}

/// Card operations.
impl Game {
    /// Deals random cards for the next street.
    pub fn draw(&self) -> Hand {
        self.deck().deal(self.street())
    }
    /// Returns the remaining deck (all cards not in play).
    pub fn deck(&self) -> Deck {
        let mut removed = Hand::from(self.board);
        for seat in self.seats.iter() {
            removed = Hand::or(removed, Hand::from(seat.cards()));
        }
        Deck::from(removed.complement())
    }
}

/// Position tracking.
impl Game {
    /// Index of the player to act.
    fn actor_idx(&self) -> Position {
        (self.dealer + self.ticker) % self.n()
    }
    fn actor_ref(&self) -> &Seat {
        self.seats
            .get(self.actor_idx())
            .expect("index should be in bounds bc modulo")
    }
    fn actor_mut(&mut self) -> &mut Seat {
        let index = self.actor_idx();
        self.seats
            .get_mut(index)
            .expect("index should be in bounds bc modulo")
    }
}

/// Stack and SPR calculations.
impl Game {
    /// Total chips in play (pot + all stacks).
    pub fn total(&self) -> Chips {
        self.pot() + self.seats().iter().map(|s| s.stack()).sum::<Chips>()
    }
    /// Effective stack (minimum stack for heads-up).
    ///
    /// In N-way pots this would be the second-largest stack;
    /// for heads-up it's simply the smaller of the two.
    pub fn effective(&self) -> Chips {
        self.seats.iter().map(|s| s.stack()).min().unwrap_or(0)
    }
    /// Stack-to-pot ratio (effective stack / pot).
    pub fn spr(&self) -> f32 {
        match self.pot() {
            0 => 0.0,
            p => self.effective() as f32 / p as f32,
        }
    }
    /// Maximum stake among all players this street.
    fn stakes(&self) -> Chips {
        self.seats
            .iter()
            .map(|s| s.stake())
            .max()
            .expect("non-empty seats")
    }
    /// True if this is a preflop opening spot (no player actions yet).
    /// Used to interpret Odds(n,1) as nBB rather than nx pot.
    #[allow(dead_code)]
    fn is_opening(&self) -> bool {
        self.street() == Street::Pref && self.pot() == Self::sblind() + Self::bblind()
    }
}

/// Blind configuration.
impl Game {
    /// Returns the blind posting actions [SB, BB].
    pub const fn blinds() -> [Action; 2] {
        [Action::Blind(Self::sblind()), Action::Blind(Self::bblind())]
    }
    /// Big blind size.
    pub const fn bblind() -> Chips {
        rbp_core::B_BLIND
    }
    /// Small blind size.
    pub const fn sblind() -> Chips {
        rbp_core::S_BLIND
    }
}

/// Abstraction interface: mapping between concrete Actions and abstract Edges.
impl Game {
    /// Returns all available edges for current game state.
    /// Expands legal actions into the discretized edge space.
    pub fn choices(&self, depth: usize) -> Path {
        self.legal()
            .into_iter()
            .flat_map(|action| self.unfold(depth, action))
            .collect()
    }
    /// Expands an action into edges using street-specific bet grids.
    /// Non-raise actions map 1:1; raises expand to all grid sizes.
    fn unfold(&self, depth: usize, action: Action) -> Vec<Edge> {
        match action {
            Action::Raise(_) => Edge::raises(self.street(), depth),
            _ => vec![Edge::from(action)],
        }
    }
    /// Converts an abstract [`Edge`] into a concrete [`Action`].
    /// The resulting action may be illegal; use [`Self::snap`] to coerce.
    pub fn actionize(&self, edge: Edge) -> Action {
        match edge {
            Edge::Fold => Action::Fold,
            Edge::Draw => self.reveal(),
            Edge::Call => Action::Call(self.to_call()),
            Edge::Check => Action::Check,
            Edge::Shove => Action::Shove(self.to_shove()),
            Edge::Open(n) => Action::Raise(n * rbp_core::B_BLIND),
            Edge::Raise(_) => Action::Raise(edge.into_chips(self.pot())),
        }
    }
    /// Converts a concrete [`Action`] into an abstract [`Edge`].
    /// Raise amounts snap to the closest grid size; other actions map directly.
    pub fn edgify(&self, action: Action, depth: usize) -> Edge {
        match action {
            Action::Fold => Edge::Fold,
            Action::Check => Edge::Check,
            Action::Draw(_) => Edge::Draw,
            Action::Call(_) => Edge::Call,
            Action::Blind(_) => Edge::Call,
            Action::Shove(_) => Edge::Shove,
            Action::Raise(chips) => self.snap_to_edge(chips, depth),
        }
    }
    /// Snaps a chip amount to the nearest edge in the grid.
    fn snap_to_edge(&self, chips: Chips, depth: usize) -> Edge {
        let edges = Edge::raises(self.street(), depth);
        edges
            .into_iter()
            .min_by_key(|e| (e.into_chips(self.pot()) as i32 - chips as i32).abs())
            .unwrap_or(Edge::Shove)
    }
    /// Maps an action to the nearest legal action in the current state.
    ///
    /// Used for CFR traversal where canonical edges may not correspond to
    /// legal actions due to stack/pot differences from prior streets.
    /// Semi-recursive: aggressive actions cascade through the fallback chain
    /// `Raise → Shove → Call → passive`.
    ///
    /// # Mapping rules
    ///
    /// - `Raise(x)` where `x >= to_shove()` → recurse with `Shove`
    /// - `Raise(x)` where `x < to_raise()` → `Raise(to_raise())`
    /// - `Raise(_)` when `!may_raise()` → recurse with `Shove`
    /// - `Shove` when `!may_shove()` → recurse with `Call`
    /// - `Call` when `!may_call()` → `passive()`
    /// - `Check` when `!may_check()` → `Call` or `Fold`
    /// - `Fold` when `!may_fold()` → `Check`
    pub fn snap(&self, action: Action) -> Action {
        match action {
            Action::Raise(x) if x >= self.to_shove() => self.snap(self.shove()), //
            Action::Raise(_) if !self.may_raise() => self.snap(self.shove()),    //
            Action::Raise(x) if x < self.to_raise() => self.raise(),             //
            Action::Raise(x) => Action::Raise(x),                                //
            Action::Shove(_) if self.may_shove() => self.shove(),                //
            Action::Shove(_) if self.may_call() => self.calls(),                 // ? unnecessary
            Action::Shove(_) => self.passive(),                                  // ? unreachable
            Action::Call(_) if self.may_call() => self.calls(),                  // ? unnecessary
            Action::Call(_) if self.may_shove() => self.shove(),                 // ? unnecessary
            Action::Call(_) => self.passive(),                                   // ? unnecessary
            Action::Check if self.may_check() => Action::Check,                  // ? self.passive()
            Action::Check if self.may_call() => self.calls(),                    // ? self.passive()
            Action::Check => self.folds(),                                       // ? self.passive()
            Action::Fold if self.may_fold() => Action::Fold,                     // ? self.passive()
            Action::Fold => Action::Check,                                       // ? self.passive()
            Action::Draw(_) | Action::Blind(_) => action,
        }
    }
}


impl std::fmt::Display for Game {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for seat in self.seats.iter() {
            writeln!(
                f,
                "{:>3} {:>3} {:<6}",
                seat.state(),
                seat.cards(),
                seat.stack()
            )?;
        }
        writeln!(f, "Pot   {}", self.pot())?;
        writeln!(f, "Board {}", self.board())?;
        Ok(())
    }
}
/// Infinite iterator over actions across games.
///
/// Yields each `Action` played, resetting to a fresh game when busted.
/// Never terminates — use `.take(n)` to bound iteration.
pub struct Perpetual(Game);
impl Perpetual {
    pub fn new(game: Game) -> Self {
        Self(game)
    }
}
impl Iterator for Perpetual {
    type Item = Action;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let actions = self.0.legal();
            if !actions.is_empty() {
                let action = actions[rand::random_range(0..actions.len())];
                self.0 = self.0.apply(action);
                return Some(action);
            }
            self.0 = self.0.clone().continuation().unwrap_or_else(Game::root);
        }
    }
}

/// Iterator over completed hands in a session.
///
/// Yields the terminal `Game` state at the end of each hand.
/// Stops when a player busts (can't cover the big blind).
pub struct Hands(Game);
impl Hands {
    pub fn new(game: Game) -> Self {
        Self(game)
    }
}
impl Iterator for Hands {
    type Item = Game;
    fn next(&mut self) -> Option<Self::Item> {
        while !self.0.must_stop() {
            let actions = self.0.legal();
            let action = actions[rand::random_range(0..actions.len())];
            self.0 = self.0.apply(action);
        }
        let terminal = self.0.clone();
        self.0 = self.0.clone().continuation()?;
        Some(terminal)
    }
}

/// Iterator over actions in a session.
///
/// Yields each `Action` played across multiple hands.
/// Stops when a player busts (can't cover the big blind).
pub struct Session(Game);
impl Session {
    pub fn new(game: Game) -> Self {
        Self(game)
    }
}
impl Iterator for Session {
    type Item = Action;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let actions = self.0.legal();
            if !actions.is_empty() {
                let action = actions[rand::random_range(0..actions.len())];
                self.0 = self.0.apply(action);
                return Some(action);
            }
            self.0 = self.0.clone().continuation()?;
        }
    }
}

impl Game {
    /// Infinite iterator over actions, resetting on bust.
    pub fn perpetual(self) -> Perpetual {
        Perpetual::new(self)
    }
    /// Iterator over completed hands, stopping when busted.
    pub fn hands(self) -> Hands {
        Hands::new(self)
    }
    /// Iterator over actions, stopping when busted.
    pub fn session(self) -> Session {
        Session::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fold all active players until the hand is terminal
    fn fold_to_terminal(mut game: Game) -> Game {
        while !game.must_stop() {
            game = game.apply(Action::Fold);
        }
        game
    }

    /// advance to the next hand by folding everyone in the current hand
    fn next_hand(game: Game) -> Option<Game> {
        fold_to_terminal(game).continuation()
    }

    /// reach flop with only SB and BB by folding UTG through BTN (4 folds)
    fn heads_to_flop(mut game: Game) -> Game {
        for _ in 0..4 {
            game = game.apply(Action::Fold);
        }
        game = game.apply(Action::Call(1)); // SB calls
        game = game.apply(Action::Check);   // BB checks
        let flop = game.deck().deal(Street::Pref);
        game.apply(Action::Draw(flop))
    }

    /// UTG(dealer+3) acts first preflop in 6-max
    #[test]
    fn test_root() {
        let game = Game::root();
        assert_eq!(game.board().street(), Street::Pref);
        assert_eq!(game.actor().state(), State::Betting);
        assert_eq!(game.pot(), Game::sblind() + Game::bblind());
        // UTG = (dealer+3)%N acts first preflop
        assert_eq!(game.turn(), Turn::Choice((game.dealer + 3) % game.n()));
    }

    #[test]
    fn everyone_folds_pref() {
        // need 5 folds (UTG through SB) for only BB to remain
        let game = fold_to_terminal(Game::root());
        assert!(game.is_everyone_folding());
        assert!(game.is_everyone_alright());
        assert!(!game.is_everyone_calling());
        assert!(game.must_deal()); // ambiguous
        assert!(game.must_stop());
    }

    #[test]
    fn everyone_folds_flop() {
        // reach flop with SB+BB only, then SB raises and BB folds
        let game = heads_to_flop(Game::root());
        let raise = game.to_raise();
        let game = game.apply(Action::Raise(raise)); // SB raises
        let game = game.apply(Action::Fold);          // BB folds
        assert!(game.is_everyone_folding());
        assert!(game.is_everyone_alright());
        assert!(!game.is_everyone_calling());
        assert!(game.must_deal()); // ambiguous
        assert!(game.must_stop());
    }

    /// 6-max: 4 fold preflop leaving SB+BB; check-check through all streets
    #[test]
    fn history_of_checks() {
        // After root: pot=3, UTG(3) to act
        let game = Game::root();
        assert!(game.board().street() == Street::Pref);
        assert!(game.pot() == 3);
        assert!(!game.must_post());
        assert!(!game.must_stop());
        assert!(!game.must_deal());
        assert!(!game.is_everyone_alright());
        assert!(!game.is_everyone_calling());
        assert!(!game.is_everyone_touched());
        assert!(!game.is_everyone_matched()); // stakes unmatched: UTG stake=0, BB stake=2

        // UTG folds
        let game = game.apply(Action::Fold);
        assert!(!game.is_everyone_touched());
        assert!(!game.is_everyone_matched()); // UTG+1 still hasn't matched

        // UTG+1, CO, BTN fold (3 more)
        let game = game.apply(Action::Fold).apply(Action::Fold).apply(Action::Fold);
        // now SB(1) is to act
        assert_eq!(game.turn(), Turn::Choice(1));
        assert!(!game.is_everyone_touched());

        // SB calls 1 (pot=4)
        let game = game.apply(Action::Call(1));
        assert!(game.board().street() == Street::Pref);
        assert!(game.pot() == 4);
        assert!(!game.must_stop());
        assert!(!game.must_deal());
        assert!(!game.is_everyone_alright());
        assert!(!game.is_everyone_touched());
        assert!(game.is_everyone_matched()); // SB and BB both at stake=2

        // BB checks (pot=4) — everyone touched, must deal
        let game = game.apply(Action::Check);
        assert!(game.board().street() == Street::Pref);
        assert!(game.pot() == 4);
        assert!(!game.must_stop());
        assert!(game.must_deal());
        assert!(game.is_everyone_alright());
        assert!(game.is_everyone_calling());
        assert!(game.is_everyone_touched());
        assert!(game.is_everyone_matched());

        // Flop
        let flop = game.deck().deal(game.board().street());
        let game = game.apply(Action::Draw(flop));
        assert!(game.board().street() == Street::Flop);
        assert!(game.pot() == 4);
        assert!(!game.must_stop());
        assert!(!game.must_deal());
        assert!(!game.is_everyone_alright());
        assert!(!game.is_everyone_calling());
        assert!(!game.is_everyone_touched());
        assert!(game.is_everyone_matched()); // stakes reset to 0

        // SB checks
        let game = game.apply(Action::Check);
        assert!(!game.must_deal());
        assert!(!game.is_everyone_touched());
        assert!(game.is_everyone_matched());

        // BB checks — must deal
        let game = game.apply(Action::Check);
        assert!(game.board().street() == Street::Flop);
        assert!(game.pot() == 4);
        assert!(!game.must_stop());
        assert!(game.must_deal());
        assert!(game.is_everyone_alright());
        assert!(game.is_everyone_calling());
        assert!(game.is_everyone_touched());

        // Turn
        let turn = game.deck().deal(game.board().street());
        let game = game.apply(Action::Draw(turn));
        assert!(game.board().street() == Street::Turn);
        assert!(!game.is_everyone_touched());

        // SB checks
        let game = game.apply(Action::Check);
        assert!(!game.is_everyone_touched());

        // BB raises (re-opens betting)
        let raise = game.to_raise();
        let game = game.apply(Action::Raise(raise));
        assert!(game.board().street() == Street::Turn);
        assert!(!game.must_stop());
        assert!(!game.must_deal());
        assert!(!game.is_everyone_alright());
        assert!(!game.is_everyone_calling());
        assert!(game.is_everyone_touched()); // acted once
        assert!(!game.is_everyone_matched()); // BB hasn't matched

        // BB calls
        let to_call = game.to_call();
        let game = game.apply(Action::Call(to_call));
        assert!(game.board().street() == Street::Turn);
        assert!(!game.must_stop());
        assert!(game.must_deal());
        assert!(game.is_everyone_alright());
        assert!(game.is_everyone_calling());
        assert!(game.is_everyone_touched());
        assert!(game.is_everyone_matched());

        // River
        let rive = game.deck().deal(game.board().street());
        let game = game.apply(Action::Draw(rive));
        assert!(game.board().street() == Street::Rive);
        assert!(!game.must_stop());
        assert!(!game.must_deal());
        assert!(!game.is_everyone_alright());
        assert!(!game.is_everyone_touched());
        assert!(game.is_everyone_matched());

        // SB checks
        let game = game.apply(Action::Check);
        assert!(!game.is_everyone_touched());

        // BB checks — terminal
        let game = game.apply(Action::Check);
        assert!(game.board().street() == Street::Rive);
        assert!(game.must_stop());
        assert!(!game.must_deal());
        assert!(game.is_everyone_alright());
        assert!(game.is_everyone_calling());
        assert!(game.is_everyone_touched());
        assert!(game.is_everyone_matched());
    }

    /// next() resets game state correctly after terminal
    #[test]
    fn next_after_fold() {
        let game = fold_to_terminal(Game::root());
        assert!(game.must_stop());
        let next = game.continuation().expect("can continue");
        assert_eq!(next.street(), Street::Pref);
        assert_eq!(next.pot(), Game::sblind() + Game::bblind());
        assert_eq!(next.board(), Board::empty());
        assert_eq!(next.dealer, 1); // rotated from 0
        // UTG with dealer=1 is at (1+3)%6=4
        assert_eq!(next.turn(), Turn::Choice(4));
        assert!(!next.is_everyone_touched());
    }

    /// dealer rotates correctly across multiple hands, wrapping at N=6
    #[test]
    fn dealer_rotation() {
        let game = Game::root();
        assert_eq!(game.dealer, 0);
        let game = next_hand(game).unwrap();
        assert_eq!(game.dealer, 1);
        let game = next_hand(game).unwrap();
        assert_eq!(game.dealer, 2);
        // after 6 full rotations, wraps back to 0
        let mut game = Game::root();
        for _ in 0..6 {
            game = next_hand(game).unwrap();
        }
        assert_eq!(game.dealer, 0);
    }

    /// ticker resets correctly for each new hand (initial ticker=1, 2 blinds → ticker=3)
    #[test]
    fn ticker_reset_on_next() {
        let g0 = Game::root();
        let g1 = next_hand(g0).unwrap();
        let g2 = next_hand(g1).unwrap();
        assert_eq!(g0.ticker, g1.ticker);
        assert_eq!(g1.ticker, g2.ticker);
        assert_eq!(g0.ticker, 3); // initial=1, SB post→2, BB post→3
    }

    /// is_everyone_touched works for dealer=1 (6-max rotation)
    #[test]
    fn touched_with_rotated_dealer() {
        let game = next_hand(Game::root()).unwrap();
        assert_eq!(game.dealer, 1);
        assert!(!game.is_everyone_touched());
        // With dealer=1: UTG=(1+3)%6=4. Fold 4 players, SB calls, BB checks.
        let mut game = game;
        for _ in 0..4 {
            game = game.apply(Action::Fold);
        }
        assert!(!game.is_everyone_touched());
        let game = game.apply(Action::Call(1)); // SB=(1+1)%6=2 calls
        assert!(!game.is_everyone_touched());
        let game = game.apply(Action::Check); // BB=(1+2)%6=3 checks
        assert!(game.is_everyone_touched());
        assert!(game.must_deal());
    }

    /// multi-street hand with rotated dealer
    #[test]
    fn full_hand_rotated_dealer() {
        let game = next_hand(Game::root()).unwrap();
        assert_eq!(game.dealer, 1);
        // fold 4 (UTG=4, UTG+1=5, CO=0, BTN=1), SB=2 calls, BB=3 checks
        let mut game = game;
        for _ in 0..4 {
            game = game.apply(Action::Fold);
        }
        let game = game.apply(Action::Call(1)).apply(Action::Check);
        assert!(game.must_deal());
        let flop = game.deck().deal(Street::Pref);
        let game = game.apply(Action::Draw(flop));
        assert_eq!(game.street(), Street::Flop);
        // SB=(1+1)%6=2 acts first postflop
        assert_eq!(game.turn(), Turn::Choice(2));
        assert!(!game.is_everyone_touched());
        let game = game.apply(Action::Check).apply(Action::Check);
        assert!(game.is_everyone_touched());
        assert!(game.must_deal());
    }

    /// six consecutive hands cycle dealer through all positions
    #[test]
    fn five_hands_sequence() {
        let mut game = Game::root();
        for i in 0..6 {
            assert_eq!(game.dealer, i % 6);
            assert_eq!(game.pot(), Game::sblind() + Game::bblind());
            assert_eq!(game.street(), Street::Pref);
            assert!(!game.is_everyone_touched());
            // UTG = (dealer+3)%6 acts first
            assert_eq!(game.turn(), Turn::Choice((game.dealer + 3) % game.n()));
            game = next_hand(game).unwrap();
        }
    }

    /// preflop action order: UTG first, then postflop SB first
    #[test]
    fn symmetric_preflop_action() {
        // dealer=0: UTG(3) first preflop; after 4 folds, SB(1) and BB(2) reach flop
        let g0 = Game::root();
        assert_eq!(g0.dealer, 0);
        assert_eq!(g0.turn(), Turn::Choice(3)); // UTG first
        let g0 = heads_to_flop(g0);
        // SB(1) acts first on flop
        assert_eq!(g0.turn(), Turn::Choice(1));

        // dealer=1: UTG(4) first preflop; SB(2) acts first postflop
        let mut g1 = next_hand(Game::root()).unwrap();
        assert_eq!(g1.dealer, 1);
        assert_eq!(g1.turn(), Turn::Choice(4)); // UTG=(1+3)%6=4
        for _ in 0..4 {
            g1 = g1.apply(Action::Fold);
        }
        let g1 = g1.apply(Action::Call(1)).apply(Action::Check);
        let flop = g1.deck().deal(Street::Pref);
        let g1 = g1.apply(Action::Draw(flop));
        assert_eq!(g1.turn(), Turn::Choice(2)); // SB=(1+1)%6=2
    }

    /// SB acts first on flop for any dealer position
    #[test]
    fn flop_actor_both_dealers() {
        // dealer=0: SB=1 acts first on flop
        let g0 = heads_to_flop(Game::root());
        assert_eq!(g0.turn(), Turn::Choice(1));

        // dealer=1: SB=2 acts first on flop
        let mut g1 = next_hand(Game::root()).unwrap();
        assert_eq!(g1.dealer, 1);
        for _ in 0..4 {
            g1 = g1.apply(Action::Fold);
        }
        let g1 = g1.apply(Action::Call(1)).apply(Action::Check);
        let flop = g1.deck().deal(Street::Pref);
        let g1 = g1.apply(Action::Draw(flop));
        assert_eq!(g1.turn(), Turn::Choice(2));
    }

    /// all six players shove leads to all-in showdown
    #[test]
    fn allin_showdown() {
        let mut game = Game::root();
        while !game.must_stop() && !game.must_deal() {
            let shove = game.to_shove();
            game = game.apply(Action::Shove(shove));
        }
        assert!(game.is_everyone_shoving());
        assert!(game.must_stop() || game.must_deal());
    }

    /// UTG shoves then everyone folds — only UTG remains (shoving counts as not-folded)
    #[test]
    fn allin_fold() {
        let mut game = Game::root();
        let shove = game.to_shove(); // UTG shoves
        game = game.apply(Action::Shove(shove));
        while !game.must_stop() {
            game = game.apply(Action::Fold);
        }
        assert!(game.must_stop());
        assert!(game.is_everyone_folding()); // only UTG (Shoving) remains → count=1
    }

    /// raise-reraise keeps action open, next actor is CO(5)
    #[test]
    fn raise_reraise() {
        let g0 = Game::root(); // UTG(3) acts
        let r1 = g0.to_raise();
        let g1 = g0.apply(Action::Raise(r1)); // UTG raises → UTG+1(4) acts
        let r2 = g1.to_raise();
        let g2 = g1.apply(Action::Raise(r2)); // UTG+1 re-raises → CO(5) acts
        assert!(!g2.must_deal());
        assert!(!g2.is_everyone_alright());
        assert_eq!(g2.turn(), Turn::Choice(5)); // CO is next
        assert!(g2.may_raise() || g2.may_call());
    }

    /// BB wins pot when all others fold preflop
    #[test]
    fn stacks_after_fold() {
        let game = fold_to_terminal(Game::root());
        assert!(game.must_stop());
        let settlements = game.settlements();
        // BB (pos 2) wins pot=3
        assert_eq!(settlements[2].pnl().reward(), 3);
        assert_eq!(settlements[2].won(), 1);   // BB net +1
        assert_eq!(settlements[1].won(), -1);  // SB lost blind
        assert_eq!(settlements[0].won(), 0);   // BTN nothing at risk
        assert_eq!(settlements[3].won(), 0);   // UTG nothing at risk
    }

    /// SB wins after raising on flop and BB folds
    #[test]
    fn stacks_after_flop_bet_fold() {
        let game = heads_to_flop(Game::root()); // pot=4, SB+BB on flop
        let raise = game.to_raise(); // SB raises (min=2 on empty flop)
        let game = game.apply(Action::Raise(raise)); // SB raises
        let game = game.apply(Action::Fold);          // BB folds
        assert!(game.must_stop());
        let settlements = game.settlements();
        // SB(1): spent=1+1+2=4, wins pot=4+2=6, won=2
        assert_eq!(settlements[1].won(), 2);
        // BB(2): spent=2, wins=0, won=-2
        assert_eq!(settlements[2].won(), -2);
        // BTN(0): no chips risked
        assert_eq!(settlements[0].won(), 0);
    }

    /// dealer rotates correctly across hands with non-trivial preflop
    #[test]
    fn multi_hand_with_betting() {
        let g0 = fold_to_terminal(Game::root());
        let g1 = g0.continuation().unwrap();
        assert_eq!(g1.dealer, 1);
        assert_eq!(g1.pot(), Game::sblind() + Game::bblind());
        let g1 = fold_to_terminal(g1);
        let g2 = g1.continuation().unwrap();
        assert_eq!(g2.dealer, 2);
        assert_eq!(g2.pot(), Game::sblind() + Game::bblind());
    }

    /// UTG faces full BB amount, can fold/call/raise/shove
    #[test]
    fn legal_preflop_options() {
        let game = Game::root(); // UTG(3) acts; to_call=2 (UTG stake=0, BB stake=2)
        let legal = game.legal();
        assert!(legal.contains(&Action::Fold));
        assert!(legal.contains(&Action::Call(2))); // UTG calls full 2bb
        assert!(legal.iter().any(|a| matches!(a, Action::Raise(_))));
        assert!(legal.iter().any(|a| matches!(a, Action::Shove(_))));
        assert!(!legal.contains(&Action::Check));
    }

    /// BB can check after everyone limps (SB already covered)
    #[test]
    fn legal_bb_can_check() {
        let mut game = Game::root();
        for _ in 0..4 {
            game = game.apply(Action::Fold);
        } // UTG-BTN fold
        game = game.apply(Action::Call(1)); // SB calls; now BB's turn
        let legal = game.legal();
        assert!(legal.contains(&Action::Check));
        assert!(!legal.contains(&Action::Fold));
    }

    /// SB acts first on flop; can check or raise, not fold
    #[test]
    fn legal_flop_options() {
        let game = heads_to_flop(Game::root());
        let legal = game.legal();
        assert!(legal.contains(&Action::Check));
        assert!(legal.iter().any(|a| matches!(a, Action::Raise(_))));
        assert!(!legal.contains(&Action::Fold));
    }

    /// check-check through all four streets reaches terminal river
    #[test]
    fn terminal_river_showdown() {
        let mut game = heads_to_flop(Game::root()); // already at flop
        // flop: SB+BB check
        game = game.apply(Action::Check).apply(Action::Check);
        for street in [Street::Flop, Street::Turn] {
            let cards = game.deck().deal(street);
            game = game
                .apply(Action::Draw(cards))
                .apply(Action::Check)
                .apply(Action::Check);
        }
        assert_eq!(game.street(), Street::Rive);
        assert!(game.must_stop());
        assert!(!game.must_deal());
    }

    /// twelve hands cycle dealer through two full rotations of 6
    #[test]
    fn ten_hands_alternation() {
        let mut game = Game::root();
        for i in 0..12 {
            assert_eq!(game.dealer, i % 6);
            assert_eq!(game.turn(), Turn::Choice((game.dealer + 3) % game.n()));
            game = next_hand(game).unwrap();
        }
    }

    /// UTG min-raise: to_raise = relative(2) + required(max(1,2)) = 4
    #[test]
    fn min_raise_size() {
        let game = Game::root(); // UTG: stake=0, BB=2, SB=1
        // relative=2-0=2, marginal=2-1=1, required=max(1,2)=2, to_raise=4
        assert_eq!(game.to_raise(), 4);
        let game = game.apply(Action::Raise(4)); // UTG raises to 4 → UTG+1 acts
        // UTG+1: stake=0, UTG=4, BB=2. relative=4, marginal=4-2=2, required=2, to_raise=6
        assert_eq!(game.to_raise(), 6);
    }

    /// pot increments correctly through call and raise sequences
    #[test]
    fn pot_tracking() {
        let game = Game::root(); // pot=3 (blinds)
        assert_eq!(game.pot(), 3);
        let game = game.apply(Action::Call(2)); // UTG calls 2
        assert_eq!(game.pot(), 5);
        let game = game.apply(Action::Raise(4)); // UTG+1 raises 4
        assert_eq!(game.pot(), 9);
        let game = game.apply(Action::Call(4)); // CO calls 4
        assert_eq!(game.pot(), 13);
    }

    /// all-in pot sums to total chips; losers can't cover bblind
    #[test]
    fn bust_prevents_next() {
        let mut game = Game::root();
        while !game.must_stop() && !game.must_deal() {
            let shove = game.to_shove();
            game = game.apply(Action::Shove(shove));
        }
        assert!(game.is_everyone_shoving());
        while !game.must_stop() {
            let cards = game.deck().deal(game.street());
            game = game.apply(Action::Draw(cards));
        }
        let rewards: Vec<i16> = game.settlements().iter().map(|s| s.pnl().reward()).collect();
        assert_eq!(rewards.iter().sum::<i16>(), 600); // 6 × 100bb fully distributed
    }

    /// actor_idx: dealer=0, ticker=3 → UTG(3); after UTG+1 fold → CO(5)
    #[test]
    fn actor_idx_wrapping() {
        let game = Game::root();
        assert_eq!(game.actor_idx(), 3); // UTG: (0+3)%6=3
        let game = game.apply(Action::Call(2)); // UTG calls → UTG+1(4)
        assert_eq!(game.actor_idx(), 4);
        let game = game.apply(Action::Fold); // UTG+1 folds → CO(5)
        assert_eq!(game.actor_idx(), 5);
    }

    /// snap preserves legal actions unchanged
    #[test]
    fn snap_legal_unchanged() {
        let game = Game::root();
        game.legal()
            .iter()
            .inspect(|&&action| assert_eq!(game.snap(action), action))
            .count();
    }

    /// snap coerces oversized raise to shove
    #[test]
    fn snap_raise_to_shove_too_large() {
        let game = Game::root();
        let shove = game.to_shove();
        assert_eq!(game.snap(Action::Raise(Chips::MAX)), game.shove());
        assert_eq!(game.snap(Action::Raise(shove)), game.shove());
    }

    /// snap coerces undersized raise to min-raise
    #[test]
    fn snap_raise_to_minim_too_small() {
        let game = Game::root();
        let minraise = game.to_raise();
        assert_eq!(game.snap(Action::Raise(1)), Action::Raise(minraise));
        assert_eq!(game.snap(Action::Raise(0)), Action::Raise(minraise));
    }

    /// snap coerces fold to check when not facing bet (BB's option)
    #[test]
    fn snap_fold_to_check_not_facing_bet() {
        let mut game = Game::root();
        for _ in 0..4 {
            game = game.apply(Action::Fold);
        }
        game = game.apply(Action::Call(1)); // SB calls; now BB's turn
        assert!(!game.may_fold());
        assert!(game.may_check());
        assert_eq!(game.snap(Action::Fold), Action::Check);
    }

    /// snap coerces check to call when facing bet
    #[test]
    fn snap_check_to_call_facing_bet() {
        let game = Game::root(); // UTG faces 2bb bet
        assert!(!game.may_check());
        assert!(game.may_call());
        assert_eq!(game.snap(Action::Check), game.calls());
    }
}
