#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef RUMMIKUB_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

// ============================================================================
// Common Rummikub environment for DQN / PPO / ISMCTS / MCCFR
// ----------------------------------------------------------------------------
// Research rules used here
//   * 2~4 players
//   * Jokers excluded -> 104 tiles (2 copies of 4 colors x 1..13)
//   * Initial hand: 14 tiles
//   * Initial registration: melds made ONLY from the player's own hand,
//     total tile-value >= 30; registration ends that turn.
//   * After registration, a whole turn is represented as ONE action:
//       current table (all tiles) + at least one tile from hand
//       -> any complete partition into valid Runs/Groups.
//     Therefore ordinary new-meld submission, append, split, merge,
//     Run<->Group transfer, and arbitrary complete table rearrangement are
//     all covered by the same exhaustive final-partition generator.
//   * DRAW is always a legal choice while pool is non-empty.
//   * If pool is empty, DRAW becomes PASS. If every player consecutively
//     passes, the player(s) with minimum remaining hand value win; exact ties
//     are treated as a draw (winner() == -2).
//
// IMPORTANT
//   max_legal_actions == 0  : exhaustive generation (no artificial cap).
//   max_legal_actions > 0   : generation stops at that many actions and
//                             generation_truncated() becomes true.
//   Exhaustive rearrangement can be combinatorially expensive in late-game
//   positions. That is inherent in the action-space definition, not a rule
//   shortcut in this environment.
//
// Python-facing core API
//   reset(seed, start_player)
//   current_player()
//   observation(viewer)
//   legal_action_features()
//   legal_action_descriptions()
//   step(action_index)
//   clone()
//   determinize(observer, seed)
//   terminal(), winner(), terminal_utility(player)
//
// Baseline storage variant in this file
//   * Player hands are stored as ordinary int[4][13]-equivalent grids
//     (HandGrid = color x number count), NOT bitmasks.
//   * Game rules, candidate generation, state/action encodings, reward logic,
//     and Python-facing API are intentionally kept equivalent to the bitmask
//     environment so win-rate experiments differ only in hand storage.
//
// State encoding (702 floats)
//   own hand counts                         52
//   table: 34 meld slots x 19 features     646
//     [type R, type G, 4 color bits, 13 number bits]
//   self registration                       1
//   opponent hand counts (max 3)            3
//
// NOTE: opponent registration flags and pool remaining ratio are intentionally
// excluded from the learning observation by research design. They remain
// internal environment state because the game rules still need them.
//
// Action encoding (699 floats)
//   hand tiles consumed                     52
//   resulting table: 34 x 19               646
//   draw/pass flag                           1
// ============================================================================

namespace rummikub2d {

constexpr int MAX_PLAYERS = 4;
constexpr int COLORS = 4;
constexpr int NUMBERS = 13;
constexpr int TILE_TYPES = COLORS * NUMBERS;        // 52
constexpr int TOTAL_TILES = 104;
constexpr int INITIAL_HAND = 14;
constexpr int MAX_MELDS = TOTAL_TILES / 3;          // 34
constexpr int MELD_FEATURE_DIM = 2 + COLORS + NUMBERS; // 19
constexpr int TABLE_FEATURE_DIM = MAX_MELDS * MELD_FEATURE_DIM; // 646
constexpr int STATE_DIM = TILE_TYPES + TABLE_FEATURE_DIM + 1 + 3; // 702
constexpr int ACTION_DIM = TILE_TYPES + TABLE_FEATURE_DIM + 1;             // 699

static constexpr char COLOR_ARRAY[5] = "RYBK";

struct Tile {
    int num = 0;      // 1..13
    char color = '?'; // R/Y/B/K
};

// Ordinary 2-dimensional hand representation used for the baseline experiment.
// grid[color][number_index] stores the physical count 0, 1, or 2.
using HandGrid = std::array<std::array<int, NUMBERS>, COLORS>;

struct Meld {
    char type = 'N'; // 'R' or 'G'
    std::array<Tile, 13> tiles{};
    int count = 0;
};

struct RewardConfig {
    // No training reward is fixed by the C++ environment.
    // Python should set these values before DQN/PPO training.
    float hand_normal = 0.0f;      // tiles 3..11, excluding 12
    float hand_2_12 = 0.0f;        // tiles 2 and 12
    float hand_1_13 = 0.0f;        // tiles 1 and 13

    float table_len3 = 0.0f;       // each length-3 meld
    float table_len4 = 0.0f;       // each length-4 meld
    float table_len5plus = 0.0f;   // each length >= 5 meld

    float win = 0.0f;
    float loss = 0.0f;
    float tie = 0.0f;
};

struct RewardComponents {
    // Raw, unweighted components. These are returned for experiment logging.
    int hand_normal_used = 0;
    int hand_2_12_used = 0;
    int hand_1_13_used = 0;

    int table_len3_delta = 0;
    int table_len4_delta = 0;
    int table_len5plus_delta = 0;

    // +1 = acting player wins, -1 = loses, 0 = nonterminal/tie.
    int terminal_outcome = 0;
};

struct StepResult {
    std::vector<float> next_state;
    float reward = 0.0f;
    float hand_reward = 0.0f;
    float table_reward = 0.0f;
    float terminal_reward = 0.0f;

    RewardComponents components;
    bool done = false;
    int winner = -1;       // -1 ongoing, -2 terminal tie, >=0 unique winner
    int acting_player = -1;
    int next_player = -1;
};

struct ActionInternal {
    bool draw = false;
    bool register_after = false;

    HandGrid final_hand{};
    int final_hand_count = 0;

    std::vector<Meld> final_table;
    std::vector<float> features;
    std::string description;
};

class RummikubEnv {
public:
    explicit RummikubEnv(int players = 2, std::size_t max_legal_actions = 0)
        : total_players_(players), max_legal_actions_(max_legal_actions) {
        if (players < 2 || players > MAX_PLAYERS)
            throw std::invalid_argument("players must be in [2, 4]");
        reset(1u, 0);
    }

    RummikubEnv(const RummikubEnv &) = default;
    RummikubEnv &operator=(const RummikubEnv &) = default;

    std::vector<float> reset(std::uint32_t seed, int start_player = 0) {
        if (start_player < 0 || start_player >= total_players_)
            throw std::invalid_argument("start_player out of range");

        rng_.seed(seed);
        for (auto &grid : hand_) {
            for (auto &row : grid) row.fill(0);
        }
        std::memset(hand_count_, 0, sizeof(hand_count_));
        std::memset(registered_, 0, sizeof(registered_));

        table_melds_.clear();
        pool_.clear();
        pool_.reserve(TOTAL_TILES);

        for (int copy = 0; copy < 2; ++copy) {
            for (int n = 1; n <= NUMBERS; ++n) {
                for (int c = 0; c < COLORS; ++c) {
                    pool_.push_back(Tile{n, COLOR_ARRAY[c]});
                }
            }
        }
        std::shuffle(pool_.begin(), pool_.end(), rng_);
        pool_pos_ = 0;

        current_player_ = start_player;
        consecutive_empty_pool_passes_ = 0;
        done_ = false;
        winner_ = -1;
        winning_mask_ = 0;
        cached_actions_.clear();
        generation_truncated_ = false;

        for (int round = 0; round < INITIAL_HAND; ++round) {
            for (int p = 0; p < total_players_; ++p) {
                if (!deal_one(p))
                    throw std::runtime_error("pool exhausted during initial deal");
            }
        }

        if (!validate_state())
            throw std::runtime_error("invalid initial state after reset");

        return observation(current_player_);
    }

    // ------------------------------- Public API -----------------------------

    int players() const { return total_players_; }
    int current_player() const { return current_player_; }
    bool terminal() const { return done_; }
    int winner() const { return winner_; }
    int winning_mask() const { return winning_mask_; }
    int state_dim() const { return STATE_DIM; }
    int action_dim() const { return ACTION_DIM; }
    int max_melds() const { return MAX_MELDS; }
    int pool_remaining() const {
        return static_cast<int>(pool_.size() - pool_pos_);
    }
    bool generation_truncated() const { return generation_truncated_; }

    void set_reward_config(const RewardConfig &config) {
        reward_config_ = config;
    }

    RewardConfig reward_config() const {
        return reward_config_;
    }

    void set_reward_weights(
        float hand_normal,
        float hand_2_12,
        float hand_1_13,
        float table_len3,
        float table_len4,
        float table_len5plus,
        float win,
        float loss,
        float tie = 0.0f) {

        reward_config_.hand_normal = hand_normal;
        reward_config_.hand_2_12 = hand_2_12;
        reward_config_.hand_1_13 = hand_1_13;
        reward_config_.table_len3 = table_len3;
        reward_config_.table_len4 = table_len4;
        reward_config_.table_len5plus = table_len5plus;
        reward_config_.win = win;
        reward_config_.loss = loss;
        reward_config_.tie = tie;
    }

    void set_max_legal_actions(std::size_t n) {
        max_legal_actions_ = n;
        cached_actions_.clear();
    }

    std::vector<float> observation(int viewer) const {
        if (viewer < 0 || viewer >= total_players_)
            throw std::out_of_range("viewer out of range");

        std::vector<float> out;
        out.reserve(STATE_DIM);

        // 1) Own hand: each tile type count 0/1/2 -> 0/0.5/1.
        auto own = hand_counts(viewer);
        for (int i = 0; i < TILE_TYPES; ++i)
            out.push_back(own[i] / 2.0f);

        // 2) Exact table meld structure in fixed canonical slots.
        append_table_features(out, table_melds_);

        // 3) Own registration flag.
        out.push_back(registered_[viewer] ? 1.0f : 0.0f);

        // 4) Opponent hand counts in cyclic order, padded to 3 slots.
        int slots = 0;
        for (int offset = 1; offset < total_players_; ++offset) {
            int p = (viewer + offset) % total_players_;
            out.push_back(hand_count_[p] / static_cast<float>(TOTAL_TILES));
            ++slots;
        }
        while (slots < 3) {
            out.push_back(0.0f);
            ++slots;
        }

        if (static_cast<int>(out.size()) != STATE_DIM)
            throw std::runtime_error("STATE_DIM mismatch");
        return out;
    }

    std::vector<std::vector<float>> legal_action_features() {
        ensure_not_terminal();
        ensure_actions_cached();
        std::vector<std::vector<float>> out;
        out.reserve(cached_actions_.size());
        for (const auto &a : cached_actions_) out.push_back(a.features);
        return out;
    }

    std::vector<std::string> legal_action_descriptions() {
        ensure_not_terminal();
        ensure_actions_cached();
        std::vector<std::string> out;
        out.reserve(cached_actions_.size());
        for (const auto &a : cached_actions_) out.push_back(a.description);
        return out;
    }

    std::size_t legal_action_count() {
        ensure_not_terminal();
        ensure_actions_cached();
        return cached_actions_.size();
    }

    StepResult step(std::size_t action_index) {
        ensure_not_terminal();
        ensure_actions_cached();
        if (action_index >= cached_actions_.size())
            throw std::out_of_range("action_index out of range");

        const int actor = current_player_;
        ActionInternal action = cached_actions_[action_index];
        cached_actions_.clear();

        std::array<int, TILE_TYPES> before_hand = hand_counts(actor);
        TableShapeCounts before_table_shape = table_shape_counts(table_melds_);

        int outcome = apply_action(actor, action); // 1 play, 0 real draw, -1 pass

        if (!validate_state())
            throw std::runtime_error("step produced an invalid state");

        std::array<int, TILE_TYPES> after_hand = hand_counts(actor);
        RewardComponents components = reward_components(
            before_hand, after_hand, before_table_shape, table_shape_counts(table_melds_));

        if (hand_count_[actor] == 0) {
            done_ = true;
            winner_ = actor;
            winning_mask_ = (1 << actor);
            components.terminal_outcome = +1;
        } else {
            if (outcome == -1) ++consecutive_empty_pool_passes_;
            else consecutive_empty_pool_passes_ = 0;

            if (pool_remaining() == 0 &&
                consecutive_empty_pool_passes_ >= total_players_) {
                finish_blocked_game();
                if (winner_ == actor) components.terminal_outcome = +1;
                else if (winner_ == -2) components.terminal_outcome = 0;
                else components.terminal_outcome = -1;
            }
        }

        float hand_r = weighted_hand_reward(components);
        float table_r = weighted_table_reward(components);
        float terminal_r = weighted_terminal_reward(components);

        if (!done_) current_player_ = (current_player_ + 1) % total_players_;

        StepResult result;
        result.reward = hand_r + table_r + terminal_r;
        result.hand_reward = hand_r;
        result.table_reward = table_r;
        result.terminal_reward = terminal_r;
        result.components = components;
        result.done = done_;
        result.winner = winner_;
        result.acting_player = actor;
        result.next_player = done_ ? -1 : current_player_;
        result.next_state = done_ ? observation(actor) : observation(current_player_);
        return result;
    }

    RummikubEnv clone() const {
        RummikubEnv copy(*this);
        copy.cached_actions_.clear();
        copy.generation_truncated_ = false;
        return copy;
    }

    // Create a perfect-information sample consistent with observer's
    // information set. Observer's hand, public table, player hand counts,
    // registration flags, current player and pool size remain unchanged.
    RummikubEnv determinize(int observer, std::uint32_t seed) const {
        if (observer < 0 || observer >= total_players_)
            throw std::out_of_range("observer out of range");

        RummikubEnv copy(*this);
        copy.cached_actions_.clear();
        copy.generation_truncated_ = false;

        std::vector<Tile> hidden;

        // Unknown opponent hands.
        for (int p = 0; p < total_players_; ++p) {
            if (p == observer) continue;
            auto counts = copy.hand_counts(p);
            for (int idx = 0; idx < TILE_TYPES; ++idx) {
                for (int k = 0; k < counts[idx]; ++k)
                    hidden.push_back(tile_from_index(idx));
            }
        }

        // Unknown pool remainder.
        for (std::size_t i = copy.pool_pos_; i < copy.pool_.size(); ++i)
            hidden.push_back(copy.pool_[i]);

        std::mt19937 local_rng(seed);
        std::shuffle(hidden.begin(), hidden.end(), local_rng);

        // Preserve only the public counts of opponent hands.
        int target_counts[MAX_PLAYERS]{};
        for (int p = 0; p < total_players_; ++p)
            target_counts[p] = copy.hand_count_[p];

        for (int p = 0; p < total_players_; ++p) {
            if (p == observer) continue;
            for (int c = 0; c < COLORS; ++c)
                copy.hand_[p][c].fill(0);
            copy.hand_count_[p] = 0;
        }

        std::size_t pos = 0;
        for (int p = 0; p < total_players_; ++p) {
            if (p == observer) continue;
            for (int k = 0; k < target_counts[p]; ++k) {
                if (pos >= hidden.size())
                    throw std::runtime_error("determinize: hidden pool underflow");
                copy.add_tile_to_hand(p, hidden[pos++]);
            }
        }

        copy.pool_.assign(hidden.begin() + static_cast<std::ptrdiff_t>(pos),
                          hidden.end());
        copy.pool_pos_ = 0;
        copy.rng_.seed(seed ^ 0x9E3779B9u);

        if (!copy.validate_state())
            throw std::runtime_error("determinize created invalid state");
        return copy;
    }

    // Terminal payoff suitable as a common final objective for evaluation,
    // ISMCTS rollouts, and 2-player MCCFR.
    float terminal_utility(int player) const {
        if (!done_) throw std::runtime_error("terminal_utility on nonterminal state");
        if (player < 0 || player >= total_players_)
            throw std::out_of_range("player out of range");
        if (winner_ == -2) return 0.0f;
        return winner_ == player ? 1.0f : -1.0f;
    }

    int hand_value(int player) const {
        if (player < 0 || player >= total_players_)
            throw std::out_of_range("player out of range");
        auto counts = hand_counts(player);
        int sum = 0;
        for (int idx = 0; idx < TILE_TYPES; ++idx) {
            int n = index_number(idx);
            sum += counts[idx] * n;
        }
        return sum;
    }

    int hand_size(int player) const {
        if (player < 0 || player >= total_players_)
            throw std::out_of_range("player out of range");
        return hand_count_[player];
    }

    bool registered(int player) const {
        if (player < 0 || player >= total_players_)
            throw std::out_of_range("player out of range");
        return registered_[player] != 0;
    }

    std::string information_state_key(int viewer) const {
        if (viewer < 0 || viewer >= total_players_)
            throw std::out_of_range("viewer out of range");
        std::ostringstream oss;
        oss << "V" << viewer << "|CP" << current_player_ << '|';

        auto counts = hand_counts(viewer);
        for (int x : counts) oss << x;
        oss << '|';

        auto table = canonical_table(table_melds_);
        for (const auto &m : table) oss << meld_key(m) << '/';
        oss << '|';

        for (int offset = 1; offset < total_players_; ++offset) {
            int p = (viewer + offset) % total_players_;
            oss << hand_count_[p] << ',';
        }
        oss << "|R" << registered_[viewer];
        return oss.str();
    }

    bool validate_state() const {
        if (total_players_ < 2 || total_players_ > MAX_PLAYERS) return false;
        if (current_player_ < 0 || current_player_ >= total_players_) return false;
        if (table_melds_.size() > static_cast<std::size_t>(MAX_MELDS)) return false;

        std::array<int, TILE_TYPES> total{};

        // Table.
        for (const auto &m : table_melds_) {
            if (!validate_meld(m)) return false;
            for (int i = 0; i < m.count; ++i) {
                int idx = tile_index(m.tiles[i]);
                if (idx < 0) return false;
                if (++total[idx] > 2) return false;
            }
        }

        // Hands (ordinary 2-D array: [color][number]).
        for (int p = 0; p < total_players_; ++p) {
            int real = 0;
            for (int c = 0; c < COLORS; ++c) {
                for (int n = 0; n < NUMBERS; ++n) {
                    int cnt = hand_[p][c][n];
                    if (cnt < 0 || cnt > 2) return false;
                    real += cnt;
                    total[c * NUMBERS + n] += cnt;
                    if (total[c * NUMBERS + n] > 2) return false;
                }
            }
            if (real != hand_count_[p]) return false;
        }

        // Remaining pool.
        for (std::size_t i = pool_pos_; i < pool_.size(); ++i) {
            int idx = tile_index(pool_[i]);
            if (idx < 0) return false;
            if (++total[idx] > 2) return false;
        }

        // Every physical tile type must still exist exactly twice globally.
        for (int idx = 0; idx < TILE_TYPES; ++idx)
            if (total[idx] != 2) return false;

        return true;
    }

private:
    // ------------------------------ Core state ------------------------------

    int total_players_ = 2;
    int current_player_ = 0;
    std::size_t max_legal_actions_ = 0; // 0 = exhaustive

    // Per-player hand. Each hand itself is a conventional 4 x 13 array.
    std::array<HandGrid, MAX_PLAYERS> hand_{};
    int hand_count_[MAX_PLAYERS]{};
    int registered_[MAX_PLAYERS]{};

    std::vector<Meld> table_melds_;

    std::vector<Tile> pool_;
    std::size_t pool_pos_ = 0;
    std::mt19937 rng_;

    int consecutive_empty_pool_passes_ = 0;
    bool done_ = false;
    int winner_ = -1;
    int winning_mask_ = 0;

    bool generation_truncated_ = false;
    std::vector<ActionInternal> cached_actions_;

    RewardConfig reward_config_{};

    struct Candidate {
        Meld meld;
        std::array<int, TILE_TYPES> use{}; // each entry is 0/1 for one meld
        int score = 0;
        std::string key;
    };

    // ------------------------------ Tile helpers ----------------------------

    static int color_index(char c) {
        for (int i = 0; i < COLORS; ++i)
            if (COLOR_ARRAY[i] == c) return i;
        return -1;
    }

    static int tile_index(const Tile &t) {
        int c = color_index(t.color);
        if (c < 0 || t.num < 1 || t.num > NUMBERS) return -1;
        return c * NUMBERS + (t.num - 1);
    }

    static Tile tile_from_index(int idx) {
        if (idx < 0 || idx >= TILE_TYPES)
            throw std::out_of_range("tile index out of range");
        int c = idx / NUMBERS;
        int n = idx % NUMBERS;
        return Tile{n + 1, COLOR_ARRAY[c]};
    }

    static int index_number(int idx) {
        return (idx % NUMBERS) + 1;
    }

    static bool validate_run(const Meld &m) {
        if (m.type != 'R' || m.count < 3 || m.count > 13) return false;
        int c = color_index(m.tiles[0].color);
        if (c < 0) return false;
        int prev = -1;
        for (int i = 0; i < m.count; ++i) {
            if (color_index(m.tiles[i].color) != c) return false;
            if (m.tiles[i].num < 1 || m.tiles[i].num > 13) return false;
            if (i > 0 && m.tiles[i].num != prev + 1) return false;
            prev = m.tiles[i].num;
        }
        return true;
    }

    static bool validate_group(const Meld &m) {
        if (m.type != 'G' || m.count < 3 || m.count > 4) return false;
        int num = m.tiles[0].num;
        if (num < 1 || num > 13) return false;
        int used_colors = 0;
        for (int i = 0; i < m.count; ++i) {
            if (m.tiles[i].num != num) return false;
            int c = color_index(m.tiles[i].color);
            if (c < 0) return false;
            int bit = 1 << c;
            if (used_colors & bit) return false;
            used_colors |= bit;
        }
        return true;
    }

    static bool validate_meld(const Meld &m) {
        return m.type == 'R' ? validate_run(m)
             : m.type == 'G' ? validate_group(m)
             : false;
    }

    static int meld_score(const Meld &m) {
        int sum = 0;
        for (int i = 0; i < m.count; ++i) sum += m.tiles[i].num;
        return sum;
    }

    static int color_mask_of(const Meld &m) {
        int mask = 0;
        for (int i = 0; i < m.count; ++i)
            mask |= 1 << color_index(m.tiles[i].color);
        return mask;
    }

    static int number_mask_of(const Meld &m) {
        int mask = 0;
        for (int i = 0; i < m.count; ++i)
            mask |= 1 << (m.tiles[i].num - 1);
        return mask;
    }

    static std::string meld_key(const Meld &m) {
        std::ostringstream oss;
        oss << m.type << ':' << color_mask_of(m) << ':' << number_mask_of(m);
        return oss.str();
    }

    static bool meld_less(const Meld &a, const Meld &b) {
        if (a.type != b.type) return a.type < b.type;
        int ac = color_mask_of(a), bc = color_mask_of(b);
        if (ac != bc) return ac < bc;
        int an = number_mask_of(a), bn = number_mask_of(b);
        if (an != bn) return an < bn;
        return a.count < b.count;
    }

    static std::vector<Meld> canonical_table(std::vector<Meld> table) {
        std::sort(table.begin(), table.end(), meld_less);
        return table;
    }

    std::array<int, TILE_TYPES> hand_counts(int player) const {
        std::array<int, TILE_TYPES> out{};
        for (int c = 0; c < COLORS; ++c) {
            for (int n = 0; n < NUMBERS; ++n) {
                out[c * NUMBERS + n] = hand_[player][c][n];
            }
        }
        return out;
    }

    std::array<int, TILE_TYPES> table_counts() const {
        std::array<int, TILE_TYPES> out{};
        for (const auto &m : table_melds_) {
            for (int i = 0; i < m.count; ++i) {
                int idx = tile_index(m.tiles[i]);
                ++out[idx];
            }
        }
        return out;
    }

    static int sum_counts(const std::array<int, TILE_TYPES> &a) {
        int sum = 0;
        for (int x : a) sum += x;
        return sum;
    }

    static bool can_consume(const std::array<int, TILE_TYPES> &remaining,
                            const Candidate &cand) {
        for (int i = 0; i < TILE_TYPES; ++i)
            if (cand.use[i] > remaining[i]) return false;
        return true;
    }

    static void consume(std::array<int, TILE_TYPES> &remaining,
                        const Candidate &cand,
                        int sign) {
        // sign = -1 consume, +1 restore
        for (int i = 0; i < TILE_TYPES; ++i)
            if (cand.use[i]) remaining[i] += sign * cand.use[i];
    }

    static void satisfy_required(std::array<int, TILE_TYPES> &required,
                                 const Candidate &cand,
                                 int sign) {
        // sign = -1 when selecting; +1 when undoing. Only occurrences that
        // were still required are changed. Caller supplies a saved copy when
        // exact restoration is needed; this helper is not used for undo.
        if (sign != -1) return;
        for (int i = 0; i < TILE_TYPES; ++i) {
            if (cand.use[i] && required[i] > 0) --required[i];
        }
    }

    static bool all_zero(const std::array<int, TILE_TYPES> &a) {
        for (int x : a) if (x != 0) return false;
        return true;
    }

    void add_tile_to_hand(int player, const Tile &t) {
        int c = color_index(t.color);
        int n = t.num - 1;
        if (c < 0 || n < 0 || n >= NUMBERS)
            throw std::runtime_error("invalid tile in add_tile_to_hand");
        if (hand_[player][c][n] >= 2)
            throw std::runtime_error("attempted to add third identical tile");
        ++hand_[player][c][n];
        ++hand_count_[player];
    }

    bool deal_one(int player) {
        if (pool_pos_ >= pool_.size()) return false;
        add_tile_to_hand(player, pool_[pool_pos_++]);
        return true;
    }

    static void counts_to_hand_grid(const std::array<int, TILE_TYPES> &counts,
                                    HandGrid &grid,
                                    int &total) {
        for (auto &row : grid) row.fill(0);
        total = 0;
        for (int idx = 0; idx < TILE_TYPES; ++idx) {
            int cnt = counts[idx];
            if (cnt < 0 || cnt > 2)
                throw std::runtime_error("invalid tile count in hand encoding");
            int c = idx / NUMBERS;
            int n = idx % NUMBERS;
            grid[c][n] = cnt;
            total += cnt;
        }
    }

    // --------------------------- Meld/candidate generation ------------------

    static std::vector<Candidate>
    enumerate_candidates(const std::array<int, TILE_TYPES> &available) {
        std::vector<Candidate> out;

        // Every contiguous run interval of length >= 3.
        for (int c = 0; c < COLORS; ++c) {
            for (int start = 0; start < NUMBERS; ++start) {
                bool possible = true;
                for (int end = start; end < NUMBERS; ++end) {
                    int idx = c * NUMBERS + end;
                    if (available[idx] == 0) possible = false;
                    if (!possible) break;
                    if (end - start + 1 < 3) continue;

                    Candidate cand;
                    cand.meld.type = 'R';
                    for (int n = start; n <= end; ++n) {
                        int pos = cand.meld.count++;
                        cand.meld.tiles[pos] = Tile{n + 1, COLOR_ARRAY[c]};
                        cand.use[c * NUMBERS + n] = 1;
                    }
                    cand.score = meld_score(cand.meld);
                    cand.key = meld_key(cand.meld);
                    out.push_back(cand);
                }
            }
        }

        // Every valid 3-color / 4-color group.
        for (int n = 0; n < NUMBERS; ++n) {
            for (int mask = 0; mask < (1 << COLORS); ++mask) {
                int bits = 0;
                for (int c = 0; c < COLORS; ++c)
                    if (mask & (1 << c)) ++bits;
                if (bits != 3 && bits != 4) continue;

                bool possible = true;
                for (int c = 0; c < COLORS; ++c) {
                    if ((mask & (1 << c)) &&
                        available[c * NUMBERS + n] == 0) {
                        possible = false;
                        break;
                    }
                }
                if (!possible) continue;

                Candidate cand;
                cand.meld.type = 'G';
                for (int c = 0; c < COLORS; ++c) {
                    if (!(mask & (1 << c))) continue;
                    int pos = cand.meld.count++;
                    cand.meld.tiles[pos] = Tile{n + 1, COLOR_ARRAY[c]};
                    cand.use[c * NUMBERS + n] = 1;
                }
                cand.score = meld_score(cand.meld);
                cand.key = meld_key(cand.meld);
                out.push_back(cand);
            }
        }

        std::sort(out.begin(), out.end(), [](const Candidate &a, const Candidate &b) {
            return a.key < b.key;
        });
        return out;
    }

    // ----------------------------- Action encoding --------------------------

    static void append_meld_features(std::vector<float> &out, const Meld &m) {
        out.push_back(m.type == 'R' ? 1.0f : 0.0f);
        out.push_back(m.type == 'G' ? 1.0f : 0.0f);

        int cm = color_mask_of(m);
        for (int c = 0; c < COLORS; ++c)
            out.push_back((cm & (1 << c)) ? 1.0f : 0.0f);

        int nm = number_mask_of(m);
        for (int n = 0; n < NUMBERS; ++n)
            out.push_back((nm & (1 << n)) ? 1.0f : 0.0f);
    }

    static void append_table_features(std::vector<float> &out,
                                      const std::vector<Meld> &table_in) {
        auto table = canonical_table(table_in);
        if (table.size() > static_cast<std::size_t>(MAX_MELDS))
            throw std::runtime_error("table contains too many melds");

        for (const auto &m : table) append_meld_features(out, m);
        for (std::size_t i = table.size(); i < static_cast<std::size_t>(MAX_MELDS); ++i)
            for (int k = 0; k < MELD_FEATURE_DIM; ++k) out.push_back(0.0f);
    }

    std::vector<float> encode_action(const std::array<int, TILE_TYPES> &before_hand,
                                     const ActionInternal &a) const {
        std::vector<float> out;
        out.reserve(ACTION_DIM);

        std::array<int, TILE_TYPES> after{};
        for (int c = 0; c < COLORS; ++c) {
            for (int n = 0; n < NUMBERS; ++n) {
                after[c * NUMBERS + n] = a.final_hand[c][n];
            }
        }

        for (int i = 0; i < TILE_TYPES; ++i) {
            int used = std::max(0, before_hand[i] - after[i]);
            out.push_back(used / 2.0f);
        }

        append_table_features(out, a.final_table);
        out.push_back(a.draw ? 1.0f : 0.0f);

        if (static_cast<int>(out.size()) != ACTION_DIM)
            throw std::runtime_error("ACTION_DIM mismatch");
        return out;
    }

    static std::string action_key(const ActionInternal &a) {
        // The hand is still read directly from the 4x13 2-D array.
        // Encode each 0/1/2 count as one byte instead of converting all 52
        // integers through ostringstream.  This removes representation-irrelevant
        // string-formatting overhead while preserving exactly the same duplicate
        // action criterion.  No bitmask is used here.
        std::string key;
        key.reserve(2 + TILE_TYPES + 1 + a.final_table.size() * 12);
        key.push_back(a.draw ? 'D' : 'P');
        key.push_back(a.register_after ? '1' : '0');

        for (int c = 0; c < COLORS; ++c) {
            for (int n = 0; n < NUMBERS; ++n) {
                const int cnt = a.final_hand[c][n];
                if (cnt < 0 || cnt > 2)
                    throw std::runtime_error("invalid 2-D hand count in action_key");
                key.push_back(static_cast<char>('0' + cnt));
            }
        }

        key.push_back('|');
        auto table = canonical_table(a.final_table);
        for (const auto &m : table) {
            key += meld_key(m);
            key.push_back('/');
        }
        return key;
    }

    bool reached_action_cap(std::size_t current_size) {
        if (max_legal_actions_ > 0 && current_size >= max_legal_actions_) {
            generation_truncated_ = true;
            return true;
        }
        return false;
    }

    void add_action(std::vector<ActionInternal> &out,
                    std::unordered_set<std::string> &seen,
                    const std::array<int, TILE_TYPES> &before_hand,
                    ActionInternal action) {
        auto key = action_key(action);
        if (!seen.insert(key).second) return;
        if (reached_action_cap(out.size())) return;
        action.final_table = canonical_table(action.final_table);
        action.features = encode_action(before_hand, action);
        out.push_back(std::move(action));
    }

    ActionInternal make_draw_action(int player) const {
        ActionInternal a;
        a.draw = true;
        a.register_after = registered_[player] != 0;
        a.final_hand = hand_[player];
        a.final_hand_count = hand_count_[player];
        a.final_table = table_melds_;
        a.description = pool_remaining() > 0 ? "DRAW" : "PASS(pool empty)";
        return a;
    }

    // ------------------------- Registration action search -------------------

    void registration_dfs(
        int player,
        const std::vector<Candidate> &cands,
        std::array<int, TILE_TYPES> &remaining_hand,
        int start_idx,
        int score,
        std::vector<Meld> &selected,
        const std::array<int, TILE_TYPES> &before_hand,
        std::vector<ActionInternal> &out,
        std::unordered_set<std::string> &seen) {

        if (reached_action_cap(out.size())) return;

        if (score >= 30 && !selected.empty()) {
            ActionInternal a;
            a.draw = false;
            a.register_after = true;
            counts_to_hand_grid(remaining_hand,
                                a.final_hand,
                                a.final_hand_count);
            a.final_table = table_melds_;
            a.final_table.insert(a.final_table.end(), selected.begin(), selected.end());
            a.description = "REGISTER: sum=" + std::to_string(score) +
                            ", melds=" + std::to_string(selected.size());
            add_action(out, seen, before_hand, std::move(a));
            if (reached_action_cap(out.size())) return;
        }

        if (selected.size() + table_melds_.size() >= static_cast<std::size_t>(MAX_MELDS))
            return;

        // Nondecreasing candidate indices remove permutation duplicates while
        // still allowing the same meld twice when both copies exist.
        for (int i = start_idx; i < static_cast<int>(cands.size()); ++i) {
            const auto &cand = cands[i];
            if (!can_consume(remaining_hand, cand)) continue;

            consume(remaining_hand, cand, -1);
            selected.push_back(cand.meld);
            registration_dfs(player, cands, remaining_hand, i,
                             score + cand.score, selected,
                             before_hand, out, seen);
            selected.pop_back();
            consume(remaining_hand, cand, +1);

            if (reached_action_cap(out.size())) return;
        }
    }

    // ---------------------- Complete rearrangement search -------------------

    // Choose a currently-uncovered mandatory table tile with the smallest
    // number of feasible candidate melds (MRV heuristic).
    int choose_required_pivot(
        const std::array<int, TILE_TYPES> &required,
        const std::array<int, TILE_TYPES> &remaining,
        const std::vector<Candidate> &cands,
        const std::array<std::vector<int>, TILE_TYPES> &contains) const {

        int best_tile = -1;
        int best_choices = std::numeric_limits<int>::max();

        for (int t = 0; t < TILE_TYPES; ++t) {
            if (required[t] <= 0) continue;
            int choices = 0;
            for (int ci : contains[t]) {
                if (can_consume(remaining, cands[ci])) ++choices;
            }
            if (choices < best_choices) {
                best_choices = choices;
                best_tile = t;
                if (choices == 0) break;
            }
        }
        return best_tile;
    }

    void optional_hand_melds_dfs(
        int player,
        const std::vector<Candidate> &cands,
        std::array<int, TILE_TYPES> &remaining_hand,
        int start_idx,
        std::vector<Meld> &selected,
        int root_hand_count,
        const std::array<int, TILE_TYPES> &before_hand,
        std::vector<ActionInternal> &out,
        std::unordered_set<std::string> &seen) {

        if (reached_action_cap(out.size())) return;

        int unused = sum_counts(remaining_hand);
        int used_hand = root_hand_count - unused;

        // All mandatory table tiles were already covered before entering this
        // function. Every remaining tile is therefore an unused hand tile.
        if (used_hand > 0) {
            ActionInternal a;
            a.draw = false;
            a.register_after = true;
            counts_to_hand_grid(remaining_hand,
                                a.final_hand,
                                a.final_hand_count);
            a.final_table = selected;
            a.description = "PLAY/REARRANGE: used_hand=" +
                            std::to_string(used_hand) +
                            ", melds=" + std::to_string(selected.size());
            add_action(out, seen, before_hand, std::move(a));
            if (reached_action_cap(out.size())) return;
        }

        if (selected.size() >= static_cast<std::size_t>(MAX_MELDS)) return;

        // Add any further melds made solely from still-unused hand tiles.
        for (int i = start_idx; i < static_cast<int>(cands.size()); ++i) {
            const auto &cand = cands[i];
            if (!can_consume(remaining_hand, cand)) continue;

            consume(remaining_hand, cand, -1);
            selected.push_back(cand.meld);
            optional_hand_melds_dfs(player, cands, remaining_hand, i,
                                    selected, root_hand_count, before_hand,
                                    out, seen);
            selected.pop_back();
            consume(remaining_hand, cand, +1);

            if (reached_action_cap(out.size())) return;
        }
    }

    void cover_required_dfs(
        int player,
        const std::vector<Candidate> &cands,
        const std::array<std::vector<int>, TILE_TYPES> &contains,
        std::array<int, TILE_TYPES> &required,
        std::array<int, TILE_TYPES> &remaining,
        std::vector<Meld> &selected,
        int root_hand_count,
        const std::array<int, TILE_TYPES> &before_hand,
        std::vector<ActionInternal> &out,
        std::unordered_set<std::string> &seen,
        int repeated_pivot = -1,
        int min_candidate_for_same_pivot = 0) {

        if (reached_action_cap(out.size())) return;
        if (selected.size() > static_cast<std::size_t>(MAX_MELDS)) return;

        if (all_zero(required)) {
            // At this moment remaining contains only hand tiles, because every
            // original table occurrence has been consumed exactly once.
            optional_hand_melds_dfs(player, cands, remaining, 0,
                                    selected, root_hand_count, before_hand,
                                    out, seen);
            return;
        }

        int pivot = choose_required_pivot(required, remaining, cands, contains);
        if (pivot < 0) return;

        int start_filter = (pivot == repeated_pivot)
                         ? min_candidate_for_same_pivot : 0;

        bool any = false;
        for (int ci : contains[pivot]) {
            if (ci < start_filter) continue;
            const auto &cand = cands[ci];
            if (!can_consume(remaining, cand)) continue;

            any = true;
            auto required_before = required;
            consume(remaining, cand, -1);
            satisfy_required(required, cand, -1);
            selected.push_back(cand.meld);

            bool same_pivot_still_required = required[pivot] > 0;
            cover_required_dfs(
                player, cands, contains,
                required, remaining, selected,
                root_hand_count, before_hand,
                out, seen,
                same_pivot_still_required ? pivot : -1,
                same_pivot_still_required ? ci : 0);

            selected.pop_back();
            required = required_before;
            consume(remaining, cand, +1);

            if (reached_action_cap(out.size())) return;
        }

        (void)any; // no feasible candidate -> dead branch
    }

    std::vector<ActionInternal> generate_actions_for_player(int player) {
        generation_truncated_ = false;
        std::vector<ActionInternal> out;
        std::unordered_set<std::string> seen;

        const auto before_hand = hand_counts(player);

        // DRAW/PASS is always represented explicitly as action index 0.
        {
            ActionInternal draw = make_draw_action(player);
            draw.features = encode_action(before_hand, draw);
            seen.insert(action_key(draw));
            out.push_back(std::move(draw));
        }

        if (reached_action_cap(out.size())) return out;

        if (!registered_[player]) {
            auto remaining = before_hand;
            auto cands = enumerate_candidates(remaining);
            std::vector<Meld> selected;
            registration_dfs(player, cands, remaining, 0, 0,
                             selected, before_hand, out, seen);
            return out;
        }

        // COMPLETE NORMAL TURN:
        // required = every occurrence currently on table.
        // remaining = table occurrences + player's hand occurrences.
        auto required = table_counts();
        auto remaining = required;
        for (int i = 0; i < TILE_TYPES; ++i) remaining[i] += before_hand[i];

        auto cands = enumerate_candidates(remaining);
        std::array<std::vector<int>, TILE_TYPES> contains;
        for (int ci = 0; ci < static_cast<int>(cands.size()); ++ci) {
            for (int t = 0; t < TILE_TYPES; ++t) {
                if (cands[ci].use[t]) contains[t].push_back(ci);
            }
        }

        std::vector<Meld> selected;

        if (all_zero(required)) {
            // Empty table: any one-or-more hand meld combination is a legal play.
            auto rem_hand = before_hand;
            optional_hand_melds_dfs(player, cands, rem_hand, 0,
                                    selected, sum_counts(before_hand), before_hand,
                                    out, seen);
        } else {
            cover_required_dfs(player, cands, contains,
                               required, remaining, selected,
                               sum_counts(before_hand), before_hand,
                               out, seen);
        }

        return out;
    }

    void ensure_actions_cached() {
        if (cached_actions_.empty())
            cached_actions_ = generate_actions_for_player(current_player_);
        if (cached_actions_.empty())
            throw std::runtime_error("no legal action generated");
    }

    // ------------------------------- Apply action ---------------------------

    int apply_action(int player, const ActionInternal &a) {
        if (a.draw) {
            if (pool_remaining() > 0) {
                if (!deal_one(player))
                    throw std::runtime_error("draw failed despite nonempty pool");
                return 0;
            }
            return -1;
        }

        hand_[player] = a.final_hand;
        hand_count_[player] = a.final_hand_count;
        table_melds_ = canonical_table(a.final_table);
        if (a.register_after) registered_[player] = 1;
        return 1;
    }

    // ------------------------------ Reward / terminal -----------------------

    struct TableShapeCounts {
        int len3 = 0;
        int len4 = 0;
        int len5plus = 0;
    };

    static TableShapeCounts table_shape_counts(const std::vector<Meld> &table) {
        TableShapeCounts counts;
        for (const auto &m : table) {
            if (m.count == 3) ++counts.len3;
            else if (m.count == 4) ++counts.len4;
            else if (m.count >= 5) ++counts.len5plus;
        }
        return counts;
    }

    static RewardComponents reward_components(
        const std::array<int, TILE_TYPES> &before,
        const std::array<int, TILE_TYPES> &after,
        const TableShapeCounts &before_table,
        const TableShapeCounts &after_table) {

        RewardComponents c;

        for (int i = 0; i < TILE_TYPES; ++i) {
            int removed = std::max(0, before[i] - after[i]);
            if (removed == 0) continue;

            int num = index_number(i);
            if (num == 1 || num == 13) c.hand_1_13_used += removed;
            else if (num == 2 || num == 12) c.hand_2_12_used += removed;
            else c.hand_normal_used += removed;
        }

        c.table_len3_delta = after_table.len3 - before_table.len3;
        c.table_len4_delta = after_table.len4 - before_table.len4;
        c.table_len5plus_delta = after_table.len5plus - before_table.len5plus;
        return c;
    }

    float weighted_hand_reward(const RewardComponents &c) const {
        return c.hand_normal_used * reward_config_.hand_normal
             + c.hand_2_12_used * reward_config_.hand_2_12
             + c.hand_1_13_used * reward_config_.hand_1_13;
    }

    float weighted_table_reward(const RewardComponents &c) const {
        return c.table_len3_delta * reward_config_.table_len3
             + c.table_len4_delta * reward_config_.table_len4
             + c.table_len5plus_delta * reward_config_.table_len5plus;
    }

    float weighted_terminal_reward(const RewardComponents &c) const {
        if (c.terminal_outcome > 0) return reward_config_.win;
        if (c.terminal_outcome < 0) return reward_config_.loss;
        if (done_ && winner_ == -2) return reward_config_.tie;
        return 0.0f;
    }

    void finish_blocked_game() {
        done_ = true;
        int best_value = std::numeric_limits<int>::max();
        winning_mask_ = 0;

        for (int p = 0; p < total_players_; ++p) {
            int v = hand_value(p);
            if (v < best_value) {
                best_value = v;
                winning_mask_ = (1 << p);
            } else if (v == best_value) {
                winning_mask_ |= (1 << p);
            }
        }

        if ((winning_mask_ & (winning_mask_ - 1)) == 0) {
            for (int p = 0; p < total_players_; ++p) {
                if (winning_mask_ & (1 << p)) {
                    winner_ = p;
                    return;
                }
            }
        }
        winner_ = -2;
    }

    void ensure_not_terminal() const {
        if (done_) throw std::runtime_error("environment is terminal");
    }
};

} // namespace rummikub2d

// ============================================================================
// pybind11 binding
// Build example (Linux/macOS, Python environment with pybind11 installed):
//   c++ -O3 -std=c++17 -shared -fPIC -DRUMMIKUB_PYBIND [pybind11 includes]
//       rummikub_env_2d.cpp -o rummikub_env_2d[python extension suffix]
//
// On Windows, CMake or setuptools/scikit-build is recommended.
// ============================================================================

#ifdef RUMMIKUB_PYBIND
PYBIND11_MODULE(rummikub_env_2d, m) {
    m.doc() = "2-D-array no-joker Rummikub environment for DQN/PPO/ISMCTS/MCCFR";
    m.attr("STATE_DIM") = rummikub2d::STATE_DIM;
    m.attr("ACTION_DIM") = rummikub2d::ACTION_DIM;
    m.attr("TOTAL_TILES") = rummikub2d::TOTAL_TILES;

    py::class_<rummikub2d::RewardConfig>(m, "RewardConfig")
        .def(py::init<>())
        .def_readwrite("hand_normal", &rummikub2d::RewardConfig::hand_normal)
        .def_readwrite("hand_2_12", &rummikub2d::RewardConfig::hand_2_12)
        .def_readwrite("hand_1_13", &rummikub2d::RewardConfig::hand_1_13)
        .def_readwrite("table_len3", &rummikub2d::RewardConfig::table_len3)
        .def_readwrite("table_len4", &rummikub2d::RewardConfig::table_len4)
        .def_readwrite("table_len5plus", &rummikub2d::RewardConfig::table_len5plus)
        .def_readwrite("win", &rummikub2d::RewardConfig::win)
        .def_readwrite("loss", &rummikub2d::RewardConfig::loss)
        .def_readwrite("tie", &rummikub2d::RewardConfig::tie);

    py::class_<rummikub2d::RewardComponents>(m, "RewardComponents")
        .def_readonly("hand_normal_used", &rummikub2d::RewardComponents::hand_normal_used)
        .def_readonly("hand_2_12_used", &rummikub2d::RewardComponents::hand_2_12_used)
        .def_readonly("hand_1_13_used", &rummikub2d::RewardComponents::hand_1_13_used)
        .def_readonly("table_len3_delta", &rummikub2d::RewardComponents::table_len3_delta)
        .def_readonly("table_len4_delta", &rummikub2d::RewardComponents::table_len4_delta)
        .def_readonly("table_len5plus_delta", &rummikub2d::RewardComponents::table_len5plus_delta)
        .def_readonly("terminal_outcome", &rummikub2d::RewardComponents::terminal_outcome);

    py::class_<rummikub2d::StepResult>(m, "StepResult")
        .def_readonly("next_state", &rummikub2d::StepResult::next_state)
        .def_readonly("reward", &rummikub2d::StepResult::reward)
        .def_readonly("hand_reward", &rummikub2d::StepResult::hand_reward)
        .def_readonly("table_reward", &rummikub2d::StepResult::table_reward)
        .def_readonly("terminal_reward", &rummikub2d::StepResult::terminal_reward)
        .def_readonly("components", &rummikub2d::StepResult::components)
        .def_readonly("done", &rummikub2d::StepResult::done)
        .def_readonly("winner", &rummikub2d::StepResult::winner)
        .def_readonly("acting_player", &rummikub2d::StepResult::acting_player)
        .def_readonly("next_player", &rummikub2d::StepResult::next_player);

    py::class_<rummikub2d::RummikubEnv>(m, "RummikubEnv")
        .def(py::init<int, std::size_t>(),
             py::arg("players") = 2,
             py::arg("max_legal_actions") = 0)
        .def("reset", &rummikub2d::RummikubEnv::reset,
             py::arg("seed"), py::arg("start_player") = 0)
        .def("players", &rummikub2d::RummikubEnv::players)
        .def("current_player", &rummikub2d::RummikubEnv::current_player)
        .def("terminal", &rummikub2d::RummikubEnv::terminal)
        .def("winner", &rummikub2d::RummikubEnv::winner)
        .def("winning_mask", &rummikub2d::RummikubEnv::winning_mask)
        .def("state_dim", &rummikub2d::RummikubEnv::state_dim)
        .def("action_dim", &rummikub2d::RummikubEnv::action_dim)
        .def("max_melds", &rummikub2d::RummikubEnv::max_melds)
        .def("pool_remaining", &rummikub2d::RummikubEnv::pool_remaining)
        .def("generation_truncated", &rummikub2d::RummikubEnv::generation_truncated)
        .def("set_reward_config", &rummikub2d::RummikubEnv::set_reward_config,
             py::arg("config"))
        .def("reward_config", &rummikub2d::RummikubEnv::reward_config)
        .def("set_reward_weights", &rummikub2d::RummikubEnv::set_reward_weights,
             py::arg("hand_normal"),
             py::arg("hand_2_12"),
             py::arg("hand_1_13"),
             py::arg("table_len3"),
             py::arg("table_len4"),
             py::arg("table_len5plus"),
             py::arg("win"),
             py::arg("loss"),
             py::arg("tie") = 0.0f)
        .def("set_max_legal_actions", &rummikub2d::RummikubEnv::set_max_legal_actions)
        .def("observation", &rummikub2d::RummikubEnv::observation,
             py::arg("viewer"))
        .def("legal_action_features", &rummikub2d::RummikubEnv::legal_action_features)
        .def("legal_action_descriptions", &rummikub2d::RummikubEnv::legal_action_descriptions)
        .def("legal_action_count", &rummikub2d::RummikubEnv::legal_action_count)
        .def("step", &rummikub2d::RummikubEnv::step,
             py::arg("action_index"))
        .def("clone", &rummikub2d::RummikubEnv::clone)
        .def("determinize", &rummikub2d::RummikubEnv::determinize,
             py::arg("observer"), py::arg("seed"))
        .def("terminal_utility", &rummikub2d::RummikubEnv::terminal_utility,
             py::arg("player"))
        .def("hand_value", &rummikub2d::RummikubEnv::hand_value,
             py::arg("player"))
        .def("hand_size", &rummikub2d::RummikubEnv::hand_size,
             py::arg("player"))
        .def("registered", &rummikub2d::RummikubEnv::registered,
             py::arg("player"))
        .def("information_state_key", &rummikub2d::RummikubEnv::information_state_key,
             py::arg("viewer"))
        .def("validate_state", &rummikub2d::RummikubEnv::validate_state);
}
#endif

#ifndef RUMMIKUB_PYBIND
#include <iostream>

// Standalone smoke test. A small cap is used only so the test completes fast;
// set max_legal_actions=0 in research runs when exhaustive action generation
// is required.
int main() {
    rummikub2d::RummikubEnv env(2, 2000);
    // Reward weights intentionally remain zero in standalone mode.
    // Training code should set them from Python via set_reward_weights().
    auto s = env.reset(42, 0);

    std::cout << "state_dim=" << s.size()
              << " expected=" << env.state_dim() << '\n';
    std::cout << "valid=" << env.validate_state() << '\n';

    for (int turn = 0; turn < 12 && !env.terminal(); ++turn) {
        int p = env.current_player();
        auto actions = env.legal_action_features();
        std::cout << "turn=" << turn
                  << " player=" << p
                  << " legal=" << actions.size()
                  << " truncated=" << env.generation_truncated()
                  << " pool=" << env.pool_remaining() << '\n';

        // Smoke test chooses DRAW/PASS (index 0).
        auto r = env.step(0);
        std::cout << "  reward=" << r.reward
                  << " done=" << r.done
                  << " winner=" << r.winner << '\n';
    }

    std::cout << "final_valid=" << env.validate_state() << '\n';

    // Determinization must preserve validity and the observer's information.
    auto det = env.determinize(0, 12345);
    std::cout << "determinized_valid=" << det.validate_state() << '\n';
    return 0;
}
#endif
