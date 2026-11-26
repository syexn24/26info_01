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

namespace rummikub {

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

    std::array<int, COLORS> final_hand1{};
    std::array<int, COLORS> final_hand2{};
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
        std::memset(hand1_, 0, sizeof(hand1_));
        std::memset(hand2_, 0, sizeof(hand2_));
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

        search_steps_ = 0;
        search_budget_exhausted_ = false;

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
    std::size_t search_steps_used() const {
        return search_steps_;
    }

    std::size_t max_search_steps() const {
        return max_search_steps_;
    }

    bool search_budget_exhausted() const {
        return search_budget_exhausted_;
    }

    void set_max_search_steps(std::size_t n) {
        max_search_steps_ = n;

        // 이미 만들어진 action cache는
        // 이전 탐색 제한을 기준으로 생성되었으므로 제거한다.
        cached_actions_.clear();
    }

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
            for (int c = 0; c < COLORS; ++c) {
                copy.hand1_[p][c] = 0;
                copy.hand2_[p][c] = 0;
            }
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

        // Hands.
        for (int p = 0; p < total_players_; ++p) {
            int real = 0;
            for (int c = 0; c < COLORS; ++c) {
                if (hand2_[p][c] & ~hand1_[p][c]) return false;
                for (int n = 0; n < NUMBERS; ++n) {
                    int bit = 1 << n;
                    if (hand1_[p][c] & bit) {
                        ++real;
                        if (++total[c * NUMBERS + n] > 2) return false;
                    }
                    if (hand2_[p][c] & bit) {
                        ++real;
                        if (++total[c * NUMBERS + n] > 2) return false;
                    }
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

    int hand1_[MAX_PLAYERS][COLORS]{};
    int hand2_[MAX_PLAYERS][COLORS]{};
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

    // ------------------------------------------------------------
    // Action generation search budget
    //
    // max_search_steps_ == 0 : unlimited
    //
    // legal action의 최종 개수뿐 아니라
    // DFS 내부에서 검사하는 후보/재귀 탐색량 자체를 제한한다.
    // ------------------------------------------------------------

    std::size_t max_search_steps_ = 100000;
    std::size_t search_steps_ = 0;

    bool search_budget_exhausted_ = false;

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
                int bit = 1 << n;
                int cnt = 0;
                if (hand1_[player][c] & bit) ++cnt;
                if (hand2_[player][c] & bit) ++cnt;
                out[c * NUMBERS + n] = cnt;
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
        int bit = 1 << (t.num - 1);
        if (!(hand1_[player][c] & bit)) hand1_[player][c] |= bit;
        else if (!(hand2_[player][c] & bit)) hand2_[player][c] |= bit;
        else throw std::runtime_error("attempted to add third identical tile");
        ++hand_count_[player];
    }

    bool deal_one(int player) {
        if (pool_pos_ >= pool_.size()) return false;
        add_tile_to_hand(player, pool_[pool_pos_++]);
        return true;
    }

    static void counts_to_hand_masks(const std::array<int, TILE_TYPES> &counts,
                                     std::array<int, COLORS> &h1,
                                     std::array<int, COLORS> &h2,
                                     int &total) {
        h1.fill(0);
        h2.fill(0);
        total = 0;
        for (int idx = 0; idx < TILE_TYPES; ++idx) {
            int cnt = counts[idx];
            if (cnt < 0 || cnt > 2)
                throw std::runtime_error("invalid tile count in hand encoding");
            int c = idx / NUMBERS;
            int n = idx % NUMBERS;
            int bit = 1 << n;
            if (cnt >= 1) h1[c] |= bit;
            if (cnt == 2) h2[c] |= bit;
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
                int bit = 1 << n;
                int cnt = 0;
                if (a.final_hand1[c] & bit) ++cnt;
                if (a.final_hand2[c] & bit) ++cnt;
                after[c * NUMBERS + n] = cnt;
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
        std::ostringstream oss;
        oss << (a.draw ? 'D' : 'P') << '|'
            << (a.register_after ? '1' : '0') << '|';
        for (int c = 0; c < COLORS; ++c)
            oss << a.final_hand1[c] << ',' << a.final_hand2[c] << ';';
        oss << '|';
        auto table = canonical_table(a.final_table);
        for (const auto &m : table) oss << meld_key(m) << '/';
        return oss.str();
    }

    // ============================================================================
    // Action generation limit
    // ============================================================================

    bool reached_action_cap(std::size_t current_size) {

        if (
            max_legal_actions_ > 0
            && current_size >= max_legal_actions_
        ) {

            generation_truncated_ = true;

            return true;
        }

        return false;
    }


    // ------------------------------------------------------------
    // 새로운 탐색 예산 초기화
    // ------------------------------------------------------------

    void reset_search_budget() {

        search_steps_ = 0;

        search_budget_exhausted_ = false;
    }


    // ------------------------------------------------------------
    // 탐색 1회 소비
    //
    // max_search_steps_ == 0 이면 제한 없음.
    //
    // 제한에 도달하면
    // generation_truncated_도 true로 만들어
    // 기존 Python 코드의 generation_truncated() 통계와
    // 호환되도록 한다.
    // ------------------------------------------------------------

    bool consume_search_step() {

        if (max_search_steps_ == 0) {

            ++search_steps_;

            return true;
        }


        if (
            search_steps_
            >= max_search_steps_
        ) {

            search_budget_exhausted_ = true;

            generation_truncated_ = true;

            return false;
        }


        ++search_steps_;

        return true;
    }


    // ------------------------------------------------------------
    // 현재 action generation을 즉시 중단해야 하는가?
    // ------------------------------------------------------------

    bool should_stop_generation(
        std::size_t current_action_count
    ) {

        if (
            reached_action_cap(
                current_action_count
            )
        ) {

            return true;
        }


        if (
            search_budget_exhausted_
        ) {

            return true;
        }


        return false;
    }


    // ============================================================================
    // Add action
    // ============================================================================

    void add_action(
        std::vector<ActionInternal> &out,
        std::unordered_set<std::string> &seen,
        const std::array<int, TILE_TYPES> &before_hand,
        ActionInternal action
    ) {

        if (
            should_stop_generation(
                out.size()
            )
        ) {

            return;
        }


        auto key = action_key(
            action
        );


        // 중복 action은 즉시 제거
        if (
            !seen.insert(
                key
            ).second
        ) {

            return;
        }


        if (
            should_stop_generation(
                out.size()
            )
        ) {

            return;
        }


        action.final_table = canonical_table(
            action.final_table
        );


        action.features = encode_action(
            before_hand,
            action
        );


        out.push_back(
            std::move(action)
        );
    }


    // ============================================================================
    // DRAW / PASS
    // ============================================================================

    ActionInternal make_draw_action(
        int player
    ) const {

        ActionInternal a;

        a.draw = true;

        a.register_after = (
            registered_[player] != 0
        );


        for (
            int c = 0;
            c < COLORS;
            ++c
        ) {

            a.final_hand1[c] =
                hand1_[player][c];

            a.final_hand2[c] =
                hand2_[player][c];
        }


        a.final_hand_count =
            hand_count_[player];


        a.final_table =
            table_melds_;


        a.description = (
            pool_remaining() > 0

            ? "DRAW"

            : "PASS(pool empty)"
        );


        return a;
    }


    // ============================================================================
    // Registration action search
    // ============================================================================

    void registration_dfs(
        int player,
        const std::vector<Candidate> &cands,
        std::array<int, TILE_TYPES> &remaining_hand,
        int start_idx,
        int score,
        std::vector<Meld> &selected,
        const std::array<int, TILE_TYPES> &before_hand,
        std::vector<ActionInternal> &out,
        std::unordered_set<std::string> &seen
    ) {

        // --------------------------------------------------------
        // 전체 generation 중단 검사
        // --------------------------------------------------------

        if (
            should_stop_generation(
                out.size()
            )
        ) {

            return;
        }


        // 재귀 진입 자체도 탐색량 1회로 계산
        if (
            !consume_search_step()
        ) {

            return;
        }


        // --------------------------------------------------------
        // 등록 점수 30 이상
        // --------------------------------------------------------

        if (
            score >= 30
            && !selected.empty()
        ) {

            ActionInternal a;

            a.draw = false;

            a.register_after = true;


            counts_to_hand_masks(
                remaining_hand,

                a.final_hand1,

                a.final_hand2,

                a.final_hand_count
            );


            a.final_table =
                table_melds_;


            a.final_table.insert(
                a.final_table.end(),

                selected.begin(),

                selected.end()
            );


            a.description =
                "REGISTER: sum="
                + std::to_string(score)
                + ", melds="
                + std::to_string(
                    selected.size()
                );


            add_action(
                out,
                seen,
                before_hand,
                std::move(a)
            );


            if (
                should_stop_generation(
                    out.size()
                )
            ) {

                return;
            }
        }


        // --------------------------------------------------------
        // table meld 개수 제한
        // --------------------------------------------------------

        if (
            selected.size()
                + table_melds_.size()

            >= static_cast<std::size_t>(
                MAX_MELDS
            )
        ) {

            return;
        }


        // --------------------------------------------------------
        // 후보 탐색
        // --------------------------------------------------------

        for (
            int i = start_idx;
            i < static_cast<int>(
                cands.size()
            );
            ++i
        ) {

            if (
                should_stop_generation(
                    out.size()
                )
            ) {

                return;
            }


            // 후보 하나를 검사하는 것도
            // 탐색 비용으로 계산한다.
            if (
                !consume_search_step()
            ) {

                return;
            }


            const auto &cand =
                cands[i];


            if (
                !can_consume(
                    remaining_hand,
                    cand
                )
            ) {

                continue;
            }


            consume(
                remaining_hand,
                cand,
                -1
            );


            selected.push_back(
                cand.meld
            );


            registration_dfs(
                player,
                cands,
                remaining_hand,
                i,
                score + cand.score,
                selected,
                before_hand,
                out,
                seen
            );


            selected.pop_back();


            consume(
                remaining_hand,
                cand,
                +1
            );


            // 하위 재귀에서 탐색 예산을 소진했으면
            // 현재 재귀도 즉시 종료한다.
            if (
                should_stop_generation(
                    out.size()
                )
            ) {

                return;
            }
        }
    }


    // ============================================================================
    // Choose required pivot
    //
    // 현재 테이블에서 아직 처리되지 않은 tile 중
    // 가능한 meld 후보 수가 가장 적은 tile을 선택한다.
    // ============================================================================

    int choose_required_pivot(
        const std::array<int, TILE_TYPES> &required,
        const std::array<int, TILE_TYPES> &remaining,
        const std::vector<Candidate> &cands,
        const std::array<
            std::vector<int>,
            TILE_TYPES
        > &contains
    ) {

        int best_tile = -1;

        int best_choices =
            std::numeric_limits<int>::max();


        for (
            int t = 0;
            t < TILE_TYPES;
            ++t
        ) {

            if (
                required[t] <= 0
            ) {

                continue;
            }


            int choices = 0;


            for (
                int ci : contains[t]
            ) {

                // pivot 계산 과정의 후보 검사도
                // 탐색 budget에 포함한다.
                if (
                    !consume_search_step()
                ) {

                    return -1;
                }


                if (
                    can_consume(
                        remaining,
                        cands[ci]
                    )
                ) {

                    ++choices;
                }
            }


            if (
                choices < best_choices
            ) {

                best_choices =
                    choices;

                best_tile =
                    t;


                if (
                    choices == 0
                ) {

                    break;
                }
            }
        }


        return best_tile;
    }


    // ============================================================================
    // Optional hand meld search
    //
    // 기존 테이블을 모두 유효하게 덮은 이후,
    // 아직 사용하지 않은 손패로 추가 meld를 만드는 탐색.
    // ============================================================================

    void optional_hand_melds_dfs(
        int player,
        const std::vector<Candidate> &cands,
        std::array<int, TILE_TYPES> &remaining_hand,
        int start_idx,
        std::vector<Meld> &selected,
        int root_hand_count,
        const std::array<int, TILE_TYPES> &before_hand,
        std::vector<ActionInternal> &out,
        std::unordered_set<std::string> &seen
    ) {

        if (
            should_stop_generation(
                out.size()
            )
        ) {

            return;
        }


        if (
            !consume_search_step()
        ) {

            return;
        }


        int unused =
            sum_counts(
                remaining_hand
            );


        int used_hand =
            root_hand_count
            - unused;


        // --------------------------------------------------------
        // 손패를 최소 1장 사용했다면
        // 하나의 합법적 PLAY/REARRANGE action
        // --------------------------------------------------------

        if (
            used_hand > 0
        ) {

            ActionInternal a;

            a.draw = false;

            a.register_after = true;


            counts_to_hand_masks(
                remaining_hand,

                a.final_hand1,

                a.final_hand2,

                a.final_hand_count
            );


            a.final_table =
                selected;


            a.description =
                "PLAY/REARRANGE: used_hand="
                + std::to_string(
                    used_hand
                )
                + ", melds="
                + std::to_string(
                    selected.size()
                );


            add_action(
                out,
                seen,
                before_hand,
                std::move(a)
            );


            if (
                should_stop_generation(
                    out.size()
                )
            ) {

                return;
            }
        }


        if (
            selected.size()
            >= static_cast<std::size_t>(
                MAX_MELDS
            )
        ) {

            return;
        }


        // --------------------------------------------------------
        // 남은 손패로 추가 meld 생성
        // --------------------------------------------------------

        for (
            int i = start_idx;
            i < static_cast<int>(
                cands.size()
            );
            ++i
        ) {

            if (
                should_stop_generation(
                    out.size()
                )
            ) {

                return;
            }


            if (
                !consume_search_step()
            ) {

                return;
            }


            const auto &cand =
                cands[i];


            if (
                !can_consume(
                    remaining_hand,
                    cand
                )
            ) {

                continue;
            }


            consume(
                remaining_hand,
                cand,
                -1
            );


            selected.push_back(
                cand.meld
            );


            optional_hand_melds_dfs(
                player,
                cands,
                remaining_hand,
                i,
                selected,
                root_hand_count,
                before_hand,
                out,
                seen
            );


            selected.pop_back();


            consume(
                remaining_hand,
                cand,
                +1
            );


            if (
                should_stop_generation(
                    out.size()
                )
            ) {

                return;
            }
        }
    }


    // ============================================================================
    // Complete rearrangement search
    // ============================================================================

    void cover_required_dfs(
        int player,
        const std::vector<Candidate> &cands,
        const std::array<
            std::vector<int>,
            TILE_TYPES
        > &contains,
        std::array<int, TILE_TYPES> &required,
        std::array<int, TILE_TYPES> &remaining,
        std::vector<Meld> &selected,
        int root_hand_count,
        const std::array<int, TILE_TYPES> &before_hand,
        std::vector<ActionInternal> &out,
        std::unordered_set<std::string> &seen,
        int repeated_pivot = -1,
        int min_candidate_for_same_pivot = 0
    ) {

        // --------------------------------------------------------
        // 즉시 중단 검사
        // --------------------------------------------------------

        if (
            should_stop_generation(
                out.size()
            )
        ) {

            return;
        }


        if (
            !consume_search_step()
        ) {

            return;
        }


        if (
            selected.size()
            > static_cast<std::size_t>(
                MAX_MELDS
            )
        ) {

            return;
        }


        // --------------------------------------------------------
        // 기존 table의 모든 tile을 이미 덮었음
        // --------------------------------------------------------

        if (
            all_zero(
                required
            )
        ) {

            optional_hand_melds_dfs(
                player,
                cands,
                remaining,
                0,
                selected,
                root_hand_count,
                before_hand,
                out,
                seen
            );


            return;
        }


        // --------------------------------------------------------
        // 다음 mandatory tile 선택
        // --------------------------------------------------------

        int pivot =
            choose_required_pivot(
                required,
                remaining,
                cands,
                contains
            );


        // choose_required_pivot 도중
        // budget을 모두 사용했을 수 있음.
        if (
            search_budget_exhausted_
        ) {

            return;
        }


        if (
            pivot < 0
        ) {

            return;
        }


        int start_filter =
            (
                pivot
                == repeated_pivot
            )

            ? min_candidate_for_same_pivot

            : 0;


        // --------------------------------------------------------
        // pivot을 포함하는 meld 후보 탐색
        // --------------------------------------------------------

        for (
            int ci : contains[pivot]
        ) {

            if (
                should_stop_generation(
                    out.size()
                )
            ) {

                return;
            }


            if (
                !consume_search_step()
            ) {

                return;
            }


            if (
                ci < start_filter
            ) {

                continue;
            }


            const auto &cand =
                cands[ci];


            if (
                !can_consume(
                    remaining,
                    cand
                )
            ) {

                continue;
            }


            auto required_before =
                required;


            consume(
                remaining,
                cand,
                -1
            );


            satisfy_required(
                required,
                cand,
                -1
            );


            selected.push_back(
                cand.meld
            );


            bool same_pivot_still_required =
                required[pivot] > 0;


            cover_required_dfs(
                player,
                cands,
                contains,
                required,
                remaining,
                selected,
                root_hand_count,
                before_hand,
                out,
                seen,

                same_pivot_still_required
                    ? pivot
                    : -1,

                same_pivot_still_required
                    ? ci
                    : 0
            );


            selected.pop_back();


            required =
                required_before;


            consume(
                remaining,
                cand,
                +1
            );


            // 자식 DFS에서 상한에 도달하면
            // 남은 후보를 더 보지 않고 즉시 unwind
            if (
                should_stop_generation(
                    out.size()
                )
            ) {

                return;
            }
        }
    }


    // ============================================================================
    // Generate actions for current player
    // ============================================================================

    std::vector<ActionInternal>
    generate_actions_for_player(
        int player
    ) {

        // --------------------------------------------------------
        // 매 state마다 모든 generation 상태 초기화
        // --------------------------------------------------------

        generation_truncated_ =
            false;


        reset_search_budget();


        std::vector<ActionInternal>
            out;


        std::unordered_set<std::string>
            seen;


        const auto before_hand =
            hand_counts(
                player
            );


        // --------------------------------------------------------
        // DRAW / PASS
        //
        // 항상 action index 0으로 보장한다.
        // 따라서 탐색 budget 때문에 DRAW가 사라지는 일은 없다.
        // --------------------------------------------------------

        {
            ActionInternal draw =
                make_draw_action(
                    player
                );


            draw.features =
                encode_action(
                    before_hand,
                    draw
                );


            seen.insert(
                action_key(
                    draw
                )
            );


            out.push_back(
                std::move(draw)
            );
        }


        if (
            reached_action_cap(
                out.size()
            )
        ) {

            return out;
        }


        // --------------------------------------------------------
        // 아직 등록 전
        // --------------------------------------------------------

        if (
            !registered_[player]
        ) {

            auto remaining =
                before_hand;


            auto cands =
                enumerate_candidates(
                    remaining
                );


            std::vector<Meld>
                selected;


            registration_dfs(
                player,
                cands,
                remaining,
                0,
                0,
                selected,
                before_hand,
                out,
                seen
            );


            return out;
        }


        // --------------------------------------------------------
        // 등록 이후의 일반 턴
        //
        // 현재 table의 모든 tile은 반드시 최종 table에도 존재해야 한다.
        //
        // remaining =
        //      기존 table tile
        //      + 현재 player의 hand tile
        //
        // 이들을 run/group으로 재분할한다.
        // --------------------------------------------------------

        auto required =
            table_counts();


        auto remaining =
            required;


        for (
            int i = 0;
            i < TILE_TYPES;
            ++i
        ) {

            remaining[i]
                += before_hand[i];
        }


        // --------------------------------------------------------
        // 가능한 run/group 후보 생성
        //
        // enumerate_candidates 자체는
        // 전체 DFS에 비해 후보 공간이 작기 때문에
        // 현재는 search budget 대상으로 두지 않는다.
        // --------------------------------------------------------

        auto cands =
            enumerate_candidates(
                remaining
            );


        // --------------------------------------------------------
        // 각 tile이 어느 candidate meld에 들어가는지 index 생성
        // --------------------------------------------------------

        std::array<
            std::vector<int>,
            TILE_TYPES
        > contains;


        for (
            int ci = 0;
            ci < static_cast<int>(
                cands.size()
            );
            ++ci
        ) {

            for (
                int t = 0;
                t < TILE_TYPES;
                ++t
            ) {

                if (
                    cands[ci].use[t]
                ) {

                    contains[t].push_back(
                        ci
                    );
                }
            }
        }


        std::vector<Meld>
            selected;


        // --------------------------------------------------------
        // table이 비어 있는 경우
        //
        // 손패만으로 새 meld를 생성한다.
        // --------------------------------------------------------

        if (
            all_zero(
                required
            )
        ) {

            auto rem_hand =
                before_hand;


            optional_hand_melds_dfs(
                player,
                cands,
                rem_hand,
                0,
                selected,
                sum_counts(
                    before_hand
                ),
                before_hand,
                out,
                seen
            );
        }


        // --------------------------------------------------------
        // 기존 table이 있는 경우
        //
        // 기존 tile을 모두 포함하는 유효 partition을 탐색한다.
        // --------------------------------------------------------

        else {

            cover_required_dfs(
                player,
                cands,
                contains,
                required,
                remaining,
                selected,
                sum_counts(
                    before_hand
                ),
                before_hand,
                out,
                seen
            );
        }


        return out;
    }


    // ============================================================================
    // Cache
    // ============================================================================

    void ensure_actions_cached() {

        if (
            cached_actions_.empty()
        ) {

            cached_actions_ =
                generate_actions_for_player(
                    current_player_
                );
        }


        if (
            cached_actions_.empty()
        ) {

            throw std::runtime_error(
                "no legal action generated"
            );
        }
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

        for (int c = 0; c < COLORS; ++c) {
            hand1_[player][c] = a.final_hand1[c];
            hand2_[player][c] = a.final_hand2[c];
        }
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

} // namespace rummikub

// ============================================================================
// pybind11 binding
// Build example (Linux/macOS, Python environment with pybind11 installed):
//   c++ -O3 -std=c++17 -shared -fPIC -DRUMMIKUB_PYBIND [pybind11 includes]
//       rummikub_env.cpp -o rummikub_env[python extension suffix]
//
// On Windows, CMake or setuptools/scikit-build is recommended.
// ============================================================================

#ifdef RUMMIKUB_PYBIND
PYBIND11_MODULE(rummikub_env, m) {
    m.doc() = "Common no-joker Rummikub environment for DQN/PPO/ISMCTS/MCCFR";
    m.attr("STATE_DIM") = rummikub::STATE_DIM;
    m.attr("ACTION_DIM") = rummikub::ACTION_DIM;
    m.attr("TOTAL_TILES") = rummikub::TOTAL_TILES;

    py::class_<rummikub::RewardConfig>(m, "RewardConfig")
        .def(py::init<>())
        .def_readwrite("hand_normal", &rummikub::RewardConfig::hand_normal)
        .def_readwrite("hand_2_12", &rummikub::RewardConfig::hand_2_12)
        .def_readwrite("hand_1_13", &rummikub::RewardConfig::hand_1_13)
        .def_readwrite("table_len3", &rummikub::RewardConfig::table_len3)
        .def_readwrite("table_len4", &rummikub::RewardConfig::table_len4)
        .def_readwrite("table_len5plus", &rummikub::RewardConfig::table_len5plus)
        .def_readwrite("win", &rummikub::RewardConfig::win)
        .def_readwrite("loss", &rummikub::RewardConfig::loss)
        .def_readwrite("tie", &rummikub::RewardConfig::tie);

    py::class_<rummikub::RewardComponents>(m, "RewardComponents")
        .def_readonly("hand_normal_used", &rummikub::RewardComponents::hand_normal_used)
        .def_readonly("hand_2_12_used", &rummikub::RewardComponents::hand_2_12_used)
        .def_readonly("hand_1_13_used", &rummikub::RewardComponents::hand_1_13_used)
        .def_readonly("table_len3_delta", &rummikub::RewardComponents::table_len3_delta)
        .def_readonly("table_len4_delta", &rummikub::RewardComponents::table_len4_delta)
        .def_readonly("table_len5plus_delta", &rummikub::RewardComponents::table_len5plus_delta)
        .def_readonly("terminal_outcome", &rummikub::RewardComponents::terminal_outcome);

    py::class_<rummikub::StepResult>(m, "StepResult")
        .def_readonly("next_state", &rummikub::StepResult::next_state)
        .def_readonly("reward", &rummikub::StepResult::reward)
        .def_readonly("hand_reward", &rummikub::StepResult::hand_reward)
        .def_readonly("table_reward", &rummikub::StepResult::table_reward)
        .def_readonly("terminal_reward", &rummikub::StepResult::terminal_reward)
        .def_readonly("components", &rummikub::StepResult::components)
        .def_readonly("done", &rummikub::StepResult::done)
        .def_readonly("winner", &rummikub::StepResult::winner)
        .def_readonly("acting_player", &rummikub::StepResult::acting_player)
        .def_readonly("next_player", &rummikub::StepResult::next_player);

    py::class_<rummikub::RummikubEnv>(m, "RummikubEnv")
        .def(py::init<int, std::size_t>(),
             py::arg("players") = 2,
             py::arg("max_legal_actions") = 0)
        .def("reset", &rummikub::RummikubEnv::reset,
             py::arg("seed"), py::arg("start_player") = 0)
        .def("players", &rummikub::RummikubEnv::players)
        .def("current_player", &rummikub::RummikubEnv::current_player)
        .def("terminal", &rummikub::RummikubEnv::terminal)
        .def("winner", &rummikub::RummikubEnv::winner)
        .def("winning_mask", &rummikub::RummikubEnv::winning_mask)
        .def("state_dim", &rummikub::RummikubEnv::state_dim)
        .def("action_dim", &rummikub::RummikubEnv::action_dim)
        .def("max_melds", &rummikub::RummikubEnv::max_melds)
        .def("pool_remaining", &rummikub::RummikubEnv::pool_remaining)
        .def("generation_truncated", &rummikub::RummikubEnv::generation_truncated)
        .def(
            "search_steps_used",
            &rummikub::RummikubEnv::search_steps_used
        )

        .def(
            "max_search_steps",
            &rummikub::RummikubEnv::max_search_steps
        )

        .def(
            "search_budget_exhausted",
            &rummikub::RummikubEnv::search_budget_exhausted
        )

        .def(
            "set_max_search_steps",
            &rummikub::RummikubEnv::set_max_search_steps,
            py::arg("n")
        )
        .def("set_reward_config", &rummikub::RummikubEnv::set_reward_config,
             py::arg("config"))
        .def("reward_config", &rummikub::RummikubEnv::reward_config)
        .def("set_reward_weights", &rummikub::RummikubEnv::set_reward_weights,
             py::arg("hand_normal"),
             py::arg("hand_2_12"),
             py::arg("hand_1_13"),
             py::arg("table_len3"),
             py::arg("table_len4"),
             py::arg("table_len5plus"),
             py::arg("win"),
             py::arg("loss"),
             py::arg("tie") = 0.0f)
        .def("set_max_legal_actions", &rummikub::RummikubEnv::set_max_legal_actions)
        .def("observation", &rummikub::RummikubEnv::observation,
             py::arg("viewer"))
        .def("legal_action_features", &rummikub::RummikubEnv::legal_action_features)
        .def("legal_action_descriptions", &rummikub::RummikubEnv::legal_action_descriptions)
        .def("legal_action_count", &rummikub::RummikubEnv::legal_action_count)
        .def("step", &rummikub::RummikubEnv::step,
             py::arg("action_index"))
        .def("clone", &rummikub::RummikubEnv::clone)
        .def("determinize", &rummikub::RummikubEnv::determinize,
             py::arg("observer"), py::arg("seed"))
        .def("terminal_utility", &rummikub::RummikubEnv::terminal_utility,
             py::arg("player"))
        .def("hand_value", &rummikub::RummikubEnv::hand_value,
             py::arg("player"))
        .def("hand_size", &rummikub::RummikubEnv::hand_size,
             py::arg("player"))
        .def("registered", &rummikub::RummikubEnv::registered,
             py::arg("player"))
        .def("information_state_key", &rummikub::RummikubEnv::information_state_key,
             py::arg("viewer"))
        .def("validate_state", &rummikub::RummikubEnv::validate_state);
}
#endif

#ifndef RUMMIKUB_PYBIND
#include <iostream>

// Standalone smoke test. A small cap is used only so the test completes fast;
// set max_legal_actions=0 in research runs when exhaustive action generation
// is required.
int main() {
    rummikub::RummikubEnv env(2, 2000);
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
