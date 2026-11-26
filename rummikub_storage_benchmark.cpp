#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cmath>

using Clock = std::chrono::steady_clock;
static constexpr int C = 4;
static constexpr int N = 13;
static constexpr int MAX_MELDS = 34;
static constexpr char COLORS[C] = {'R','Y','B','K'};

struct Tile { int num; int color; };

struct MaskMeld {
    char type = 'N'; // R/G
    uint8_t color_mask = 0;
    uint16_t number_mask = 0; // low 13 bits
    uint8_t count = 0;
};

struct ArrayMeld {
    char type = 'N';
    int cell[C][N]{}; // deliberately ordinary int 2-D array
    int count = 0;
};

struct MaskHand {
    // Same idea as the uploaded environment: two physical copies represented
    // by separate per-color integer masks.
    int copy1[C]{};
    int copy2[C]{};
};

struct ArrayHand {
    int cell[C][N]{}; // value 0,1,2
};

static volatile uint64_t g_sink = 0;
static volatile uint32_t g_case_idx = 0;

static inline int popcount16(uint16_t x) { return __builtin_popcount((unsigned)x); }
static inline int popcount8(uint8_t x) { return __builtin_popcount((unsigned)x); }

MaskMeld make_mask_run(int color, int lo, int hi) {
    MaskMeld m; m.type='R'; m.color_mask=(uint8_t)(1u<<color);
    for (int n=lo;n<=hi;++n) m.number_mask |= (uint16_t)(1u<<(n-1));
    m.count=(uint8_t)(hi-lo+1); return m;
}
MaskMeld make_mask_group(int num, uint8_t cm) {
    MaskMeld m; m.type='G'; m.color_mask=cm; m.number_mask=(uint16_t)(1u<<(num-1));
    m.count=(uint8_t)popcount8(cm); return m;
}
ArrayMeld to_array(const MaskMeld& m) {
    ArrayMeld a; a.type=m.type; a.count=m.count;
    for (int c=0;c<C;++c) if (m.color_mask&(1u<<c))
        for (int n=0;n<N;++n) if (m.number_mask&(1u<<n)) a.cell[c][n]=1;
    return a;
}

// ------------------------------------------------------------------
// 1) validate_run_set equivalent
// ------------------------------------------------------------------
bool validate_run_mask(const MaskMeld& m) {
    if (m.type!='R' || m.count<3 || m.count>N) return false;
    if (popcount8(m.color_mask)!=1) return false;
    if (popcount16(m.number_mask)!=m.count) return false;
    uint16_t x=m.number_mask;
    int lo=__builtin_ctz((unsigned)x);
    int hi=31-__builtin_clz((unsigned)x);
    uint16_t expected=(uint16_t)(((1u<<(hi-lo+1))-1u)<<lo);
    return x==expected;
}
bool validate_run_array(const ArrayMeld& m) {
    if (m.type!='R' || m.count<3 || m.count>N) return false;
    int used_color=-1, cnt=0, first=-1, last=-1;
    for (int c=0;c<C;++c) for (int n=0;n<N;++n) if (m.cell[c][n]) {
        ++cnt; if (used_color<0) used_color=c; else if (used_color!=c) return false;
        if (first<0) first=n; last=n;
    }
    if (cnt!=m.count || last-first+1!=cnt) return false;
    for (int n=first;n<=last;++n) if (!m.cell[used_color][n]) return false;
    return true;
}

// ------------------------------------------------------------------
// 2) validate_group_set equivalent
// ------------------------------------------------------------------
bool validate_group_mask(const MaskMeld& m) {
    return m.type=='G' && (m.count==3 || m.count==4) &&
           popcount16(m.number_mask)==1 && popcount8(m.color_mask)==m.count;
}
bool validate_group_array(const ArrayMeld& m) {
    if (m.type!='G' || (m.count!=3 && m.count!=4)) return false;
    int used_num=-1, cnt=0;
    bool color_seen[C]{};
    for (int c=0;c<C;++c) for (int n=0;n<N;++n) if (m.cell[c][n]) {
        ++cnt; if (used_num<0) used_num=n; else if (used_num!=n) return false;
        if (color_seen[c]) return false; color_seen[c]=true;
    }
    return cnt==m.count;
}

// ------------------------------------------------------------------
// 3/4) can_append_* equivalents
// ------------------------------------------------------------------
bool can_append_run_mask(const MaskMeld& m, Tile t) {
    if (!validate_run_mask(m) || !(m.color_mask&(1u<<t.color))) return false;
    int lo=__builtin_ctz((unsigned)m.number_mask)+1;
    int hi=32-__builtin_clz((unsigned)m.number_mask);
    return t.num==lo-1 || t.num==hi+1;
}
bool can_append_run_array(const ArrayMeld& m, Tile t) {
    if (!validate_run_array(m)) return false;
    int first=-1,last=-1;
    for (int n=0;n<N;++n) if (m.cell[t.color][n]) { if(first<0) first=n; last=n; }
    if (first<0) return false;
    return t.num==first || t.num==last+2; // convert 1-based tile num vs 0-based index
}
bool can_append_group_mask(const MaskMeld& m, Tile t) {
    if (!validate_group_mask(m) || m.count>=4) return false;
    int num=__builtin_ctz((unsigned)m.number_mask)+1;
    return t.num==num && !(m.color_mask&(1u<<t.color));
}
bool can_append_group_array(const ArrayMeld& m, Tile t) {
    if (!validate_group_array(m) || m.count>=4) return false;
    int target=-1;
    for (int c=0;c<C;++c) for (int n=0;n<N;++n) if (m.cell[c][n]) { target=n+1; break; }
    if (target!=t.num) return false;
    return m.cell[t.color][t.num-1]==0;
}

// ------------------------------------------------------------------
// 5/6) find_appendable_* equivalents
// ------------------------------------------------------------------
int find_appendable_runs_mask(const std::vector<MaskMeld>& table, Tile t) {
    int k=0; for (const auto& m:table) if (m.type=='R' && can_append_run_mask(m,t)) ++k; return k;
}
int find_appendable_runs_array(const std::vector<ArrayMeld>& table, Tile t) {
    int k=0; for (const auto& m:table) if (m.type=='R' && can_append_run_array(m,t)) ++k; return k;
}
int find_appendable_groups_mask(const std::vector<MaskMeld>& table, Tile t) {
    int k=0; for (const auto& m:table) if (m.type=='G' && can_append_group_mask(m,t)) ++k; return k;
}
int find_appendable_groups_array(const std::vector<ArrayMeld>& table, Tile t) {
    int k=0; for (const auto& m:table) if (m.type=='G' && can_append_group_array(m,t)) ++k; return k;
}

// ------------------------------------------------------------------
// Hand conversion / candidate generation (find_new_sets equivalent)
// ------------------------------------------------------------------
ArrayHand hand_to_array(const MaskHand& h) {
    ArrayHand a;
    for (int c=0;c<C;++c) for (int n=0;n<N;++n) {
        int b=1<<n; a.cell[c][n]=((h.copy1[c]&b)?1:0)+((h.copy2[c]&b)?1:0);
    }
    return a;
}
int find_new_sets_mask(const MaskHand& h) {
    int count=0;
    // Runs: every contiguous interval length >=3 where at least one copy exists.
    for (int c=0;c<C;++c) {
        uint16_t avail=(uint16_t)(h.copy1[c] | h.copy2[c]);
        for (int s=0;s<N;++s) {
            if (!(avail&(1u<<s))) continue;
            for (int e=s;e<N && (avail&(1u<<e));++e) if (e-s+1>=3) ++count;
        }
    }
    // Groups: enumerate the same 3/4-color masks as the uploaded candidate generator.
    for (int n=0;n<N;++n) {
        uint8_t cm=0; for (int c=0;c<C;++c) if ((h.copy1[c]|h.copy2[c])&(1<<n)) cm|=(1u<<c);
        for (int mask=0; mask<(1<<C); ++mask) {
            int bits=__builtin_popcount((unsigned)mask);
            if (bits!=3 && bits!=4) continue;
            if ((mask & cm)==mask) ++count;
        }
    }
    return count;
}
int find_new_sets_array(const ArrayHand& h) {
    int count=0;
    for (int c=0;c<C;++c) {
        for (int s=0;s<N;++s) {
            if (h.cell[c][s]==0) continue;
            for (int e=s;e<N && h.cell[c][e]>0;++e) if (e-s+1>=3) ++count;
        }
    }
    for (int n=0;n<N;++n) {
        for (int mask=0; mask<(1<<C); ++mask) {
            int bits=__builtin_popcount((unsigned)mask);
            if (bits!=3 && bits!=4) continue;
            bool ok=true;
            for (int c=0;c<C;++c) if ((mask&(1<<c)) && h.cell[c][n]==0) { ok=false; break; }
            if (ok) ++count;
        }
    }
    return count;
}

// ------------------------------------------------------------------
// 8) validate_rearranged_table equivalent: validate every final meld.
// ------------------------------------------------------------------
bool validate_table_mask(const std::vector<MaskMeld>& table) {
    if ((int)table.size()>MAX_MELDS) return false;
    for (const auto& m:table) {
        if (m.type=='R') { if(!validate_run_mask(m)) return false; }
        else if (m.type=='G') { if(!validate_group_mask(m)) return false; }
        else return false;
    }
    return true;
}
bool validate_table_array(const std::vector<ArrayMeld>& table) {
    if ((int)table.size()>MAX_MELDS) return false;
    for (const auto& m:table) {
        if (m.type=='R') { if(!validate_run_array(m)) return false; }
        else if (m.type=='G') { if(!validate_group_array(m)) return false; }
        else return false;
    }
    return true;
}

// ------------------------------------------------------------------
// 9) try_ai_rearrange_patterns equivalent, represented here by candidate
// generation + a bounded exact-cover DFS. This mirrors the uploaded file's
// enumerate_candidates + cover_required_dfs structure conceptually.
// ------------------------------------------------------------------
struct MCandidate { uint64_t bits=0; }; // 52 tile types, one occurrence each

std::vector<MCandidate> candidates_mask(const MaskHand& h) {
    std::vector<MCandidate> out;
    for (int c=0;c<C;++c) {
        uint16_t avail=(uint16_t)(h.copy1[c]|h.copy2[c]);
        for (int s=0;s<N;++s) {
            uint16_t nm=0;
            for (int e=s;e<N && (avail&(1u<<e));++e) {
                nm|=(1u<<e); if(e-s+1>=3){ uint64_t b=0; for(int n=s;n<=e;++n)b|=1ull<<(c*N+n); out.push_back({b}); }
            }
        }
    }
    for(int n=0;n<N;++n){
        uint8_t cm=0; for(int c=0;c<C;++c) if((h.copy1[c]|h.copy2[c])&(1<<n)) cm|=(1u<<c);
        for(int mask=0;mask<(1<<C);++mask){
            int pc=__builtin_popcount((unsigned)mask); if(pc!=3&&pc!=4) continue;
            if((mask & cm)!=mask) continue; uint64_t b=0; for(int c=0;c<C;++c) if(mask&(1<<c)) b|=1ull<<(c*N+n); out.push_back({b});
        }
    }
    return out;
}
struct ACandidate { int cell[C][N]{}; };
std::vector<ACandidate> candidates_array(const ArrayHand& h) {
    std::vector<ACandidate> out;
    for(int c=0;c<C;++c) for(int s=0;s<N;++s){
        if(h.cell[c][s]==0) continue; ACandidate cur{};
        for(int e=s;e<N && h.cell[c][e]>0;++e){ cur.cell[c][e]=1; if(e-s+1>=3) out.push_back(cur); }
    }
    for(int n=0;n<N;++n) for(int mask=0;mask<(1<<C);++mask){
        int pc=__builtin_popcount((unsigned)mask); if(pc!=3&&pc!=4) continue; bool ok=true; ACandidate a{};
        for(int c=0;c<C;++c) if(mask&(1<<c)){ if(h.cell[c][n]==0){ok=false;break;} a.cell[c][n]=1; }
        if(ok) out.push_back(a);
    }
    return out;
}

uint64_t rearrange_mask(const MaskHand& h, uint64_t required, int budget) {
    auto cs=candidates_mask(h); int steps=0; uint64_t solutions=0;
    std::function<void(uint64_t,int)> dfs=[&](uint64_t rem,int start){
        if(steps++>=budget) return; if((rem&required)==0){++solutions; return;}
        int p=__builtin_ctzll(rem&required);
        for(int i=start;i<(int)cs.size();++i){ if(steps>=budget) return; uint64_t b=cs[i].bits; if(!(b&(1ull<<p)))continue; if((b&rem)!=b)continue; dfs(rem^b,i+1); }
    };
    uint64_t avail=0; for(int c=0;c<C;++c) for(int n=0;n<N;++n) if((h.copy1[c]|h.copy2[c])&(1<<n)) avail|=1ull<<(c*N+n);
    if(required==0) required=avail; dfs(avail,0); return solutions + (uint64_t)steps;
}
uint64_t rearrange_array(const ArrayHand& h, const ArrayHand& required, int budget) {
    auto cs=candidates_array(h); int steps=0; uint64_t solutions=0;
    ArrayHand rem=h;
    std::function<void(int)> dfs=[&](int start){
        if(steps++>=budget) return;
        int pc=-1,pn=-1; for(int c=0;c<C&&pc<0;++c)for(int n=0;n<N;++n)if(required.cell[c][n]>0&&rem.cell[c][n]>0){pc=c;pn=n;break;}
        if(pc<0){++solutions;return;}
        for(int i=start;i<(int)cs.size();++i){
            if(steps>=budget)return; if(!cs[i].cell[pc][pn])continue; bool ok=true;
            for(int c=0;c<C&&ok;++c)for(int n=0;n<N;++n)if(cs[i].cell[c][n]&&rem.cell[c][n]==0){ok=false;break;}
            if(!ok)continue; for(int c=0;c<C;++c)for(int n=0;n<N;++n)if(cs[i].cell[c][n])--rem.cell[c][n];
            dfs(i+1); for(int c=0;c<C;++c)for(int n=0;n<N;++n)if(cs[i].cell[c][n])++rem.cell[c][n];
        }
    };
    dfs(0); return solutions+(uint64_t)steps;
}

std::vector<MaskMeld> make_table() {
    std::vector<MaskMeld> t;
    t.push_back(make_mask_run(0,1,5));
    t.push_back(make_mask_run(1,6,10));
    t.push_back(make_mask_run(2,2,4));
    t.push_back(make_mask_group(7,0b0111));
    t.push_back(make_mask_group(11,0b1011));
    return t;
}
MaskHand make_hand(uint32_t seed, int tiles=14) {
    std::mt19937 rng(seed); std::vector<int> deck; deck.reserve(104);
    for(int copy=0;copy<2;++copy)for(int c=0;c<C;++c)for(int n=0;n<N;++n)deck.push_back(c*N+n);
    std::shuffle(deck.begin(),deck.end(),rng); MaskHand h;
    for(int i=0;i<tiles;++i){int idx=deck[i],c=idx/N,n=idx%N,b=1<<n; if(!(h.copy1[c]&b))h.copy1[c]|=b; else h.copy2[c]|=b;}
    return h;
}

struct Stat { double mean_ns, median_ns, p95_ns; };
template<class F>
Stat bench(F&& f, int warmup, int repeats, int inner) {
    for(int i=0;i<warmup;++i) g_sink += (uint64_t)f();
    std::vector<double> v; v.reserve(repeats);
    for(int r=0;r<repeats;++r){ auto st=Clock::now(); uint64_t x=0; for(int i=0;i<inner;++i)x+=(uint64_t)f(); auto en=Clock::now(); g_sink+=x;
        v.push_back(std::chrono::duration<double,std::nano>(en-st).count()/inner); }
    std::sort(v.begin(),v.end()); double mean=std::accumulate(v.begin(),v.end(),0.0)/v.size();
    return {mean,v[v.size()/2],v[(size_t)std::floor(0.95*(v.size()-1))]};
}

void row(std::ofstream& out, const std::string& fn,const std::string& backend,const Stat&s,int payload,const std::string& tc,const std::string& sc,int inner){
    out<<fn<<','<<backend<<','<<std::fixed<<std::setprecision(3)<<s.mean_ns<<','<<s.median_ns<<','<<s.p95_ns<<','<<payload<<','<<'"'<<tc<<'"'<<','<<'"'<<sc<<'"'<<','<<inner<<"\n";
}

int main(int argc,char**argv){
    std::string path = argc>1?argv[1]:"benchmark_results.csv";
    int repeats=200, warmup=100; std::ofstream out(path);
    out<<"function,backend,mean_ns,median_ns,p95_ns,representation_bytes,theoretical_time,theoretical_extra_space,inner_iterations\n";
    auto mt=make_table(); std::vector<ArrayMeld> at; for(auto&m:mt)at.push_back(to_array(m));
    // Paired datasets prevent the compiler from folding a benchmark into a constant.
    std::vector<MaskMeld> run_cases; std::vector<ArrayMeld> run_cases_a;
    std::vector<MaskMeld> group_cases; std::vector<ArrayMeld> group_cases_a;
    for(int rep=0; rep<32; ++rep){
        for(int c=0;c<C;++c){ int lo=1+(rep%8); int hi=std::min(13,lo+2+(rep%4)); auto m=make_mask_run(c,lo,hi); run_cases.push_back(m); run_cases_a.push_back(to_array(m)); }
        for(int n=1;n<=13;++n){ uint8_t cm=(rep%2)?0b0111:0b1011; auto m=make_mask_group(n,cm); group_cases.push_back(m); group_cases_a.push_back(to_array(m)); }
    }
    std::vector<MaskHand> hands; std::vector<ArrayHand> hands_a;
    for(uint32_t sd=40; sd<104; ++sd){ auto h=make_hand(sd,22); hands.push_back(h); hands_a.push_back(hand_to_array(h)); }
    auto mh=hands[2]; auto ah=hands_a[2]; Tile tr{6,0}; Tile tg{7,3};
    int fast=2000, med=300, slow=1;

    row(out,"validate_run_set","bitmask",bench([&]{auto i=(g_case_idx++)%run_cases.size(); return validate_run_mask(run_cases[i]);},warmup,repeats,fast),sizeof(MaskMeld),"O(1) for fixed-width mask; O(1) word ops","O(1)",fast);
    row(out,"validate_run_set","2d_array",bench([&]{auto i=(g_case_idx++)%run_cases_a.size(); return validate_run_array(run_cases_a[i]);},warmup,repeats,fast),sizeof(ArrayMeld),"O(C*N)","O(1)",fast);
    row(out,"validate_group_set","bitmask",bench([&]{auto i=(g_case_idx++)%group_cases.size(); return validate_group_mask(group_cases[i]);},warmup,repeats,fast),sizeof(MaskMeld),"O(1)","O(1)",fast);
    row(out,"validate_group_set","2d_array",bench([&]{auto i=(g_case_idx++)%group_cases_a.size(); return validate_group_array(group_cases_a[i]);},warmup,repeats,fast),sizeof(ArrayMeld),"O(C*N)","O(1)",fast);
    row(out,"can_append_run_tile","bitmask",bench([&]{auto i=(g_case_idx++)%run_cases.size(); Tile t{1+(int)(g_case_idx%13),(int)(g_case_idx%C)}; return can_append_run_mask(run_cases[i],t);},warmup,repeats,fast),sizeof(MaskMeld),"O(1)","O(1)",fast);
    row(out,"can_append_run_tile","2d_array",bench([&]{auto i=(g_case_idx++)%run_cases_a.size(); Tile t{1+(int)(g_case_idx%13),(int)(g_case_idx%C)}; return can_append_run_array(run_cases_a[i],t);},warmup,repeats,fast),sizeof(ArrayMeld),"O(C*N)","O(1)",fast);
    row(out,"can_append_group_tile","bitmask",bench([&]{auto i=(g_case_idx++)%group_cases.size(); Tile t{1+(int)(g_case_idx%13),(int)(g_case_idx%C)}; return can_append_group_mask(group_cases[i],t);},warmup,repeats,fast),sizeof(MaskMeld),"O(1)","O(1)",fast);
    row(out,"can_append_group_tile","2d_array",bench([&]{auto i=(g_case_idx++)%group_cases_a.size(); Tile t{1+(int)(g_case_idx%13),(int)(g_case_idx%C)}; return can_append_group_array(group_cases_a[i],t);},warmup,repeats,fast),sizeof(ArrayMeld),"O(C*N)","O(1)",fast);
    row(out,"find_appendable_runs","bitmask",bench([&]{return find_appendable_runs_mask(mt,tr);},warmup,repeats,med),sizeof(MaskMeld)*mt.size(),"O(M)","O(1)",med);
    row(out,"find_appendable_runs","2d_array",bench([&]{return find_appendable_runs_array(at,tr);},warmup,repeats,med),sizeof(ArrayMeld)*at.size(),"O(M*C*N)","O(1)",med);
    row(out,"find_appendable_groups","bitmask",bench([&]{return find_appendable_groups_mask(mt,tg);},warmup,repeats,med),sizeof(MaskMeld)*mt.size(),"O(M)","O(1)",med);
    row(out,"find_appendable_groups","2d_array",bench([&]{return find_appendable_groups_array(at,tg);},warmup,repeats,med),sizeof(ArrayMeld)*at.size(),"O(M*C*N)","O(1)",med);
    row(out,"find_new_sets","bitmask",bench([&]{auto i=(g_case_idx++)%hands.size(); return find_new_sets_mask(hands[i]);},warmup,repeats,med),sizeof(MaskHand),"O(C*N^2 + N*(C + 2^C))","O(1) excluding output",med);
    row(out,"find_new_sets","2d_array",bench([&]{auto i=(g_case_idx++)%hands_a.size(); return find_new_sets_array(hands_a[i]);},warmup,repeats,med),sizeof(ArrayHand),"O(C*N^2 + N*2^C*C)","O(1) excluding output",med);
    row(out,"validate_rearranged_table","bitmask",bench([&]{return validate_table_mask(mt);},warmup,repeats,med),sizeof(MaskMeld)*mt.size(),"O(M)","O(1)",med);
    row(out,"validate_rearranged_table","2d_array",bench([&]{return validate_table_array(at);},warmup,repeats,med),sizeof(ArrayMeld)*at.size(),"O(M*C*N)","O(1)",med);

    // Required set for bounded rearrangement: use all available tile types.
    uint64_t reqm=0; for(int c=0;c<C;++c)for(int n=0;n<N;++n)if((mh.copy1[c]|mh.copy2[c])&(1<<n))reqm|=1ull<<(c*N+n);
    ArrayHand reqa=ah;
    int budget=2000;
    row(out,"try_ai_rearrange_patterns","bitmask",bench([&]{return rearrange_mask(mh,reqm,budget);},10,60,slow),sizeof(MaskHand),"Exponential in candidate partitions; bounded here by search_budget","O(depth + K) candidates",slow);
    row(out,"try_ai_rearrange_patterns","2d_array",bench([&]{return rearrange_array(ah,reqa,budget);},10,60,slow),sizeof(ArrayHand),"Exponential in candidate partitions; bounded here by search_budget","O(depth*C*N + K*C*N)",slow);
    out.close();

    std::ofstream mem(path.substr(0,path.find_last_of('.'))+"_space.csv");
    mem<<"object,bitmask_bytes,2d_array_bytes,array_over_bitmask_ratio\n";
    mem<<"hand,"<<sizeof(MaskHand)<<','<<sizeof(ArrayHand)<<','<<(double)sizeof(ArrayHand)/sizeof(MaskHand)<<"\n";
    mem<<"meld,"<<sizeof(MaskMeld)<<','<<sizeof(ArrayMeld)<<','<<(double)sizeof(ArrayMeld)/sizeof(MaskMeld)<<"\n";
    mem<<"table_5_melds,"<<sizeof(MaskMeld)*5<<','<<sizeof(ArrayMeld)*5<<','<<(double)sizeof(ArrayMeld)/sizeof(MaskMeld)<<"\n";
    mem.close();

    std::cerr<<"wrote "<<path<<" and space CSV; sink="<<g_sink<<"\n";
    return 0;
}
