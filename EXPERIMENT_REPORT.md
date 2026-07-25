# Differentiable ZDOCK — 学習・評価レポート

微分可能 ZDOCK（`docking_score_elec` の 156 学習可能パラメータ）を、リポジトリ
自身の FFT 探索が生成したデコイに対する **条件付き energy-based ランキング学習**
として訓練し、単一複合体（1KXQ）および多複合体ベンチマーク（DB5.5）で評価した
記録である。GPT の研究提案を評価したうえで実装計画を立て、段階的に実行した。

- 実装コア（変更なしで再利用）: `zdock.score`（スコア）、`zdock.search`（FFT 探索）、
  `zdock.dockq`（DockQ/RMSD）、`zdock.train`（Adam ループ・損失群）。
- 本レポートで追加/変更した部分: `zdock.dataset`、`zdock.evaluate`、
  `zdock.score` の特徴量分解、`zdock.train` の損失、`scripts/`。

---

## 1. GPT 研究提案の評価（賛同 / 非賛同）

### 賛同した点（採用）
- **DiffDock 移植ではなく「FFT で全 pose を評価する条件付き EBM」として学習する**基本
  方針。本リポジトリの構造（`docking_search` でデコイ生成 → `docking_score_elec`
  で微分可能に再スコア）とそのまま一致する。
- **multi-positive basin loss**（near-native pose 集合を positive とする InfoNCE）。
  FFT の回転・並進離散化に強く、hard argmax を微分せずに済む。→ `loss_basin` を実装。
- **hard-negative margin**（高スコアの非 native を直接罰する）。→ 既存
  `loss_margin_hard_negatives` を採用。
- **DockQ を連続 ranking target として前計算**。→ `zdock.dockq` を使用。
- **residual/prior 正則化 `‖θ − θ_ZDOCK‖²`**。→ `loss_param_prior` を実装。
- **leakage 回避・DB5.5 外部評価**の方針。→ DB5.5 で複合体単位 split を実装
  （ただし §7 の限界に注意）。

### 非賛同・後回しにした点（理由つき）
- **「まず PINDER 全体で事前学習」を出発点にすること**。長期的には正しいが、本
  リポジトリにはデータ配管が皆無で、apo/predicted alignment や oracle DockQ 推定は
  大規模 ETL になる。GPT 自身の「§10 最小構成」を実際の出発点として採用した。
- **quality head（別 sigmoid 回帰 L_Q）**。現モデルは 156 パラメータの固定線形/双線形
  形で、head を載せる共有特徴（NN エンコーダ）が無い。線形モデルは過学習すら
  できず calibration はボトルネックでない。→ エンコーダ学習まで延期。
- **L_funnel（SO(3) denoising score-matching）+ Gaussian splatting**。連続
  refinement を製品にする場合のみ必要。FFT ランキングには範囲外。→ 延期。
- **swap 対称 loss L_swap**。ZDOCK のスコアは構造的に非対称（受容体のみ 3.4Å 表面層
  と Coulomb ポテンシャルを持つ）。受容体固定運用では非対称エンコードと衝突。→ 不採用。
- **learnable grid encoder**。最終目標だが 156 パラメータとは別モデル。→ 延期。

---

## 2. 実装計画（段階）

- **Stage 0**: 環境構築・既存テストで実装の健全性を確認。
- **Stage 1**: 欠けていたデータ配管を **リポジトリ自身の FFT 探索で自己完結生成**
  （外部 ZDOCK バイナリ不要）。1KXQ で before/after 評価。
- **Stage 2**: 賛同した損失（basin / margin / prior）を実装し、損失比較。
- **Stage 3**: DB5.5 をダウンロードし、多複合体で **複合体単位 train/test 汎化評価**。

---

## 3. 実装内容（ファイル別）

### 新規
- **`src/zdock/dataset.py`** — 自己完結デコイ生成＋ラベル付け。
  - `prepare_protein_from_pdbms` / `prepare_protein_from_pdb` / `prepare_protein`:
    受容体・リガンドの原子から特徴量（radius, SASA, atomtype_id, charge_id）を導出。
  - `parse_pdb_plain`: 標準 PDB（DB5.5）クリーニングパーサー。第 1 モデルのみ、
    水素・HETATM・水・非標準残基・altLoc(≠blank/'A')・atomtype/charge LUT 外の原子を
    ドロップ（例外にせずカウント）。
  - `generate_decoys`: グローバルランダム回転 + native 近傍 cone を混ぜて FFT 探索を
    実行し、候補 pose を受容体 de-center 座標系で返す。
  - `label_decoys`: 各 pose の RMSD・DockQ を native 配置に対して計算（DockQ は pose
    方向にチャンク化して OOM 回避）。
  - **フレーム整合**（本パイプラインの肝）:
    - `rec_dec = rec_xyz_raw − rec_com`
    - `lig_ref = orient(lig_xyz_raw, iface_mass)`（FFT 探索が回す参照リガンド）
    - `native_lig = lig_xyz_raw − rec_com`（rec_dec 座標系での native 配置）
    - デコイ = `rotate(lig_ref, q) + t`（同座標系）
    - `q*`（native 姿勢）は `kabsch_quaternion(lig_ref, native_lig)`、`t*` は重心差。
      cone を `q*` 周りに置くことで positive を必ず候補集合に含める。

- **`src/zdock/evaluate.py`** — ランキング品質評価。
  - `score_poses`, `evaluate_ranking`, `format_report`。
  - CAPRI 風 success@K を **RMSD 閾値と DockQ 閾値の両定義**で報告（本リポジトリの
    DockQ は原子レベル近似のため、絶対値の裏取りとして RMSD を併記）。

- **`scripts/build_decoy_dataset.py`** — `*.pdb.ms`（ZDOCK 形式）と DB5.5 平 PDB の
  両対応で統合 h5 を書き出す。複合体ごとに prep/生成/ラベルを try/except で保護し、
  OOM はスキップ。`zdock.data` が期待するスキーマ + `dockq` データセット。

- **`scripts/run_experiment.py`** — 単一複合体で pose を train/val 分割し、
  baseline → 学習 → 保持 pose で再評価。

- **`scripts/run_db55.py`** — DB5.5 複合体単位 train/test 汎化実験（§5.3、§6.3）。

### 変更
- **`src/zdock/score.py`** — `docking_score_elec` / `_score_ligand_chunk` に
  `return_components` を追加。スコアを pose 毎特徴量に厳密分解して返す（§4.3）。
- **`src/zdock/train.py`** — `loss_basin`（multi-positive InfoNCE）、`loss_param_prior`
  （L2 anchor）、損失モード `basin` / `combined`（basin + λ·margin + λ·prior）を追加。
- **`src/zdock/data.py`** — 統合 h5 から `dockq` フィールドを読み込むよう拡張。

既存テスト（95 件）は変更後も全通過を維持。

---

## 4. 計算内容（詳細）

### 4.1 スコアモデル
pose `f` のスコア: `score[f] = α · S_SC[f] + S_IFACE[f] + β · S_ELEC[f]`。
- `S_SC`: 形状相補性（受容体/リガンドの複素 SC 格子の相関）。
- `S_IFACE`: 12 原子種 × 12 の界面統計ポテンシャル `Σ_ij iface_ij · T_ij`。
- `S_ELEC`: Coulomb 静電（受容体ポテンシャル格子 × リガンド電荷格子）。
- 学習可能: α(1) + iface(144) + charge(11) = 156。β は charge とスケール冗長なので固定(3.0)。

### 4.2 デコイ生成とラベル
- 各複合体で受容体を de-center、リガンドを orient し、
  `random_quaternions`（グローバル）+ `rotation_cone(q*)`（native 近傍）で回転集合を作り、
  `docking_search`（FFT）で各回転の全並進を評価し top-`ntop` を取得。さらに `t*` の
  near-native pose を明示追加。
- 各 pose を rec_dec 座標系に復元し、`ligand_rmsd_to_native` と `dockq_batch` で
  RMSD・DockQ を付与。`hit_mask = rmsd ≤ 5.0Å`。
- 生成設定: 1KXQ = (random 3000, cone 400, ntop 2000)。DB5.5 = (random 1500,
  cone 300, ntop 1500, cone_deg 25°, spacing 3.0Å)。

### 4.3 特徴量分解（学習高速化の肝）
スコアは (α, iface) について線形、charge について双線形である。pose 毎に
- `S_SC[f]`（スカラー、パラメータ非依存）
- `T[f, 12, 12]`（`S_IFACE = Σ_ij iface_ij · T`）
- `S_ELEC[f]`（charge をデフォルト固定した固定スカラー特徴）

を **1 回の格子演算で抽出**（`docking_score_elec(..., return_components=True)`）して
キャッシュすれば、学習時は微小なテンソル縮約
`score = α·S_SC + <iface, T> + β·S_ELEC` だけで済む。これにより多複合体でも毎
エポックの格子再計算が不要となり、DB5.5 全体の学習が数十秒で収束する。
分解が本体スコアと一致することを検証済み（float32 で相対誤差 ~1e-7）。

### 4.4 学習設定
- 損失: `combined = basin(T=0.5) + 0.5·margin + λ_prior·prior`。
- 1KXQ: `train()`（全 pose、Adam lr 0.02、300 epoch）。
- DB5.5: `run_db55.py` の専用ループ（α + 144 IFACE を学習、charge はデフォルト固定）、
  Adam lr 0.05、複合体ミニバッチ 16、2000 step。

---

## 5. 計算結果

### 5.1 1KXQ before/after（保持 val pose 960 個）
| 設定 | 最良 pose ランク | top-1 RMSD | top-1 DockQ | top-5 最良 DockQ |
|---|---|---|---|---|
| baseline（デフォルト ZDOCK） | 905 / 960 | 59.3Å | 0.007 | 0.009 |
| trained（combined, 300ep） | **14** | **1.48Å** | **0.622** | **0.940** |

学習後 α=0.0287、‖Δiface‖=6.483、‖Δcharge‖=2.667。保持 pose で success@1 を達成。

### 5.2 損失関数比較（1KXQ, 保持 val, 最良 pose ランク=小さいほど良い）
| 損失 | 最良ランク | top-1 RMSD | top-1 DockQ |
|---|---|---|---|
| baseline | 905 | 59.3Å | 0.007 |
| `basin` 単独 | 88 | 2.66Å | 0.431 |
| `dockq_margin` 単独 | 27 | 4.75Å | 0.360 |
| **`combined`** | **14** | **1.48Å** | **0.622** |

→ GPT 提案の「basin + hard-negative margin + prior を組み合わせる」方針が実データで最良。

### 5.3 DB5.5 複合体間汎化（本題）
- 271 複合体（bound）から 266 複合体のデコイを 7×RTX A6000 並列で生成（残り 5 は
  巨大複合体で OOM スキップ）。複合体単位で **train 173 / test 93** に分割。
- 学習に一切使っていない **test 93 複合体**での top-K 成功率:

| K | baseline RMSD | trained RMSD | baseline DockQ | trained DockQ |
|---|---|---|---|---|
| top-1 | 0.0% | **71.0%** | 0.0% | **89.2%** |
| top-5 | 0.0% | **90.3%** | 0.0% | **94.6%** |
| top-10 | 0.0% | **90.3%** | 0.0% | **96.8%** |
| top-50 | 0.0% | 97.8% | 0.0% | 100% |
| top-100 | 0.0% | **100%** | 0.0% | **100%** |

- 学習後 α=0.0228、‖Δiface‖=11.527、mean best-DockQ@top1 = 0.452。
- ランダム分割上では train-set top-1 RMSD 70.5% と test 71.0% がほぼ一致し、一見
  「汎化している」ように見えた。**しかしこれは interface leakage による楽観バイアス
  だった**（5.4 で検証）。

### 5.4 PINDER interface-deleaked split（leakage 検証）
5.3 のランダム複合体分割は interface をクラスタリングしていないため、類似 interface が
train/test に跨り得る。そこで **PINDER**（FoldSeek/MMseqs による interface 類似度
クラスタリング＋iAlign による deleaking で構築された gold-standard split）を導入し、
**設計上 train と test の間に interface leakage が無い**条件で測り直した。

- `pinder` パッケージで index を取得。`fastpdb`↔`biotite 1.7` 非互換のため
  `PinderSystem(id, pdb_engine="biotite")` で holo 単量体 PDB を取得し、DB5.5 と同一の
  自作パーサ／featurization に流した（`scripts/build_decoy_dataset.py --format pinder`）。
- **train**: `split==train` から別クラスタ 300 件をサンプル（test とクラスタ重複 0）。
  **test**: 標準ベンチマーク **PINDER-S**（250 クラスタ代表）。7 GPU 並列生成後、
  featurization 時の OOM を除いて **train 226 / test 241 複合体**。
- 同一パイプライン・同一損失（`basin + 0.5·margin + 0.1·prior`, lr=0.01, 3000 epoch）:

| K | baseline TEST DockQ | trained **TEST(deleaked)** DockQ | trained **TRAIN** DockQ |
|---|---|---|---|
| top-1 | 0.4% | **1.2%** | **82.7%** |
| top-5 | 0.4% | 2.5% | 93.8% |
| top-10 | 1.2% | 2.9% | 96.0% |
| top-100 | 1.7% | **4.1%** | **99.6%** |

- baseline は **train でも test でも** ほぼ 0%（top-1 DockQ 0.4%）。学習で train は
  0.4%→**82.7%** に上がるが、deleaked test は 0.4%→**1.2%** しか上がらない。
- **結論**: 156 パラメータは訓練複合体の interface 化学を「暗記」するが、**未知
  interface には全く汎化しない**。5.3 の 71% は**ほぼ全てが leakage（過学習）由来の
  楽観バイアス**であり、deleaked 条件での真の汎化性能は baseline+数 % に留まる。
  → これが「deleak した split を作ると何が分かるか」の答え。leakage 除去は必須で、
  現行モデル／損失は未知 interface への転移をまだ達成していない。

### 5.5 Iterative hard-negative mining（汎化改善の検証）
現在のパラメータで TRAIN 複合体の FFT 探索を再実行し、高スコアだが
`DockQ < 0.23` の pose を訓練 pool に追加するループを実装した
（`scripts/run_pinder_hardneg.py`）。TEST の候補集合は固定し、TEST は checkpoint
選択に使わない。

最初の実装ではラウンドごとに ZDOCK 初期値から再学習していたうえ、raw score の標準偏差
が約 500〜2,000 なのに `temperature=0.5`, `margin=1.0`, 共通 lr=0.01 を使っていた。
その結果、TEST top-1 DockQ が round ごとに `21.2% → 5.4% → 0.0% → 86.7% → 0.0%`
と激しく振動した。round 3 の 86.7% は次ラウンドで消失し、再現性のある改善ではない。

そこで以下を修正して再試行した。

- 前ラウンドのパラメータと Adam state を継承（random restart を廃止）。
- 損失計算時のみ複合体内で score を標準化。正の定数による変換なので pose ranking は不変。
- `alpha_lr=1e-5`, `iface_lr=5e-4`、gradient clipping、`0 <= alpha <= 0.1`。
- near-native positive は round 0 の集合に固定し、追加するのは negative のみ。
- TRAIN 275件を fit 220 / validation 55 に固定分割。validation候補集合はround 0で固定し、
  miningにも学習にも使わず、validation lossでearly stopping。
- seed 0/1/2をGPU 3台で独立実行（4 mining rounds）。

安定化後の held-out PINDER-S TEST 結果:

| seed | round 0 top-1 / top-10 / top-100 | round 4 top-1 / top-10 / top-100 |
|---|---|---|
| 0 | 0.0% / 0.0% / 0.4% | 0.0% / 0.0% / 0.4% |
| 1 | 0.0% / 0.0% / 0.4% | 0.0% / 0.0% / 0.4% |
| 2 | 0.0% / 0.0% / 0.8% | 0.0% / 0.0% / 0.8% |

mining により mean pool size は 1,900 から約 3,586 に増え、固定パラメータでの TRAIN
top-1 DockQ は約 26〜29% から約 4〜5% に低下したため、モデルが誤って高く評価する
**実際に難しい negative の採掘には成功**した。しかし各 mining round の更新は固定
validation lossを改善せず、early stoppingでround 0 checkpointへ戻された。したがって
この条件では **hard-negative mining単独によるdeleaked top-K向上は0ポイント**。
不足していたのはnegativeの難しさではなく、220複合体から未知interfaceへ移せる表現力・
データ量・正則化である。

#### 5.5.1 なぜ hard-negative mining が効かなかったか（仮説）

**データ不足は有力な主因候補**。pose は1複合体あたり数千あっても同じinterfaceから
生成された強相関サンプルなので、統計的な独立単位は pose 数ではなく interface cluster
数である。fit は220クラスタ、学習対象はalpha 1個＋IFACE 144個の計145パラメータで、
約1.5 interface/parameterにすぎない。しかも各interfaceが観測する12×12 atom-type pair
は一部だけなので、個々のIFACE成分の実効観測数はさらに少ない。round 0でTRAIN top-1
DockQが約26〜29%まで上がる一方、deleaked TESTが0%という差も高分散／過適合と整合する。

ただし、現実験だけから「単にデータが少ないだけ」とは断定できない。競合仮説は以下。

1. **validation分布の不一致**: fitにはcurrent modelで採掘したadversarial negativeを追加
   したが、validationはdefault model由来の候補集合に固定した。mined分布への改善が
   default分布のvalidation lossを悪化させ、全更新が棄却された可能性がある。
2. **偽negative／対称性**: 原子対応を固定した近似DockQでは、homomerの対称等価poseや
   別の妥当な結合modeを`DockQ < 0.23`として採掘し得る。高スコアの正しいposeをnegative
   として押し下げれば、矛盾した教師になる。
3. **損失の外れ値感度**: margin lossは全positive中の`min(score_pos)`を基準にするため、
   basin端の低品質positive 1件でも制約が決まる。hard negative追加時に「最悪positiveを
   全negativeより上へ」という過度に厳しい／ノイズ感度の高い問題になる。
4. **replay比率**: positive集合は固定したがnegative poolは約1,500→約3,200へ増えた。
   marginを全negative平均する現在の実装では、難例の量と勾配寄与がroundごとに変わる。
5. **表現力不足**: globalな12×12 IFACE行列とalphaだけでは、残基環境、曲率、水素結合
   方向性、柔軟性、複合体タイプ依存性を表せない。データを増やしても同じ係数で相反する
   interfaceを順位付けできないなら、これはvarianceではなくbiasの問題。
6. **診断不足**: early stopping後のcheckpointだけを保存したため、mined TRAIN lossを
   下げた候補がvalidationをどれだけ悪化させたかのtrajectoryが残っていない。

**切り分け実験（優先順）**:

1. **scaling law**: interface cluster数を220/500/1,000/2,000以上へ増やし、同一の固定
   PINDER-S TEST、3 seed、同一pose数でround 0とmining 1 roundを比較する。Nとともに
   TEST top-Kおよびmining gainが上がればデータ不足説を支持し、早期飽和すれば表現力説。
2. **IFACE coverage**: 144成分ごとの非zero接触complex数とtrain/val/test分布を測定する。
   rare成分が多ければ低rank残差（例: `W=W0+UV^T`, rank 2/4）やtype tyingで実効自由度を
   下げ、小規模データで比較する。
3. **固定adversarial validation**: validation複合体をround 0 modelで一度だけmineし、
   default候補＋adversarial候補を固定した複合validationでcheckpoint選択する。
4. **label audit**: mined上位negativeをhomomer/heteromer別に抽出し、対称鎖入替えを許した
   DockQ/contact recoveryで再評価する。偽negative率を測定してから再学習する。
5. **robust loss / balanced replay**: `min_pos`をpositive scoreの10% quantileへ変更し、
   各更新でfixed positive・random negative・hard negativeを一定比率でsampleする。

このため次の最重要実験は候補1の**クラスタ数scaling law**であり、これを行わずに
「データ不足」または「モデル不足」のどちらかへ結論を固定すべきではない。

#### 5.5.2 scaling-law実験の計算時間見積り

fit 220クラスタ、round 0＋4 mining roundsの安定化実験は約89分/GPUだった。round 0は
全275件（fit 220＋validation 55）、以降はfit 220件を探索する。実測からround 0＋
mining 1回は約45分/seedであり、探索時間がcluster数にほぼ線形と仮定すると:

| fit clusters | 1 seed | 3 seed（3 GPU並列時のwall time） |
|---:|---:|---:|
| 220 | 約45分（実施済） | 約45分（実施済） |
| 500 | 約1.5〜2時間 | 約1.5〜2時間 |
| 1,000 | 約3〜4時間 | 約3〜4時間 |
| 2,000 | 約6〜8時間 | 約6〜8時間 |

220/500/1,000/2,000の12 jobs（4規模×3 seed）を7 GPUへ割り当てると、理想的な総wall
timeは最大jobに支配され約7〜9時間。実装変更、PINDER取得、失敗再実行を含めて
**約8〜11時間**を見込む。

ただし現在のscriptは全`PreparedProtein`をGPUに常駐させるため、500件以上ではVRAM不足
になる。実行前に (1) proteinをCPU保持、(2) 1複合体ずつGPUへ転送、(3) pose feature
poolのみGPU/CPU cacheへ保存、というstreaming化が必要（実装・smoke test 約1〜2時間）。
まず500件×3 seedをpilot（約2〜3時間、実装込み）し、増加傾向が見えた場合のみ
1,000/2,000へ進む段階的実行が費用対効果に優れる。

### 5.6 interface cluster数のscaling law（§5.5.1の切り分け実験1）

#### 5.6.1 問いと仮説

§5.5で fit 220 interface cluster のhard-negative miningは deleaked TEST を
1点も改善しなかった。§5.5.1はその主因候補として (A) データ不足（145
parameterに対し220クラスタ）と (B) 表現力・損失・ラベルの問題を挙げたが、
現実験だけでは切り分けられなかった。

**検証する予測**:
- (A)が正しければ、独立interface cluster数 N を増やすと round 0 の held-out
  性能と hard-negative mining gain が N とともに（log(N)的に）上昇する。
- (B)が正しければ、N を約10倍にしても held-out 性能は早期に飽和する。

他はすべて固定して N のみを 220 / 500 / 1,000 / 2,000 と変える。

#### 5.6.2 streaming実装（実行の前提）

§5.5.2で予告したとおり、`run_pinder_hardneg.py` は全 `PreparedProtein` と全
feature pool を GPU に常駐させるため数百複合体で頭打ちになる。新しい
`scripts/run_pinder_scaling.py` は次のstreaming構成にした。

1. **prepared complexのCPU disk cache**（`src/zdock/prep_cache.py`）。PDB
   parse → atom type/radius/SASA → orient → Kabsch native pose は seed にも
   round にも依存しないので一度だけ計算し、`sha1(PINDER id)[:20]` を
   ファイル名（2階層fan-out）にして保存する。PINDER の id は長く、そのまま
   ファイル名にすると壊れやすいのでhash化した。id は payload 内に保存し、
   load時に照合するのでhash衝突は例外になる（silent corruptionにしない）。
   全 seed・全 N がこの cache を共有する。
2. **1複合体ずつGPUへ**。`PreparedProtein.to()` を新設し（`src/zdock/dataset.py`、
   tensor field をまとめて移動、整数idフィールドはdtype変換しない）、
   cache→GPU→FFT探索→label→feature抽出→**CPUへ退避**→即解放、を複合体単位で回す。
   pose座標（`(F, N_lig, 3)`）は featurize 直後に破棄する。
3. **feature poolはhost常駐**。1複合体あたり `S_SC(F)`, `T(F,12,12)`,
   `S_ELEC(F)`, `RMSD(F)`, `DockQ(F)` と origin ラベルで約1.1 MB（round 0,
   F=1,900）／約2.3 MB（cap 4,000）。2,500複合体でも約6 GBでhost側に収まる。
4. **学習時のみmini-batchをGPUへ**。batch 16複合体で約37 MBのH2D転送。
   validation・TESTも複合体単位で転送して評価する。
5. **OOM安全化**。`generate_decoys` の `rot_chunk_size`、`docking_score_elec`
   の `frame_chunk_size`、DockQ の `pose_chunk` を段階的に半減して最大3回
   retryし、それでも失敗したら **ID・原子数・段階** を `skipped.jsonl` に
   記録して skip する（run は落とさない）。DockQ の dense
   `(chunk, N_rec, N_lig)` テンソルは複合体サイズから逆算して chunk を決める。
6. **成果物をround・seedごとに分離保存**: `split.json`、`coverage.json`、
   `baseline_test.json`、`round{r}_metrics.json`、`round{r}_trajectory.csv`、
   `round{r}_ckpt.pt`、`skipped.jsonl`、`summary.csv`。job は
   `(N, seed)` ごとに別ディレクトリなので同時書き込みは起きない。

**VRAMプロファイル（実測、12複合体、A6000）**: peak は FFT の回転バッチが支配的で、
原子数とは単純比例しない（格子サイズは分子の空間的広がりで決まる）。

| `rot_chunk` | worst-complex peak | 12複合体合計時間 |
|---:|---:|---:|
| 32（旧既定） | 35.6 GiB | 60.5 s |
| **8（新既定）** | **24.7 GiB** | **58.9 s** |
| 4 | 12.4 GiB | 76.1 s |

`rot_chunk=8` は時間を悪化させずに peak を約30%下げるので既定にした。
`rot_chunk=4` はさらに半減するが約29%遅い。

#### 5.6.3 データ選定（deterministic・nested・leakage管理）

`scripts/pinder_scaling_select.py` が PINDER 2024-02 index から master list を作る。

- `split == "train"` かつ `holo_R and holo_L`。
- `chain_R`/`chain_L`/id に `UNDEFINED` を含む系を除外（1,560,682 → 1,411,207行）。
- **cluster_id ごとに1系だけ**採用（統計単位は PDB entry ではなく interface
  cluster）。代表は cluster 内で id 辞書順最小 = deterministic。→ **31,646
  eligible cluster**。
- 固定 PINDER-S TEST 250 idの cluster_id と重複するものを除外（実測 overlap = 0。
  PINDER のdeleaked splitが設計上保証している通り）。
- 固定 seed 20260725 で全 eligible cluster を shuffle し、先頭4,000件を master list
  として出力（`data/scaling/master_ids.txt`）。

各条件は「preparation に成功した master list の**接頭辞**」を使うので、
500 ⊂ 1,000 ⊂ 2,000 は自動的に nested になる。validation は接頭辞内で
`index % 5 == 0` の複合体（=20%）とし、これも接頭辞安定なので
validation集合も fit集合も N をまたいで nested になる。

必要total数（fit N に対し validation を別途20%確保）:

| fit N | validation | total |
|---:|---:|---:|
| 220 | 55 | 275 |
| 500 | 125 | 625 |
| 1,000 | 250 | 1,250 |
| 2,000 | 500 | 2,500 |

**準備結果（測定値）**: master list 先頭2,900件を `scripts/prep_pinder_cache.py`
で14 worker（7 GPU×2）並列に準備し、**2,900件すべて成功（fail 0 / OOM 0）**、
所要約18分、cache約550 MB。したがって usable cluster 数は必要な2,500を上回り、
どの条件も fit 数を保証できた（`data/scaling/prep_manifest.jsonl`）。
なお §5.4/§5.5 で使った旧 train 300 id のうち68件は `UNDEFINED` chain を含む
ため今回の master list には入らない。旧 220 実験とはサンプルが異なるので、
**旧結果は別サンプルの歴史的アンカーとして扱い**、今回は同一手続きの
N=220 条件を新たに走らせて低N端を内的に整合させた。

#### 5.6.4 科学的条件（§5.5の安定化設定を維持）

- 前roundのparameterとAdam stateを継承（random restartしない）。
- loss計算時のみ複合体内でscoreを標準化（正の定数によるaffine変換なので
  pose rankingは不変＝報告する全指標に影響しない）。
- `alpha_lr=1e-5`, `iface_lr=5e-4`, `grad_clip=5.0`, `0 <= alpha <= 0.1`。
- 損失 `basin(T=0.5) + 0.5·margin(1.0) + 0.1·prior`。
- 探索・pose数は全条件同一: `n_random_rot=1500`, `n_cone=400`, `ntop=1500`
  （round 0 pool = 1,900 pose）、`pool_cap=4000`。
- positive集合は round 0 で固定。mining round は `DockQ < 0.23` の negative
  だけを追加する（コード側で「mining前後でpositive数が変わっていない」ことを
  複合体ごとに assert）。
- validation候補集合は round 0 で固定し、mining も学習もしない。
- TEST は §5.4 で default parameter から一度だけ生成した固定 pool（PINDER-S
  241複合体、2,400 pose/複合体）を全条件で再利用。**checkpoint選択・round選択・
  hyperparameter調整には一切使わない**。
- checkpoint選択は固定validation lossのearly stoppingのみ。
- 最適化step数は N に比例させ、条件間で **epoch数を揃えた**:
  `steps = max(1500, 100 · ceil(N/16))`（N=220で1,500 = §5.5と同じ、
  N=500で3,200、N=1,000で6,300、N=2,000で12,500）。step数固定にすると
  大きい N ほど学習不足になり、scaling結果に交絡するため。
  validation間隔は `max(50, steps/100)`、patience 8。

#### 5.6.5 実行コマンド

```bash
export PINDER_BASE_DIR=$PWD/external/pinder
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/home/yasu/tmp/ddock-tmp        # repo外

# 1) master list（deterministic・nested・test cluster重複なし）
uv run python scripts/pinder_scaling_select.py --n-master 4000

# 2) PreparedProteinのdisk cache（seed/N間で共有、1回だけ）
uv run python scripts/prep_pinder_cache.py --limit 2900 \
    --gpus 0,1,2,3,4,5,6 --jobs-per-gpu 2

# 3) 単一job（N, seedごとに独立プロセス・独立出力）
CUDA_VISIBLE_DEVICES=0 uv run python -u scripts/run_pinder_scaling.py \
    --n-fit 500 --seed 0 --rounds 1

# 4) fleet実行（1 GPU 1 job、長いjobから割り当て）
uv run python -u scripts/run_scaling_fleet.py \
    --jobs 500:0,500:1,500:2,220:0,220:1,220:2 --gpus 0,1,2,3,4,5
uv run python -u scripts/run_scaling_fleet.py \
    --jobs 2000:0,2000:1,2000:2,1000:0,1000:1,1000:2 --gpus 0,1,2,3,4,5

# 5) 集計 + 構造不変条件の検証（nested / fit∩val=∅ / TEST非混入）
uv run python scripts/aggregate_scaling.py --runs-dir data/scaling/runs
```

#### 5.6.6 検証（実施済）

- `python -m py_compile`: 新規4スクリプト＋2モジュールで通過。
- 既存＋新規テスト: **111 passed, 1 skipped**（`uv run pytest -q`, 361 s）。
  新規 `tests/test_scaling_streaming.py`（16件）で検証した内容:
  - `PreparedProtein.to()` の往復無損失、整数idフィールドをfloat castしない。
  - disk cacheの往復無損失、hash衝突検出（例外）、破損ファイルはcache missとして扱う。
  - **CPU featureとGPU featureの一致**（`S_SC`/`T`/`S_ELEC`, rtol 2e-4）。
  - 特徴分解 `score_from_feats` が `docking_score_elec` 本体と一致（rtol 1e-8）。
  - loss用の per-complex 標準化が pose ranking を変えない（argsort一致）。
  - `cap_pool` が positive を1つも落とさない。
  - mining が positive数を増やさない（pool composition counterで確認）。
  - IFACE coverage が pose数ではなく complex数を数えている。
  - split が nested（fit・validation とも）／fit∩val=∅／必要数不足なら明示エラー。
- 少数IDでのstreaming smoke run（N=8, round 0+1, GPU 1枚, 1.1分）で
  round 0→mining→学習→TEST評価まで通し、pool composition が
  `pos 366 / rand-neg 1534 / hard-neg 1529` と期待通りになることを確認。
- run 本体でも実行時 assert を入れてある（fit∩val=∅、fit/val に TEST id が
  混入しない、fit数が要求Nと一致、mining前後でpositive数不変）。

#### 5.6.7 【重要】scaling実験の中断とスコア関数のバグ発見

**scaling実験は round 0 の途中で全ジョブを停止した。** 実験の前提であるスコア関数の
実装に、ZDOCK 原論文と照合して重大な欠落が見つかったためである。完走したのは
N=220 seed2 の 1 job のみ（§5.6.8）。

発見の経緯（すべて測定値にもとづく）:

1. **success@K が床に張り付いていた**。241 TEST 複合体で刻みは 1/241 = 0.41 pp、
   観測値は 0.0〜1.7% で scaling 傾向を検出できる分解能がない。連続指標
   （Mann-Whitney AUC）を導入すると baseline 0.0106 / 学習後 0.0065 と、
   **偶然（0.5）を大きく下回った**。
2. **固定 pool の positive はすべて注入されたものだった**。TEST pool 2,400 pose の
   うち index 0–1999 が FFT 探索出力、2000–2399 が注入 near-native cone。positive
   （DockQ≥0.23）は平均 382.2 個/複合体で、**FFT 由来は 0.067 個**、FFT 部分に
   positive を持つ複合体は 60 件中 2 件のみ。§5.3〜§5.6 の success@K は「探索が自ら
   低評価した注入 pose を上位へ持ち上げられるか」を測っており、ドッキング成功率ではない。
3. **探索の到達可能上限は高い**。「サンプル回転の最良 × 格子に載せた native 並進」の
   DockQ は 20 複合体すべてで 0.23 を超え（平均 0.76〜0.90）、正解は探索空間に存在する。
   にもかかわらず探索 recall は 0.0%。
4. **格子間隔が ZDOCK 仕様から逸脱していた**。全 ZDOCK 論文が 1.2 Å を用いるのに対し
   `docking_score_elec` / `docking_search` の既定は 3.0 Å（`geom.generate_grid` の既定
   1.2 のみが正しく残っていた）。ただし 1.2 Å に直しても recall は 0.0% のままで、
   **格子は主因ではなかった**。
5. **真の原因: PSC の有利項が丸ごと欠落していた**。Chen & Weng 2003 Eq.(3)(4) の
   `S_PSC = Re[R·L] = Σ Re[R]Re[L] − Σ Im[R]Im[L]` のうち実装には第2項（衝突罰則）
   しかなく、第1項（カットオフ内の受容体–リガンド**原子ペア総数**）が存在しなかった。
   スコアは「接触するほど下がる」量になり、最大化する探索は一貫して**接触しない pose**
   を最上位に返していた。独立測定でも純粋な接触数 `Σ n_ij` の AUC は **0.998** と
   全項中最強だった。
6. **符号規約の不整合**。Chen et al. 2003 p.81 が明示する「PSC は高いほど良い / ACE は
   負が有利」の非互換が未解決のまま加算されていた（IFACE 単独 AUC 0.127、符号反転で
   0.873）。

**棄却した仮説**（記録のため）:

- 並進格子（3.0 Å）が主犯 → 到達可能上限が 100% 閾値超えで棄却。
- 回転サンプリング不足が主犯 → Chen & Weng 2003 Fig.3(a) は Δ=20°（1,800 回転）でも
  N_P=1000 で success rate 約 80% を報告。本実装の 1,900 回転は同等なので棄却。
- IFACE の表そのものが符号反転 → Mintseris 2007 Eq.(11)(13)(14) と照合し、符号規約・
  転置・受容体重み付け・リガンド二値化すべて論文と一致。表は正しく、問題は**組合せ時の
  符号統一の欠落**だった。
- ρ=3.5 / β=3.0 が論文値からの逸脱 → **誤り**。3.5 と 3.0 は Chen et al. 2003 Eq.(2) /
  p.82 の値。最初に参照した Chen & Weng 2002（GSC、ρ=9、β=0.06）は旧版で、バージョンを
  取り違えた。一度 GSC 版へ書き換えたが撤回した。

**測定に用いた decoy 構成の失敗も記録する**。IFACE の識別能を測る際、(a) 受容体
bounding box 内の一様乱数並進 → 大半が貫通 pose、(b) キャッシュ済み TEST pool → decoy が
スコアで選抜済み、(c) 表面接触 decoy → 接触数だけで AUC 0.998 と自明に分離、といずれも
交絡した。接触数を揃えた帯での測定は 8 複合体中 1 件しか帯が重ならず統計的に弱い。
**IFACE の逆相関が「表の誤り」か「接触数効果」かは未決着**であり、今回の修正は
Chen et al. 2003 p.81 の記述にもとづく組合せ規約の統一である。

**修正内容**（数式と論文対応は `doc/scoring_function.tex` / `.pdf`）:

| 項目 | 修正前 | 修正後 | 出典 |
|---|---|---|---|
| 格子間隔 | 3.0 Å | 1.2 Å | Chen & Weng 2003 Methods |
| PSC 有利項 `Re[R]Re[L]` | **欠落** | 実装（D=3.6 Å） | 同 Eq.(3)(4) |
| PSC 罰則 ρ / ρ² | 3.5 / 12.25 | 同（正しかった） | Chen et al. 2003 Eq.(2) |
| 溶媒接触層 3.4 Å | あり（GSC 由来） | 削除（PSC は定義しない） | Chen & Weng 2003 |
| IFACE 符号 | 未統一 | σ=−1 で統一 | Chen et al. 2003 p.81 |
| β | 3.0 | 同（正しかった） | 同 p.82 |
| α | 0.01 | 1.0 | Eq.(2) に α は無い |

ρ・D・α・β・iface・charge はすべて微分可能テンソルとして露出し、初期値は論文値。

**修正の効果**（12 複合体、デフォルトパラメータ、学習なし、1,900 一様回転、h=1.2 Å、
top-2000、注入なし。到達可能上限は全条件で 100% が閾値超え）:

| 実装 | recall | 探索最良 DockQ | s@1 | s@10 | s@100 |
|---|---:|---:|---:|---:|---:|
| 罰則のみ, h=3.0 Å | 0.0% | 0.107 | 0% | 0% | 0% |
| 罰則のみ, h=1.2 Å | 0.0% | 0.066 | 0% | 0% | 0% |
| ＋符号統一 | 16.7% | 0.116 | 0% | 0% | 0% |
| **PSC 完全実装** | **66.7%** | **0.649** | **17%** | **42%** | **50%** |

**既存結論への影響**: §5.1〜§5.6 のすべての数値（1KXQ の top-1 DockQ 0.622、DB5.5 の
71%、PINDER deleaked の 1.2%、hard-negative mining の 0.0%）は、**形状相補性の有利項を
欠いたスコア関数の上で**得られたものである。「未知 interface へ汎化しない」という
§5.4/§5.5 の結論も、この前提の上でしか成立しない。修正後のスコア関数で取り直す必要がある。

**Julia 参照テストの削除**: `tests/test_phase5.py`（スコアの一致）と
`tests/test_phase6_grad.py`（有限差分勾配の一致）を削除した。両者の参照 h5 は PSC
有利項を欠いた実装から生成されており、**参照そのものが誤っている**。論文準拠の実装と
一致しないのが正しい状態であり、Julia ソースはリポジトリに無く参照を再生成できない。
これらを残すと、正しい実装が「失敗」と表示され続け、回帰検知の役に立たない。

#### 5.6.8 中断時点で得られた scaling データ

N=220 seed2 のみ完走（44.6 分、skip 0）。**バグのあるスコア関数上での結果**である点に注意。

| | baseline | round 0 | round 1 (mined) |
|---|---:|---:|---:|
| TEST DockQ@1 | 0.4% | 0.0% | 0.0% |
| TEST DockQ@100 | 1.7% | 0.4% | 0.4% |
| FIT DockQ@1 | — | 26.8% | 0.0% |

mining gain は全 K で 0.0 pp。§5.5 の fit 220 の結果（TRAIN 26〜29% / TEST 0.0%）を
独立サンプルで再現した。round 1 は採択 0 / 棄却 8 で round 0 のパラメータへ完全復帰し、
棄却候補の平均は fit loss 7.336 / val loss 5.067（出発点の val loss は 4.071）。mined
fit 分布での改善が default 分布の validation を悪化させており、§5.5.1 の対抗仮説1
（validation 分布の不一致）を支持する測定値である。

IFACE coverage（fit 220 / val 55 / test 241）はいずれも zero 成分 0/144、成分あたり
非zero複合体の中央値 193/48/240、平均被覆率 0.877/0.879/0.990。**144 成分すべてがほぼ
全複合体から信号を受けており**、§5.5.1 の候補2（rare 成分による実効自由度不足）は
支持されない。

#### 5.6.9 回転サンプリング仮説の棄却と、評価条件の欠陥

§5.6.7 で PSC 有利項を実装した直後の測定（12複合体、recall 66.7%）には**評価条件の
欠陥**があった。`validate_grid_spacing.py` は回転集合を `1,500 一様 + 400 cone` で
作っており、cone は native 姿勢 q* の周り25°である。つまり q* が候補集合へ漏れて
いた。cone を外した正直な条件で測り直した結果:

| 指標 | cone あり | **cone なし** |
|---|---:|---:|
| success@1 | 16.7% | **8.3%** |
| success@10 | 41.7% | **16.7%** |
| success@100 | 50.0% | **41.7%** |
| recall | 66.7% | **58.3%** |
| 平均 到達可能上限 | 0.902 | **0.566** |
| 平均 探索最良 DockQ | 0.649 | **0.398** |

低K側の約半分は cone 由来だった。**0% → 8.3〜50% が正味の改善**である。

失敗4件（6gxp/5jpq/7jhy/8b3p）で回転数ラダーを測定（cone なし、一様ランダム）:

| n_rot | 平均最近傍角 | 到達可能上限 | 上限≥0.23 | recall | 探索最良 |
|---:|---:|---:|---:|---:|---:|
| 1,900 | 11.1° | 0.55 | 100% | 25% | 0.14 |
| 3,600 | 11.1° | 0.55 | 100% | 0% | 0.12 |
| 14,400 | 6.3° | 0.79 | 100% | 0% | 0.13 |
| 54,000 | 3.9° | 0.73 | 100% | **0%** | 0.11 |

回転数を28倍にし最近傍角を 11.1°→3.9° に下げても **recall は 0% のまま**、探索最良は
むしろ微減。**回転サンプリング不足は主因ではない**（要因1を棄却）。

副次的な観察: `random_quaternions` は seed 固定で prefix-stable（先頭が一致）。
1,900→3,600 では4複合体とも最近傍回転が更新されず上限が動かなかった。また
1,900→3,600 で recall が 25%→0% に低下したのは、候補増加で 8b3p のヒットが
top-2000 から押し出されたためで、論文の観察と一致する（Chen & Weng 2003:
"finer angular intervals tend to rank the best hits lower than coarser angular
intervals"、Δ=4° が N_P>100 で最低の success rate）。

#### 5.6.10 3点比較による項別寄与と、診断で作り込んだ誤り

**先に、私（実行者）が作った診断上の誤りを2件記録する。**

1. **到達不可能な pose との比較**。当初「正解 pose」を `rotate(lig_ref, q_cone) + t*`
   で作ったが、この回転は探索の一様回転集合に無く、並進も FFT 格子に載っていない。
   探索が到達できない理想 pose と比較していたため、全複合体で「探索1位のほうが
   スコアが低い」という矛盾した結果が出た。
2. **pose デコードの誤り**。`res.scores` は降順ソート済みなので `argmax` は常に 0 で、
   `quats[top]` は「最良 pose を生んだ回転」ではない。正しくは
   `quats[res.quat_indices[top]]`。並進だけ正しく回転が無関係な pose を採点しており、
   FFT スコア +986.6 に対し直接評価 −8,614 という乖離が出ていた。

この乖離を追う過程で **FFT 探索と直接評価の整合性を確認した**（重要な副産物）:

| 検証 | 結果 |
|---|---|
| 探索出力 top-20 の直接再評価（n_rot=400） | 最大相対誤差 0.0000〜0.0001 |
| 格子が pose ごとに再構築される疑い | 棄却（3 pose とも同一 shape・同一原点、単独=一括） |
| FFT 巡回相関の wrap-around 疑い | 棄却（top-2000 のうち格子外 0件） |
| 探索出力 top-2000 の直接再評価（n_rot=1,900） | 平均誤差 0.08、最大 42.5（スケール約1,000） |

残差は `docking_score_elec` が渡された pose から格子を再構築するため位置合わせが
わずかに異なることによる。実害は無いが既知の差として記録する。

**回転の冗長度**（top-2000 に含まれる相異なる回転数、12複合体）: 257〜737種類、
最頻回転の占有は 21〜62 pose。1回転が上位を埋め尽くす状態ではなく、**論文準拠の
「回転あたり1並進」だけで recall が大きく改善する見込みは薄い**と判断した（実測でも
そのとおりだった、§5.6.11）。8b3p のみ相異なる回転が257と際立って少なく最頻62 pose
で、homodimer の対称性による解の集中を示唆する。

#### 5.6.11 論文準拠への2つの修正と、パラメータ学習への引き継ぎ

論文との残る差を2点実装した。

**A. 回転あたり保持並進数を1に**（`docking_search(trans_per_rotation=1)`、既定）

> Chen & Weng 2003, Methods: "Only the best translational orientation is kept for
> every rotational orientation. **We used to keep the top 10** translational
> orientations for each rotation. We subsequently discovered that these 10
> translations are extremely similar, and **keeping only the best one** helped to
> remove false positives without affecting the ranking of the best hit."

「10」は旧版（Chen & Weng 2002）の挙動で、現行 ZDOCK は1である。これは非最大抑制で
あって純粋な top-K ランキングではないため、**success@K の意味が変わる**
（pose-top-K と rotation-top-K は別の量）。両規約を併記する。

**B. Hopf ファイブレーション格子**（`zdock.rotation_grid.hopf_quaternions`）

ZDOCK は均等 Euler 角セット（Julie C. Mitchell 提供）を使い、任意姿勢と最近傍の
角距離が Δ 以下であることを保証する。このセットは配布されていないため、
Yershova, Jain, LaValle & Mitchell (2010), *IJRR* 29(7):801–812 の Hopf 分解
（S³ = S² × S¹、HEALPix × 等間隔円）で同等の格子を生成した。著者に Mitchell を
含む同系譜の手法である。外部依存なしで実装（HEALPix RING の閉形式）。

被覆半径の実測（8,000ランダム姿勢の最近傍角）:

| セット | N | 平均 | p95 | **最大** |
|---|---:|---:|---:|---:|
| **Hopf nside=3** | 1,944 | 9.4° | **13.2°** | **16.8°** |
| 一様ランダム | 1,900 | 11.1° | 17.9° | 25.9° |
| **Hopf nside=4** | 4,608 | 7.0° | 9.9° | **12.2°** |
| 一様ランダム | 4,608 | 8.2° | 13.4° | 20.6° |
| **Hopf nside=9** | 52,488 | 3.1° | 4.4° | **5.5°** |
| 一様ランダム | 52,488 | 3.6° | 5.8° | 8.8° |

同じ点数で**最大角が 25.9°→16.8°（−35%）**。平均より裾の改善が大きい。Hopf
nside=3（1,944点）は Δ≈17° 相当で、論文の「Δ=20°、1,800点」とほぼ対応する。

**2×2 比較**（12複合体、cone なし、デフォルトパラメータ、学習なし）:

| 条件 | recall | 上限 | 探索最良 | s@1 | s@10 | s@100 | s@500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 一様1,900 / pose-topK | 58% | 0.57 | 0.40 | 8% | 17% | 42% | 50% |
| 一様1,900 / 1並進 | 67% | 0.57 | 0.42 | 8% | 17% | 42% | 58% |
| Hopf 1,944 / pose-topK | 42% | 0.65 | 0.38 | 8% | 25% | 33% | 33% |
| **Hopf 1,944 / 1並進（完全準拠）** | **75%** | **0.65** | **0.53** | 8% | 17% | 25% | 42% |

**recall は 58%→75%、探索最良 DockQ は 0.40→0.53、到達可能上限は 0.57→0.65 に改善
したが、success@1 は全条件で 8%（1/12）で不動**。正解 pose は返ってくるように
なったが、上位に置けていない。**問題はサンプリングでも冗長性でもなくランキング**。
なお s@100/s@500 の比較は規約が違うため直接比較してはならない（1並進条件の top-500
は「500種類の異なる回転」である）。

**項別寄与**（Hopf 1,944 / 1並進、recall による群分け、12複合体）。3点比較:
① 厳密 (q\*,t\*) ② 到達可能な最良（最近傍サンプル回転 × 格子スナップ t\*）
③ 探索の1位。

(A) `③ − ②`（正なら誤った pose を選好）:

| group | n | α·有利項 | −α·衝突 | α·PSC | IFACE | β·ELEC | 合計 |
|---|---:|---:|---:|---:|---:|---:|---:|
| recall>0 | 9 | +102 | **+742** | +843 | +139 | 0.0 | **+982** |
| recall=0 | 3 | **+706** | −180 | +525 | **+366** | −0.2 | **+891** |

(B) `② − ①`（離散化のコスト）:

| group | n | α·有利項 | −α·衝突 | IFACE | 合計 |
|---|---:|---:|---:|---:|---:|
| recall>0 | 9 | +109 | **−966** | −8 | **−865** |
| recall=0 | 3 | +17 | −72 | −17 | −71 |

(C) 絶対値:

| group | n | ideal | reach | won | DockQ id/re/won | 最近傍角 |
|---|---:|---:|---:|---:|---|---:|
| recall>0 | 9 | 680 | **−185** | 797 | 1.00 / 0.66 / 0.11 | 9.0° |
| recall=0 | 3 | 112 | 41 | 932 | 1.00 / 0.61 / 0.02 | 8.7° |

**解釈**: 2群で機構が異なる。

- **recall>0（9件）**: 離散化で失う865点の**ほぼ全て（966点）が衝突罰則**。剛体のまま
  9°回転をスナップすると正解 pose の界面が受容体へ食い込む。実構造は側鎖が動いて
  接触するため、剛体近似では必然的に重なる。そこへ ρ⁴=150/セル の罰則がかかり、
  **正解 pose だけが致命傷を負う**。誤った pose は表面を軽く撫でるだけなので無傷。
  極端な例が 3wlb で、到達可能な正解のスコアが **−4,770**。
- **recall=0（3件）**: 衝突ではなく**有利項（+706）と IFACE（+366）が誤った pose を
  支持**。接触ペア数の数え方（D）と界面ポテンシャルの重みの問題。

論文の ρ=3.5 は均等 Euler 角セット＋refinement を前提とした値である。本実装は
refinement 前の段階なので、この罰則の強さが正解を潰している。

**パラメータ学習で優先的に動かすべき順序**（測定にもとづく）:

1. **ρ**（PSC 罰則スケール、現在 3.5）— ρ⁴ の4乗依存で感度が最大。剛体近似の
   食い込みを許容する方向へ下げる余地が最も大きい
2. **α**（PSC 全体の重み、現在 1.0）— PSC と IFACE の比
3. **D**（有利項カットオフ、現在 3.6 Å）— recall=0 群で有利項が誤答を支持
4. **IFACE 144成分** — recall=0 群で +366

いずれも微分可能テンソルとして露出済みで、初期値は論文値。**学習前段階の目標は
達成**した。

**現時点の到達点**（12複合体、完全論文準拠、デフォルトパラメータ、学習なし）:

| 指標 | 修正前 | 現在 |
|---|---:|---:|
| 探索 recall | 0.0% | **75%** |
| 探索最良 DockQ | 0.066 | **0.53** |
| success@1 | 0% | **8%** |
| success@10 | 0% | **17%** |
| 被覆半径 max | 25.9° | **16.8°** |

論文（Chen & Weng 2003 Fig.3a、Δ=20°）は N_P=10 で約25%、N_P=100 で約45% であり、
本実装（17% / 25%）はまだ下回る。ただしテストセット（PINDER interface-deleaked
12件 vs 論文の49件ベンチマーク）も成功判定（DockQ≥0.23 vs interface RMSD≤2.5 Å）も
異なるため直接比較はできない。

#### 5.6.12 テストが検出した実バグ（記録）

論文準拠への修正後にテストスイートを回したところ、**4件の実バグを検出**した。削除
すべき Julia 参照テストとは別に、これらは実装側の誤りである。

1. **`iface_score_matrix` の未挿入** — パッチスクリプトが `AssertionError` で中断し、
   ヘルパー関数が `score.py` に入らないまま5スクリプトが import していた。
2. **`docking_search_sc`（SC 専用探索）の取り残し** — `sc_rho` / `psc_d` 引数の欠落
   （`NameError`）、`sc_cell_volume_factor` の未適用で**スコアが 15.6 = (3.0/1.2)³ 倍
   ずれる**、`G.real - G.imag` のまま（Eq.(4) は実部のみ）。
3. **PSC 半径スケーリングの非対称** — 受容体は素の vdW 半径、リガンドは `√1.5·r` の
   ままだった。Chen & Weng 2003 Fig.1(b) は原子円そのものを描いており、
   `√1.5` / `√0.8` は旧 GSC の構成。両側を素の vdW に統一した。
4. **同点処理が `rot_chunk_size` に依存** — 回転あたり1並進にすると多数の回転が同一
   スコアになり、`topk` が位置で同点を解決するためチャンク分割で結果が変わっていた。
   `(quat index, translation index)` による決定論的タイブレークを実装。

**特に重要だったのは、キャッシュ特徴からのスコア再構成の不整合**である。
`docking_score_elec` は `IFACE_SIGN` を内部で適用するが、`score_from_feats` は生の
表を使っていた。このまま学習していれば、**FFT 探索がランク付けするスコアとは別の
目的関数を最適化していた**。`zdock.score.iface_score_matrix()` を単一の真実の源として
追加し、`run_pinder_scaling` / `run_pinder_hardneg` / `run_pinder` / `run_db55` /
`eval_test_pool` の5スクリプトを差し替えた。

テスト側にも旧規約の残骸があり、`test_search.py` が参照値を手組みする箇所で
`G.real - G.imag` と生の iface 行列を使っていた。これらはテスト側の誤りとして修正した
（本体は正しい）。

**限界**: 12複合体は統計的に弱く、いずれも 3.0 Å 換算で15万 voxel 以下の小型複合体
に偏っている。パラメータ学習に進む前に、より大きく多様な集合でこのベースラインを
測り直すことが望ましい。

---

### 5.7 11体並列サブエージェントによる独立バグ監査（2026-07-25）

#### 5.7.1 目的・方法・限界

**問い**: §5.6 で「学習前段階は達成」と結論したが、それは自己申告である。独立した
検証者が、原著論文・数学的整合性・実験衛生・測定スクリプトの論理という異なる角度
から見たときに、同じ結論に到達するか。

**方法**: 読み取り専用（リポジトリ改変禁止）のサブエージェント11体を並列起動し、
互いに独立した担当範囲を与えた。各体には「主張を測定で裏づけること」「論文の
どの文が根拠かを引用すること」「決着しないものは決着しないと書くこと」を課した。

| # | 担当範囲 | 主要参照 |
|---|---|---|
| 1 | PSC 項 | Chen & Weng 2003 Eq.(1)-(4), Fig.1 |
| 2 | IFACE 項 | Mintseris et al. 2007 Eq.(1)-(3),(8)-(14) |
| 3 | ELEC 項・部分電荷・β | Chen & Weng 2002, Chen et al. 2003 |
| 4 | FFT 並進探索の数学 | （論文非依存、恒等式検証） |
| 5 | 格子構成・原子→格子の離散化 | — |
| 6 | 回転表現・サンプリング | Yershova 2010, HEALPix |
| 7 | パラメータの学習可能性（勾配） | doc/scoring_function.tex |
| 8 | 評価機構（DockQ・success統計） | Basu & Wallner 2016 |
| 9 | データ集合・実験衛生・再現性 | PINDER index |
| 10 | 測定スクリプトの論理 | — |
| 11 | 文書の内部整合性（レフェリー） | EXPERIMENT_REPORT.md, .tex |

**この監査自体の限界**（先に明記する）:
- 各体の測定は多くが 1KXQ / 1AY7 / 4lzf など**少数の複合体**で行われている。
  効果量の桁は信頼できるが、コーパス全体の平均値ではない。
- 監査は**修正を行っていない**。以下の「測定値」は現行コードの挙動であり、
  修正後にどう動くかは未測定である。
- Mintseris Supporting Table I / II、Gabb et al. 1997 は `ref/` に無く、
  12原子タイプの割り当てと距離依存誘電率は**検証不能**として残った。
- 重複した指摘は下で1件に統合したが、独立に再現された事実は「複数体が確認」と注記
  した。同一エージェントが自分の誤りを再現した可能性は排除できていない。

#### 5.7.2 是認された部分（監査で問題が見つからなかったもの）

先に、**攻撃されて生き残った**部分を記録する。

- **FFT 相互相関の恒等式**（`search.py:713-715`）。`ifft(fft(R)·conj(fft(conj(L))))[t]
  = Σ_m R[m]L[m−t]` を任意の複素 `L` について導出・数値検証（誤差 1.1e-14）。
  素朴な `conj(fft(L))` は別物（同テストで最大差 39.2）であり、コードは正しい。
- **並進のデコード**（`search.py:349-357, 773-783`）。デルタ関数による作業例で符号・
  大きさとも確認。返る並進は「リガンドに**加える**変位」であり、呼び出し側の用法と
  一致。`k·spacing` の変位なので**格子原点は打ち消し合う**（§5.6 で懸念した
  「到達不能な理想 pose」問題は、格子原点に起因しない）。
- **`quat_indices` のデコード**。6箇所すべてで正しい回転と並進が対になっている
  （§5.6.10 で記録した私の診断バグの再発は無し）。
- **complex64 の精度は十分**。1.74M セルで FFT 丸め誤差は SC で最大 0.130、
  total で 2.0e-3。top-1 は float32/float64 で一致し、**top-2000 の index 集合は
  対称差 0**。ランキングが依存するスコア差（top1−top2 で 1.25〜47.8）より2桁小さい。
- **IFACE の符号規約と添字順序**。`_IFACE_IJ` が ACE 規約（有利=負）であることを
  5本の独立証拠で確認（疎水×疎水が最も負、同電荷が最も正／native 界面の厳密ペア和
  −432.3 に対しランダム decoy は +0.084±0.234 per contact）。`IFACE_SIGN=-1` の適用は
  **ちょうど1回**。Eq.(12)/(13) の i=ligand, j=receptor も両経路で正しく、
  非対称行列を使うテストが通る。
- **Hopf 回転格子**。単位ノルム 2.2e-16、SO(3) を二重被覆せず、`n_psi=6·nside` は
  被覆効率の最適点（0.287、他の n_psi で 0.225〜0.289）。HEALPix 南極冠の index が
  鏡像になっているが**点集合は同一**（集合誤差 2e-16）なので実害なし。
- **Kabsch / 四元数規約**。`geom.rotate` は標準の転置（逆回転）だが
  `scipy_rotations_to_quaternions(as_inverse=True)` が打ち消しており、
  `rotate(lig_ref,q*)+t* == native_lig` が 2.5e-14 で成立。到達可能性の数値は
  正しい規約の上に乗っている。
- **`sc_cell_volume_factor` の3経路一貫性**、**キャッシュ特徴からのスコア再構成の
  厳密一致**（`max|full − (α·sc + Σ M⊙T + β·elec)| = 0.0`）、
  **`T` が ρ・D に非依存**であること。
- **nested 分割は disk 上の実データで成立**（220⊂500⊂1000⊂2000、fit∩val=∅、
  TEST 重複 0、seed 0/1/2 で byte 一致）。

#### 5.7.3 確定した実装バグ（深刻度順）

以下はすべて**測定つき**である。丸括弧内は指摘した監査体。

**S1. ELEC の open-space マスクが界面のポテンシャルを消している**
（`score.py:699-700`, `search.py:454`／3体が独立に確認）

```python
open_space_mask = (rec_sc_real == 0) & (rec_sc_imag == 0)
```

初期実装では `rec_sc_real` は GSC の表面層グリッド（殻の上で1）だったのでこの式は
正しく「open space」を意味した。PSC 書き換え（`24db5f3`）で `Re[R]` の意味が
**「open space かつ受容体原子から `radius+3.6 Å` 以内の原子数」**に変わったため、
現在このマスクは「**どの受容体原子からも 5.4 Å 以上離れたセル**」＝バルク溶媒だけを
残す。接触帯を丸ごと 0 にしている。

| | open space セル | Σ\|V\| | max\|V\| |
|---|---:|---:|---:|
| 論文どおり（`im==0`） | 1,349,522 | 8723.2 | 1.199 |
| 現行コード | 1,302,237 | **1755.0 (−80%)** | 0.467 |

生き残る最近接セルは受容体原子から **5.12 Å**。vdW 接触は 3.2〜4.5 Å なので
**界面のリガンド原子は例外なく V=0 を見る**。別の体は 4lzf で「リガンド電荷の
70.4% がマスク領域外」「S_ELEC 0.107 vs 0.619」、さらに別の体は 1AY7 で
「−0.0702 vs −0.945（**93% が消える**）」を測定。§5.6.11 の項別分解で
`β·ELEC` が 0.0/−0.2 だったのは、この直接の帰結である。

**学習段階への影響が最大**：β と 11 個の電荷パラメータは、接触している全ての pose で
勾配が**厳密に 0** になる（測定：shift=0〜9 Å で `beta.grad=0.0`, `|charge.grad|=0.0`。
11〜13 Å でようやく非零）。

修正: `open_space_mask = (rec_sc_imag <= 0)`（両ファイル同時）。

**S2. ELEC の符号が最大化の向きに対して反転している**（`score.py:535,548`,
`search.py:338`）

本実装は**最大化**する（`topk`）。PSC は「高いほど良い」に、IFACE は `IFACE_SIGN=-1`
で揃えてある。ELEC だけ揃っていない。2003 Eq.(2) は `Im[L] = −1 ×(atom charge)` で、
「より負が有利」の規約（p.81）。最大化系では ELEC 寄与は **`−β Σ V q`** でなければ
ならない。コードは `+q` を置いて `+β Σ V q` を加算している。

```
q_lig = −1（引力）: score_elec = −0.14506
q_lig = +1（斥力）: score_elec = +0.14506
```

β=+3.0 なので**探索は静電的な斥力を報酬にしている**。しかもこれは学習では吸収
できない：`S_ELEC` は電荷 LUT について**二次**なので LUT の大域符号反転は
`S_ELEC` を変えず、β は凍結されている。`score.py:610-612` の docstring は
「ligand stores −q」と書いてあり、**コードが自分の docstring と逆**。
S1 と S2 は同時に直す必要がある（片方だけでは大きさか向きのどちらかが誤ったまま）。

**S3. PSC の surface/core 優先順位が論文と逆**（`score.py:303-305`,
`doc/scoring_function.tex:83`）

```python
core_b = core > 0
shell_b = (shell > 0) & ~core_b     # core が勝つ
```

Chen & Weng 2003 および Chen et al. 2003 は同じ文を繰り返している：
*"The 'solvent excluding surface layer of a protein' is defined by the grid points
corresponding to surface atoms. **All other** grid points corresponding to any core
atoms are in the protein 'core'."* — 表面層が**先に**定義され、core は残余である。
**surface が優先**。

1KXQ で表面原子と core 原子の両方に覆われるセルは受容体 4,176（占有セルの 17.2%）、
リガンド 1,065（18.8%）。これはまさに界面近傍のセルである。

| ρ | 罰則（現行=core優先） | 罰則（論文=surface優先） |
|---|---:|---:|
| 3.0 | 351.0 | 279.0 (−20.5%) |
| 3.5 | 502.25 | 379.8 (−24.4%) |

native pose の `S_PSC` は **168.75 → 291.2（+73%）**。PSC が寛容に扱うはずの殻を
系統的に過剰罰則している。`doc/scoring_function.tex:83` にも「core taking precedence」
と書いてあるので、タイポではなく意図的な誤りである。§5.6.11 で「ρ⁴=150/セルの罰則が
正解 pose だけを潰している」と結論したが、**その罰則の 20〜24% はこのバグ由来**で
あり、ρ を下げる前にこちらを直すべきである。

**S4. 原子→格子の割り当てが nearest ではなく floor**（`spread.py:47-49`／2体が確認）

```python
ix = torch.ceil((xyz[:, 0] - x_min) / dx).long() - 1     # = floor
```

`generate_grid` は格子点を `x_min + i·h` に置くので、`ceil(u)-1` は floor であり、
常に**下側**の格子点に落とす。関数名 `_nearest_cell_indices` が偽り。Julia 実装の
忠実な移植だが、ZDOCK/論文からの逸脱である。

1KXQ native pose、厳密ペア和 −432.31（1683 接触）に対する IFACE の格子近似:

| spacing | 格子上の値 | 誤差 | `(count>0)` で失う原子 |
|---|---:|---:|---:|
| 0.6 Å | −381.86 | +11.7% | 0/916 |
| **1.2 Å** | **−288.40** | **+33.3%** | 10/916 |
| 1.8 Å | −232.24 | +46.3% | 135/916 |

誤差は**すべて大きさの損失**（系統誤差）。1.2 Å で分解すると、二値化の寄与は
1.6 pt に過ぎず、**スナップが 31.7 pt**。平均変位は
**(−0.629, −0.603, −0.598) Å ≈ −h/2/軸**。事前に +h/2 ずらすと誤差は
−4.3% に潰れる（spacing にほぼ非依存＝純粋な離散化残差の指紋）。

影響範囲は nearest 散布を使う3つのリガンド格子すべて（IFACE `L`、PSC `Re[L]`、
ELEC `Q_L`）。受容体側の格子はすべて真の原子座標から作られるので、**受容体と
リガンドが半セルずれている**。FFT の並進はセル単位なので**探索では回収できず**、
1.2 Å では到達可能 pose に `h·√3/2 = 1.04 Å` の下限誤差が残る。

修正は `torch.round(...)`。ただし Julia 参照との bit 一致が壊れ、
`tests/test_search.py:583` が現行挙動を固定しているため、フラグ化して再測定が要る。

**S5. マイニングは 3.0 Å で探索し、特徴量は 1.2 Å で作っている**
（`dataset.py:300`, `run_pinder_scaling.py:180-196`, `run_pinder_hardneg.py:78-85`）

`grep -n spacing` が4つの `run_*.py` で**0ヒット**。したがって
`generate_decoys` → `docking_search` は既定 **3.0 Å**、
`docking_score_elec(..., return_components=True)` は `SC_REFERENCE_SPACING` = **1.2 Å**。
git で回帰であることが確認できる：

| commit | `docking_score_elec` | `generate_decoys` |
|---|---|---|
| `9f56952` | 3.0 | 3.0（整合） |
| `b9c9334` | 1.2 | **3.0（取り残し）** |

spacing 修正が片方だけ動いた。結果として、**top-2000 の足切りは 3.0 Å のスコアで
行われ、学習損失は 1.2 Å の特徴量で計算されている**。「hard negative」は誰も
学習も報告もしていないスコア関数の下で hard である。§5.6 のプールはすべてこの
状態で作られている。

**S6. `docking_search_sc` のリガンド側で `sc_rho` が黙って無視される**
（`search.py:188-197, 964-967`）

`_build_ligand_sc_grids_batch` に `sc_rho` 引数が無く、リガンドは常に 3.5 を使う。
受容体側は `sc_rho` を尊重する。`sc_rho=3.0` を渡すと受容体 `Im∈{0,3,9}`、
リガンド `Im∈{0,3.5,12.25}` になり、Eq.(3) の `Im[R]=Im[L]` を破る。
現在の呼び出し元はテストのみなので既報の結果は無効化されないが、ρ の掃引を
仕掛けた瞬間に嘘の値を出す罠。

**S7. `docking_search` の出力に sentinel pose が混じる**
（`search.py:682-684, 745-754`／2体が確認）

top-k バッファを `score=-inf, quat=0, flat=0` で事前充填し、`keep = min(ntop, …)` は
常に `ntop`。`trans_per_rotation=1` では候補は `n_rot` 個しか生成されないので、
`n_rot=1900, ntop=2000` の既定では**返る 2000 件のうち 100 件が placeholder**で、
「受容体重心に置いた参照リガンド」としてデコードされる。`eval_search_test.py` と
`validate_grid_spacing.py` は `isfinite` で濾していないので recall/top-K 統計に入る。

**影響の実測**: sentinel pose の DockQ は境界5複合体で 0.004〜0.057。したがって
**§5.6.11 の 75% recall と 2×2 表は汚染されていない**。ただし「top-2000」「n_out=2000」
という表記は誤りで、実体は 1,944（Hopf）または 1,900（一様）＋ sentinel である。

**S8. 格子の padding にリガンドの x 方向の広がりを使い、`orient()` が符号つき質量で
重心を取る**（`geom.py:206, 99-103, 135`）

`orient()` に渡す「質量」は IFACE 表の第0列で、**12成分中6個が負**。したがって
慣性テンソルは半正定値でなく、SVD の特異値順が「最長軸が先」を意味しない。
prep cache 300複合体で測定:

| | x が最長軸でない | 最悪 S/max_extent |
|---|---:|---:|
| iface 質量（現行） | **65/300 (21.7%)** | **0.299** |
| 一様質量 | 3/300 (1.0%) | 0.869 |

実害の測定: **native pose で 7/300 の複合体がリガンド原子を箱の外に出す**
（最大 12.1% の原子が脱落、`_in_bounds` が黙って捨てる）。うち1件は非接触ペアなので
差し引いても実接触例が残る。FFT の巻き込みも 4lzf で確認（300 並進のうち 15 で
直接計算と不一致、最大 691.6、正の alias も存在）。ただし実際に返る上位 pose では
FFT と直接計算が `rel.diff = 0.0000` で一致しており、**現時点で勝者は汚染されていない**。

さらに `decenter` は `w.sum()` で割るため、`Σw ≈ 0` のとき重心が発散する。
測定：`|centroid|` 中央値 7.8 Å、>10 Å が 121/300、**最悪 22,707 Å**（`4ez1`, Σw=0.002）。
`lig_ref` は原点まわりで回転させられるので、この場合リガンドは格子の外に飛ぶ。

副作用として padding が `2·S` になり、箱が必要以上に大きい（中央値 2.8e6 voxel、
最大 2.4e8 voxel）。`validate_grid_spacing.py` の OOM リトライの直接原因。

**S9. 対称性を扱わないため、完全に正しい pose が DockQ ≈ 0.01 と評価される**
（`dockq.py` 全体）

chain permutation / 対称性の最大化がリポジトリのどこにも無い（`grep` 済み）。
両側同一 UniProt の割合: 固定 TEST 241件で **27.4%**、scaling コーパス 2900件で
**19.8%**（`pinder_train_ids.txt` では 0% — train/test の組成非対称も注記に値する）。

役割入替え pose `L' = T⁻¹(rec)` を作って測定（14複合体）:

| ケース | n | 現行 DockQ | L-RMSD |
|---|---:|---:|---:|
| 厳密 C2（180°, involutory） | 3 | **1.000** | 0.0 Å |
| 非 involutory（2.6°〜173.9°） | **11 (79%)** | **0.005〜0.020** | 34〜70 Å |

**コーパスの約 20% × 非C2 79% ≈ 15% の複合体が、完全に正しい pose を ~0.01 と
採点される**。これは §5.5.1 の仮説2（真の positive を hard negative として掘って
しまう）を仮説から**実証された機構**に格上げする。

**S10. DockQ の原子レベル近似で閾値 0.23 が正典の 0.332 に相当する**
（`dockq.py:174-210`）

| 項 | 正典 (Basu & Wallner 2016) | 本実装 |
|---|---|---|
| Fnat | native **残基-残基**接触 | native **原子-原子**対 |
| iRMSD | 界面残基の主鎖、10 Å、両鎖、**Kabsch 重ね合わせ後** | 全リガンド原子、8 Å、リガンドのみ、**重ね合わせなし** |
| LRMSD | 主鎖 | 全原子（実質一致） |

1KXQ の90摂動で測定: iRMSD の膨張は「重ね合わせ省略 1.95×」×「8 Å 全原子マスク
1.11×」= 中央値 2.10×。**本実装 0.23 ⟺ 正典 0.332**、正典 0.23 ⟺ 本実装 0.169。
正典で acceptable の pose の **9.3% (7/75)** が本実装で 0.23 を割る（偽陽性は 0/68）。
つまり報告中の success@K はすべて**過小報告**側に偏っている。順位は保存される
（Spearman 0.997）。

付随: `term_i, term_l ≥ 0` なので **fnat=0 かつ iRMSD→∞ でも L-RMSD ≤ 5.72 Å なら
DockQ ≥ 0.23 に到達できる**。native 接触を1つも回収していない「positive」がありうる。

**S11. `n_native == 0` の複合体が DockQ を 2/3 で頭打ちにされる**（`dockq.py:160-182,210`）

`fnat=0` のまま3で割るので、**厳密な native pose が 1.0 でなく 0.667** になる。
prep cache 150件で 2件（1.3%）が該当（`3jap` は2鎖の最近接距離が 106.5 Å で
そもそも接触していない）。TEST 241件には無いが、fit/validation に使う 2900件には
含まれる。

#### 5.7.4 測定・解析の誤り（既報の数値を無効化するもの）

**M1. `diagnose_failures.py:94` の `rho=0` は有利項も同時に解放する**

これは §5.6 の途中で私が独立に気づいて記録した事実だが、監査体が定量化した:
`rho=0` にすると `sc_encode` の `im ≡ 0` になり、`re = counts * (im <= 0)` の
マスクが**全セルで真**になるので受容体内部の有利項カウントが漏れる。

| ρ | `Re.sum()` | `Re` nnz | `Im` nnz |
|---|---:|---:|---:|
| 3.5 | 11419 | 3730 | 502 |
| **0.0** | **14768 (+29%)** | 4232 | 0 |
| 1e-6 | 11419 | 3730 | 502 |

したがって `fav` と `clash` は**同じ量 Δ だけ両方過大**（`psc` と `total` は正しい）。
Δ は食い込みの単調増加関数なので、まさに診断対象の「食い込んだ勝者 pose」で
最大になる。**§5.6.11 の表 A/B の `fav`・`clash` 列は再測定が必要**
（`total` 列と表 C は影響なし）。修正は `_call(1e-6)`。

**M2. §5.6.11 表 A（到達可能 pose との比較）は恒等式であって測定ではない**

`reach = (quats[nearest], t_snap)` は探索の候補集合の**要素**であり、
`docking_search` は全候補の argmax を返す。したがって

```
score(won) = max over all (q_i,t) ≥ score(reach)
```

は数学的恒等式で、`gap_total ≥ 0` は**必ず**成り立つ。表の脚注
「正=スコアが本当に誤った pose を好んでいる」は反証不能である。
さらに `dockq_reach < 0.23` の複合体では、この gap は native 選好について何も
語らない。正しい定義は「候補集合の中で DockQ≥閾値を満たす pose のうち**最高スコア**の
もの」との差、あるいはその pose の**順位**。

**M3. `diagnose_score_terms.py:221` が生の IFACE 表を使っている**

```python
imat = iface.view(12, 12).T          # iface_score_matrix() を通していない
```

`docking_score_elec` は `IFACE_SIGN=-1` を適用するので、このスクリプトの `IF` は
**探索がランク付けする IFACE 項の符号反転**である。よって
`auc_IFACE` と `auc_NEG_IFACE` のラベルは入れ替わっており、
セクション3「COMBINED SCORE」は全行が別の目的関数を評価している。
§5.6.12 で「5スクリプトを `iface_score_matrix` に差し替えた」と書いたが、
**このスクリプトは差し替え漏れ**である。

**M4. 同じスクリプトの AUC に同順位補正が無い**（`diagnose_score_terms.py:68-77`）

| 入力 | 正しい値（scipy） | 当該関数 |
|---|---:|---:|
| 同点なし | 0.788906 | 0.788906 |
| 整数の同点 | 0.857812 | 0.861875 |
| 全同点 | **0.500000** | **0.700000** |

入力順にも依存する（同一データの5通りの置換で 0.8591〜0.8712）。
**採点対象は整数の接触数**、すなわち同点だらけの入力である。したがって
**§5.6.7 の項目5「純粋な接触数 Σn_ij の AUC は 0.998」と項目6「IFACE 単独 0.127、
符号反転で 0.873」は無効**であり、M3 とあわせて再測定を要する。

**M5. `eval_test_pool.py` の `mann_whitney_auc` は CUDA で例外を投げる**
（`:74, :77`）

`torch.arange` / `torch.zeros` に `device=` が無い。`main()` の既定は
`--device cuda` である。両コミット（`24db5f3`, `b9c9334`）に存在。
なお rank-sum と同順位補正のロジック自体は scipy と厳密一致で**正しい**
（M4 の壊れた実装とは別物）。

**M6. `data/scaling/eval_pool/*` の AUC は §5.6.12 の符号修正より前の出力**

出力は 7月25日 13:02、`eval_test_pool.py` の修正は 20:40（`b9c9334`）。
つまり §5.6.7 項目1 の「baseline 0.0106 / 学習後 0.0065」および eval_pool 表全体は
**符号反転した再構成で作られている**。引用前に再生成が必要。

**M7. 固定 TEST プールは中立な候補集合ではなく、baseline スコアに対する敵対的集合**

構成（`dataset.py:313-341`）: 3000 一様回転 + `q*` まわり 25° cone 400 → FFT 探索の
上位 2000 → **同じ cone 400 個を `t*` で再注入**。実ファイル（241×2400）の測定:

- **95.4%（230/241）の複合体は、探索由来 2000 pose の中に positive を1つも持たない**
- 探索部分の L-RMSD ≤ 5 Å は 0.24%（中央値 33.8 Å）
- baseline パラメータで **72.6% の複合体が first-hit rank ちょうど 2001**、
  95.0% が >2000 — すなわち探索由来 pose 全部が注入 positive 全部を上回る

negatives は（ほぼ）同じ汎関数の arg-top-2000 であり、positives はその汎関数が
捨てた pose である。**baseline success@K ≈ 0 はほぼ恒等式**であり、
平均 AUC 0.0106 は「negatives が positives を上回るように選ばれた」ことの表示である。

なおこの測定は §5.6.7 の結論（PSC 有利項の欠落でスコアが非接触 pose を好んでいた）
を**独立に裏づける**証拠でもある。

**M8. §5.4/§5.6 の success@K の差はすべて統計的に有意でない**

刻みは 1/241 = 0.415 pp、観測範囲は 0.0〜4.1%。

| 比較 | 検定 | p |
|---|---|---:|
| §5.4 top-1 0.4%→1.2% | Fisher exact 両側 | **0.623** |
| §5.4 top-10 1.2%→2.9% | 同 | **0.339** |
| §5.4 top-100 1.7%→4.1% | 同 | **0.173** |
| baseline vs N220_seed2 @1 | McNemar 厳密 | 1.000 |
| 同 @10 | 同 | 0.250 |
| 同 @100 | 同 | 0.250 |

検出力: n=241、α=0.05、80% で、1.7% を起点に検出可能な最小率は **6.5%**。
McNemar が p<0.05 に達するには**一方向の不一致が6件以上（≥2.5 pp）**必要。
報告中のどの効果もこの床を下回る。

**より検出力の高い統計は既に計算済みで、比較に使われていない**:

| 指標 | baseline → round 0 | Wilcoxon p |
|---|---|---:|
| Mann-Whitney AUC | 0.0106 → 0.0065 | **8.2e-10** |
| first_hit_pct | 0.8082 → 0.8203 | **1.9e-05** |
| mean best DockQ@100 | 0.0642 → 0.0358 | **1.4e-34** |

**向きに注意**: この3指標はいずれも「学習後のほうが**悪い**」と言っている。
床に張りついた success@K 表がそれを隠していた。（ただし M6 によりこれらの
具体値は再生成待ち。また round1 は round0 と bit 一致なので独立な反復ではない。）

**M9. 被覆半径の「max」は下方バイアスのある標本最大値**（`rotation_grid.py:386-408`）

上限を標本で推定すれば必ず過小になる。SO(3) の穴の体積が ε³ で縮むので、
誤差は n_probe^(−1/3) でしか閉じない。2M プローブ＋局所上昇による真値（これも
下界）と比較:

| 格子 | 報告値 | 真値（≳） | 誤差 |
|---|---:|---:|---:|
| Hopf nside=3 (1944) | 16.8° | **18.5°** | −9% |
| Hopf nside=4 (4608) | 12.2° | **13.9°** | −12% |
| Hopf nside=9 (52488) | 5.5° | **6.2°** | −13% |

加えて `validate_grid_spacing.py:202` は `n_probe=4000` を渡しているが、
`doc/scoring_function.tex:345` は「over 8000 random orientations」と書いている。

**相対的な主張（−35%）は頑健**（一様1900 → 28.33°、Hopf 1944 → 18.53°）。
ただし論文の Δ は**保証された上界**で、こちらは**確率的な下界**なので、
同じ量を同じ方法で測っていない。公平な等価性の述べ方は充填効率で、
論文の集合が 0.248〜0.308、Hopf が 0.29 と実質同等である。

**M10. `validate_grid_spacing.py` の `--n-cone` 既定は 400 のまま**（`:153`）

スクリプト自身のコメント（`:228-230`）が「`--n-cone 0` が正直な条件」「非零の cone は
`q*` を候補集合に漏らし ceiling と recall を両方水増しする」と書いているのに、
既定値は変更されなかった。`:262` のバナー「DEFAULT ZDOCK PARAMETERS — search output
only, nothing injected」は**注入**については真だが**回転集合**について沈黙している。

#### 5.7.5 学習段階に効く欠陥

**L1. α の clamp が「論文準拠」とされる値の 10 分の 1 で頭打ちになる**
（`run_pinder_scaling.py:329, 500, 539`、`run_pinder_hardneg.py:195, 311`／3体が確認）

`doc/scoring_function.tex:230` と §5.6.7 は α の初期値を **1.0**（2003 Eq.(2) には
α が無く PSC のスケールは ρ が決めるため）と宣言した。しかし学習スクリプトは
`alpha0 = 0.01`、`--alpha-max` 既定 `0.1`、`alpha.clamp_(max=0.1)` のままである。
**最適値が実行可能領域の外にある**。α が境界に張りついた場合、「学習しても
改善しない」という結論は箱制約と交絡する。診断スクリプト
（`diagnose_failures.py`, `validate_grid_spacing.py`）だけが 1.0 に更新済み。

**L2. `D`（PSC 有利項カットオフ）は微分不可能**（`score.py:341-384, 596`,
`spread.py:155-213`／2体が確認）

`psc_d` は `radius + psc_d` → `_neighbors_indices` に入り、そこで
`float(rcut.max().item())`（detach）と `d2 < rcut_sq`（bool）に消費される。
出力は `flat[within]` と重み1なので、D からスコアへの float 経路が存在しない。

```
D=3.5    S=−11138.3025      D=3.6     S=−11111.3025
D=3.55   S=−11124.3025      D=3.6001  S=−11111.3025   (dS = 0)
psc_d.grad = None
```

**したがって `doc/scoring_function.tex:239-241` の「ρ, D, α, β, e_ij, q_m は
すべて微分可能テンソルとして露出」は D について誤り**であり、
§5.6.11 の優先順位リスト項目3（D）も、勾配で動かせるという前提が崩れる。
D は grid search でしか調整できない（あるいは soft cutoff への置換が要る）。

**L3. ρ は微分可能で、しかも意味のある自由度である**（訂正・良い知らせ）

監査の前提（「ρ はマスクにしか効かないので勾配が死んでいるのでは」）は**誤り**だった。
占有は vdW 半径で決まり ρ は値なので、これは論文どおりである。autograd と中心差分が
11桁一致し、`S_SC(ρ)` は厳密な4次式:

```
S_SC(rho) = 152 − 28·rho² − 42·rho³ − 15·rho⁴      （1次項なし）
```

すなわち `c_pair − (n_ss ρ² + n_sc ρ³ + n_cc ρ⁴)`。ρ の勾配ステップは
surface-surface / surface-core / core-core の重みを有利ペア数に対して**再配分**する。
ただし ρ=0 でのみ `(im<=0)` マスクが不連続（`S_SC(0)=259.0` vs 外挿 152.0）なので、
最適化器を 0 に張りつかせてはいけない。

**キャッシュ経路で ρ を学習可能にする最小変更**: `f.sc` の代わりに
`(c_pair, n_ss, n_sc, n_cc)` の4スカラーを保存すれば厳密に再構成できる（検証済み）。

**L4. 現行の学習スクリプトが実際に学習できるのは 145 個だけ**

`alpha`(1) + `iface_flat`(144)。`beta` は `requires_grad` 無しで optimizer にも
入っていない。`charge` はパラメータですらない（`charge_dummy = torch.zeros(0)`、
`f.elec` は既定 LUT で焼き込み済みの定数特徴）。`sc_rho`, `psc_d` は `f.sc` に
焼き込まれており**構造的に学習不可能**。`doc/scoring_function.tex` の
「β … can be optimised」は現状の実装を記述していない。

**L5. 1 pose だけの複合体で損失が NaN になる**
（`run_pinder_scaling.py:147`, `run_pinder_hardneg.py:123`）

`scale = s.detach().std().clamp_min(1.0)` — 要素1個の `torch.std` は `nan`、
`clamp_min` は NaN を伝播する。1つの単一 pose 複合体が損失を NaN にし、
`clip_grad_norm_` と Adam が α と iface を**恒久的に汚染**する。
修正: `std(unbiased=False)` または `nan_to_num`。

**L6. `train.py:433-435` の `loss_param_prior` がタンパク質ループの内側にある**

`"combined"` 分岐で prior がデータ項に対して複合体数倍される（`.backward()` も
per-protein で N 回累積）。`run_pinder_scaling.py:282` はバッチにつき1回適用しており、
**2つの経路が食い違っている**。

**L7. `refine_poses_gradient` は支配項に対して勾配を持たない**
（`score.py:485-509` vs `:514-547`, `search.py:808`）

`scatter_mode="trilinear"` は IFACE と ELEC のリガンド散布しか切り替えず、
PSC は nearest のまま。測定（4lzf）:

```
mode=trilinear  SC only     max|dscore/dxyz| = 0.0      ← grad_fn すら無い
mode=trilinear  IFACE only  max|dscore/dxyz| = 5.93
mode=trilinear  ELEC only   max|dscore/dxyz| = 0.18
```

`refine_poses_gradient` は目的関数の 2.4%（IFACE+ELEC）だけを見て Adam 上昇を行い、
97.6%（衝突罰則を含む形状項）は**力を及ぼさない**。精密化が pose を立体重なりへ
追い込み、既定 nearest で再計算した `scores_final` はむしろ悪化しうる。

#### 5.7.6 再現性・実験衛生

**R1. `prep_manifest.jsonl` は自分の worker ファイルから再生成できない**
（`prep_pinder_cache.py:54, 145`）

- `prep_manifest.jsonl` は 2900 レコード、`prep_manifest/worker*.jsonl` は **2886**。
- 欠けている 14 件（rank 0-13）は `smokeworker_worker0{0,1,2}.jsonl` にあり、
  マージの glob `worker*.jsonl` が拾わない。
- その 14 件は既に cache にあるので、再実行しても `:54` の early-continue で
  **レコードが出力されない**。

シミュレーション結果:

```
usable 2728 → 14件を失うと 2714
N=500: fit id の 134/500 が変わり、validation は 125/125（100%）が入れ替わる
保存済み N1000 fit ⊇ 再計算 N500 fit :  False
```

14 mod 5 = 4 なので stride の位相が回り、nested 構造が壊れる。
**disk 上の既存結果は内部整合だが、前向きの再現性が失われている。**

**R2. voxel フィルタが未登録 id に対して fail-open**（`run_pinder_scaling.py:427`）

`voxels.get(pid, 0) > max_vox` は**不在 id を 0 voxel = 常に適格**として扱う。
`compute_grid_sizes.py` は当時 `ok` だった id しか表に入れない。cache を
`--limit` を増やして拡張すると、新規複合体がフィルタを素通りする。
現時点では 2900 件すべてに voxel 登録があるので実害なし。

**R3. cache の破損が黙って N を減らす**（`prep_cache.py:72-75`,
`run_pinder_scaling.py:651`）

`load_prepared` は破損ファイルに `None` を返し、`try_mine` は
`stage="cache_load"` にし、救済パスは `stage == "mine"` だけを再試行する。
`assert len(fit_ids) == args.n_fit` は id リストを見ているので通り、
`fit_pools = [pools[p] for p in fit_ids if p in pools]` が**黙って N−1 で走る**。
`/home` が 97% 使用中で、これは短い write が起きる条件そのものである。
現行の run は `skipped=0` なので発生していない。

**R4. 分割を定義する成果物が gitignore されている**

`master_ids.txt`(132K), `master_clusters.csv`(348K), `grid_voxels.json`(128K),
`prep_manifest.jsonl`(456K), 各 `split.json` — 合計 1.1 MB 未満。
これらは**実験の定義そのもの**であり、R1 により byte 単位で再生成できない。
CLAUDE.md が禁じているのは「大きい」生成物であって、これらは大きくない。

**R5. GPU 上の結果は bit 再現しない**

選択は完全に決定的（`select_split` に RNG 無し、seed 0/1/2 で byte 一致）。
一方、同一プロセス・同一 seed・同一 chunk で 3複合体×2回:

```
6f8g__A1_...  poses identical=True  scores identical=False  max|Δ|=1.5e-05 (rel 3.7e-08)
```

pose 集合は安定（決定論的タイブレークが効いている）。報告値を動かす大きさでは
ないが、「同じコマンドで bit 同一」ではない。

**R6. `cluster_id` 非重複は思ったより弱い保証**

PINDER index で `cluster_id` は文字どおり `cluster_{L}_{R}`（鎖クラスタの対）である。
「cluster_id 重複なし」は「train が test の (受容体族, リガンド族) の**組**を再利用
しない」だけを意味する。実際の deleak は `split == "train"` が担っている。

N=2000 の選択（2500 系）で単鎖の重なりを測定:
- **149/2500 (6.0%)** が TEST の単量体と1つ以上のクラスタを共有
- **10/2500 (0.4%)** が受容体・リガンド**両方**の族を共有（対は異なる）

学習するのは大域 145 パラメータだけで複合体ごとの容量が無いため、この実験に
限れば無視は擁護できる（0.4% は success@K の量子化 0.41 pp 未満）。ただし
「cluster_id の disjoint が leakage 対策である」という書き方は避けるべき。

**R7. voxel フィルタが学習コーパスだけを縮めており、TEST は縮めていない**

| | n | 中央値 voxel | 平均総原子数 |
|---|---:|---:|---:|
| 採用 | 2728 | 174,179 | 3487 |
| 除外 | 172 (5.9%) | 3,195,127 | 9182 |

除外された 5.9% は系統的に約 2.6 倍大きい。しかも評価コホートは同じフィルタを
受けていない（学習コーパスの原子数中央値 2576 vs cache 済み TEST 178件で 3444）。
**scaling 曲線は小-中型で当てはめ、大型で評価している**。train→test の差の一部は
サイズのドメインシフトであって汎化ギャップだけではない。

**R8. 「TEST セット」が2つあり、どちらも生存バイアスを受けている**

`pinder_test_ids.txt` 250 → `test_feats.pt` **241**（9件が parse/OOM で脱落）
→ `prep_manifest_test.jsonl` **178 ok / 72 fail**（`eval_search_test.py` はこちら）。
`run_pinder_scaling.py:567` の assert は**部分集合**判定なので、縮んでも黙って通る。
プール指標と end-to-end 指標は**異なる母集団**の上で計算されている。

**R9. `aggregate_scaling.py` は構造違反を検出しても終了コード 0**（`:96-97`）

`PROBLEMS FOUND` と印字したあと CSV を書いて正常終了する。加えて
`splits[...]` は metrics の `continue` の**後**で埋まるので、未完了 run の
壊れた split は検査されない。

**R10. `tests/test_scaling_streaming.py` は本物だが浅い**

nested / 失敗スキップ / oversized 除外 / voxel 表欠落 / cache 不足 の各テストは
**独立に組んだ fixture** で書かれており、実装を変えれば落ちる（トートロジーではない）。
一方で覆っていないもの:
- **TEST 非混入は一度もテストされていない**（assert は `main()` にあり、テストは通らない）
- **manifest が変わったときの nested 性**（＝R1 そのもの）はテストされていない
- seed 非依存性のテストが無い（`_Args` に `seed` フィールドが無い）
- fail-open な voxel 参照が exercise されない（fixture が全 id に voxel を与える）
- `test_mining_appends_negatives_without_growing_positives` は
  production の `absorb()` ではなく `pool.cat()` を呼び、negative フィルタを
  テスト本体で再実装しているので**トートロジーに近い**

**R11. 「positive は増えない」assert はトートロジー**（`run_pinder_scaling.py:593`）

`dockq < thr` の行しか cat せず、`cap_pool` は positive を全部残す構成なので
`after_pos == before_pos` は破れようがない。加えて early-return 経路では
そもそも実行されない。

#### 5.7.7 論文との相違（バグではないが記録すべきもの）

- **ρ=3.5 は Eq.(3)/(4) の罰則値を再現しない**。PSC 論文は
  surf-surf/surf-core/core-core = −9/−27/−81 と明記し、有利項の重みは +1。
  コードは ρ=3.5 で **−12.25/−42.88/−150.06** を出す（比も 1:3:9 でなく 1:3.5:12.25）。
  ρ=3.5 は Chen et al. 2003 Eq.(2) の値だが、**そこでは有利項の重みが 1.334 に
  引き上げられている**（罰則:報酬比は Eq.(3) で 9.0、Eq.(2) で 9.18 とほぼ同じ）。
  現行は Eq.(3) のチャネル配置と報酬重み 1.0 に Eq.(2) の ρ を混ぜており、
  比が **12.25** ＝ どちらの論文より 36% 厳しい。
  選択肢は (a) ρ=3.0 に戻す、(b) ρ=3.5 のまま有利項を 1.334 にする。
- **`Re[L_PSC]` が 1 でなくカウント**（`score.py:378-383` ほか）。Eq.(3) は
  「リガンド原子の最近接格子なら 1」と明記。1.2 Å では 916 原子中 12 個（1.31%）が
  セルを共有するだけだが、**有利項は 2.83% 変わる**（671 vs 652）——共有セルが
  `Re[R]` の高い領域に集中するため。docstring の「1.2 Å ではほぼ同一」は過小評価。
- **距離依存誘電率が無い**。2002 は Gabb et al. 1997 の方式を採ると書いており、
  そこでは ε=4r（6 Å 以下）から ε=80（8 Å）へ ramp する。コードは裸の `q/r` を
  8 Å で hard cutoff。8 Å という数字自体が Gabb 由来である。
  Gabb 論文が `ref/` に無いため**未決着**。
- **部分電荷が CHARMM19 でない**。2002 は CHARMM19 と明記。しかも本リポジトリの
  `.ms` ファイルは最終列に ZDOCK 自身の CHARMM19 電荷を持ち、`io.py` が
  `PdbAtoms.charge_col` に読んでいるが**誰も使っていない**。

  | 原子 | 現行 LUT | `.ms` 列 |
  |---|---:|---:|
  | 主鎖 N | **+0.50** | **−0.15**（符号が逆） |
  | 主鎖 C | 0.00 | +0.60 |
  | CA | 0.00 | +0.10 |
  | LYS NZ | +1.00 | +0.75 |

  1KXQ 受容体 3908 原子のうち 2810 が異なり、**正味電荷 −19.10 e vs −6.57 e**。
  うち −11 e は「主鎖 O 496個 × −0.5 に対し主鎖 N 474個 × +0.5」という純粋な人工物。
  加えて (a) 終端 `else: id=8 (q=0)` が SER OG / THR OG1 / TYR OH / HIS 環 N /
  CYS SG / 主鎖 C をすべて 0 にする、(b) C 末端が `O`(−0.5) と `OXT`(−1.0) の
  両方を得て −1.5 e になる、(c) `is_first_n` が全原子リストに対する単一フラグなので
  **多鎖受容体では A 鎖の N 末端しか認識されない**。
- **PSC の半径スケーリングは未決着**。現行は両側とも素の vdW。2002 の GSC は
  core に `√1.5·r`、surface に `√0.8·r` を使い、2003 は「罰則パラメータは以前の
  GSC からそのまま持ってきた」と書いている。Fig.1(b) の説明は
  *"For clarity, we use a grid spacing that equals atom diameter"* と明示的に
  模式図であり、半径**スケール**を決定しない。`score.py:489-490` のコメントは
  Fig.1(b) が裏づける以上の確信を主張している。S3（surface 優先）と組み合わせると
  「薄く柔らかい皮 + 太い core」が一貫した組で、現行は**両軸とも逆**である。
- **`rotation_cone` は cone 内で一様でなく中心に 24.7 倍集中**
  （`rotation_grid.py:138-141, 162`）。Haar 密度は (1−cos θ) に比例するので
  θ 一様は中心を過剰重み付けする。docstring は逆のことを書いている。
  cone は `q*` を漏らすので、この 25 倍の集中は漏れの強さを 25 倍にする。
  `generate_decoys` 経由の全プールがこれを持つ。
- **`rotation_cone(q, 0)` は `IndexError`**（`:163`）。n=0 は「漏れ無し」という
  **正直な設定**なのに API が罰する。`validate_grid_spacing.py:231` はこのために
  ガードを足している。

#### 5.7.8 既報の記述に対する訂正（レフェリー監査より）

CLAUDE.md の方針に従い、**過去の記述を消さずに訂正を並置**する。

1. **§5.6.7 の効果表と `doc/scoring_function.tex:371-392`**：
   キャプション「注入なし・native 由来回転なし」は**5行中3行で偽**。
   行1-2 は `grid_spacing_validation_summary.json`＝`--n-cone 400` の既定実行で、
   しかも **20複合体**（12ではない）。行3 も cone 込み。cone 無しは行4-5 のみ。
   12複合体・cone 込みの値は 0.099 / 0.064。
2. **報告 :493-497 の「固定 pool の positive は**すべて**注入されたもの」**：
   「382.2個/複合体・FFT 由来 0.067個・60件中2件」は**shard 順の先頭 60 複合体**でしか
   再現しない。241複合体での正しい値は **377.87 / 11.25 / 11複合体 (4.6%)**
   （うち1件は FFT 由来 positive を 1,337 個持つ）。定性的な主張（positive の約3%が
   探索由来）は生き残るが、「すべて」は偽。
3. **報告 :738 の「離散化で失う865点のほぼ全て（966点）が衝突罰則」**：
   まず 966 > 865 で算術的に奇妙（有利項 +109 が相殺）。より重要なのは、
   9複合体の平均 −966 の **76% が 3wlb 1件**（8,698 中 6,614）であること。
   **中央値の複合体は総計 −150、衝突寄与 −113**。1件（6gxp）は離散化差が**正**（+111）。
   3wlb を除くと群平均は 衝突 −260 / 総計 −218。IFACE 列の「−8」は
   −124〜+98 の平均、すなわちノイズを数値として報告している。
4. **報告 :618「回転サンプリング不足は主因ではない（要因1を棄却）」**：
   強すぎる。(a) n=4 で、うち3件は基準密度で既に recall 0 なので動けるのは1件だけ。
   観測された 25%→0% はその1件が DockQ 0.237 で閾値を跨いだだけ。
   (b) ceiling は n=1,900 で既に 100% だったので、この ladder はランキング経路しか
   検定できない。(c) §5.6.11 は**同じ点数**でも被覆半径の良い格子が recall を
   動かすことを示している。棄却されるのは「回転**点数**の不足」であって
   「回転集合の**質**」ではない。
5. **報告 :608「失敗4件（6gxp/5jpq/7jhy/8b3p）」**：
   `rot_all12_n1900.csv` で `n_pos_out = 0` は **4zsh, 6gxp, 5jpq, 7jhy, 6wkk の5件**。
   **8b3p は `n_pos_out = 2`（最良 DockQ 0.237）で失敗ではない**。4zsh と 6wkk が
   欠けている。
6. **報告 :705-707 と :760-766「recall 58%→75%」**：
   2×2 の**最悪の隅**（一様＋pose-top-K）と**最良の隅**（Hopf＋NMS）を対にしている。
   同じ表の中で Hopf 単独は recall 58%→**42%**、NMS 単独は 58%→67% であり、
   2因子の符号は一貫していない。「修正前→現在」表も cone 汚染された n=20 の
   「前」（0.066）と cone 無し n=12 の「後」（0.53）を混ぜている。n=12 では
   これらは 1〜4 複合体の差で、9/12 の二項 95% CI は概ね 43〜95%。
7. **§5.6.7 の「IFACE 表は正しく、問題は符号統一の欠落だった（棄却した仮説）」と
   8行後の「IFACE 逆相関は未決着」は両立しない**。前者は論文の読み合わせであって
   測定ではないので、「棄却した仮説」に列挙すべきでない。
   また .tex の §Sign conventions が σ=−1 と主張する一方、§Desolvation は
   Eq.(11) を `e_ij = ln(n_ij/c)` と正の対数比で印字している。監査の結論は
   「**Mintseris の式は印字どおりなら +ln(n/c) だが、寄託された表はその符号反転
   （ACE 規約）である**」——コードは正しく、根拠の書き方が不正確。
8. **§5.6.11 の被覆半径**：16.8° → **≈18.5°**、12.2° → **≈13.9°**、5.5° → **≈6.2°**。
   使われた n_probe は 8000 ではなく **4000**。−35% の相対主張は維持される（M9）。
9. **§5.6.11 の「いずれも微分可能テンソルとして露出済み」**：D は誤り（L2）。
10. **§5.6.7 の「α=1.0 に修正」**：診断スクリプトのみ。学習・デコイ生成経路は
    0.01（上限 0.1）のまま（L1）。
11. **報告 :504「格子は主因ではなかった」**：これを支える 1.2 Å 再試験は
    **有利項が欠落した壊れたスコア**の上で行われた。PSC 修正後に 3.0 Å で
    再試験していない。正しくは「欠陥スコア上では 1.2 Å でも recall 0% だった。
    修正後のスコアでの格子依存性は未測定」。
12. **未報告の null run**：`gv_alpha1.0`（recall 0.0%、best 0.099）、
    `gv_alpha0.1`（recall 8.3%、**s@1 = 8.3%** ＝ 最終「完全準拠」構成と同じ top-1 率を
    罰則のみのスコアで達成）、`gv_pair_a1`（`IFACE_PAIR_OFFSET` 二重計上版、recall 0.0%）
    が `data/scaling/` に保存されているのに報告に無い。CLAUDE.md の
    「失敗・null も記録する」に反する。特に α=0.1 の結果は
    「success@1 は不動」の文脈として重要。
13. **§5.1-§5.5, §6, §7 に PSC 欠陥の注記が無い**。この caveat は §5.6.7 の中にしか
    無く、§7 項目1 は今も「ランダム分割の 71% は leakage 由来と判明…**確定した**」と
    書いている。§7 で読み終える読者は誤解する。加えて **71.0% / 89.2% の
    出力ファイルはリポジトリのどこにも保存されていない**（§5.4 と違い追跡不能）。
14. **追跡性の欠落**：§5.6.2 の VRAM 表、§5.6.3 の選択件数、§5.6.6 の「361 s」に
    対応する保存ログが無い。§5.6.3 の「2,900件すべて成功」に対し
    `prep_main.log` は "attempted 2886 ok 2886"（残り14件は記録されていない2回目の
    パス＝R1 の原因）。:840 の「95 passed」は陳腐化。

#### 5.7.9 結論と修正の優先順位

**この監査が §5.6 の結論に与える影響**:

- **「学習前段階は達成した」は取り下げる。** 少なくとも S1〜S5 は、既定経路で
  実行されるスコア関数そのものの誤りである。§5.6 の「完全論文準拠」という表現は
  正当化されない。
- **§5.6.7 の中心的発見（PSC 有利項の欠落が探索失敗の主因）は生き残る。**
  M7 の独立測定（探索由来 pose に positive を持つ複合体が 4.6% しかない）が
  これを裏づけている。
- **§5.6.11 の項別分解（表 A/B の fav・clash 列）は無効**（M1, M2）。
  表 C と total 列は有効。
- **§5.6.7 の AUC 群（0.998 / 0.127 / 0.873）は無効**（M3, M4）。
- **§5.4/§5.5/§5.6 の success@K の差はすべて有意でない**（M8）。しかも
  検出力のある指標は「学習後のほうが悪い」と言っている。
- **§5.6.11 の 75% recall と 2×2 表は汚染されていない**（S7 の sentinel は
  DockQ 0.004〜0.057 で閾値を跨がない）が、「top-2000」という表記は誤り。

**修正順序**（依存関係と影響の大きさで）:

| 順 | 項目 | 理由 |
|---|---|---|
| 1 | **S1 + S2**（ELEC マスクと符号） | 2箇所・数トークン。ELEC 信号の 80〜93% を回復し、β と電荷の勾配を復活させる。同時に直さないと大きさか向きが誤ったまま |
| 2 | **S3**（surface 優先） | 1箇所。既定経路の全スコアに効き、S_PSC が +73% 変わる。ρ の議論はこの後でないと意味がない |
| 3 | **S5**（spacing 不整合） | 全キャッシュプールと §5.6 の全数値を汚染している。修正後はプール再生成が必要 |
| 4 | **M1 + M3 + M4 + M5**（測定スクリプト4件） | いずれも数行。直さないと修正の効果を正しく測れない |
| 5 | **S4**（floor → nearest） | 効果は最大級（IFACE 33%）だが Julia 参照テストを壊すのでフラグ化・再測定が必要 |
| 6 | **L1 + L5 + L6**（α clamp・NaN・prior の位置） | 学習を回す前に必須 |
| 7 | **S7 + S6**（sentinel・`sc_rho` 未伝播） | 小さいが表記と ρ 掃引の正しさに効く |
| 8 | **R1 + R2 + R3 + R4**（再現性） | 結果は変わらないが、これを直さないと次の実験が前の実験と接続できない |
| 9 | **S8**（padding と `orient` の符号つき質量） | 現時点で勝者は汚染されていないが、7/300 で native が箱から出ており、`|centroid|` が 22,707 Å の例がある |
| 10 | **S9 + S10 + S11**（DockQ 対称性・近似・n_native=0） | 評価の意味を変える。ただし「学習前段階」の判定基準そのものを動かすので、変更前後を両方報告する必要がある |

**次の実験**（この監査から正当化されるもの）:

1. S1/S2/S3 修正後に §5.6.11 の 12複合体を **cone 無し**で再測定し、
   項別寄与を M1 修正済みスクリプトで取り直す。α=1.0 と α=0.01 の両方で。
2. ρ ∈ {3.0, 3.5} × surface/core 優先の 2×2 を測る（S3 と論文相違が交絡している）。
3. S4 をフラグで入れ、Julia 参照テストを「旧挙動」テストに改名したうえで
   recall/ceiling を比較する。
4. S9（DockQ 対称性）を実装し、TEST 241件の positive 数がどう変わるかを測る。
   §5.5.1 の仮説2（真の positive を hard negative として掘っている）の直接検定になる。
5. R1 を修正したうえで `usable_ids.txt` を凍結し、hash を `split.json` に記録して
   scaling 実験を再実行する。

**この監査の限界**（再掲・重要）: 効果量の多くは 1〜300 複合体の測定であり、
コーパス平均ではない。監査体同士の指摘は独立だが、同一の誤読を複数体が共有した
可能性は排除できていない。特に「論文がこう書いている」という主張は、
`ref/` に無い補足資料（Mintseris Supporting Table I/II、Gabb et al. 1997）に
依存する部分について**検証できていない**。

---

### 5.8 §5.7 監査で確定したバグの修正（2026-07-25）

§5.7 の S1〜S6 と、それ以外で「明確に誤りで、直し方に判断の余地がない」項目を
修正した。**評価の再測定はまだ行っていない**ので、以下は「コードがどう変わったか」
と「単体で確認できる効果」だけの記録である。§5.7 で無効と判定した数値
（表 A/B の fav・clash 列、AUC 群、success@K の有意性）は**まだ更新されていない**。

#### 5.8.1 修正一覧

**スコア関数本体**

| 監査 | ファイル | 変更 |
|---|---|---|
| **S1** | `score.py`, `search.py` | ELEC の open-space マスクを `(re==0)&(im==0)` → `im<=0`。占有チャネルは `Im` であり、`Re` は PSC 書き換え後「`radius+D` 以内の原子数」に意味が変わっていた |
| **S2** | `score.py` (`ELEC_LIGAND_SIGN`, `ligand_partial_charge`), `search.py` 3箇所 | リガンド側の電荷堆積を `+q` → `−q`（Chen et al. 2003 Eq.(2) の `Im[L_PSC+ELEC] = −1 × charge`）。3経路が必ず同じ符号を使うようヘルパー1本に集約 |
| **S3** | `score.py` `sc_encode` | surface/core の優先順位を反転。`shell_b = shell > 0; core_b = (core>0) & ~shell_b` |
| **S4** | `spread.py` `_nearest_cell_indices` | `ceil(u)-1`（floor）→ `round(u)`（真の最近接）。`ZDOCK_LEGACY_FLOOR_BINNING=1` で旧挙動に戻せる |
| **S5** | `dataset.py`, `run_pinder_scaling.py`, `run_pinder_hardneg.py` | `generate_decoys` の既定 spacing を 3.0 → `SC_REFERENCE_SPACING`(1.2)。両スクリプトに `--spacing` を追加し、**探索と特徴量化の両方**に同じ値を通す。`lig_xyz_for_grid=prot.lig_ref` も明示 |
| **S6** | `search.py` | `_build_ligand_sc_grids_batch` に `sc_rho` を追加し `docking_search_sc` から伝播（Eq.(3) の `Im[R]=Im[L]` を回復） |
| **S7** | `search.py` 両探索関数 | 返り値を `isfinite` で切り詰め、`-inf` 事前充填が pose として外に出ないように |
| **S11** | `dockq.py` | `n_native==0` のとき 2項で正規化（正解 pose が 0.667 でなく 1.0 になる）。`DockQComponents.n_native_contacts` を追加して呼び出し側が検出・除外できるように |

**測定スクリプト**

| 監査 | ファイル | 変更 |
|---|---|---|
| **M1** | `diagnose_failures.py` | 有利項の分離を `rho=0` → `rho=1e-6`。`rho=0` は `(im<=0)` マスクも解除して受容体内部の有利項を漏らしていた（+29%）。生の iface 行列のインライン再実装も `iface_score_matrix()` に置換 |
| **M3** | `diagnose_score_terms.py` | `iface.view(12,12).T` → `iface_score_matrix(iface)`。`auc_NEG_IFACE` → `auc_FLIPPED_IFACE` に改名（ラベルが入れ替わっていた） |
| **M4** | `diagnose_score_terms.py` | AUC に midrank の同点補正を追加。**検証: 全同点で 0.700 → 0.500、整数同点データで scipy と 6桁一致（0.377193）** |
| **M5** | `eval_test_pool.py` | `torch.arange` / `torch.zeros` に `device=scores.device`（CUDA 既定で例外を投げていた） |
| **M9** | `rotation_grid.py`, `validate_grid_spacing.py` | `covering_radius_deg` の返り値に `max_deg_lower_bound` を追加し docstring で下方バイアスを明記。既定 `n_probe` 20000→200000、既定 `seed` 0→1（grid と probe の seed 衝突回避）。printer は `max>=` 表記に |
| **M10** | `validate_grid_spacing.py`, `eval_search_ceiling.py` | `--n-cone` 既定を 400 → **0**（正直な条件）。非零のときはバナーに警告を出す。`--cov-probes` を追加 |
| — | `eval_search_test.py`, `eval_search_ceiling.py`, `compute_grid_sizes.py`, `build_decoy_dataset.py` | `--spacing` 既定 3.0 → 1.2（3スクリプトを跨いで比較できるように） |

**学習経路**

| 監査 | ファイル | 変更 |
|---|---|---|
| **L1** | `run_pinder_scaling.py`, `run_pinder_hardneg.py` | `--alpha0` を CLI 化し、`--alpha-max` の既定を `10 × alpha0` に。`alpha_max < alpha0` を assert。**初期値が実行可能領域の外に出ることが構造的に不可能になった** |
| **L5** | 同上 | `s.detach().std()` → `std(unbiased=False)`。1 pose の複合体で NaN が出て Adam が α と iface を恒久的に汚染する経路を封じた |
| **L6** | `train.py` | `"combined"` 分岐の `loss_param_prior` をタンパク質ループの外へ（epoch につき1回）。`run_pinder_scaling.mean_objective` と一致 |
| — | 同 2 スクリプト | `loss_basin` / `loss_margin_hard_negatives` に `positive_threshold=args.dockq_thr` を伝播（既定 0.23 に固定されていた） |

**データ衛生**

| 監査 | ファイル | 変更 |
|---|---|---|
| **R2** | `run_pinder_scaling.py` | voxel フィルタを **fail-closed** に。`ok` なのに voxel 表に無い id は `SystemExit` |
| **R3** | 同上 | prep cache の欠損／破損を `SystemExit` に。従来は `stage="cache_load"` として黙ってスキップされ、`len(fit_ids)==n_fit` を通したまま N−1 で走っていた |
| **R9** | `aggregate_scaling.py` | 構造検査に失敗したら `SystemExit(1)`。`split.json` を「完了 round が無い run」でも収集するよう順序変更 |
| — | `run_pinder_hardneg.py` | `torch.load(..., weights_only=True)` |

**その他（バグではないが誤解を招くもの）**

- `search.py` モジュール docstring の `real(G) − imag(G)` を `real(G)` に訂正
  （Eq.(4) は実部のみ）。`docking_score_sc_direct` も同じ式に揃え、
  `tests/test_search.py` の参照値も訂正した（テスト側が production と別の式を
  検証していた）。
- `score.py`, `search.py` の陳腐化した GSC 由来コメント（「surface が core を
  上書き」等）を実装に合わせて書き直し。
- `rotation_cone(q, 0)` の `IndexError` を修正（n=0 は「漏れ無し」という正直な
  設定なので API が罰してはいけない）。docstring の「shell を過剰重み付け」を
  「**中心**を 25 倍過剰重み付け」に訂正（実測値つき）。
- `diagnose_failures.py` の死んだ CLI（`--n-near`, `--cone-deg`）と未使用 import、
  `search.py` の未使用 `lig_partial_q` を削除。

#### 5.8.2 修正の効果（単体測定）

1KXQ（受容体 3908 原子・リガンド 916 原子、spacing 1.2 Å、float64）:

```
ELEC マスク  OLD (re==0 & im==0): 1,937,675 cells  sum|V| =  1760.3
ELEC マスク  NEW (im<=0):         1,985,033 cells  sum|V| =  8707.8   (4.95x)
```

**ELEC の信号が 4.95 倍に回復**した（監査は 1KXQ で「−80%」＝ 5.0 倍と報告して
おり一致）。リガンド堆積の符号も Eq.(2) どおりになった（q=+1.00 → 堆積 −1.00）。

接触した pose での勾配（従来は β と charge が**厳密に 0**）:

```
alpha  -116911.2500
rho    -128863.0000
iface  norm 2998.8586   nonzero 133/144
beta        +0.798291   <- 修正前は接触 pose で常に 0
charge norm   29.856371   nonzero 10/11
```

**β と 11 個の電荷が学習可能になった**（S1 の直接の帰結）。ただし §5.7 の
SUSPECTED 項目のとおり、`β·S_ELEC` の絶対値が `S_PSC` に比べて十分かは未検証で、
電荷の学習に投資する前に大きさの較正が必要である。

sentinel pose の除去（S7）:

```
n_rot=12, ntop=50 を要求  ->  返る pose 数 12（修正前は 50、うち 38 が -inf）
すべて有限、回転 index は 12 個すべて相異なる
```

#### 5.8.3 検証

- **`uv run pytest -q` : 112 passed, 1 skipped, 1 deselected**（4分55秒）。
- S4 により Julia 参照の nearest binning テスト2件が落ちたので、
  `legacy_floor_binning` fixture で**旧挙動を明示的に有効化する legacy テスト**に
  分離した（移植の忠実性は今も検証されている）。同時に新挙動を検証する
  `test_nearest_cell_is_actually_nearest` を追加した——格子点の両側 ±0.59 Å が
  同じセルに、±0.61 Å が隣のセルに割り当たること（旧 floor 実装ならこれは落ちる）。
- AUC の同点補正は scipy `mannwhitneyu` と 6 桁一致することを両実装で確認。
- 全 Python ファイルの `ast.parse` を確認。`doc/scoring_function.pdf` を再ビルド。

#### 5.8.4 文書の訂正（`doc/scoring_function.tex`）

- 「$\mathrm{core}$ taking precedence」→「$\mathrm{surf}$ taking precedence」
  （原文引用つき）。
- 「ρ, D, α, β, e_ij, q_m はすべて微分可能」→ **D を除外**し、なぜ勾配が
  恒等的に 0 なのか（ハードな距離判定 → bool 選択）と実測値を明記。
  q_m が CHARMM19 でない件も注記。
- ρ=3.5 の節に「これは PSC 論文の罰則値（−9/−27/−81）ではない」という caveat を
  追加。Eq.(2) は ρ=3.5 と**同時に**有利項を 1.334 に上げており、罰則:報酬比は
  9.0 vs 9.18 でほぼ同じだが、本実装は 12.25 になっている。
- 被覆半径の表の「max」が 4000 プローブの**標本最大値＝下界**であることを明記し、
  真値（≥18.5°/≥13.9°/≥6.2°）を併記。−35% の相対主張は維持されることも明記。
- 「Measured effect」表のキャプションに**訂正段落**を追加：上3行は
  `--n-cone 400` の cone 汚染条件で、うち2行は n=20（12ではない）。各行に
  条件をラベルした。

#### 5.8.5 意図的に修正しなかったもの（判断が要る、または方針決定が必要）

| 監査 | 内容 | 保留理由 |
|---|---|---|
| **S8** | `orient()` の符号つき質量による重心発散（最悪 22,707 Å）と、padding にリガンドの x 方向の広がりを使う件 | 直すと `lig_ref` の参照フレームが変わり、**prep cache 2900件が全部無効**になる。native が箱から出るのは 7/300 で、現時点で上位 pose は汚染されていない。cache 再生成のコストを見てから判断すべき |
| **S9** | DockQ の対称性処理（コーパスの約15%） | 評価指標の**定義**が変わる。導入するなら変更前後の両方を報告する必要があり、独立した実験として扱うべき |
| **S10** | DockQ 原子レベル近似（本実装 0.23 ⟺ 正典 0.332） | 同上。閾値の再較正か正典実装かの方針決定が要る |
| **論文相違** | ρ=3.5 vs 3.0、`Re[L_PSC]` のカウント vs 1、CHARMM19 電荷、距離依存誘電率 | いずれも「どちらの定式化を採るか」の選択であり、実験で決めるべき（§5.7.9 の次実験2に含めた）。文書には相違として明記した |
| **L7** | `refine_poses_gradient` が PSC に勾配を持たない | trilinear PSC の実装が要る。当面は「IFACE+ELEC のみの精密化」であることを認識して使う |
| **R1** | `prep_manifest.jsonl` が再生成できない | 修正方法（cache hit でもレコードを書く／glob をやめる）は自明だが、**既存の manifest を作り直すと nested 構造が変わる**。実行するタイミングを選ぶ必要がある |
| **R4** | 分割定義ファイルが gitignore されている | commit するかは判断を仰ぐべき（1.1 MB 未満） |

#### 5.8.6 次にやるべきこと

修正されたコードの上で §5.7.9 の再測定を行う必要がある。**現在レポートに載っている
探索・評価の数値は、すべて修正前のコードで得られたものである。**

1. §5.6.11 の12複合体を `--n-cone 0` で再測定（cone 既定が 0 になったので
   コマンドはそのままで正直な条件になる）。
2. `diagnose_failures.py`（M1修正済み）で項別分解を取り直し、
   §5.6.11 の表 A/B を差し替える。表 A は M2 のとおり定義自体を変える必要がある。
3. `diagnose_score_terms.py`（M3/M4修正済み）で AUC を取り直し、
   §5.6.7 の項目5・6 を差し替える。
4. `eval_test_pool.py` を再実行して §5.6.7 項目1 の AUC を再生成（M6）。
5. マイニングプールは S5 により 1.2 Å で作り直しになる。

---

## 6. 解釈と注意
- baseline が 0% なのは、候補集合が native 近傍 cone（positive）と FFT デコイ（多くは
  非 native）を混ぜており、デフォルト α=0.01・生 IFACE 行列がこれらの near-native を
  上位化できないため。**ZDOCK の実運用性能ではなく、この自己生成デコイ集合上での
  デフォルトパラメータのランキング**である点に注意。意味のある信号は before/after と
  train≈test の一致（汎化）。
- 本結果は **固定候補集合の再ランキング**の汎化であり、FFT 探索自体の改善ではない
  （デコイはデフォルト param で生成済み）。
- DockQ は原子レベル近似で、閾値は正典 CAPRI と厳密一致しない（RMSD を併記）。
- bound(holo) redocking の結果（GPT の「事前学習」段階に相当）。

---

## 7. 限界と次段階
1. **[実施済 → 5.4]** PINDER の interface-deleaked split で測り直した結果、ランダム
   分割の 71% は leakage 由来と判明。deleaked test では 1〜4% しか出ず、**汎化が本質的
   課題**であることが確定した。次に必要なのは (a) より強い正則化／実効パラメータ削減、
   (b) train 複合体数の大幅増（数百→数千クラスタ）、(c) 損失の見直し（basin/margin が
   train interface に過適合しやすい）。
2. **[実施済 → 5.5]** hard-negative mining は難例の採掘には成功したが、3 seedすべてで
   deleaked top-K改善は0ポイント。単独では汎化問題を解決しない。
3. **unbound/predicted 入力での fine-tuning**（GPT §2）。実用ドッキングでは bound は
   入力されないため、apo→holo の相対変換を教師にする段階が必要。
4. charge_score は本実験で凍結（α + IFACE のみ学習）。次段は charge の双線形特徴
   `c^T M[f] c` を抽出して同時学習可能。

---

## 8. 再現手順
```bash
uv sync
uv run pytest -q                         # 実装の健全性（95 passed）

# 1KXQ（ZDOCK .pdb.ms 形式、tests/data 同梱）
uv run python scripts/build_decoy_dataset.py --proteins 1KXQ \
    --output data/decoys.h5 --device cuda \
    --n-random-rot 3000 --n-cone 400 --ntop 2000
uv run python scripts/run_experiment.py --dataset data/decoys.h5 \
    --loss combined --epochs 300 --device cuda

# DB5.5（外部ダウンロード → 7 GPU 並列生成 → 汎化評価）
curl -sSL -o external/benchmark5.5.tgz https://zlab.wenglab.org/benchmark/benchmark5.5.tgz
tar xzf external/benchmark5.5.tgz -C external
# 複合体コードを 7 分割し、各 GPU で build_decoy_dataset.py --format pdb を並列実行
# （data/shards/shard{0..6}.h5 を生成）
uv run python scripts/run_db55.py --shards 'data/shards/shard*.h5' \
    --device cuda --epochs 2000 --lr 0.05

# PINDER interface-deleaked split（leakage 検証, 5.4）
export PINDER_BASE_DIR=$PWD/external/pinder
uv pip install pinder
# test=PINDER-S(250) / train=別クラスタ300 の id リストを data/ に書き出し済み
uv run python scripts/gen_pinder_shards.py --ids-file data/pinder_test_ids.txt \
    --tag test --out-dir data/shards_pinder --gpus 0,1,2,3,4,5,6
uv run python scripts/gen_pinder_shards.py --ids-file data/pinder_train_ids.txt \
    --tag train --out-dir data/shards_pinder --gpus 0,1,2,3,4,5,6
uv run python scripts/run_pinder.py --device cuda --epochs 3000 \
    --lr 0.01 --lambda-prior 0.1

# Iterative hard-negative mining（5.5、seed 0/1/2 を別GPUで実行）
CUDA_VISIBLE_DEVICES=0 uv run python -u scripts/run_pinder_hardneg.py \
    --seed 0 --rounds 4 --epochs-per-round 1500 --device cuda
```

生成物（`data/`, `external/`, `logs/`, `*.h5`）は `.gitignore` 済み。上記スクリプトで
再現できる。
