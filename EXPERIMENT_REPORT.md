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
