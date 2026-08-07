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

### 5.9 パラメータ学習への準備（2026-07-25）

§5.8 の修正後、**hard-negative mining ループには入らず、round 0 の単一ラウンド学習**に
進むための設計・実装。学習の実行自体はまだ行っていない（スモークのみ）。

#### 5.9.1 修正後コードでの項の大きさ較正（測定）

**問い**: ELEC が復活し PSC の符号・優先順位が直った今、3項の相対的な大きさは
どうなっているか。これが α₀ と「何を学習すべきか」を決める。

1KXQ、h=1.2 Å、Hopf nside=3（1944回転）の探索**上位500 pose**。
これは「ランカーが実際に識別しなければならない母集団」である。

| 項 | mean | std（**ランキングを決めるのはこちら**） |
|---|---:|---:|
| S_PSC | 364.6 | **102.1** |
| S_IFACE | 337.6 | **104.4** |
| β·S_ELEC (β=3) | 0.14 | **1.00** |

**測定にもとづく結論2つ**:

1. **α₀ = 1.0 が正しい。** PSC と IFACE の spread を等しくする α は **1.023**。
   `doc/scoring_function.tex` の「論文準拠 1.0」（Chen et al. 2003 Eq.(2) には α が
   無く PSC のスケールは ρ が決める）が実測でも支持された。旧来の α=0.01 は
   PSC の寄与を IFACE の 1% に潰していた。
2. **ELEC は S1 修正後もランキングにはほとんど効かない。** spread が IFACE の
   **約1%**。等しくするには β≈312 が必要で、β=3 は論文値（Chen et al. 2003 p.82）。
   したがって **β と 11電荷は勾配が復活しても今回の学習対象から外す**。
   効果が観測できる見込みが薄く、β の再較正は別の（論文からの逸脱を伴う）判断である。

**限界**: 1複合体・1母集団での測定。コーパス平均ではない。

#### 5.9.2 学習対象: α + ρ + IFACE 144 = 146 パラメータ

§5.7-L2 のとおり D は微分不可能、§5.9.1 のとおり β・電荷は寄与が小さい。
残る ρ は §5.6.11 が「ρ⁴ の衝突罰則が正解 pose だけを潰している」と結論した
**感度最大の自由度**だが、`f.sc` に焼き込まれていて構造的に学習できなかった。

**解決**: キャッシュを ρ 非依存の4スカラーに分解する（`--psc-decompose`、既定 on）。

```
sc: (F,) → (F, 4) = (c_pair, n_ss, n_sc, n_cc)
S_PSC = c_pair − k·(n_ss ρ² + n_sc ρ³ + n_cc ρ⁴)
```

`Im` は `{0, ρ, ρ²}` しか取らないので、`Σ Im_R·Im_L` は重なりの種類別セル数
（surface-surface / surface-core / core-core）の一次結合に厳密に分解できる。
これらの計数は ρ に依存しないので、**1度作ったキャッシュが全ての ρ に対して有効**。

検証: ρ=3.5 で作ったキャッシュから ρ∈{1.0, 2.0, 3.5, 5.0} で再構成し、
各 ρ で直接再スコアした値と比較（1KXQ、float64、5 pose）:

| ρ | max\|直接 − 再構成\| |
|---|---:|
| 1.0 | 6.2e−11 |
| 2.0 | 1.2e−10 |
| 3.5 | 1.4e−09 |
| 5.0 | 5.4e−09 |

丸め誤差のみ。テスト `test_score_decomposition_matches_full_score` を
ρ∈{1.0, 3.5, 6.0} でパラメータ化し、`test_rho_receives_gradient_through_cached_features`
を追加した。

**副産物として §5.7 の PSC #7 / grid #3（`sc_cell_volume_factor` を S_PSC 全体に
適用している）も解消した**。`psc_score_from_terms` は体積因子を**衝突チャネルにのみ**
掛ける。`c_pair` は原子ペア数で格子間隔に依存しないので、1つのスカラーで両方を
補正することはできない。h=1.2 Å では因子が厳密に 1 なので現行の値は変わらない。

#### 5.9.3 候補プールの作り直し: 到達可能な positive

§5.7-M7 で、固定 TEST プールが **baseline スコアに対する敵対的集合**であり
positive は全て「探索が到達できない pose」であることが判明した
（95.4% の複合体が探索由来 pose に positive を持たず、72.6% で first-hit rank が
ちょうど 2001）。同じレシピで作られた学習プールも同じ欠陥を持つ。

**新レシピ** (`generate_pool_reachable`, `--pool reachable` 既定):

```
quats = Hopf(nside=3)                    # cone なし = q* を候補集合に漏らさない
negatives = docking_search の上位 N
positives = { (q_i, t) : q_i ∈ Hopf格子で q* に最も近い K個(既定8),
                         t  ∈ snap(t*) の ±2 セルの格子点(125通り),
                         DockQ ≥ 閾値 }
```

positive を**探索自身の格子上**で列挙するので、学習分布と配備分布が一致する。
`(回転index, 整数並進セル)` で探索結果との重複を除去。

**到達可能な positive が1つも無い複合体は fit から除外し件数を記録する** ——
これは「どれだけパラメータを合わせても success が上がらない上限」そのものである。

実測（prep cache の小型複合体4件、Hopf 1944回転、±2セル×8回転=1000候補）:

| 複合体 | neg | pos候補 | pos | 最良DockQ |
|---|---:|---:|---:|---:|
| 4mjh__A1_P04792--4mjh__B1_P04792 | 1500 | 921 | 860 | 0.795 |
| 6uyo__A1_P63165--6uyo__B1_P29590 | 1500 | 944 | 922 | 0.781 |
| 6xfk__A1_P0CL43--6xfk__B1_P35672 | 1500 | 679 | 677 | 0.725 |
| 6jwm__A1_Q99619--6jwm__B1_P35228 | 1500 | 808 | 808 | 0.813 |

最良 DockQ が 1.0 でなく 0.72〜0.81 なのは、回転格子と並進格子の離散化による
**正直な上限**である（従来は正確な t* に注入していたので 1.0 が出ていた）。

#### 5.9.4 TEST プールも作り直しが必要（`scripts/build_test_pool.py`）

`data/shards_pinder/test_feats.pt` は二重に無効:
(a) 特徴量 `(S_SC, T, S_ELEC)` が修正前のスコア関数で計算されている
    ——パラメータではなく**特徴量**なので重み付けし直せない、
(b) pose が旧レシピ（cone 汚染 + 到達不能な注入）。

新スクリプトは `mine_complex` を経由するので、**TEST プールと fit プールが
1つのコード経路・1組のフラグで作られる**。差が出たらそれは意図的な引数である。
`load_test_feats` は旧形式 `(F,)` のキャッシュを ρ 学習時に**ハードエラー**にする。

TEST 8件でのスモーク（baseline パラメータ）:

| 複合体 | n | pos | **posのうち探索由来** | 最良DockQ |
|---|---:|---:|---:|---:|
| 3k1i__D1_O25709--3k1i__A1_O25448 | 2500 | 974 | **0** | 0.806 |
| 3qhy__B1_O87916--3qhy__A1_Q93T42 | 2500 | 998 | **0** | 0.595 |
| 3shg__A1_E6Z0R3--3shg__B1_E6Z0R4 | 2500 | 1000 | **0** | 0.708 |
| 8e3g__B1_P43026--8e3g__A1_P12643 | 2500 | 438 | **0** | 0.529 |
| 1zt2__A1_Q97Z83--1zt2__C1_Q9UWW1 | 2500 | 984 | **0** | 0.656 |
| 5j1l__B1_O26069--5j1l__A1_O26068 | 2500 | 1000 | **0** | 0.699 |
| 8b3s__A1_O31597--8b3s__B1_A0A164W157 | 2500 | 1000 | **0** | 0.644 |
| 3vf0__B1_Q8IY67--3vf0__A2_P18206 | 2500 | 999 | **0** | 0.584 |

**到達不能な複合体は 0/8**（＝上限は問題でない）が、
**baseline パラメータの探索が上位1500に near-native を入れられた複合体も 0/8**。
これが「今どこにいるか」の正直な測定である。再ランキングの課題は実在する。

#### 5.9.5 DockQ 対称性（S9）: パイロットでは homodimer を除外

`is_homodimer()` を追加（PINDER id の両半分の UniProt を比較）。
`--exclude-homodimer`（既定 on）。理由は §5.7-S9 の実測——
非C2 の役割入替え pose が DockQ 0.005〜0.020 と採点され、
**「正しい pose を下位化せよ」という向きのラベルノイズ**になる。
対称性対応 DockQ は評価指標の定義変更なので独立した実験として扱う。

#### 5.9.6 実装（`Params` バンドル）

`(alpha, iface)` が11箇所の関数シグネチャを貫通していたため、
`Params(alpha, rho, iface)` に束ねた。今後 β・電荷を足すときも
シグネチャの再配線が不要になる。`rho` は `[--rho-min 0.5, --rho-max 9.0]` に
clamp する（**ρ=0 では `(im<=0)` マスクが反転して S_PSC が不連続に飛ぶ**ため。
§5.7-L3 の実測: S_SC(0)=259.0 vs 4次式の外挿 152.0）。
境界に張りついたら `rho_at_bound` / `alpha_at_bound` を記録し警告を出す。

`run_pinder_scaling.py` を分岐させずに拡張した。重複はまさに S5（spacing 不整合）の
原因だったため。旧レシピは `--pool legacy` で残してある。

#### 5.9.7 コスト（実測）と既定値の変更

1.2 Å は 3.0 Å の 15.6 倍のセル数になる。1KXQ（2.0M voxel ≒ コーパス中央値）:

| spacing | 格子 | voxel | 1944回転の探索 | ピークVRAM |
|---|---|---:|---:|---:|
| 3.0 Å | 51×50×56 | 0.14M | 2.4 s | 1.4 GiB |
| 1.2 Å | 123×121×135 | 2.01M | **14.6 s** | **7.9 GiB** |

TEST 8件のスモークでは `--rot-chunk 8`（3.0 Å 時代の既定）で**ほぼ全件が OOM** し、
救済ラダーで回復した。既定を **`--rot-chunk 2` / `--frame-chunk 100`** に変更。

**`--max-grid-voxels 2000000` は 3.0 Å の voxel 表に対する閾値なので、
1.2 Å で `compute_grid_sizes.py` を回し直して設定し直す必要がある**
（`compute_grid_sizes.py` の既定 spacing も 1.2 に修正済み）。
§5.7-R7 のとおり、この足切りは学習コーパスだけを縮めて TEST は縮めないので、
サイズのドメインシフトとして報告する必要がある。

#### 5.9.8 エンドツーエンドのスモーク（N_fit=8、200ステップ、8複合体TEST）

```
round 0 (no mining) | N_fit=8 | mean pool=2430
    (pos 867 [search 31 / enumerated 835] / neg 1564)
  alpha=0.7979  rho=3.3035  ||dIface||=0.712  val_loss=5.9257  peakGPU=9.0 GiB
```

- **3つのパラメータ群すべてが動いた**。特に **ρ が 3.5 → 3.30 と下がった**——
  §5.6.11 が予測した「剛体近似の食い込みに対して衝突罰則が強すぎる」方向である。
  ただし N=8・200ステップなので、これは**配線が生きていることの確認**であって
  科学的な結果ではない。
- 学習プール側では**探索自身が平均31個の positive を見つけている**（TEST の 0 と対照的）。
- `--alpha-lr 1e-5` は α₀=0.01 前提の値で、α₀=1.0 に対して 100倍小さすぎた
  （α が 1.0→0.998 しか動かなかった）。**`alpha_lr = 1e-3 × alpha0`** を既定にし、
  相対ステップを保つようにした。修正後は 1.0→0.798。

#### 5.9.9 検証

`uv run pytest -q` → **115 passed, 1 skipped, 1 deselected**。
新規テスト2件（ρ 再構成のパラメータ化、ρ の勾配到達）を追加。

#### 5.9.10 次のステップ（実行前に必要なこと）

1. **1.2 Å で voxel 表を再生成**し `--max-grid-voxels` を設定し直す。
2. **cone なし・Hopf 格子での到達可能 ceiling** を fit コーパスのサンプルで測定。
   到達不能な複合体の割合が学習の上限を与える。
3. `build_test_pool.py` で TEST プール（250件）を作り直す。
4. N=220・seed 1つで round 0 学習を通し、健全なら N と seed を広げる。
5. 評価は §5.7-M8 のとおり **per-complex AUC の paired Wilcoxon** を主指標にする
   （success@K は床に張りついていて全ての差が n.s.）。
   加えて `eval_search_test.py` の end-to-end 再探索。

---

### 5.10 学習前の3つの前提測定（2026-07-25〜26）

§5.9.10 に挙げた3項目の実行記録。**うち1つで新しい根本原因バグが見つかり、修正した**。

#### 5.10.1 到達可能 ceiling — と、そこで見つかった `orient()` の重心発散

**問い**: FFT 探索が返せる pose は `rotate(lig_ref, q_i) + k·spacing` の形しかない。
その集合の中に閾値を超える pose が無い複合体では、**どんな (α, ρ, e_ij) でも成功しない**。
その割合はいくつか。

**方法** (`scripts/reachable_ceiling.py`): q* に最も近い K 個の格子回転 ×
snap(t*) の ±R セル格子点を列挙して DockQ の最大値を取る。探索不要
（pose 構築 + ラベルのみ）で約 1 秒/複合体。fit コーパス先頭 300 件、
Hopf nside=3（1944回転）、h=1.2 Å、fp32。

**結果（修正前）**: **292/300 = 97.3%** に到達可能な positive があった。
平均最良 DockQ 0.668、中央値 0.703。列挙設定の掃引:

| K回転 × ±Rセル | 閾値超え | 平均最良DockQ |
|---|---:|---:|
| 1×0 | 93.3% | 0.528 |
| 1×1 | 96.0% | 0.610 |
| 4×2 | 97.0% | 0.666 |
| 8×2（本番設定） | 97.3% | 0.668 |

4×2 でほぼ飽和しており、**律速は列挙の広さではなく回転格子の粗さ**である
（q* に最も近い格子回転は中央値 9.73°、p95 13.65°、最大 15.25°）。

**到達不能だった8件がすべて `|centroid(lig_ref)|` の大きい複合体だった**
（中央値 78 Å、到達可能群は 6 Å）。これは §5.7-S8（`orient()` が符号つき
IFACE 質量で重心を取り、Σw≈0 で発散する）の帰結である。

**機構**（測定）: 探索は `lig_ref` を**原点まわりで**回転させるので、
ずれた重心が梃子となり、回転格子の離散化誤差を並進誤差に増幅する。

| 複合体 | \|重心\| | 最近接格子回転の誤差 | それが生む重心のずれ |
|---|---:|---:|---:|
| 4ez1__B1_Q8WSF8--4ez1__F1_P69657 | 22,707.6 Å | 13.91° | **2,704 Å** |
| 7tgh__YA1_Q22DR7--7tgh__LC1_I7MIK1 | 4,288.1 Å | 8.70° | **618 Å** |
| 6r6h__H1_Q13617--6r6h__J1_Q03071 | 191.3 Å | 13.71° | 44.7 Å |
| 8ap6__L1_Q38CI8--8ap6__MA1_Q582T1 | 82.8 Å | 10.98° | 15.8 Å |
| 7oit__A1_A0A140TAH5--7oit__B1_Q0JRZ9（正常） | 0.8 Å | 11.57° | 0.1 Å |

並進探索が補正できるのは ±`trans_cells`×spacing = ±2.4 Å なので、
上位2件は原理的に回復不能だった。

**修正**:
1. `geom.orient()` は**幾何重心**で decenter する（慣性テンソルには従来どおり
   重みを使う。`decenter_weighted=True` で旧挙動）。
2. **既存の2900件キャッシュを作り直さずに済む厳密補正**
   `PreparedProtein.recenter_ligand()` を追加し、`prep_cache.load_prepared` が
   読み込み時に適用する。`lig_ref' = lig_ref − c`、`t*' = t* + rotate(c, q*)` で

   ```
   rotate(lig_ref', q*) + t*'  ==  native_lig
   ```

   が恒等的に保たれる（検証: native 再構成誤差は 5.7e−3 → 6.4e−3 Å で不変。
   float32 の丸めの範囲）。

**修正後の再測定（同じ300件）**:

| | 閾値超え | 平均最良DockQ | 中央値 |
|---|---:|---:|---:|
| 修正前 | 97.3% (292/300) | 0.668 | 0.703 |
| **修正後** | **99.0% (297/300)** | **0.695** | **0.720** |

到達不能8件の個別内訳:

| 複合体 | 修正前 | 修正後 | \|重心\| |
|---|---:|---:|---:|
| 4ez1 | 0.000 | **0.804** | 22,707.6 |
| 7tgh | 0.000 | **0.702** | 4,288.1 |
| 7cae | 0.163 | **0.713** | 73.5 |
| 6r6h | 0.039 | **0.653** | 191.3 |
| 8ap6 | 0.167 | **0.547** | 82.8 |
| 7w5z | 0.138 | **0.399** | 88.0 |
| 7n6g | 0.128 | 0.128 | 3.4 |
| 7k58 | 0.195 | 0.213 | 14.9 |

**結論**: 到達可能性は律速ではない（99%）。残る3件は重心が小さく別要因
（7n6g はリガンド1904原子、7k58 は4664原子で、回転格子の粗さに対して
リガンドが大きすぎる）。

**未修正のまま残した S8 の残り半分**: 慣性テンソルの符号つき重み
（x が最長軸でない 21.7%、native が箱から出る 7/300）。こちらは `lig_ref` の
**向き**が変わるので読み込み時補正ができず、キャッシュ全再生成が要る。
重心側だけで到達不能が 8→3 件に減ったので、残り3件のために全再生成する
価値は低いと判断した。記録として残す。

**テストの扱い**: `tests/test_orient.py::test_orient_matches_julia` は
Julia 参照が符号つき重心を使っているため、`decenter_weighted=True` を明示する
`..._legacy_weighted_centroid` に改名（移植の忠実性は今も検証される）。
新挙動を検証する `test_orient_centroid_is_at_the_origin` を追加。

#### 5.10.2 1.2 Å の voxel 表と、そこで見つかった別のバグ

`compute_grid_sizes.py` が **SIGKILL される**（exit 137）。原因は
`generate_grid` が実際に格子テンソルを確保することで、このコーパスの最大複合体は
3.0 Å で 1.2e10 voxel、**1.2 Å では 1.94e11 voxel ≈ 760 GB** になる。
形だけ知りたい用途のために確保しない `geom.grid_shape()` を追加し、
`compute_grid_sizes.py` をそちらに切り替えた（既存の `generate_grid` と
出力一致を確認）。

**1.2 Å での分布（2900複合体）**: 中央値 **2,791,360**、p90 1.85e7、
p95 3.43e7、p99 1.11e8、最大 **1.94e11**。

#### 5.10.3 探索コストの実測と足切り閾値の決定

Hopf 1944回転、`--rot-chunk 2`、h=1.2 Å、A6000:

| voxel | n_rec | n_lig | 探索時間 | ピークVRAM |
|---:|---:|---:|---:|---:|
| 415,662 | 685 | 85 | 13.3 s | 0.4 GiB |
| 1,123,632 | 3704 | 102 | 17.6 s | 1.4 GiB |
| 2,791,360（中央値） | 370 | 370 | 32.6 s | 3.1 GiB |
| 6,321,952 | 696 | 749 | 58.2 s | 5.9 GiB |
| 13,075,200 | 6323 | 615 | 134.5 s | 12.2 GiB |
| 22,512,943 | 758 | 758 | 267.3 s | 21.0 GiB |

**時間は voxel 数にほぼ比例し、原子数にはしない**（13.1M の複合体は受容体6323原子、
22.5M は758原子だが後者のほうが2倍遅い）。特徴量化＋DockQ ラベルを含めると
1複合体あたり概ね探索時間の2倍。

閾値候補（homodimer 除外後、N=275 = fit 220 + val 55 を1 GPU で回した場合）:

| 閾値 | コーパス除外率 | usable | N=275 実時間 | 選択の voxel 中央値 |
|---:|---:|---:|---:|---:|
| 3.0M | 48.4% | 1193 | 2.8 h | 791,700 |
| 6.0M | 30.7% | 1586 | 3.7 h | 1,124,463 |
| 15.0M | 12.3% | 2032 | 6.2 h | 2,013,077 |
| **31.25M** | **5.7%** | **2194** | **8.4 h** | **2,245,572** |

**採用: 31.25M**。3.0 Å の 2M と体積等価（2e6 × (3.0/1.2)³ = 3.125e7）で、
除外率 5.7% は §5.6 の 5.9% とほぼ同じ。**切り詰めの度合いを §5.6 と揃えることで、
その caveat（§5.7-R7）が変わらないようにした**。

#### 5.10.4 §5.7-R7 のサイズ・ドメインシフトが解消した

TEST 側の voxel 表も 1.2 Å で作成（`prep_cache_test` の178件）:
中央値 **2,336,631**、最大 2.16e7 で、**178件すべてが 31.25M 閾値以下**。

| コホート | voxel 中央値 |
|---|---:|
| 学習（31.25M 閾値適用後の N=275 選択） | 2,245,572 |
| TEST | 2,336,631 |

§5.7-R7 は「学習コーパスだけがサイズで切り詰められ、TEST は切り詰められないので
train→test の差の一部はドメインシフト」と指摘していたが、
**1.2 Å では TEST 側が閾値に一切かからないため、この交絡は消えた**。
両コホートの voxel 中央値は 4% 違いである。

#### 5.10.5 TEST プール再構築で見つかった3つ目のバグ — 特徴量化の OOM が大きい複合体を系統的に落としていた

**1回目の構築（`--frame-chunk 100`、3 GPU に分割）**: 250件中 **167件しか作れず、
83件 (33%) が OOM で脱落**した。しかも脱落は無作為ではない:

| | n | voxel 中央値 | voxel 最大 |
|---|---:|---:|---:|
| 構築成功 | 167 | 1,721,010 | 7,674,480 |
| **OOM で脱落** | 83 | **4,145,358** | 21,605,584 |

§5.7-R8 が「脱落する複合体は無作為ではない（parse に失敗する／大きいもの）」と
指摘した生存バイアスの実例である。**足切り閾値 31.25M はまったく効いておらず、
実効上限は約 7.7M voxel だった。**

**原因**: 特徴量化が作る `L_count` は `(frame_chunk × 12, nx, ny, nz)` ——
**1 pose あたり12格子**である。1.2 Å では:

| voxel | fc=100 | fc=50 | fc=25 |
|---:|---:|---:|---:|
| 2,800,000 | 12.5 GB | 6.3 GB | 3.1 GB |
| 7,700,000 | 34.4 GB | 17.2 GB | 8.6 GB |
| 22,500,000 | **100.6 GB** | 50.3 GB | 25.1 GB |

`--frame-chunk` の既定 100 は 3.0 Å 時代のプロファイル（§5.6.2）であり、
OOM 救済ラダーも `frame_chunk` を 100→50→25 と2回しか半減させないため、
大きい複合体では 25 GB を要求したまま落ち続けていた。
**探索ではなく特徴量化が律速だった**（探索は 22.5M voxel でも rot_chunk=2 で 21 GiB）。

**修正**: DockQ の `_adaptive_pose_chunk` と同様に、格子体積に連動する
`_adaptive_frame_chunk(n_voxels, budget, cap)` を追加（`--feature-budget`、
既定 1e9 要素 = 4 GiB）:

| voxel | frame_chunk | L_count |
|---:|---:|---:|
| 400,000 | 100 | 1.8 GiB |
| 2,800,000 | 29 | 3.6 GiB |
| 7,700,000 | 10 | 3.4 GiB |
| 22,500,000 | 3 | 3.0 GiB |

**ピークがコーパス全体でほぼ一定になる。** 実測でも GPU 使用量が 45.8 GiB →
7〜8 GiB に低下した。**脱落した83件を再試行したところ、skip 0 で全件構築できた。**

#### 5.10.6 完成した TEST プール（250/250）

| | 値 |
|---|---|
| 複合体数 | **250 / 250**（`data/pinder_test_ids.txt` の全件） |
| voxel 中央値 / 最大 | 2,273,965 / 21,605,584 |
| 到達可能 positive を持たない | **1 / 250** |
| **探索自身が positive を見つけた** | **0 / 250** |
| positive 数の中央値 | 1000 / 2500 pose |
| 最良到達可能 DockQ 中央値 | 0.662（最小 0.211） |
| `sc` の形 | (2500, 4) = PSC の4項分解 |

**コホートの訂正**: §5.9.10 で「新しい TEST コホートは178件」と書いたが**誤り**。
`prep_manifest_test.jsonl` が 178 ok / 72 fail としていたのに対し、
`prep_cache_test` には実際には**250件すべて**が入っていた。§5.7-R1 の
「manifest は自分の worker ファイルから再生成できず信頼できない」がここにも表れた。
したがって新コホートは **250件**であり、§5.4-§5.6 の241件コホートより*大きい*。
両者は依然として別物なので直接比較してはならない。

**この時点で確定した事実**（学習前・baseline パラメータ）:
- 到達可能性は律速ではない（249/250）。
- しかし **250件すべてで、探索の上位1500に near-native pose が1つも入らない**。
- したがって現状の課題は**純粋にランキング**であり、パラメータ学習が効くとすれば
  まさにここである。これが学習前の正直な出発点である。

#### 5.10.7 round-0 プールのディスクキャッシュ

round 0 のプールは「選択」と「baseline パラメータ」だけで決まり、
`select_split` に RNG は無く分割は seed 間で byte 一致する。したがって
**どの `--seed` でもまったく同じ採掘をする**。3 seed を並列に回すと GPU コストが
3倍になるだけなので、`--pool-cache`（既定 `data/scaling/pool_cache`）を追加した。
キーは選択・レシピ・baseline パラメータから作る:

```
n{N}_r{round}_sp{spacing}_{pool}_{rot_set}{nside}_ntop{K}_nr{K}_tc{R}
   _a{alpha0}_rho{rho0}_hd{homodimer除外}_mv{max_voxels}
```

検証（N=8、seed 0 → seed 1）: 2回目はプール構成が完全一致
（`mean pool=2309, pos 848 [search 38 / enumerated 809], neg 1462`）で
採掘をスキップし、**GPU ピーク 8.7 GiB → 0.0 GiB**。

**mining round ≥ 1 はキャッシュしない** —— そこまでに到達したパラメータに依存し、
したがって seed に依存するからである。

#### 5.10.8 実行中

`N=220 / seed 0 / --rounds 0 / alpha0=1.0` を GPU 0 で実行中。
採掘は **47.5 秒/複合体**（275複合体で約3.6時間、救済待ち 0件）。

```bash
export PINDER_BASE_DIR=$PWD/external/pinder HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/home/yasu/tmp/ddock-tmp
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_pinder_scaling.py \
  --n-fit 220 --rounds 0 --seed 0 --alpha0 1.0 \
  --grid-voxels data/scaling/grid_voxels_1.2.json --max-grid-voxels 31250000 \
  --test-cache data/shards_pinder/test_pool_reachable.pt \
  --out-dir data/scaling/runs_round0
```

完了後の評価:

```bash
# 主指標: per-complex AUC の paired Wilcoxon（success@K は床拘束で全て n.s.）
uv run python scripts/compare_conditions.py \
  --pool data/shards_pinder/test_pool_reachable.pt \
  --ckpt data/scaling/runs_round0/N220_seed0/round0_ckpt.pt --prov all
uv run python scripts/compare_conditions.py ... --prov search   # 探索由来 pose のみ

# 正直な end-to-end: 学習後パラメータで再探索（同じ Hopf 格子、cone なし）
uv run python scripts/eval_search_test.py \
  --ckpt data/scaling/runs_round0/N220_seed0/round0_ckpt.pt
```

---

### 5.11 予備プローブ（N=8）と、そこで露呈した評価の交絡（2026-07-26）

**この節の結論を先に書く: 出た数字は劇的だが、成果ではない。**
学習が獲得したのは界面の化学ではなく埋没度であり、しかもそれを自明な
接触数カウントより下手に獲得している。候補プールの構成に交絡がある。

#### 5.11.1 何をやったか

N=220 の採掘（約3.7時間）を待つ間に、「そもそもこの目的関数で AUC が動くのか」を
確かめるため、キャッシュ済みの **N=8** プール（§5.10.7）で 3000 ステップまで
学習を回した。評価は 250複合体の固定 TEST プール（§5.10.6）。

```
alpha=0.0000  rho=2.8820  ||dIface||=1.609  val_loss=2.2567  steps=2001/3000
WARNING: a parameter sits on its box constraint (alpha True, rho False)
```

#### 5.11.2 出た数字（250複合体、paired Wilcoxon）

| 指標 | baseline | 学習後 | Δ | Wilcoxon p |
|---|---:|---:|---:|---:|
| mean AUC | 0.0049 | **0.9606** | +0.9557 | 1.4e−42 |
| first_hit_pct（浅いほど良い） | 0.6041 | **0.0068** | −0.5972 | 2.0e−42 |
| mean best DockQ@1 | 0.0191 | **0.3661** | +0.3470 | 4.4e−42 |
| mean best DockQ@100 | 0.0301 | **0.5295** | +0.4994 | 2.1e−42 |
| success@1 | 0.0% | **95.6%** | — | 4.5e−72（McNemar 厳密、不一致 238/0） |

**8複合体の学習で、250件の interface-deleaked TEST に success@1 95.6%**。
この時点で信じるには良すぎるので、対照を測った。

#### 5.11.3 対照 — パラメータを一切使わないランカー

| ランキングに使う量 | mean AUC | 中央値 |
|---|---:|---:|
| フルスコア、既定パラメータ | 0.0049 | 0.0000 |
| フルスコア、**学習後** | 0.9606 | 1.0000 |
| **純粋な接触数 Σ_ij n_ij** | **0.9957** | 1.0000 |
| PSC 有利項の数 c_pair | 0.9949 | 1.0000 |
| 表面-表面衝突数の符号反転 | 0.0050 | 0.0000 |
| S_PSC 単独（ρ=3.5, α=1） | 0.0049 | 0.0000 |

**学習後モデル（0.9606）は、接触数を数えるだけ（0.9957）より劣る。**
144成分の対ポテンシャルを最適化して得たものが、パラメータ0個のカウンタに負けている。

#### 5.11.4 原因 — プールの2クラスが埋没度だけで分離している

| | 接触数 Σ_ij n_ij の中央値 |
|---|---:|
| positive（near-native、列挙） | **34,856** |
| negative（探索の上位1500） | **910** |
| 比 | **39.4×** |

**249複合体中 172件 (69%) で、negative の接触数が positive の範囲に1つも入らない。**
negative が positive の接触数レンジに入る割合の中央値は **0.000**。

機構: negative は「α=1, ρ=3.5 の既定スコア」の上位1500 だが、そのスコアの
AUC は **0.0049** ＝ **接触しない pose を積極的に好む**。したがって探索は
受容体を撫でるだけの pose を返し、near-native と完全に分離する。
§5.6.11 の「スコアが非接触 pose を支持している」が250複合体で定量化された形である。

α が学習で 0（下限）に張り付いたのも同じ理由である。PSC 項はランキングを反転
させるので、消すことが最適解だった。境界張り付きの警告（§5.9.6 で追加）が
正しく発火した。なお **α=0 では ρ の勾配も 0 になる**ので、報告された
ρ=2.882 は α が 0 に達する前に漂った値であって最適値ではない。

#### 5.11.5 何が言えて、何が言えないか

**言える**:
- 配線は生きている。146パラメータすべてに勾配が届き、学習は収束し、
  検証損失で選択され、held-out で大きく改善する。
- **既定パラメータ（α=1, ρ=3.5）は、この候補集合に対して壊滅的に悪い**
  ——AUC 0.0049 は偶然（0.5）どころか、ほぼ完全な逆相関である。
- したがって「学習前段階のスコア関数はランキングに使える状態ではない」が
  250複合体で確定した。

**言えない**:
- 学習した144成分の対ポテンシャルが、**埋没度以上の何かを寄与したか**。
  このプールでは原理的に測れない。
- success@1 95.6% を ZDOCK の性能として読むこと。同じ数字は
  「接触数で並べ替える」だけで（より良く）出る。

#### 5.11.6 対応（実装済み）

対照が二度と忘れられないよう、`scripts/compare_conditions.py` が
**すべての比較で自動的に**以下を印字するようにした:

- 接触数 Σn_ij と c_pair の AUC
- positive/negative の埋没度の比と、完全非重複の複合体数
- 25% を超えて非重複なら警告:
  *"the two classes are largely separable by burial alone, so this pool cannot
  show whether the learned pair potential contributes anything beyond contact
  count."*

#### 5.11.7 限界と、次に必要な実験

**この節の限界**: N=8 の予備プローブであり、seed 1本、mining なし。
数値の大きさは信用できるが、N=220 の結果を代表しない。

**次の実験（設計）**: 埋没度を揃えた negative が要る。候補は
1. **接触を好むスコアで negative を採掘する**（例: α=0 の IFACE のみ）。
   その上位 N は「接触は多いが native でない」pose になり、これが本来の
   hard negative である。プールが訓練対象パラメータに依存しないよう、
   採掘スコアは固定・明示の選択とする。
2. **接触数マッチ評価**: 両クラスを Σn_ij の重なる帯に制限して AUC を取る
   （§5.6.7 で試みたが「8複合体中1件しか使えない」で頓挫した方法。
   埋没度の重なりが無い現プールでは今も使えない——1 が先に要る）。
3. native から中距離の格子 pose（誤っているが埋没している）を negative に加える。

**1 を実施しない限り、「学習した対ポテンシャルの寄与」という本題は測れない。**

---

### 5.12 TEST セットの参照構造が全件破綻していた（2026-07-26）

**本プロジェクトの TEST セット由来の数値は、§5.3 から §5.11 まで、すべて無効である。**

#### 5.12.1 事実

PINDER は同じ系に対して2種類のモノマーファイルを置いている:

| ファイル | 受容体–リガンド最近接原子間距離 | 各鎖の重心 |
|---|---:|---|
| `pdbs/{system_id}.pdb` の鎖 R / L | **2.72 Å**（正常な水素結合距離） | R (−71.6, −7.3, 51.6) / L (−7.6, 4.2, −13.9) |
| `test_set_pdbs/{...}-R.pdb`, `-L.pdb` | **0.02 Å** | **R (0,0,0) / L (0,0,0)** |

`test_set_pdbs/` のモノマーは**それぞれ独立に原点へ centering された「ドッキングの入力」**であり、
複合体の座標系を持たない。そのまま2つを読み込んで重ねると、
**2本のタンパク質が同じ空間を占める**。

- 学習コーパス (2900件) は `pdbs/*-R.pdb` / `*-L.pdb`（＝複合体座標系）から作られており **正常**。
  実測: 最近接距離の中央値 2.62 Å、2.0 Å 未満は 120件中 5件 (4.2%)。
- TEST (250件) は `test_set_pdbs/` から作られており、**250/250 が 2.0 Å 未満**。
  16件は受容体とリガンドの座標が実質同一。最悪の 5dot は最近接 0.02 Å、
  **2.0 Å 未満の原子ペアが 32,069組**、受容体10,476原子のうち10,467個がリガンドの4.5 Å以内。

#### 5.12.2 これで説明がつくこと

今日ここまでに「発見」として記録したものの大半は、この1つのバグの症状だった:

| 観測 | 真の原因 |
|---|---|
| 探索が 250/250 で near-native を1つも返さない（§5.10.6） | 「native」が相互貫入した構造なので、探索が返せるはずがない |
| baseline の AUC が 0.0049（§5.11.2） | 「positive」が受容体に埋め込まれた pose |
| positive の接触数 34,856 vs negative 910（§5.11.4） | positive は相互貫入しているので接触数が最大になる |
| 学習が「接触が多いほど良い」を学ぶ（§5.11.3） | 上の直接の帰結 |
| native pose の S_PSC が −4,221,070（5dot） | 衝突セル 55,730個。スコアは**正しく**この構造を拒否している |

**したがって §5.11 の「スコア関数が非接触 pose を好む」「初期パラメータが壊れている」
という診断は誤りである。** スコア関数は、与えられた（破綻した）参照構造に対して
正しく振る舞っていた。§5.11.4-5.11.7 の解釈をここで撤回する。
測定値そのもの（表の数字）は撤回しないが、その解釈はすべてこの節に差し替える。

#### 5.12.3 影響範囲

| 節 | 内容 | 状態 |
|---|---|---|
| §5.3 | DB5.5 汎化 | TEST は DB5.5 由来なので**影響なし**（別途 leakage で破棄済み） |
| §5.4 | PINDER deleaked split の 1.2% / 82.7% | **無効** |
| §5.5 | hard-negative mining の gain | **無効** |
| §5.6 | scaling law の TEST 数値すべて | **無効** |
| §5.6.9-5.6.11 | 12複合体の recall / ceiling / 項別分解 | **要確認** —— `prep_cache_test` を使っているものは無効 |
| §5.10.6 | 新 TEST プール 250件 | **無効** |
| §5.11 | N=8 プローブの評価 | **無効** |

**学習側（fit / validation）は健全**なので、採掘したプールとキャッシュは再利用できる。

#### 5.12.4 修正方針

TEST 系の native 複合体は `pdbs/{system_id}.pdb` の鎖 R / L から取る。
すなわち受容体・リガンドとも複合体座標系で読み、`native_lig` をそこから作る。
`test_set_pdbs/` のモノマーは「unbound 入力」として使う場合にのみ意味があり、
その場合は複合体への剛体変換を別途求める必要がある（本実装は bound-bound
ドッキングなので、複合体ファイルから両鎖を取るのが正しい）。

**この修正なしに TEST 上のどんな数値も報告してはならない。**

#### 5.12.5 この失敗から

- §5.7 の11体監査は**スコア関数のコードだけを監査対象にしており、入力データの
  健全性を誰も検査しなかった**。「参照構造が物理的にありうるか」という
  1行のチェック（最近接原子間距離 > 2 Å）で即座に見つかったはずである。
- §5.10.6 で「探索が 250/250 で positive を見つけられない」という異常値を
  観測した時点で、スコア関数ではなくデータを疑うべきだった。
- 学習が「良すぎる」結果（success@1 95.6%）を出したとき、対照（接触数 AUC 0.9957）
  までは辿り着いたが、そこから更に上流に遡らなかった。

**再発防止として、prep 時に「受容体–リガンド最近接距離 < 2.0 Å なら異常」を
assert する。**

---

### 5.13 参照構造の修復と、判明した真のベースライン（2026-07-26）

**§5.12 のバグを直したところ、本実装は既定パラメータのままで
success@1 = 69.5% を出していた。** §5.4 以降ずっと「0%」と記録してきたものは
すべて破綻した参照構造の産物であった。

#### 5.13.1 修正内容

| 対象 | 内容 |
|---|---|
| `geom.orient` / `PreparedProtein.recenter_ligand` | 幾何重心で decenter（§5.10.1） |
| `dataset.prepare_protein` | **幾何 assert**: 受容体–リガンド最近接重原子距離 < 2.0 Å で `ValueError` |
| `dataset.parse_pdb_plain` | 鎖選択を追加 → **1つの複合体ファイルから両鎖**を読める |
| `prep_pinder_cache.py` | `--complex-dir`（worker への引数転送漏れも修正） |
| `scripts/check_prep_cache.py` | **新規**: 参照構造の物理的妥当性を全件検査、異常なら非ゼロ終了 |
| `run_pinder_scaling.py` | `--exclude-bad-geometry`、プールキャッシュを差分式に |

#### 5.13.2 修復の検証

`scripts/check_prep_cache.py` による全件検査:

| | TEST（修復前） | **TEST（修復後）** | 学習コーパス |
|---|---:|---:|---:|
| 検査数 | 250 | **249** | 2900 |
| 最近接距離の中央値 | 0.02 Å | **2.58 Å** | 2.65 Å |
| 立体的に不可能（< 2.0 Å） | **250 (100%)** | **0** | 83 (2.9%) |
| >90% の受容体原子が界面 | 250 (100%) | **0** | **0** |
| `(q*,t*)` が native を再現しない | — | 0 | 1 |
| \|lig_ref 重心\| > 10 Å | — | 0 | **0**（修正前は300件中121件） |

TEST 1件は本物の異常構造として新しい assert が拒否した。
学習コーパスの83件は個別の異常（相互貫入型は0件）なので除外リスト化した
（`data/scaling/excluded_bad_geometry.txt`、`select_split` が既定で外す）。

#### 5.13.3 真のベースライン（既定パラメータ α=1.0, ρ=3.5、学習なし）

249 複合体の interface-deleaked PINDER-S、Hopf nside=3（1944回転、cone なし）、
h=1.2 Å、探索上位1500。

| 指標 | 破綻構造 | **修復後（全pose）** | **修復後（探索由来のみ）** |
|---|---:|---:|---:|
| 探索が positive を発見 | **0/250** | — | **236/249 (94.8%)** |
| success@1 | 0.0% | 65.9% | **69.5%** |
| success@5 | 0.0% | 79.9% | **84.3%** |
| success@10 | 0.0% | 82.7% | **87.3%** |
| success@50 | 0.0% | 90.0% | **94.5%** |
| success@100 | 0.0% | 91.6% | **96.2%** |
| mean best DockQ@1 | 0.019 | 0.424 | **0.446** |
| first-hit の深さ | 60.4% | 2.9% | **1.1%** |

「探索由来のみ」は探索が positive を持つ236複合体に対する率。
13件（探索が失敗）を含めたコーパス全体では
**success@1 = 0.695 × 236/249 = 65.9%** で全pose版と一致する。

**埋没度の交絡も解消した**:

| | 破綻構造 | 修復後（全pose） | 修復後（探索由来） |
|---|---:|---:|---:|
| positive/negative の接触数比（中央値） | **39.4×** | 2.6× | **1.7×** |
| 完全非重複の複合体 | **172/249** | 1/249 | 21/236 |
| 接触数だけの AUC | 0.9957 | 0.837 | 0.891 |

接触数の AUC が依然高いのは「正解は相手にべったり接触する」という
物理的に正しい事実であり、以前のような自明な分離ではない。

#### 5.13.4 AUC は主指標として不適切だった（訂正）

§5.7-M8 で「success@K は床拘束なので per-complex AUC を主指標にすべき」と
結論したが、**修復後のプールではこれが逆になる**。

既定パラメータでの AUC を、positive の質で切り分けた（249複合体）:

| positive の定義 | 該当複合体 | mean AUC | median |
|---|---:|---:|---:|
| DockQ ≥ 0.23 | 249 | **0.0845** | 0.072 |
| DockQ ≥ 0.4 | 249 | 0.2799 | 0.268 |
| DockQ ≥ 0.6 | 194 | 0.6787 | 0.688 |
| **DockQ ≥ 0.8** | 31 | **0.9897** | 0.999 |

AUC が低かったのは、**±2セル列挙が「閾値ぎりぎりの粗悪な positive」を
942個も positive クラスに流し込み、スコアがそれらを正しく下位に落としていた**
ためである。本当に良い pose（DockQ ≥ 0.8）はほぼ完璧に上位化されている（0.990）。

したがって修復後のプールでは **success@K が主指標**（もはや床拘束ではない、
69.5%）であり、AUC を使うなら positive の質を揃えて報告しなければならない。

#### 5.13.5 列挙半径を ±2 → ±1 に変更

24 TEST 複合体、8回転 ×(2R+1)³ 並進の中央値:

| R | 候補数 | positive | 良質(≥0.6) | 最良DockQ | positiveの中央DockQ |
|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 8 | 1 | 0.630 | 0.380 |
| **1** | **216** | **216** | **8** | **0.671** | **0.383** |
| 2 | 1000 | 982 | 14 | 0.686 | 0.352 |

±1 は候補数が 4.6分の1 になるのに天井はほぼ同じ（0.671 vs 0.686）で、
positive の中央 DockQ はむしろ良い。既定値を 1 に変更した。

**設計上の反省**: ±2 という半径は「到達可能性の ceiling が飽和するか」だけを
見て決めており（4×2 で飽和することは §5.10.1 で測った）、
**そこに入る pose の質を一度も見ていなかった**。ceiling を測る目的なら
「1つでも良いものが届くか」で正しいが、学習データとして使うなら
「入るもの全部の質」を見なければならない。
列挙 positive の 98.4% は DockQ 0.23〜0.6 の粗悪品で、
「受容体に 2.4 Å めり込んだ pose が正解だ」と教えることになっていた。

なお修復後は**探索だけで 236/249 の複合体に positive がある**ので、
列挙の存在理由（探索が全滅するので正解を手で入れる必要がある）自体が
ほぼ消えた。列挙が無いと positive ゼロになるのは13件のみである。

#### 5.13.6 なぜ気づけなかったか — DockQ は参照の破綻を検出できない

同じ複合体（5dot）で、native pose に対する DockQ:

```
BROKEN 構造 : DockQ=1.0000  fnat=1.000  iRMSD=0.000  | 最近接 0.02 A, 2A未満 32,069組
FIXED  構造 : DockQ=1.0000  fnat=1.000  iRMSD=0.000  | 最近接 2.72 A, 2A未満      0組
```

**DockQ は pose と参照の相対比較なので、参照そのものが物理的にありえなくても
満点を返す。** 自己言及的であり、原理的に入力の健全性は測れない。
`tests/test_dockq.py` の「同一 pose → DockQ 1.0」も、式が正しいことの確認で
あって入力が正しいことの確認ではない。両方通っていたので安心してしまった。

異常を告げていたのは**すべて DockQ とは独立な絶対量**だった——
探索が 250/250 で失敗、positive の接触数が39倍、native pose の S_PSC が −422万。
これらを見てなおスコア関数を疑い続けたのが誤りであった。

§5.7 の11体監査は `dockq.py` について「対称性を扱っていない」「原子レベル近似で
閾値がずれる」という**式の**問題を的確に指摘したが、
**入力データが物理的に可能かは8つの角度のどれからも検査されなかった。**

#### 5.13.7 無効化される過去の結論（再掲・確定）

§5.12.3 の一覧のとおり、§5.4 / §5.5 / §5.6 / §5.10.6 / §5.11 の
TEST 由来の数値はすべて無効。特に:

- 「PINDER deleaked TEST で success@1 が 0〜1.7%」→ **実際は 69.5%**
- 「探索が near-native を返さない」→ **236/249 で返している**
- 「PSC の衝突罰則が正解 pose を潰している」（§5.6.11）→
  **潰していたのは相互貫入した偽の『正解』であって、スコアは正しかった**
- §5.11 の「初期パラメータが壊れている」→ **撤回**

学習側（fit / validation）の採掘結果とキャッシュは健全なので再利用できるが、
列挙半径を ±1 に変えたため作り直す。

#### 5.13.8 実行中

`N=220 / seed 0 / --rounds 0 / alpha0=1.0 / trans_cells=1` を GPU 0 で採掘中。
選択は usable 2142（幾何破綻83件の除外で 2194→2142）、fit 220 + val 55。

このジョブが読み込んだ TEST プールは ±2 版（構造は修復済み）である。
**チェックポイントの選択は固定 validation の損失のみで行い TEST は印字にしか
使わないので学習には影響しない**が、正式な評価は完成後に ±1 版のプールに対して
`compare_conditions.py` で取り直す。

---

### 5.14 最初の意味のあるパラメータ学習（2026-07-26）

§5.13 で参照構造を修復し、初めて健全な土台の上で学習を回した。
**結論を先に書く: success@1 が 65.9% → 70.7% に改善し（249複合体の
end-to-end 再探索、13勝1敗、McNemar p=0.0018）、3 seed で再現した。
ただし最大の発見は数字ではなく、「損失にどの pose を見せるか」だけで
結果が改善から壊滅まで振れたことである。**

#### 5.14.1 設定

| | |
|---|---|
| データ | fit 220 + validation 55 複合体（幾何破綻83件と homodimer を除外した 2142件から決定的に選択） |
| プール | 1複合体あたり中央値 1,696 pose（探索由来1,518 + 列挙178）、positive 178（探索由来13 / 列挙164） |
| 学習対象 | α + ρ + IFACE 144 = **146** |
| 損失 | `L_basin + 0.5·L_margin + 0.1·L_prior`（§5.9.2 のまま、論文提案どおり） |
| 最適化 | Adam、最大1500ステップ、固定 validation 損失で early stopping（patience 8）、TEST は一切見ない |
| 評価 | 249複合体の固定 TEST プール（§5.13）＋ **学習後パラメータでの end-to-end 再探索** |

採掘は `--mine-shard i/3 --mine-only` で3 GPU に分割し、約3.6時間 → **1.3時間**に短縮した
（round 0 のプールは seed 非依存なので、これは実験内容を変えず実時間だけ縮める）。
3 seed の学習はキャッシュから各2分程度で、GPU をほぼ使わない。

#### 5.14.2 決定的だったのは「損失に見せる pose の範囲」

最初の学習（`--loss-prov all`、プール全体を損失に見せる）は**破滅的だった**:

| 損失が見る pose | α | ‖ΔIFACE‖ | 固定プール success@1（探索由来 pose で評価） |
|---|---:|---:|---|
| 全部（列挙 positive 込み） | **0.0014** | 2.348 | 69.5% → **34.7%**（86敗4勝、McNemar p=4.3e−21） |
| **探索由来のみ** | **0.982** | 0.996 | 69.5% → **74.6%**（13勝1敗、p=0.0018） |

同じデータ・同じ損失関数・同じハイパーパラメータで、**モデルに見せる pose を変えただけ**である。

機構は明快である。列挙 positive は fit プールの positive の **92%**（中央値 164/178）を占め、
探索が実際に見つけた positive は中央値 **13個**しかない。損失は
「列挙 positive を探索 negative より上げよ」に支配される。その最も安上がりな解が
**α → 0、すなわち形状相補性項を丸ごと切ること**だった。残るのは IFACE ≈ 接触数で、
列挙 pose（正解の隣）と探索 pose を分けるには接触数で十分だが、
**すでに接触している探索 pose 同士から正しい結合部位を選ぶには形状項が要る**。

紛らわしいのは、`--loss-prov all` のモデルが**プール全体で評価すると改善して見える**
ことである（65.9% → 76.7%、p=6e−4）。列挙 positive を上位化する能力が上がるからで、
それは配備時には存在しない pose である。**評価指標を1つだけ見ていたら「改善した」と
報告していた。**

これは §5.11 で「学習データの大半が配備時に存在しない pose だ」と書きながら、
それを外さずに走らせた私の設計ミスである。

#### 5.14.3 結果（論文準拠モデル、146パラメータ、3 seed）

**固定プール、探索由来 pose のみ、236複合体**

| seed | α | ρ | ‖ΔIFACE‖ | val loss | success@1 | 勝/敗 | McNemar p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9822 | 3.6118 | 0.996 | 2.7236 | 74.6% | 13/1 | 0.0018 |
| 1 | 1.0285 | 3.6059 | 0.725 | 2.7035 | 74.2% | 11/0 | 0.00098 |
| 2 | 1.0190 | 3.6069 | 0.794 | 2.6968 | 74.2% | 11/0 | 0.00098 |

平均 ± SD: **success@1 = 74.3 ± 0.2 %**（baseline 69.5%、**+4.8 pp**）、
best DockQ@1 = 0.4691 ± 0.0009（baseline 0.4457）。
**seed 間のばらつきが極めて小さく**、α 0.98〜1.03 / ρ 3.606〜3.612 に収束している。

**end-to-end 再探索（学習後パラメータで FFT 探索そのものを回す、249複合体、seed 0）**

これが配備条件そのものである。α が変われば探索が返す候補集合自体が変わる。

| K | baseline | 学習後 | 勝 | 敗 | McNemar p |
|---:|---:|---:|---:|---:|---:|
| **1** | 65.9% | **70.7%** | 13 | 1 | **0.0018** |
| 5 | 79.9% | 82.3% | 8 | 2 | 0.109 |
| **10** | 82.7% | **86.3%** | 9 | **0** | **0.0039** |
| 50 | 89.6% | 91.6% | 6 | 1 | 0.125 |
| **100** | 91.2% | **93.6%** | 6 | **0** | **0.031** |

| 指標 | baseline | 学習後 | Wilcoxon p |
|---|---:|---:|---:|
| **first-hit 順位** | 94.0 | **67.0** | **6.2e−09** |
| recall（正解を1つでも返す） | 94.8% | 96.4% | — |
| 返却集合の最良 DockQ | 0.599 | 0.607 | 0.202 |

**固定プールの +4.8 pp が、探索を回しても +4.8 pp（65.9→70.7%）で再現した。**
K=10 と K=100 では1件も悪化していない。効果の実体は first-hit 順位が
94位 → 67位に上がったことで、これが最も有意である。

#### 5.14.4 衝突罰則の共線性制約（**この節の結論は §5.14.7 で撤回された**）

> **撤回（2026-07-26、同日）**: 以下の「制約を外すと良くなる（free 75.0% >
> rho 74.3%）」という結論は**交絡していた**。free と rho は衝突重みだけでなく
> IFACE も異なる。同じ IFACE を保ったまま衝突重みだけ入れ替えた直接の paired 比較
> では success@1 は **8勝6敗 p=0.79 で有意差なし**であり、AUC はむしろ
> **学習した重みのほうが悪い**（0.8667 vs 0.9138、Wilcoxon p=3.3e−26）。
> 詳細と要因分解は §5.14.7。以下は測定記録として残す。


スコアは **148次元の線形モデル**として厳密に書ける（数値検証済み、誤差 5e−12）:

$$S = \alpha c_{\text{pair}} - (w_{ss} n_{ss} + w_{sc} n_{sc} + w_{cc} n_{cc})
      + \sum_{ij} M_{ij}(e) T_{ij} + \beta S_{\text{ELEC}}$$

Chen & Weng 2003 は $Im \in \{0, \rho, \rho^2\}$ とするので $w_k = \alpha\rho^k$、
すなわち**対数空間で3つの重みが共線**（自由度2）である。これを外すと自由度3になる。
`--psc-mode free`。重みは対数で持つ（正値性の保証、スケールフリーな学習率、
rho モードが free モードの線形部分空間になる）。

| モデル | val loss | success@1（固定プール、3 seed） | best DockQ@1 | best DockQ@10 |
|---|---:|---:|---:|---:|
| baseline | — | 69.5% | 0.4457 | 0.5545 |
| rho (146) | 2.697〜2.724 | 74.3 ± 0.2 % | 0.4691 ± 0.0009 | 0.5720 ± 0.0017 |
| **free (148)** | **2.676〜2.684** | **75.0 ± 0.0 %** | **0.4765** | **0.5832** |

free は baseline に対し **16勝3敗（p=0.0044）**、K=10 では 13勝2敗（p=0.0074）。

**学習された重み（seed 0）**、論文値 $(12.25, 42.88, 150.06)$ に対して:

| | α | $w_{ss}$ | $w_{sc}$ | $w_{cc}$ | 論文値との比 | 含意される ρ |
|---|---:|---:|---:|---:|---|---|
| rho seed0 | 0.982 | 12.81 | 46.28 | 167.14 | (1.05, 1.08, 1.11) | (3.61, 3.61, 3.61) |
| **free seed0** | 0.896 | 13.03 | 40.09 | 150.01 | **(1.06, 0.94, 1.00)** | **(3.81, 3.55, 3.60)** |

制約付きは ρ 1個で動かすので3つが**必ず同方向**（すべて 5〜11% 増）。
制約を外すと**方向が分かれた**——表面–表面は +6%、表面–core は −6%、core–core は不変。
含意される ρ で見ると $w_{ss}$ だけが共線性から外れている。

**3 seed で方向が一致した**（追試、2026-07-26）:

| 重み | 論文値との比 | 含意される ρ | 方向 |
|---|---:|---:|---|
| $w_{ss}$（表面–表面） | **1.077 ± 0.012** | 3.809 ± 0.008 | 3/3 増 |
| $w_{sc}$（表面–core） | **0.954 ± 0.017** | 3.557 ± 0.007 | 3/3 減 |
| $w_{cc}$（core–core） | 1.000 ± 0.000 | 3.584 ± 0.013 | **動かない** |

共線性の破れ（含意 ρ の ss と cc の差）は **+0.225 ± 0.007**（rho モードでは定義上 0）。

**訂正: $w_{cc}$ が「動かない」のは結果ではなくデータ不足である。**
クランプ境界は $[0.057, 5967]$ で学習値 150.08 は完全に内側（人工物ではない）。
真の理由は fit プール 220複合体 / 373,023 pose の重なりセル数の分布にある:

| 重なりの種類 | 総和 | 非ゼロ pose の割合 | pose あたり平均 | 罰則総寄与に占める割合 |
|---|---:|---:|---:|---:|
| 表面–表面 $n_{ss}$ | 10,363,293 | **97.9%** | 27.7 | **68.7%** |
| 表面–core $n_{sc}$ | 1,156,655 | 39.0% | 3.08 | 26.8% |
| core–core $n_{cc}$ | **67,490** | **1.4%** | **0.18** | **5.5%** |

勾配は $\partial L/\partial \log w_k \propto w_k \sum_f n_k$ なので、
$n_{cc}$ が実質ゼロだから $w_{cc}$ は初期値のまま prior に固定される。**判定不能**である。

物理的には当然で、core 原子は埋没原子（SASA ≤ 1 Å²）なので、
**両タンパクの埋没原子が同じセルを共有するには深い相互貫入が要る**。
探索が返す候補にも正解近傍にもそんな pose はほとんど無い。

**したがって Chen & Weng の罰則階層のうち最大の $-\rho^4 = -150$（core–core）は、
現実の候補集合ではほぼ発火していない。** 実質7割を表面–表面が担っている。

**free モードの実質的な自由度は2**（$w_{ss}, w_{sc}$）で rho モードと同数である。
それでも良くなったのは自由度の数ではなく**方向**で、rho は3つを同方向にしか
動かせないのに対し free は $w_{ss}\uparrow$ / $w_{sc}\downarrow$ を取れた。
$\rho$ による単一パラメータ化は、実際に発火する2成分の比
$w_{sc}/w_{ss} = \rho$ を固定する強い制約であり、学習はこれを
$3.5 \to 40.09/13.03 = 3.08$ に下げたがっている。

**解釈（仮説、未検証）**: Chen & Weng は「表面の重なりは側鎖が動けば解消できるので
寛容に」という設計思想を述べているが、いま測っているのは bound-bound（構造変化なし）
なので、この設定では表面の重なりが誤 pose の証拠になりやすい。
**unbound では逆になる可能性があり、未測定である。**

#### 5.14.5 限界

| | |
|---|---|
| **bound-bound** | 250/250 で受容体・リガンドが同一 PDB 由来。unbound の難易度は未測定であり、**success@1 65.9%/70.7% を ZDOCK 論文の unbound 性能と比較してはならない** |
| **free モードは固定プールでしか評価していない** | `sc_encode` は ρ 1個で $Im$ を作るので、独立3重みで探索を回すには FFT 経路の一般化（surf/core を別チャネルにして相関を3本）が要る |
| 損失に寄与しない複合体 | 220中72件（33%）が探索由来 positive を持たず、`--loss-prov search` では勾配ゼロ |
| DockQ の対称性未対応 | §5.7-S9、コーパスの約15%で正しい pose が negative とラベルされる |
| β・11電荷 | 凍結（ランキング寄与 約1%） |
| mining は round 0 のみ | 学習後パラメータでの再採掘はしていない |
| end-to-end 再探索は seed 0 のみ | 固定プールでは3 seed 確認済みだが、再探索は1 seed |

#### 5.14.6 次段階

1. ~~free モードの3 seed で重みの方向が一致するか~~ → **完了**（一致した。上記）
2. **FFT 経路を一般化**して free も end-to-end で測る。`sc_encode` は $Im$ を
   $\rho$ 1個で作るので、独立3重みには surf/core を別チャネルにして相関を
   3本取る必要がある（複素 FFT が1本増える）
3. **mining round 1** —— §5.14.2 の分布ずれは、本来 hard-negative mining が
   解決する問題そのものである。学習後パラメータで再採掘すれば学習分布と配備分布が
   一致し、勾配ゼロの72件の一部も救済されうる
4. **N の scaling law**（N=500, 1000）—— 本来の目的。健全な学習が確認できたので
   初めて答えられる状態になった

#### 5.14.7 要因分解の再解釈（2026-07-26）

共同学習した seed 0 checkpoint の IFACE と PSC パラメータを固定 TEST プール上で
post-hoc に差し替えた。これは「各ブロックだけを凍結条件で再学習した ablation」では
なく、**共同最適化された解の cross-swap** である。

| 条件 | $\alpha$ | clash $(w_{ss},w_{sc},w_{cc})$ | success@1 | AUC | baseline との勝/敗 |
|---|---:|---:|---:|---:|---:|
| baseline | 1.000 | (12.2, 42.9, 150.1) | 69.5% | 0.8696 | — |
| free checkpoint の IFACE、既定 PSC | 1.000 | (12.2, 42.9, 150.1) | **76.7%** | **0.9111** | 20/3 |
| free checkpoint の PSC、既定 IFACE | 0.896 | (13.0, 40.1, 150.0) | 67.8% | 0.8074 | 2/6 |
| free checkpoint 全部 | 0.896 | (13.0, 40.1, 150.0) | 75.0% | 0.8667 | 16/3 |
| free IFACE、$\alpha=0.896,\rho=3.5$ | 0.896 | (11.0, 38.4, 134.4) | 75.8% | 0.9138 | 18/3 |
| rho checkpoint 全部 | 0.982 | (12.8, 46.3, 167.1) | 74.6% | 0.8761 | 13/1 |
| rho IFACE、$\alpha=0.982,\rho=3.5$ | 0.982 | (12.0, 42.1, 147.4) | 73.7% | 0.9005 | 12/2 |

**測定事実。** 236件での成功数は baseline / IFACE / PSC / full =
164 / 181 / 160 / 177。したがって成功率スケールの2×2 interaction
`full - IFACE - PSC + baseline` は丸め前でも **0.0 pp** だった。ただし複合体ごとの
interaction は16件で非ゼロで、complex bootstrap 95% CI は
**−3.39〜+3.39 pp**。平均が相殺したのであって、ブロックが全poseで独立という意味ではない。

この結果は「この共同解の改善は PSC ではなく IFACE 側に局在する」を強く支持する。
一方で「IFACE だけを学習するのが最適」と結論するには、PSCを最初から凍結して
IFACEだけを再学習し、validation 選択と end-to-end 探索をやり直す必要がある。
また「論文 clash」と記した後半2条件は、絶対的な既定重みではなく
$w_k=\alpha\rho^k$ として学習後 $\alpha$ で一括縮小した条件である。

**IFACE 内部の post-hoc 分解（free seed 0、既定 PSC）。**
学習差分 $\Delta e$ を grand mean、受容体/リガンド atom-type の行列主効果、
double-centered pair residual に分けて固定プールを再採点した。

| IFACE 差分 | success@1 | AUC | baseline との勝/敗 |
|---|---:|---:|---:|
| 全差分 | 76.7% | 0.9111 | 20/3 |
| 一様成分のみ（接触数ボーナス） | 69.9% | 0.8778 | 1/0 |
| 行＋列の atom-type 主効果 | 74.6% | 0.9049 | 14/2 |
| double-centered pair residual | 71.6% | 0.8841 | 7/2 |
| 対称化した全差分 | 76.7% | 0.9117 | 21/4 |

純接触数 ranker は AUC 0.8911 と高いが success@1 は **31.4%**、PSC pair count
単独も36.9%だった。したがって IFACE の改善を単なる接触数の再発見だけで説明する
ことはできない。ただし改善の大半が特異的な144 pair interaction ではなく
12+12個の atom-type 主効果で再現するため、「化学的な対ポテンシャルを学んだ」
という主張もまだ強すぎる。低次元の行＋列モデルと、対称化した pair residual を
独立に再学習する対照が必要である。

**残る交絡と機構仮説。**

- 固定プールは baseline パラメータの top-1500 で条件付けられている。clash feature の
  範囲が切り詰められ、baseline 選択を collider とする相関反転が起こりうる。
  IFACE-only も含め、最終判断は全249件の end-to-end 再探索で行う。
- 69.5% は baseline が positive を返した236件に条件付けた率で、全249件では65.9%。
  配備指標の主分母は後者である。
- `loss_margin_hard_negatives` は **最低スコアの positive** を全 negative より上げる。
  DockQ 0.23 境界の衝突的な pose が1つあるだけで、clash を弱める勾配になりうる。
  margin 無し、DockQ閾値/quality weighting、best-positive または上位quantile margin
  を比較すべきである。
- 同じ split と同じ round-0 pool の3 optimizer seed 一致は、サンプル安定性ではない。
  complex bootstrap、別 master-list order、学習複合体の jackknife が必要である。
- 236件中 **62件が same-UniProt homodimer** で、現 DockQ は対称解を扱えない。
  IFACE cross-swap の勝/敗のうち6/3件がこの群だったが、除外群でも14勝0敗なので
  改善そのものはこの交絡だけでは説明できない。確認評価は symmetry-aware DockQ
  または事前規定した heteromer subset を用いる。
- TEST は多数の設計判断に反復利用済みで、今回の McNemar p 値も探索的である。
  モデル選択を凍結した後、未使用 cohort または outer split で確認する。

**round 1 に関する訂正。** 現在の `absorb()` は round 1 で新規 pose のうち
DockQ < 0.23 の negative だけを追加し、新しく見つかった positive は捨てる。
したがって §5.14.6 の「勾配ゼロの72件の一部も救済」は**現実装では起こらない**。
さらに free mode は FFT 探索経路が未実装で、pool cap の採点も既定 `rho` mode の
`Params` を構築している。round 1 の前に「hard-negative-only」か
「on-policy pool refresh（positiveも更新）」かを分け、後者なら別実験として扱う。

**再解析コマンドと計算環境。** 既存の
`data/shards_pinder/test_pool_reachable.pt`、`runs_free/.../round0_ckpt.pt`、
`runs_v3/.../round0_ckpt.pt` を CPU で読み、
`scripts/compare_conditions.py::score_pool` と同じ倍精度採点を用いた
（seed 0、bootstrap 20,000回、seed 0）。新規探索・学習は行っていない。

**結論。** 現時点の最も狭く defensible な結論は、
「共同学習された round-0 解を baseline 生成固定プールへ適用すると、top-1 改善は
IFACE パラメータの差し替えで再現し、PSC 差し替え単独では再現しない」である。
独立 IFACE-only 学習、全249件 end-to-end、未使用確認 cohort までは
「clash 学習は一般に有害」「free 制約解除が有効」とは主張しない。

**優先する次実験。**

1. 既定 PSC を凍結した IFACE-only 学習を3 optimizer seedで行い、全249件を同じ
   Hopf grid で end-to-end 再探索する。行＋列、対称 pair residual も対照にする。
2. margin loss の標的（最低positive）を ablate し、PSC-only の符号反転が消えるか測る。
3. 小規模な unbound pilot で IFACE gain と clash の方向が保存されるか先に確認する。
4. mining は契約を修正して round0/round1 を同一seed・同一Nで比較する。
5. 学習レシピを凍結してから、複数の master-list order を持つ N scaling を行う。

#### 5.14.8 IFACE のみ学習が最良（`--freeze-psc`、3 seed、end-to-end 確認済み）

§5.14.7 が留保した「PSC を最初から凍結して IFACE だけ再学習する」を実施した。
α と衝突重みを論文値（$\alpha=1$, $\rho=3.5$）に固定し、144 の対項だけを学習する。

**固定プール、探索由来 pose のみ、236複合体、3 seed の平均 ± SD**
（baseline: success@1 69.5%、AUC 0.8696、best DockQ@1 0.4457）

| 学習対象 | success@1 | AUC | best DockQ@1 | 勝/敗 |
|---|---:|---:|---:|---:|
| α + ρ + IFACE（146） | 74.3 ± 0.2 % | 0.8719 ± 0.0041 | 0.4691 ± 0.0009 | 13/1 |
| α + 3衝突重み + IFACE（148） | 75.0 ± 0.0 % | 0.8667 ± 0.0000 | 0.4765 ± 0.0000 | 16/3 |
| **IFACE のみ（144、PSC 凍結）** | **77.5 ± 0.0 %** | **0.8924 ± 0.0009** | **0.4893 ± 0.0001** | **21/2** |

学習は3 seed とも 1500 ステップを完走し（early stopping せず）、
‖ΔIFACE‖ = 1.551〜1.572 と高度に一致した。
**PSC を同時に学習しないほうが良く、AUC では IFACE-only だけが baseline を明確に
上回る**（他の2条件は 0.867〜0.872 で横ばい、free は baseline 未満）。

**end-to-end 再探索（学習後パラメータで FFT 探索そのものを回す、249複合体、seed 0）**

| K | baseline | α+ρ+IFACE | **IFACE のみ** | IFACE のみ vs baseline |
|---:|---:|---:|---:|---|
| **1** | 65.9% | 70.7% | **73.5%** | **21勝2敗 p=6.6e−05** |
| 5 | 79.9% | 82.3% | **85.1%** | 14勝1敗 p=0.00098 |
| 10 | 82.7% | 86.3% | **87.6%** | 13勝1敗 p=0.0018 |
| 50 | 89.6% | 91.6% | **92.8%** | 9勝1敗 p=0.021 |
| 100 | 91.2% | 93.6% | **94.8%** | 9勝0敗 p=0.0039 |

| 指標 | baseline | α+ρ+IFACE | IFACE のみ | Wilcoxon p |
|---|---:|---:|---:|---:|
| 返却集合の最良 DockQ | 0.5986 | 0.6071 | **0.6100** | 0.0086 |
| recall | 94.8% | 96.4% | **96.8%** | — |
| first-hit 順位 | 94.0 | 67.0 | **62.4** | 3.6e−10 |

**固定プールの結論が end-to-end でも保存された。** α と ρ が論文値のままなので
探索が返す候補集合は baseline とほぼ同じであり、**差は純粋に対ポテンシャルによる
並べ替えである。**

##### 学習された差分の中身 — 「化学を学んだ」は言い過ぎ

§5.14.7 の分解を IFACE-only チェックポイントで独立に再現した。
学習差分 $\Delta e$ を grand mean・行/列の主効果・double-centered residual に分ける:

| IFACE 差分 | success@1 | AUC | baseline との勝/敗 |
|---|---:|---:|---|
| baseline（差分なし） | 69.5% | 0.8696 | — |
| **全差分** | **77.5%** | **0.8933** | 21/2 (p=6.6e−05) |
| 一様成分のみ（接触数ボーナス） | 69.9% | 0.8585 | 1/0 (p=1) |
| **行+列の主効果（12+12個）** | **74.2%** | 0.8852 | 12/1 (p=0.0034) |
| pair residual（144の相互作用） | 71.6% | 0.8748 | 7/2 (**p=0.18**) |
| 対称化した全差分 | 75.8% | 0.8945 | 18/3 (p=0.0015) |

**改善の 59%（+4.7 / +8.0 pp）は原子タイプごとの「行＋列の主効果」24個で再現し、
144成分の特異的な対相互作用は単独では有意でない（p=0.18）。**

したがって学習の主成分は **「どの原子タイプが界面に居やすいか」という1体的な傾向**
であって、「どの原子タイプ同士が好き合うか」という2体的な特異性ではない。
**「化学的な対ポテンシャルを学習した」という主張は現時点の証拠より強い。**
低次元の行＋列モデルを独立に学習する対照が必要である。

> **訂正（2026-07-27、§5.14.12）。** 直前の段落の「144成分の特異的な対相互作用は
> 単独では有意でない（p=0.18）」を根拠に pair specificity の必要性を疑うのは
> **誤りである**。$p=0.18$ は「pair residual **のみ**が baseline を上回るか」の
> 検定であって、「pair residual を additive に**加える**増分」ではない。
> 低次元モデルを独立に学習する対照（この段落が要求したもの）を §5.14.12 で実施した
> 結果、**additive(23) → full(144) の増分は 3 seed とも exact McNemar $p=0.0215$、
> AUC で Wilcoxon $p\sim10^{-7}$ と有意である**。
> 「改善の約58%が1体的成分で再現できる」は正しいままだが、
> **「144成分は不要」は誤り**である。残り約42%を担う121次元の pair residual の寄与は
> 有意である。あわせて、1体的成分は12次元の対称 additive で足り、
> 23次元へ増やしても改善は無い（完全な null）。

##### 残る限界（§5.14.7 の指摘を含む）

- **bound-bound** である（250/250 が同一 PDB 由来）。unbound は未測定。
- **TEST は多数の設計判断に反復使用済み**で、p 値は探索的である。
  レシピを凍結してから未使用コホートで確認すべきである。
- 236件中 **62件が same-UniProt homodimer** で DockQ が対称解を扱えない
  （§5.7-S9）。ただし除外群でも IFACE 差替えは 14勝0敗で、改善自体は
  この交絡では説明できない。
- 3 seed の一致は**optimizer seed** の一致であってサンプル安定性ではない。
  complex bootstrap / master-list order の変更 / 学習複合体の jackknife が要る。
- end-to-end は seed 0 のみ。
- **`absorb()` の契約**: round 1 は新規 pose のうち DockQ < 閾値の negative しか
  取り込まず、新たに見つかった positive を捨てる（コードで検証）。したがって
  §5.14.6 の「mining round 1 で勾配ゼロの72件が救済されうる」は現実装では起こらない。
  round 1 の前に「hard-negative のみ」か「on-policy でプール全体を更新」かを
  決める必要がある。

##### 次段階（優先順）

1. **margin loss の標的**（最低スコアの positive）を ablate し、PSC 同時学習が
   悪化する機構を確認する
2. **低次元（行＋列）モデルの独立学習**——144成分が本当に必要かの対照
3. **unbound pilot**——IFACE gain がドメインを越えるか
4. `absorb()` の契約を決めてから **mining round 1**
5. レシピ凍結後に **N の scaling law**

#### 5.14.9 実験1設計: IFACE の低次元対照（2026-07-26、実行前）

**問いを2つに分ける。**

1. **適応自由度の問い**: 公開済み IFACE 表 $e_0$ は固定して保持したまま、
   学習差分に144自由度が必要か。それとも
   $\Delta e_{ij}=g+r_i+c_j$ の additive subspace だけで十分か。
2. **pair specificity の問い**: 最終スコアそのものに pair-specific residual が
   必要か。

提案中の $e=e_0+g+r_i+c_j$ が直接答えるのは **1だけ**である。両モデルとも
$e_0$ の144成分と、その double-centered pair residual を配備時に保持するため、
低次元更新が full 更新と同等でも「化学的pair potentialは不要」
「最終モデルに144成分は不要」とは結論できない。問い2には、$e_0$ 自身を
additive subspace へ射影し、pair residual を除いた最終モデルを別に評価する。

##### パラメータ化と識別性

$$
\mathcal A=\{\Delta E=g\mathbf1\mathbf1^\top
 +r\mathbf1^\top+\mathbf1c^\top:
\mathbf1^\top r=\mathbf1^\top c=0\}.
$$

raw 25変数には2次元の gauge
（$r\leftarrow r+a$, $c\leftarrow c+b$, $g\leftarrow g-a-b$）があり、
関数空間の次元は **23**。priorを有効な144表に課せば予測値は一意になるが、
optimizerには平坦方向と座標スケール差が残る。特に同じ learning rate で
raw $g,r,c$ をAdam更新すると、$g$ の勾配は144セル、各 $r_i,c_j$ は12セルの
勾配和になり、full 144座標と最適化条件が揃わない。

したがって $\sum r=\sum c=0$ は課す。実装は毎stepの事後centeringより、
12次元の zero-sum orthonormal contrast $U\in\mathbb R^{12\times11}$ を用いる
23次元の直交基底を推奨する:

$$
\Delta E=
\theta_g{\mathbf1\mathbf1^\top\over12}
{(U\theta_r)\mathbf1^\top\over\sqrt{12}}
{\mathbf1(U\theta_c)^\top\over\sqrt{12}}.
$$

このとき
$\|\Delta E\|_F^2=\theta_g^2+\|\theta_r\|^2+\|\theta_c\|^2$ で、
識別性と座標スケールを同時に揃えられる。簡易実装としてraw $r,c$ をforward時に
zero-centerすることもできるが、2つのnull座標とAdamのmomentは残る。

##### prior と最適化の公平性

主比較では両モデルに同じ**関数空間ノルム**

$$\lambda\|\Delta E\|_F^2$$

を課す。次元数に応じて $\lambda$ を25/144倍する操作はしない。
現在の `loss_param_prior(p.iface_vec(), p0.iface_vec())` はこの定義に合う。
zero-sum gaugeでは同じpriorは
$\lambda(144g^2+12\|r\|^2+12\|c\|^2)$ であり、
raw coefficient norm $\lambda(g^2+\|r\|^2+\|c\|^2)$ は別のpriorなので使わない。

ただし同じ $\lambda$ は「既存のfullモデルと同じ摂動コスト」という対照であって、
各モデルに最良の汎化性能を与えるとは限らない。以下を分ける:

- **主解析**: 現在の $\lambda=0.1$、同じloss、同じfit/validation split。
- **感度解析**: 両モデルそれぞれ同じ事前規定gridから $\lambda$ とlearning rateを
  validationだけで選ぶ。探索した候補数と予算を揃える。

full IFACE-only は1500 stepでearly stopせず終了しており、まだ改善中だった可能性がある。
同じstep数は同じ最適化品質を意味しないため、両モデルともvalidation曲線がplateauする
十分大きな上限を与え、同じearly-stopping ruleを用いる。低次元側だけ収束が速い状態を
capacity差と誤認しない。

現 `_val_loss()` はpriorを含む `mean_objective()` をそのままcheckpoint選択に使う。
これは「予測validation loss」ではなく「validation loss + parameter preference」であり、
$\lambda$ 比較時に小ノルムを評価側でも再度優遇する。既存条件を忠実に再現する主解析では
両モデルに同じ規則を使うが、best-achievable比較では学習目的にpriorを残しつつ、
checkpoint/$\lambda$選択はpriorを除いたvalidation ranking lossで行う感度解析を設ける。
この規則を変える場合はfull 144も同じコードで再学習し、過去checkpointと混ぜない。

##### 比較条件

最低限の直交したモデル族は次のとおり。

| 条件 | 最終表 | 答える問い |
|---|---|---|
| fixed baseline | $e_0$ | 学習なし |
| scalar update | $e_0+g$ | 接触数スケールだけで十分か |
| additive update | $e_0+\mathcal A$ | 23更新自由度で十分か |
| residual update | $e_0+\mathcal R$ | pair-specific更新だけで効くか |
| full update | $e_0+\mathcal A+\mathcal R$ | 144更新自由度の上限 |
| absolute additive | $P_\mathcal A(e_0)+\mathcal A$ | 最終スコアにpair residualが必要か |

$\mathcal R$ は行和・列和がゼロの double-centered residual（121次元）で、
$\mathcal A$ とFrobenius内積で直交する。物理的な受容体/リガンド交換対称性も別軸で
確認する。PINDERのR/L役割が任意または組成依存なら、非対称な $r_i,c_j$ は
datasetの役割規約を利用できるため、
$\Delta e_{ij}=g+a_i+a_j$（12実効次元）と role-swap augmentation を含める。
対称fullは78次元、対称double-centered residualは66次元である。

rank-1 residualはこの直交分解の後に行う。$uv^\top$ は非凸でscale/rotationの
識別性も加わり、「低次元だから勝った」のか「最適化に失敗した」のか分けにくいため、
最初の対照としては不適切。

##### 統計と判定

- 固定プール236件はscreening、主評価は同じHopf gridによる**全249件
  end-to-end success@1**。複合体が統計単位で、poseを独立標本として数えない。
- full対low-dimensionalの最終予測を同じ複合体で比較する exact paired McNemar は、
  モデルがnestedでも有効。nestednessが問題になるのは学習likelihoodの尤度比検定であり、
  held-out 0/1予測のpaired比較ではない。
- 「fullが有意に良い」は `full vs additive` の直接McNemarと、paired差のCIで評価する。
  baselineとの差が各々有意でも、両者の差を示したことにはならない。
- post-hoc residual-onlyのbaseline比 $p=0.18$ は、residualの**additiveへの増分**を
  検定していない。また「+4.7/+8.0 pp = 59%」は非線形なargmax指標の差の比で、
  分散説明率や因果的寄与率ではない。いずれも実験動機であって判定結果には使わない。
- 「fullが有意でない」だけでは144自由度が不要とは言えない。
  事前に非劣性margin $\delta$（例: end-to-end success@1で1〜2 pp）を定め、
  $p_\mathrm{additive}-p_\mathrm{full}$ のpaired cluster bootstrapまたはpaired
  score CIの下限が $-\delta$ を上回ることを要求する。249件では狭いmarginの
  非劣性は低powerになりうるため、必要なら未使用評価複合体を増やす。
- 3 seedは同じ学習集合・同じpoolのoptimizer反復であり、$n=3$の独立な統計標本ではない。
  seedの集約法をvalidationだけで事前規定し、別master-list orderまたは
  training-complex bootstrapで学習標本不確実性を測る。
- 現PINDER-S TESTは設計に反復利用済みなので、ここでのp値は探索的。
  $\lambda$、loss、model familyを凍結後、未使用outer cohortで確認する。

##### 実験順序

低次元性の結論はlossに条件付く。worst-positive marginがfull residualだけに
境界positiveの癖を学ばせている可能性があるため、実験1とmargin ablationを完全に
直列化しない。安価な固定pool学習で

$$
\{\text{additive},\text{full}\}\times
\{\text{current margin},\text{marginなし/quality-aware}\}
$$

の2×2を先に行い、dimension×loss interactionを見る。その後validationだけで
レシピを凍結し、選んだ条件をend-to-end評価する。

続いて unbound pilot、mining契約の修正とround-1 pilot、最後に複数master-list orderの
N scalingとする。miningを最終pipelineに含めるか決めずにN scalingを先に行うと、
round-0専用のscaling curveになる。現 `hard-negative-only` round 1 は新positiveを捨て、
round-0でpositiveゼロの複合体を救わないことを比較の前提に明記する。

**データ・split・loss・seed・hardware・commands。** 本節は設計レビューであり新規runは
行っていない。提案実験は§5.14.8と同じfit 220 / validation 55、search由来poseのみ、
PSC固定（$\alpha=1,\rho=3.5$）、basin + margin + priorを出発点とする。
optimizer seed 0/1/2に加え、学習標本の反復を別途設ける。固定pool段階はCPU/GPUいずれでも
同一倍精度特徴を使用し、end-to-endは同じHopf nside=3、spacing 1.2 Å、top-1500で行う。
実装とコマンドはパラメータ化・loss gridを確定してから記録する。

**結論と限界。** additive更新がfull更新に非劣性でも証明されるのは
「このデータ・loss・公開済み $e_0$ 周りでは23の適応自由度で足りる」までである。
pair chemistry自体の不要性にはabsolute-additive対照と未使用cohortが必要である。

#### 5.14.10 次元×margin factorialの学習完了とTEST解析規則（2026-07-27、評価中）

full（144）/ additive（23）/ symmetric additive（12）×
$\lambda_\mathrm{margin}=0.5/0.0$ × optimizer seed 0/1/2 の18 runが完走した。
全条件で `--freeze-psc --loss-prov search --alpha0 1.0 --n-fit 220 --rounds 0`、
1500 step、fit 220 / validation 55 / TEST 249。round-0 pool、split、PSC、lossの
margin以外、priorは同一。新規学習の失敗、skip、OOMは報告されていない。

| 次元 | val loss m5 / m0（3 seed平均） | $\|\Delta e\|$ m5 / m0 |
|---|---:|---:|
| full 144 | 2.7243 / 2.2491 | 1.561 / 1.583 |
| additive 23 | 2.7952 / 2.3228 | 1.218 / 1.244 |
| symmetric 12 | 2.7763 / 2.3028 | 1.087 / 1.127 |

m5とm0は異なる目的関数なのでval lossの絶対値を比較しない。同じmargin内ではfullが
最小だが、これはtraining familyがnestedであることと整合し、held-out top-1の優越性を
単独では示さない。

seed 0 / m5 の固定TEST pool（探索由来positiveを持つ236件）:

| 条件 | success@1 | AUC | best DockQ@1 | baselineとの勝/敗 |
|---|---:|---:|---:|---:|
| baseline | 69.5% | 0.8696 | 0.4457 | — |
| full | 77.5% | 0.8933 | 0.4893 | 21/2 |
| additive | 74.2% | 0.8852 | 0.4743 | 12/1 |
| symmetric | 73.3% | 0.8847 | 0.4674 | 10/1 |

seed 0 / m5 の additive→full の直接差は McNemar $p=0.0215$、
per-complex AUC差は Wilcoxon $p=2.1\times10^{-7}$。これはseed 0における
探索的なfullの増分であり、残りseed・margin評価前の中間結果である。

**seed集計規則。** TEST複合体はseed間で同一なので、3×236を独立標本としてpoolしない。
各条件のmean ± SDはoptimizer感度の記述量として示す。各seedで同じseed同士をpairedに
比較し、効果量・勝敗・p値3本をすべて報告し、最小pだけを選ばず範囲と方向一致を見る。
seedをまたぐ単一p値は出さない。median parameterを事後に作ると新しい未規定modelに
なるため採用しない。3-seed平均score ensembleを配備modelとして事前定義する場合に限り、
複合体あたり1予測へ集約して1回のpaired検定を行う。

**margin比較規則。** まず同じvalidation poolを、training objectiveとは独立な共通metric
（success@1、best DockQ@1、AUCなど）で再採点してm5/m0を選ぶ。TESTで選択する場合は
探索的method comparisonと明記する。TEST上で3次元すべてのm5対m0を主張するなら、
primary endpointをsuccess@1に固定し、3つのMcNemar pへHolm補正を適用する。
AUC/best DockQ/K>1はsecondaryとし、同じfamilyで追加の有意性主張をしない。

このfactorialが直接答えるmarginの問いは、**PSC固定下でworst-positive marginを加えると
IFACE学習と次元差がどう変わるか**である。PSC悪化の原因がmarginかは、PSCが全runで
凍結されているため未回答。そこを検証する最小追加は、full IFACEで
PSC trainable/frozen × m5/m0 の2×2である。

best-positive marginは1 poseだけで条件を満たしbasin lossと役割が重なるため、現時点では
追加しない。m0が全次元でm5以上ならmarginを削除して終了する。m5が一貫して有利だが
full/additive差または失敗例がworst-positiveに集中する場合だけ、validationで選ぶ
1つのrobust variant（例: positive scoreの下位10〜25%を捨てたquantile/CVaR margin）
を事前固定して追加する。TEST結果を見てbest/quantileを両方探索しない。

**評価進行状況。** 本節記載時点でseed 0 / m5以外の15 TEST評価が実行中。
未完了値から結論・model選択は行わない。全18セル完了後、上記規則で結果、null、
seed不一致を含めて追記する。主な限界は同一split/同一poolのoptimizer seedのみ、
反復利用済みPINDER-S、固定pool条件付き236件である。最終候補は全249件end-to-endと
未使用cohortで確認する。

#### 5.14.11 TESTを見る前の validation 選択（2026-07-27）

§5.14.10 の margin 比較規則に従い、**TEST を見る前に** validation split を
training objective と独立な共通指標で再採点した。margin 0.5 と 0.0 は異なる
目的関数を最適化しているので validation *loss* は比較できないが、
success@1 / AUC / best DockQ@1 は比較できる。

round-0 の pool cache は seed 非依存なので1回読めば全 run に使える。
`split.json["val_ids"]`（55件）だけが seed ごとに異なる。
探索由来 pose のみ（`prov == 0`）で採点した。

**実装**: `scratchpad/val_select.py`（下記コマンド節）。
success@1 は探索 pose 集合を持つ **55件すべて** で数える。探索が近native pose を
1つも返さなかった複合体は「未定義」ではなく失敗であり、これを落とすと
success@1 が水増しされ、しかも33件しか残らない。AUC と best DockQ@1 は
定義できる33件に限る。

| 次元 | margin | val success@1 (n=55) | val AUC (n=33) | val bestDQ@1 (n=33) |
|---|---|---:|---:|---:|
| baseline | — | 23.6% | 0.7148 | 0.2992 |
| full 144 | **0.5** | 25.5% | **0.7603** | 0.3161 |
| full 144 | 0.0 | 25.5% | 0.7561 | 0.3159 |
| additive 23 | 0.5 | 25.5% | 0.7418 | 0.3133 |
| additive 23 | 0.0 | 25.5% | 0.7367 | 0.3157 |
| symmetric 12 | 0.5 | 25.5% | 0.7439 | 0.3133 |
| symmetric 12 | 0.0 | 25.5% | 0.7389 | 0.3184 |

（3 seed 平均。seed 間の差は AUC 第4位以下で、表示桁ではほぼ動かない。）

**measured facts**

- **validation の success@1 は18条件すべてで完全に同一**である。
  13/55 → 14/55 で、反転した複合体は1件のみ、勝敗は全条件 1勝0敗、
  exact McNemar $p=1$。**n=55 では primary endpoint として解像度がない。**
  codex が推奨した「validation で success@1 を primary にして選ぶ」は、
  この split サイズでは実行できない。
- 解像度があるのは **AUC のみ**。margin 0.5 が3次元すべてで 0.0 を上回る
  （$+0.0042$ / $+0.0051$ / $+0.0050$）。次元は両 margin で
  **full > symmetric > additive** の順。
- **best DockQ@1 は逆向きの弱い証拠**を出す。additive と symmetric では
  m0 のほうがわずかに良い（$+0.0024$ / $+0.0051$）。full ではほぼ同点。

**選択（事前規定の規則に従った結果）**: primary が使えないので AUC を選択指標
とし、**margin = 0.5、次元 = full** を選ぶ。ただしこれは**単一指標による選択**
であって、best DockQ@1 は低次元側で逆を示す。この不一致を隠さずに記録する。

**この validation split は TEST と交換可能ではない（重要な caveat）**

| | 訓練/検証プール | TEST プール |
|---|---:|---:|
| 探索が近native poseを返した割合 | **65.8%** (181/275) | **94.8%** (236/249) |
| baseline success@1 | 23.6% | 69.5% |
| baseline AUC | 0.7148 | 0.8696 |

同じ mining 設定（Hopf nside=3、spacing 1.2 Å、top-1500、±1セル列挙）で
作ったにもかかわらず、訓練/検証側は探索 recall が 29 pp 低く、baseline の
success@1 は 3分の1である。したがって

- validation は TEST よりはるかに難しい複合体集合であり、
  validation で選んだ設定が TEST で最適である保証は弱い。
- §5.14.8 以降の TEST 上の数値を「難易度中立な汎化性能」と読んではならない。

原因は未特定である。仮説: (a) PINDER の train split と PINDER-S の
test split では複合体の大きさ・種類の分布が違う、(b) `--max-grid-voxels`
による大複合体のスキップが TEST 側を系統的に易しくしている、
(c) 訓練側 prep cache に §5.12 と同種の幾何破綻が残っている。

**(c) は排除された（測定済み）。** `scripts/check_prep_cache.py` を訓練側
cache（master list の先頭275件）に対して実行した:

```
data/scaling/prep_cache: 275 complexes checked (0 missing)
  closest receptor-ligand contact: median 2.66 A, min 0.24 A, p90 3.13 A
  STERICALLY IMPOSSIBLE (< 2.0 A): 9/275
  >90% of receptor atoms within 4.5 A of the ligand: 0/275
  native pose not reproduced by (q*, t*) to 0.01 A: 0/275
```

中央値 2.66 Å は修復後 TEST の 2.72 Å と同水準で、§5.12 のような全件破綻はない。
立体的に不可能な9件は**すべて既に `data/scaling/excluded_bad_geometry.txt` に
載っており（9/9）、3 seed いずれの fit 220 / validation 55 にも1件も入っていない**
（0/220、0/55）。`--exclude-bad-geometry` が意図どおり効いている。
したがって recall ギャップはデータ破綻では説明できず、(a)/(b) は未検証のまま残る。

除外された9件（受容体-リガンド最短重原子距離）:
6r6h (0.24 Å)、6b23 (0.28)、2x31 (0.51)、6hiv (0.69)、3j16 (1.33)、
5jpq (1.45)、6xyw (1.80)、8fm9 (1.82)、6swa (1.83)。
（訓練 master list は 4000件、うち83件が同ファイルで除外済み。）

#### 5.14.12 実験1の結果: 3×2×3 factorial の TEST 評価（2026-07-27）

18セルすべての固定 TEST プール評価が完了した。§5.14.10 の集計規則に従い、
3×236 をプールせず、mean ± SD は optimizer 感度の記述量として、
検定は seed ごとに3本すべて報告する。

**設定。** `--freeze-psc --loss-prov search --alpha0 1.0 --n-fit 220 --rounds 0`、
1500 step（全 run で early stop せず完走）、$\lambda_\mathrm{prior}=0.1$、
prior は関数空間ノルム $\lambda\|\Delta E\|_F^2$。
評価は `data/shards_pinder/test_pool_reachable.pt`、`--prov search`、
DockQ 閾値 0.23、249件中 **236件**が対比較可能（13件は positive か negative が無い）。
baseline は公開パラメータ（$\alpha=1.0$, $\rho=3.5$、学習なし）で
success@1 69.5%、AUC 0.8696、best DockQ@1 0.4457、first-hit 1.098%。

##### 主表（3 seed 平均 ± SD、236複合体）

| 次元 | dof | margin | success@1 | AUC | best DockQ@1 | first-hit % |
|---|---:|---|---:|---:|---:|---:|
| baseline | 0 | — | 69.49% | 0.8696 | 0.4457 | 1.098 |
| **full** | **144** | **0.5** | **77.54 ± 0.00%** | **0.8924 ± 0.0009** | **0.4893 ± 0.0001** | **0.545 ± 0.007** |
| full | 144 | 0.0 | 75.85 ± 0.00% | 0.8828 ± 0.0011 | 0.4807 ± 0.0003 | 0.624 ± 0.011 |
| additive | 23 | 0.5 | 74.15 ± 0.00% | 0.8840 ± 0.0010 | 0.4743 ± 0.0000 | 0.668 ± 0.008 |
| additive | 23 | 0.0 | 74.58 ± 0.00% | 0.8730 ± 0.0013 | 0.4739 ± 0.0000 | 0.802 ± 0.017 |
| symmetric | 12 | 0.5 | 73.59 ± 0.24% | 0.8832 ± 0.0013 | 0.4682 ± 0.0007 | 0.639 ± 0.014 |
| symmetric | 12 | 0.0 | 73.87 ± 0.24% | 0.8718 ± 0.0015 | 0.4666 ± 0.0027 | 0.773 ± 0.021 |

**6条件すべてが baseline を有意に上回る**（seed ごとの exact McNemar
$p = 6.6\times10^{-5}$〜$0.012$、勝敗は 10勝1敗〜21勝2敗）。
success@1 の optimizer seed 依存性は full/additive で **ちょうど 0**、
symmetric で ±0.24 pp（1複合体）である。

##### 主要対比: additive(23) → full(144)、margin = 0.5

margin は §5.14.11 の validation で選んだ（TEST を見る前）。

| seed | success@1 | 勝/敗 | McNemar p | AUC | Wilcoxon p |
|---:|---|---|---:|---|---:|
| 0 | 74.15% → 77.54% | 9勝1敗 | **0.0215** | 0.8852 → 0.8933 (+0.0081) | $2.1\times10^{-7}$ |
| 1 | 74.15% → 77.54% | 9勝1敗 | **0.0215** | 0.8832 → 0.8916 (+0.0084) | $5.9\times10^{-8}$ |
| 2 | 74.15% → 77.54% | 9勝1敗 | **0.0215** | 0.8837 → 0.8923 (+0.0086) | $7.6\times10^{-8}$ |

3 seed で完全に一致する（同じ TEST 複合体なので独立な3検定ではない）。
効果量は **$+3.39$ pp、複合体 paired bootstrap 95% CI $[+0.85, +5.93]$ pp**
（seed 0、20000回）。
best DockQ@1 は $+0.015$ だが Wilcoxon $p = 0.10$〜$0.14$ で**有意ではない**。
first-hit 順位は $-0.0011$〜$-0.0014$、$p = 2$〜$3\times10^{-5}$。

##### 副次対比: symmetric(12) → additive(23) は検出されず

| seed | success@1 | 勝/敗 | McNemar p | AUC delta | Wilcoxon p |
|---:|---|---|---:|---:|---:|
| 0 | 73.31% → 74.15% | 2勝0敗 | 0.50 | +0.0005 | 0.47 |
| 1 | 73.73% → 74.15% | 2勝1敗 | 1.00 | +0.0010 | 0.23 |
| 2 | 73.73% → 74.15% | 2勝1敗 | 1.00 | +0.0010 | 0.19 |

**非対称な追加11自由度の利益は検出されず、観測差は小さい**
（$+0.85$ pp、bootstrap 95% CI $[+0.00, +2.12]$ pp）。
**これは同等性の証明ではない。** §5.14.9 が事前に挙げた非劣性 margin
$\delta = 1$〜$2$ pp に対して、CI 上限 $+2.12$ pp は $\delta=2$ をわずかに超える。
つまり「12次元で足りる」とはまだ言えず、「差が検出できなかった」までである。

一方 symmetric → full は 11勝1〜2敗、$p = 0.006$〜$0.022$、
$+4.24$ pp（CI $[+1.69, +7.20]$ pp）である。
この設定で検出可能な利益をもたらしているのは、行/列の非対称性でも一様成分でもなく
**double-centered pair residual（121次元）**である。

##### §5.14.8 の主張の訂正

§5.14.8 は post-hoc 分解で「144成分の特異的な対相互作用は単独では有意でない
（$p=0.18$）」と書いた。**この検定は問いに答えていない。** $p=0.18$ は
「pair residual **のみ**（additive 成分を捨てた差分）が baseline を上回るか」
であって、「pair residual を additive に**加える**増分」ではない。
正しい増分検定は上の主要対比であり、**3 seed とも $p = 0.0215$ で有意**、
AUC では $p \sim 10^{-7}$〜$10^{-8}$ である。
この指摘は codex（gpt-5.6, herdr 経由）による。

**訂正後の言明**: 「改善の大半は1体的な additive 成分で再現できる」は依然として
正しい（baseline 超過分の $\frac{74.15-69.49}{77.54-69.49}=58\%$）。
しかし「144成分は不要」という含意は誤りである。正しい言明は

> **pair-specific な更新自由度は、additive 更新に対して固定プール success@1 を
> $+3.39$ pp 改善した（9勝1敗、exact McNemar $p=0.0215$、
> bootstrap 95% CI $[+0.85, +5.93]$ pp、3 seed 一致）。**

である。**「pair chemistry そのものが必要」とは言えない**: additive モデルも
公開済み $e_0$ の pair residual を配備時にそのまま保持しているため、
この実験が示すのは「$e_0$ の周りの**更新**に pair 自由度が要る」ことに限られる。
これを分離するには §5.14.9 の absolute-additive 対照
（$e_0$ 自身を additive 部分空間へ射影）が必要で、未実施である。
また「58% / 42%」は非線形な argmax 指標の差の比であって、分散説明率でも
因果的寄与率でもない。

##### margin の効果（実験2の主部）

各次元内で margin 0.0 → 0.5 を直接対比較した。

| 次元 | success@1 | 勝/敗 (seed 0/1/2) | McNemar p | AUC delta | Wilcoxon p |
|---|---|---|---:|---:|---:|
| full 144 | 75.85% → 77.54% (+1.69) | 4勝0敗 ×3 | 0.125 ×3 | +0.0094〜0.0098 | $\sim10^{-31}$ |
| additive 23 | 74.58% → 74.15% (−0.42) | 1勝2敗 ×3 | 1.00 ×3 | +0.0108〜0.0112 | $\sim10^{-32}$ |
| symmetric 12 | 73.87% → 73.59% (−0.28) | 0勝2敗 / 1勝1敗 / 1勝1敗 | 0.50 / 1.00 / 1.00 | +0.0112〜0.0116 | $\sim10^{-32}$ |

primary endpoint を success@1 に固定し、3次元へ Holm 補正（seed 0）:
full 補正後 $p=0.375$、symmetric $p=1$、additive $p=1$。

success@1 差の paired bootstrap 95% CI（seed 0、pp）:
full $+1.69\,[+0.42, +3.39]$、additive $-0.42\,[-1.69, +0.85]$、
symmetric $-0.85\,[-2.12, +0.00]$。
**full の CI は 0 を含まないが、exact McNemar は $p=0.125$ である。**
不一致は不一致ペアが4件しかないためで、この領域では bootstrap 区間が
楽観的すぎる。**判定は exact 検定を採る。**

**measured facts**

- **margin は success@1 をどの次元でも有意に変えない。** full での $+1.69$ pp は
  4勝0敗で、補正前でも $p=0.125$ である（exact McNemar は片方向4件では
  $0.125$ が最小 p であり、この効果量では原理的に有意になり得ない）。
- **一方 AUC は3次元すべてで一貫して改善する**
  （平均 $+0.009$〜$+0.012$、中央値 $\approx +0.006$、
  改善/悪化/不変がおよそ 195〜201 / 18〜25 / 14〜17 件）。
  Wilcoxon $p$ は $10^{-31}$ 台だが、**これは効果量ではなく符号の一貫性**を
  反映している。SciPy でも同じ桁で計算 artifact ではないが、
  以後 $p < 10^{-20}$ と丸めて扱う。示しているのは
  **全 positive-negative ペアにわたる広域順位の改善**であって、
  top-1 の改善ではない。両者は分離して読む必要がある。
  first-hit 順位も同様に3次元すべてで改善する（$p = 10^{-4}$ 台）が、
  **中央値の変化は 0** で、変化した約12〜15%の複合体に集中している。
- **次元と margin で符号の異なるパターンを観測した**: margin は full の
  success@1 を上げ (+1.69 pp)、低次元では下げる (−0.42 / −0.85 pp)。
  差の差は $+2.12$ pp（6対1、$p=0.125$）で、**交互作用が有意に示されたわけではない。**

**解釈（仮説、未検証）**: worst-positive margin は「最悪の positive をも
negative より上に押し上げる」制約であり、144次元の自由度があれば
境界 positive 専用の補正を学習できるが、12〜23次元では他の複合体を犠牲に
するしかない。観測された符号パターンと整合するが、証明はしていない。

##### 結論

以下はすべて**固定プール上の success@1、探索的に反復利用された PINDER-S TEST、
bound-bound、PSC 凍結**という条件付きの言明である。

1. **6条件すべてが公開パラメータを上回る**（21勝2敗〜10勝1敗、
   seed 0 の6比較を Holm 補正しても全条件 $p<0.05$）。
   IFACE のみの再学習で success@1 は 69.5% → 最良 77.5%。
2. **pair-specific な更新自由度は additive 更新に対して $+3.39$ pp の改善を与える**
   （9勝1敗、$p=0.0215$、CI $[+0.85,+5.93]$ pp、3 seed 一致）。
   §5.14.8 の「$p=0.18$ だから144成分は不要」という示唆を訂正する。
   ただし**pair chemistry そのものの必要性は未証明**である（両モデルとも
   $e_0$ の pair residual を保持している）。
3. **非対称な追加11自由度（sym 12 → add 23）の利益は検出されなかった**
   （$+0.85$ pp、CI $[+0.00,+2.12]$ pp）。**同等性の証明ではない。**
4. **margin は広域順位（AUC を約 $+0.01$）を一貫して改善したが、
   success@1 の改善は未確立**である（全次元で非有意、低次元では点推定が悪化）。
5. **working choice は full 144 + $\lambda_\mathrm{margin}=0.5$**。
   validation AUC で TEST を見る前に選んだ候補であり、**確定した最良モデルではない**。
   validation/TEST の難易度シフトと TEST の反復利用のため、次段階の候補として扱う。

##### 限界と、明記すべき不都合な結果

- **bound-bound のみ。** unbound は未測定。
- **固定プール上の再ランキング**であり、end-to-end 再探索ではない。
  §5.14.8 では固定プールの結論が end-to-end で保存されたが（77.5% → 73.5%）、
  今回の6条件で end-to-end を回したのは full+m5 のみ（= §5.14.8 の条件）である。
  **低次元条件と m0 条件の end-to-end は未評価。**
- **PINDER-S TEST は多数の設計判断に反復使用済み**で、p 値は探索的である。
  未使用 cohort での確認は未実施。
- **additive → full の best DockQ@1 は非有意**（$+0.015$、$p=0.10$〜$0.14$）。
  success@1 と AUC は動いたが、上位1件の構造品質そのものは確立していない。
- **first-hit 改善は中央値 0** であり、変化した約12〜15%の複合体に集中している。
  「全体的に浅く見つかるようになった」ではない。
- **validation は33件で選択を担っており、しかも TEST と順序が反転する**:
  validation では sym > add、TEST では add > sym。選択の根拠は薄い。
- **search-only loss で勾配を出す fit 複合体は 148/220** である（残り72件は
  探索由来 positive を持たず、損失に寄与しない）。実効的な学習サンプルは
  公称の 2/3 である。
- **全 run が 1500 step の上限に到達**しており、収束 plateau は未確認。
  「低次元側だけ収束が速い」を capacity 差と誤認していない保証はない。
- **3 seed は optimizer seed のみ**で、学習標本の不確実性ではない。
  success@1 の seed SD が full/additive でちょうど 0 なのは安定性の証拠だが、
  同じ fit 220 / 同じ round-0 プールを共有しているためでもある。
  master-list order 変更または training-complex bootstrap が必要である。
- **validation split は TEST と交換可能でない**（§5.14.11、探索 recall 65.8% vs 94.8%）。
- **margin が PSC 学習を悪化させたか**は未回答である。18 run すべてで PSC は凍結
  されている。最小追加は full IFACE × {PSC trainable, frozen} × {m5, m0} の 2×2。
- **absolute-additive 対照**（$e_0$ 自身を additive 部分空間へ射影）は未実施。

##### 次段階（codex と合意した順序）

1. margin を 0.5 に凍結。best-positive / quantile margin は**走らせない**
   （m5 が一貫して有利なので削除条件は満たさないが、追加variantを
   TEST を見てから選ぶのは post-hoc になる）。
2. `absorb()` の契約を決めてから mining round 1。
3. レシピ凍結後に N の scaling law（複数の master-list order で）。

##### コマンド

**「margin 0.5 / 0.0」は $\lambda_\mathrm{margin}$ である。** margin 幅そのもの
（`--margin`）は全 run で 1.0 に固定されており、変えたのは margin 項の重み
$\lambda_\mathrm{margin} \in \{0.5, 0.0\}$ である。$\lambda_\mathrm{margin}=0$ は
margin 項の除去に等しい。

```bash
# 学習 18本（GPU 0/3/6 に分散）
for mode in full add sym; do
  for lm in 0.5 0.0; do
    tag=$([ "$lm" = 0.5 ] && echo m5 || echo m0)
    for seed in 0 1 2; do
      uv run python scripts/run_pinder_scaling.py \
          --n-fit 220 --rounds 0 --seed "$seed" \
          --freeze-psc --loss-prov search --alpha0 1.0 \
          --iface-mode "$mode" --lambda-margin "$lm" \
          --pool-cache data/scaling/pool_cache \
          --exclude-bad-geometry data/scaling/excluded_bad_geometry.txt \
          --exclude-homodimer \
          --out-dir "data/scaling/runs_${mode}_${tag}"
    done
  done
done
```

その他のハイパーパラメータ（全 run 共通）:
`--margin 1.0 --lambda-prior 0.1 --iface-lr 5e-4 --alpha-lr 1e-3
--basin-temp 0.5 --batch-size 16 --epoch-passes 100 --min-steps 1500
--patience 8 --rot-set hopf --hopf-nside 3 --spacing 1.2 --mine-ntop 1500
--trans-cells 1 --near-rot 8 --max-grid-voxels 31250000 --dockq-thr 0.23`。

```bash
# TEST 評価 18セル
uv run python scripts/compare_conditions.py \
    --pool data/shards_pinder/test_pool_reachable.pt \
    --ckpt data/scaling/runs_<mode>_<tag>/N220_seed<s>/round0_ckpt.pt \
    --prov search --out-dir data/scaling/compare_fact_<mode>_<tag>_seed<s>

# 集計・対比較・validation 選択
python scripts/agg_fact.py      # -> data/scaling/factorial_summary.json
python scripts/contrast.py      # 直接対比較と Holm 補正
uv run python scripts/val_select.py   # -> data/scaling/val_selection.json

# 訓練 cache の幾何検査
uv run python scripts/check_prep_cache.py \
    --cache-dir data/scaling/prep_cache \
    --ids-file data/scaling/master_ids.txt --limit 275
```

**hardware**: 学習は1 run あたり GPU peak 0.016〜0.032 GiB、
train 1985〜3044 秒（load average 25〜77 の共有機、他ユーザの負荷で3〜4倍変動）。
TEST 評価は CPU のみ、4並列で1セルあたり約4〜5分。

#### 5.14.13 §5.14.12 の批判的レビューと主張境界（2026-07-27）

§5.14.12 の数表と直接対比は整合している。ただし結論2〜4には以下の限定が必要である。

1. **baseline比較。** 「6条件すべてがbaselineを上回る」はsuccess@1についての
   固定プール上の事実である。seed 0の6本を1 familyとしてHolm補正すると、raw pは
   $6.6\times10^{-5}, 4.9\times10^{-4}, 9.8\times10^{-4},
   1.5\times10^{-3}, 3.4\times10^{-3}, 1.17\times10^{-2}$ で、
   6条件すべてが補正後も0.05未満に残る。したがって結論は維持できるが、
   **Holm補正済み、固定プール、探索的PINDER-S** と明記する。
2. **pair specificity。** additive→full は事前に定めた主要対比で、
   success@1 $+3.39$ pp、9勝1敗、McNemar $p=0.0215$。
   complex bootstrap 95% CIは **+0.85〜+5.93 pp**
   （`scripts/contrast.py --ci`、20000 resample、seed 0。
   別実装・別 resample 数では上限が +6.4 pp 程度まで動く）で、pair-specificな
   **更新自由度がこの固定プールで増分性能を持つ**ことを支持する。
   一方、両モデルとも公開済み $e_0$ のpair residualを保持し、低次元条件の
   end-to-endも未評価なので、無限定な「pair specificityは必要」は強すぎる。
   「残り42%をpair residualが担う」も非線形argmax差の記述比にすぎず、
   因果的寄与率として扱わない。
3. **symmetric→additive。** 非有意差は同等性ではない。seed 0の観測差は
   additive優位 $+0.85$ pp、bootstrap 95% CIは約 **0〜+2.12 pp**。
   事前候補だった非劣性margin 2 ppを厳密には満たさない。
   したがって「12次元で足りる」「完全null」ではなく、
   **非対称な11自由度の利益は今回の236件では検出されず、差は小さい**
   と書く。symmetric additive自体はbaselineを約4 pp上回っており、
   1体的成分が無意味という結果でもない。
4. **margin interaction。** fullで$+1.69$ pp、additiveで$-0.42$ ppという
   difference-in-differencesは$+2.12$ ppだが、複合体ごとの符号は6対1
   （残り229件は0）で、対応する二側符号検定は $p=0.125$、
   bootstrap CI下限も0付近である。「interactionがある」は未確立で、
   **符号の異なるinteraction patternを観測した**に留める。

**AUCの極小pの監査。** seed 0のm0→m5 AUC差は、
full/additive/symmetricで平均 $+0.0094/+0.0108/+0.0112$、
中央値 $+0.0062/+0.0059/+0.0065$。符号は
201/18/17、198/24/14、195/25/16（改善/悪化/不変）で、
SciPy Wilcoxonでも $p=5.5\times10^{-32},1.2\times10^{-32},9.2\times10^{-33}$。
したがって数値実装の人工物ではなく広範な符号一貫性である。ただしAUCは全
positive-negative pairの広域順位で、primary top-1はnull。p値は `<10^{-20}` 程度に
丸め、平均・中央値・符号数を併記して実用効果と分ける。

first-hit差はAUCほど全体的ではない。中央値は3次元すべて0で、
改善/悪化/不変はfull 24/5/207、additive 30/6/200、
symmetric 29/7/200（seed 0）。したがって「first-hitも改善」は平均と
signed-rankでは支持されるが、**変化した約12〜15%の複合体に集中した効果**である。

**隠してはならないnull/不都合な結果。**

- full→additiveのbest DockQ@1増分は非有意（$p=0.10$〜0.14）。
- marginのsuccess@1効果は全次元で非有意で、低次元では点推定が僅かに悪化する。
- validationのAUCは33件だけでモデル選択を担い、success@1は解像度ゼロ。
  さらにvalidationでは symmetric > additive、TESTでは additive > symmetric と
  低次元間の順序が反転する。
- search-only lossへ実際に勾配を出すfit複合体は148/220で、名目Nより小さい。
  全18 runが1500 step上限まで走りearly stopしていないため、最適化plateauも未確認。
- full+m5以外のend-to-end、absolute-additive、unbound、未使用TESTは未評価。

**推奨する最終表現。**

- 「pair-specificな更新自由度は、additive更新に対して固定プールsuccess@1を
  +3.39 pp改善した（primary paired $p=0.0215$）。」
- 「非対称additiveの追加11自由度には検出可能な利益がなかったが、
  12次元との非劣性は未証明。」
- 「marginはglobal AUCを一貫して約0.01改善したが、success@1改善は未確立。」
- working choiceはvalidation AUCで事前選択したfull+m5。ただしvalidation/test shiftと
  反復利用TESTのため、確定モデルではなく次段階の候補とする。

#### 5.14.14 mining round 1 の契約と評価設計（2026-07-27、実行前）

**コード確認による訂正。** round 0 はfit+validationの `sel` を採掘するが、
round 1以降のtargetは `fit_ids` のみである。したがって `absorb()` はvalidation poolへ
触れず、validationとTESTはround 0で凍結される。round間のval loss、val success@1/AUC、
固定TEST指標は同じ候補集合上で比較可能である。

**採用契約: A（hard-negative only）をprimaryとする。** 研究上の問いは
「hard-negative miningが効くか」なので、新positiveも取り込むBは
on-policy pool refresh/self-trainingという別介入になる。Cはround-0 recallに応じて
教師更新規則が複合体ごとに変わり、A/Bのどちらの効果か解釈しにくいので採用しない。
Aでは探索由来positiveをround 0に固定し、新パラメータで見つかったnegativeだけを
replay poolへ加える。72/220件が勾配ゼロのままという不利益は隠さず、
「fixed-target hard-negative mining」の限界として報告する。

ただし単純なround 0 checkpoint対round 1 checkpointは、miningに加えて
**追加1500 optimizer step**も異なる。mining効果の対照は以下のmatched branchとする:

- `continue`: round-0 poolのまま、同じround-0 checkpoint/Adam stateから1500 step継続。
- `mine`: 学習後パラメータでnegativeを再採掘し、同じcheckpoint/Adam stateから
  同じ1500 step継続。

primary contrastは `mine vs continue`。`mine vs round0` はpipeline全体の変化として
secondaryにする。seed、minibatch順、step数、early-stopping rule、pool capを揃える。
同一poseの再出現で重みが暗黙に増えないよう、round間dedupまたは少なくとも
unique/survival率の記録が必要である。

**評価。**

- checkpoint/round選択: 凍結validationの共通AUCを主に用いる
  （success@1は55件で解像度ゼロ）。val lossは同一目的・同一poolなので比較可能だが、
  prior込みであるため最適化診断とする。
- 科学的primary endpoint: 同一Hopf grid・全249件の
  **end-to-end success@1**について `mine vs continue` を複合体paired McNemarで比較。
  fixed TEST pool 236件のsuccess@1/AUC/first-hitは、共通候補上のreranking機構を示す
  secondary endpoint。
- TESTでround数を選ばない。round 0/1とcontinue/mineの比較を事前固定し、
  TESTは1回の最終比較に使う。

**Bを将来別実験として行う場合のeasy-positive drift診断。**

- roundごとのsearch recall、勾配を出す複合体数、新規に救済された複合体数。
- old/new positive数、DockQ quantile、score/rank、first-hit、basin softmax mass。
- 新positiveのunique pose/basin数とround間重複率、cap後のprovenance別survival率。
- old/new positiveがmargin anchor（最低positive）になる割合とloss/gradient寄与。
- positive:random-negative:hard-negative比、複合体ごとの重み。
- frozen validationとend-to-end TESTの指標。train recallだけ上がり外部指標が上がらない
  状態をself-reinforcementと判定する。

現 `cap_pool()` はpositiveを全件残すため、Bではpositive重複・増殖によりcapを実質超過し、
loss weightingがroundごとに変わりうる。Bの前にdedupとpositive quota/balanced samplingの
契約が必要である。

**round数。** primary実験は**1 roundで止めて評価**する。改善した場合だけ、
validation規則によりround 2をsecondary learning-curveとして追加する。
新規unique hard negativeのcap生存率、validation AUC、parameter driftが飽和したら停止し、
初回から4 roundを回して最良TEST roundを選ばない。

**結論。** A + matched no-mining continuationが、hard-negative miningの増分を最も
狭く識別する。Bは72件救済という別の価値を持つが、positive更新を含む別実験として
Aの結果後に行う。

#### 5.14.15 mining round 1 実装レビュー（2026-07-27、production実行前）

**問い。** §5.14.14 の `hardneg` 対 matched no-mining continuation を、
同一round-0 parameter/Adam stateから分岐し、round-1採掘を安全にシャード・再利用できる
実装になっているかを、未コミット差分について静的レビューした。対象は
`scripts/run_pinder_scaling.py` の `--mining-contract`、`--mine-from-ckpt`、
`--resume-from`、raw cache、`absorb()`、dedup/cap/統計である。予定条件は
full 144 + margin 0.5、`--freeze-psc --loss-prov search`、fit/validation/TEST =
220/55/249、seed 0/1/2、round 1のみであり、本節では新しい学習・TEST評価はしていない。

**検証。**

```bash
git diff -- scripts/run_pinder_scaling.py
UV_CACHE_DIR=/tmp/ddock-uv-cache uv run pytest -q tests/test_scaling_streaming.py
UV_CACHE_DIR=/tmp/ddock-uv-cache uv run python -m py_compile \
  scripts/run_pinder_scaling.py
```

既存テストは20 passed, 1 skipped（66.58 s）、構文検査も成功した。ただし既存テストは
新しいresume/shard/dedup経路を直接検証しない。smoke runでの
12,205 proposed negatives中4,913（40%）がround-0との完全一致だったという数値は
実装者の実測であり、本レビューでは再実行していない。

**静的レビューで確認したこと。** intended workflow（Hopf、full、PSC凍結、
`--rounds 1`）に限れば、Phase Bはround-1 raw candidatesだけを保存し、Phase Cは
round-0 poolを復元してから`absorb`を一度だけ適用する。resume round番号は
round-0 optstateから1へ進み、保存されたbest parameterと対応するbest Adam stateも
復元される。validation poolはround 0のままなので、mine/continue間で共通である。

**production前に直すべき点。**

1. `(rmsd, DockQ, ELEC)` はpose identityではない。別poseの派生量衝突と、同一poseの
   浮動小数再計算差によるfalse negativeの両方がありうる。しかも現在の`keep`は
   candidate内で`seen`を更新しない。`generate_pool_reachable()`が既に同一性として使う
   `(rotation index, integer translation cell)`を`Feats`とcacheまで伝播する。Hopf以外では
   rotation indexだけでなくrotation/quaternion gridのidentityも必要である。
2. `n_survived_cap += max(pool_after - pool_before, 0)` はnew poseの生存数ではない。
   capが満杯ならnew negativeがold negativeを置換しても0になり、cap未満からの切詰めでも
   oldの退場分だけ過小評価する。`origin == rnd`（かexact pose key）をcap後に直接数え、
   old-negative evictionも別に記録する。
3. `--mine-from-ckpt --rounds 0`を拒否していない。この場合、学習済みparamで採掘した
   poolをfingerprintなしの共有round-0 cacheへ書ける。またrun directoryは
   `N<n>_seed<s>`だけなので、3 minerが同じ`split.json`/`skipped.jsonl`を競合上書きし、
   mine/continueも同じ`--out-dir`ならround-1成果物を上書きする。round>0を強制し、
   Phase A/B/Cと両armのartifact directoryを分離または契約名で名前空間化する。
4. optstateにはconfig、split/cache identity、minibatch generator stateがない。
   現状でも同じCLI seedで両armを新規起動すれば両者のbatch順は一致するが、
   uninterrupted continuationではなくround-0の乱数列を再生する。さらに誤った
   margin/iface mode/prior/splitでのresumeを検出できない。source config/fingerprint、
   fit/val ID hash、generator stateを保存・照合し、effective 144-vectorをnormだけでなく
   elementwiseにcheckpointと照合する。
5. cache済みIDを除いてから`targets[shard::n]`を取るため、遅れて開始したworkerや
   failed shardの再実行ではshard所有集合が変わる。最初に全fit IDsをshard分割し、
   その後そのshard内のcached IDsを除く。複数cacheに同じPIDがあれば、現在は後の値で
   silent overwriteするため、一致確認またはerrorにする。

**解釈上の限定。** `patience`によるearly stoppingがあるので両armは同一
**step budget/rule**だが、`steps_run`まで同じとは限らない。「matched-compute」と呼ぶなら
固定stepで走らせる必要があり、現状はmatched-budget controlと書く方が正確である。また
capはnew hard negativesを追加するだけでなくold negativesを置換するため、estimandは
「fixed-positive hard-negative pool refresh」の増分である。なお汎用モードでは
`mine_complex()`が学習した`rho`をsearchへ渡していないが、今回の
`--freeze-psc`条件には影響しない。

**結論。** Phase A/B/Cの中心的な分岐設計とraw-cache/`absorb`の責務分離は妥当である。
しかしpose ID、cap survival統計、round-0 cache汚染防止、artifact分離はproduction run前の
blockerとする。修正後、(i) exact dedup、(ii) cap満杯でのnew/old survivor、
(iii) 3 shardの和がfit 220を重複なく覆うこと、(iv) resume直後のparameter/Adam/batch順が
両armで一致することを小規模回帰テストで確認してから3 seedを開始する。

#### 5.14.16 mining round 1 修正の再レビュー（2026-07-27、production実行前）

**問いと範囲。** §5.14.15の5指摘に対する未コミット修正を
`scripts/run_pinder_scaling.py`、`src/zdock/dataset.py`、
`tests/test_scaling_streaming.py`について再レビューした。新しい学習、採掘、TEST評価は
行っていない。予定条件とleakage controlは§5.14.14〜§5.14.15から不変である。

**検証。**

```bash
git diff -- scripts/run_pinder_scaling.py src/zdock/dataset.py \
  tests/test_scaling_streaming.py
UV_CACHE_DIR=/tmp/ddock-uv-cache uv run pytest -q \
  tests/test_scaling_streaming.py
UV_CACHE_DIR=/tmp/ddock-uv-cache uv run python - <<'PY'
import torch
big = 1 << 21
cell = torch.zeros(3, 3, dtype=torch.int64)
qi = torch.tensor([0, 1, 2], dtype=torch.int64)
c = cell + big // 2
key = ((qi * big + c[:, 0]) * big + c[:, 1]) * big + c[:, 2]
print(key.tolist(), bool(key[0] == key[2]))
PY
```

対象テストは**27 passed, 1 skipped**（41.32 s）だった。差分中の新規
`test_*`関数は7本であり、「28 passed」ではなく28 collected中1 skippedに相当する。
全suiteは本レビューでは実行していない。

**前回指摘への対応状況。**

- cap後のnew survivorとold-negative evictionの直接集計、`rounds 0`汚染のCLI拒否、
  intended 3-shard/2-armのartifact分離、cache除去前のshard所有集合固定、
  duplicate PIDの拒否、matched-budgetへの改称は意図どおりである。
- effective 144-vectorのelementwise照合と、同じoptstateからのAdam/minibatch state復元も
 方向は正しい。ただし下記のcheckpoint整合性の限定が残る。
- pose identityをdatasetから`Feats`、index/cat/device移動、cacheまで伝播した設計は正しいが、
  **identityのpacking自体が壊れている**。

**production blocker 1: int64 pose keyのoverflow/collision。**
`_key()`は3 translation cellに各21 bitを使った後にrotation indexも左から追加するため、
必要幅が63 bitを超える。実測では同一cellに対する`qi=[0,1,2]`が
`[4611688217451692032, -4611683819403083776, 4611688217451692032]`となり、
`qi=0`と`qi=2`が衝突した。さらに有効なodd rotationのkeyが負になるため、
`_fresh_indices()`の`pose_key < 0`判定はidentity欠損と誤認してround 1を停止する。
scalar packingをやめ、`(qi, cell_x, cell_y, cell_z)`の`(N,4)`整数tensorをそのまま保存し、
dedup時だけtuple集合にするのが安全である。少なくとも
「同一cell・rotation 0/1/2が全てdistinctかつvalid」の回帰テストが必要である。

**production blocker 2: cache schemaが更新されていない。** 新しい`pose_key`を持たない
旧round-0/raw cacheは同じcache keyで再利用され、`Feats`内では`-1`になる。Phase Aの
`--rounds 0`ではこれを検出せずcheckpointを作れ、長時間のPhase B後、Phase Cの
`absorb()`で初めて停止しうる。cache keyへpose-identity schema versionを入れて旧cacheと
分離し、Phase AまたはPhase B前のpreflightで全round-0 poolのidentityを検証する。

**非blockerだが残る契約/テスト不足。**

1. `train_params()`はbest parameter/Adam stateへ巻き戻す一方、generator stateは
   最終`steps_run`時点を保存する。両arm間では同一なのでprimary contrastは壊さないが、
   「best checkpointからの厳密な継続」ではない。accepted時にbest generator stateも保存して
   一緒に戻すか、この意味を明記する。
2. `resume_identity`には`min_steps`、`epoch_passes`、`patience`、`min_delta`、
   `grad_clip`、parameter bounds、`alpha0/rho0`、`psc_decompose`、`pool`がない。
   arm間で異なるcontinuation ruleを誤指定しても検出できるよう追加する。
3. cap testは`cap_pool()`の現象を示すだけで、production `absorb()`が
   `n_new_survived_cap=5`、`n_old_neg_evicted=5`を記録することを検証していない。
   shard testも単純なlist slicingのみでcached IDsを含む実装経路を呼ばない。
   resume identity/generator、CLI guards、cache conflictにも直接テストがない。
4. intended workflowの3-shard minerは分離されたが、`--mine-from-ckpt`を
   `--mine-shard 0/1`で使う場合はPhase Aと同じrun directoryになる。minerであること自体を
   run tagへ入れる方が安全である。

**結論。** 前回の5指摘のうち2、3、5は本体として解消し、4もprimary contrastを
matchedにする方向で改善した。しかしscalar pose-key overflowとcache schema非更新は
production blockerである。両者を直して新規round-0 cacheを作り、actual
`generate_pool_reachable()`由来keyのuniqueness/round間一致をsmoke確認するまで、
Phase Bの長時間採掘は開始しない。

#### 5.14.17 mining round 1 修正の第3回レビュー（2026-07-27、production実行前）

**問いと範囲。** §5.14.16のblocker 2点、checkpoint整合性、低次元best-checkpoint
巻き戻し修正を再レビューし、overflowが§5.14.12までの既報結果へ影響する経路を確認した。
新しい学習・TEST評価は実行していない。実装者が進行中の全suiteとは別に次を実行した。

```bash
UV_CACHE_DIR=/tmp/ddock-uv-cache uv run pytest -q \
  tests/test_scaling_streaming.py
UV_CACHE_DIR=/tmp/ddock-uv-cache uv run python -m py_compile \
  scripts/run_pinder_scaling.py src/zdock/dataset.py
```

結果は**29 passed, 1 skipped**（20.54 s）で、構文検査も成功した。「30 passed」ではなく
30 collected中1 skippedである。全suiteの結果は本節にはまだ含めない。

**解消を確認した点。**

- scalar packingを廃止した`(qi,cx,cy,cz)`の`(F,4)`表現、tuple dedup、cache schema
  `_pk2`、candidate内部dedup、cap survivor/old eviction集計は方向・実装とも妥当である。
- `best_gen`をaccepted parameter/Adamと同時に保存・巻き戻すため、保存optstateは
  同一checkpoint時点の3状態を持つ。resume identityへ追加したoptimizer/stopping/pool設定も
  intended workflowを十分に拘束する。
- 低次元modeでは`Params.NAMES`ループ後に`rowcol`を明示的にbest stateへcopyしており、
  巻き戻しバグは修正されている。既報12本で最終validation checkがすべてaccepted
  （`last_accepted_gap=0`）なら、旧コードでもreturned rowcolがbestと一致するため、
  実験1の数値へ影響ゼロという判定は妥当である。

**production blocker: sentinel判定がvalidな負translationを拒否する。**
pose identityのtranslation cellは原点をまたぐため`cx/cy/cz < 0`が正常に起こる。しかし
`_fresh_indices()`とround-0 preflightは`(pose_key < 0).any()`をidentity欠損判定に使うため、
有効な`(qi, -1, 0, 0)`も拒否する。欠損rowは4成分すべて`-1`だが、有効rowは
`qi >= 0`なので、`pose_key[:,0] < 0`（shape `(F,4)`もassert）で判定できる。
legacy経路のsentinelも現在の`(F,)`ではなく`(F,4)`へ統一し、負cellを含む回帰テストを置く。

**preflightのタイミング。** 条件は`rnd == 0 and args.rounds > 0`だが、予定Phase Aは
`--rounds 0`、Phase Bはround 0をskipする。したがって「Phase Aで落ちる」は現workflowでは
成立しない。schema versionにより旧cache混入は防げるが、identity破損やround-0 pool欠損を
長時間のPhase B前に検出する目的は未達である。round-0 reachable poolでは無条件に検証するか、
Phase B開始時に対応するround-0 cacheのfit 220件・identity shape/validityをpreflightする。

**overflowの過去結果への影響。** `_key()`の誤dedupはsearch結果を生成した後、
列挙由来`pos_*`をsearch poseと照合する箇所だけにあり、search pose自体は削除しない。
したがって以下には原理上影響しない:

- `--loss-prov search`のfit/validation lossと、それによるcheckpoint選択。
- `--prov search`へ限定した固定pool success/AUC/first-hit。
- 列挙を使わない全249件end-to-end再探索。

一方、all-provenanceのpool size、positive数、reachable-positive ceiling、coverage、
pooled success/AUCは影響対象であり、同じ数値を無条件に保存してはならない。実測1複合体の
212→216（4 pose、1.9%）は列挙側の影響量であって、search-only不変性の直接検証ではない。
集計中の12複合体でold/newの`prov==0`についてpose/orderと
`sc,T,elec,rmsd,DockQ`をchecksum比較し、既存checkpointをnew pool上で
`--prov search`再評価して同値を確認する。これが一致すれば§5.14.12の主要結論は維持できる。

**テストの限定。** 1944回転testはproductionのnested `_key()`を呼ばず同じ`torch.cat`を
test内で再実装し、return-contract testもsource文字列検査である。今回の実装は読んだ限り
正しいが、将来の回帰防止にはkey構築をmodule-level helperへ出して直接testする方が強い。

**結論。** overflowとcache version自体は解消したが、負translationの誤拒否が新しい
production blockerである。sentinel判定・preflight時点を直し、fit 220/val 55の新round-0
cache完全性、12複合体search-only同値、全suite成功を確認してからPhase Bを開始する。

#### 5.14.18 mining round 1 修正の第4回レビュー（2026-07-27、production実行前）

**問いと範囲。** §5.14.17の負translation sentinelとpreflight timingの修正を再レビューし、
overflowの既報結果への影響を直接検証する方法、およびPhase B開始条件を確定した。
新しい学習・TEST評価は実行していない。

```bash
UV_CACHE_DIR=/tmp/ddock-uv-cache uv run pytest -q \
  tests/test_scaling_streaming.py
UV_CACHE_DIR=/tmp/ddock-uv-cache uv run python -m py_compile \
  scripts/run_pinder_scaling.py src/zdock/dataset.py
```

対象テストは**31 passed, 1 skipped**（16.79 s）、構文検査も成功した。したがって
「32 passed」ではなく32 collected中1 skippedである。全suiteは本レビューでは未確認。

**blocker解消を確認。**

- productionとtestが共通の`pose_identity(qi, cell)`を使い、scalar overflowはない。
- `has_pose_identity()`はshape `(F,4)`とrotation-index列だけを検査するため、有効な
  負translation cellを受理し、sentinel rowを拒否する。
- `_fresh_indices()`も同helperを通り、candidate内部・round間のtuple dedupを行う。
- Phase Aはround 0後に`sel`全件、Phase BはFFT前にround-0 cacheのfit 220件、Phase Cは
  resume直後に`sel`全件について、pool欠損とidentity欠損を検査する。予定workflowで
  実際に発火する位置である。

intended production経路（`--pool reachable --rot-set hopf`）に新しいblockerは見つからなかった。
ただしlegacy分岐のsentinelは説明と異なり、まだ`torch.full((F,), -1)`で
`Feats.pose_key`の`(F,4)`契約を破る。legacy hard-negative miningはidentityを持たず
いずれ拒否されるので今回のPhase Bには影響しないが、`(F,4)`へ統一し、
`POSE_IDENTITY_MISSING`を実際に使う。あるいはround>0 + legacyをCLIで最初から拒否する。

**overflow影響の中間測定（実装者測定、未完了）。**

| complex | generated | old kept | exact-key kept | erroneous drops |
|---|---:|---:|---:|---:|
| 2io2 | 216 | 212 | 216 | 4 |
| 7oit | 216 | 88 | 210 | 122 |
| 4zsh | 216 | 140 | 211 | 71 |
| 5oen | 216 | 212 | 215 | 3 |
| 6hvo | 216 | 143 | 209 | 66 |
| 8env | 216 | 216 | 216 | 0 |

6複合体で266/1296候補（20.5%）を余分に削除しており、0〜56.5%とheterogeneousである。
これは列挙候補数への影響であって、正例数への影響とは限らない。全12件では
erroneously dropped poseのDockQ分布、`DockQ>=0.23`数、all-provenanceのpooled指標と
reachability統計への差も記録する。

**search-only不変性の直接検証。** 旧cacheにpose keyがなくても、`prov`とmodel-visible
featuresはあるため比較できる。対応する旧cacheと新`_pk2` cacheについて複合体ごとに
`prov==0`を選び、pose数・row orderと`sc,T,elec,rmsd,DockQ`のshapeおよび
`torch.equal`/checksumを比較する。丸め差があればmax-abs差と既存checkpointでのscore/rank
一致を併記する。これは「学習と固定pool評価が実際に見たtensor」の最も直接的な監査である。

pose座標自体も確認するなら、FFT searchを一度だけ実行して同じ`result`から
`neg_poses`を構築し、old packed-key maskとnew tuple-key maskを列挙側だけに分岐させる。
両方の`poses[:n_search]`が同一の`neg_poses`であることをassertする。searchを二度実行すると
GPU nondeterminismまで混ざるので、一つのsearch resultを共有してdedupだけを変える。

コード上、dedupは`result`から`neg_poses`を作った後の列挙`pos_*` maskだけに作用し、
search poseはcatの前半として不変である。したがって上のmodel-visible tensor監査が一致すれば
`--loss-prov search`学習、`--prov search`固定pool評価、end-to-end探索への影響なしという
結論は十分である。一方all-provenanceのpool/positive/coverage/reachability/pooled metricは
訂正対象になりうる。

**Phase B開始条件。** コード上のproduction blockerは解消した。新schemaでPhase Aを完走し、
Phase A logで`sel=275`のpreflight、Phase B dry startでfit 220 preflight、全suite成功を
確認したらPhase Bを開始してよい。search-only監査は過去結果の訂正作業であり、新しい
round-1採掘の正しさを左右しないため並行実行可能である。

#### 5.14.19 pose 識別子オーバーフローの実測と影響（2026-07-27）

§5.14.15〜§5.14.18 で codex が指摘したバグの実測記録である。
**これは round-1 の実装作業中に見つかった既存バグであり、私が今回導入したものではない。**

##### 何が壊れていたか

`generate_pool_reachable()` は「探索が既に返した pose と同じ列挙 pose を捨てる」ために
`(rotation index, integer translation cell)` を1つの int64 へ詰めていた:

$$\text{key} = ((q_i \cdot 2^{21} + c_x)\cdot 2^{21} + c_y)\cdot 2^{21} + c_z$$

$q_i$ が bit 63 に来るため int64 を溢れる。**実測**:

```
qi   : [0, 1, 2, 3, 4, 5]           (同一 cell)
key  : [ 4611688217451692032, -4611683819403083776,
         4611688217451692032, -4611683819403083776,
         4611688217451692032, -4611683819403083776]
6個の異なる回転 -> 2個のキー
hopf nside=3: 1944 回転 -> 2 個のキー
負のキー: 3/6 (奇数回転すべて)
```

つまり**キーは回転インデックスの偶奇しか区別していなかった**。
結果として、列挙 pose は「並進セルが同じで回転の偶奇が同じ」探索 pose が1つでもあれば
重複として捨てられていた。

##### 誤って捨てられた量（測定）

`scripts/measure_pose_key_overflow.py`。同一の FFT 探索結果に対して旧キーと
新しい `(N,4)` 識別子の両方で重複除去を適用し、残る列挙候補数を比較する。
探索を2回走らせると GPU の非決定性が混ざるため、探索結果は共有し dedup だけを変える。

| complex | 生成 | 旧キーで残存 | 正しい識別子で残存 | 誤削除 |
|---|---:|---:|---:|---:|
| 2io2 | 216 | 212 | 216 | 4 |
| 7oit | 216 | 88 | 210 | **122 (56.5%)** |
| 4zsh | 216 | 140 | 211 | 71 |
| 5oen | 216 | 212 | 215 | 3 |
| 6hvo | 216 | 143 | 209 | 66 |
| 8env | 216 | 216 | 216 | 0 |
| 6gxp | 216 | 216 | 216 | 0 |
| 7a6h | 216 | 212 | 216 | 4 |
| 5dyd | 216 | 124 | 209 | 85 |
| 4bts | 216 | 216 | 216 | 0 |
| 1h6f | 216 | 216 | 216 | 0 |
| 7jhy | 216 | 211 | 216 | 5 |
| **合計** | **2592** | **2206** | **2566** | **360 (13.9%)** |

**複合体差が極端である**（0〜56.5%）。均一な縮小ではないので、
「全体に少し少ない」という補正では済まない。

##### 影響範囲（コード上の論拠と、必要な実測）

この重複除去は探索結果から `neg_poses` を構築した**後**に、列挙側 `pos_*` を
探索 pose と照合する箇所だけに作用する。探索 pose 自体は削除されず、
`poses = cat([neg_poses, pos_poses])` の前半をそのまま占める。したがって

**原理上、影響しない**:

- `--loss-prov search` の fit/validation 損失と、それによる checkpoint 選択
- `--prov search` に限定した固定プールの success@1 / AUC / first-hit
  （§5.14.8、§5.14.12 の主要結果すべて）
- 列挙を使わない全249件の end-to-end 再探索

**影響しうる（訂正対象）**:

- all-provenance のプールサイズ、positive 数、`n_pos_enumerated`
- IFACE coverage、reachability 統計
- `--prov all` の pooled 指標

コード読解だけで終わらせず、`scripts/audit_pose_key_fix.py` で旧キャッシュと
新 `_pk2` キャッシュの `prov == 0` 行について pose 数・行順・
`sc, T, elec, rmsd, DockQ` を突き合わせた。以下がその実測である。

##### 監査の実測結果（2026-07-27、両キャッシュに存在する260複合体）

**探索由来 pose（`prov == 0`）**

| 項目 | 結果 |
|---|---|
| pose 数 | **260/260 で一致**（すべて 1500 → 1500） |
| `sc`（PSC 4成分） | 259/260 でビット単位一致 |
| `T`（接触行列） | 259/260 でビット単位一致 |
| `rmsd`, `DockQ` | 259/260 でビット単位一致 |
| `elec` | **259/260 が $\le 10^{-6}$**（中央値 $2.4\times10^{-7}$、p99 $4.8\times10^{-7}$） |

**all-provenance（変わることが予測されていた量）**

| 量 | 旧 | 新 | 変化 |
|---|---:|---:|---:|
| プールサイズ | 441,133 | 445,794 | +1.1% |
| positive 総数 | 45,982 | 50,623 | **+10.1%** |
| うち探索由来 | **3,408** | **3,408** | **±0** |
| うち列挙由来 | 42,574 | 47,215 | +10.9% |
| positive ゼロの複合体 | 0 | 0 | ±0 |
| 探索 positive ゼロの複合体 | 89 | 89 | ±0 |

**探索由来 positive が 3408 で完全に一致し、増えたのは列挙由来だけである。**
これは「重複除去は列挙側にしか作用しない」というコード上の主張の直接確認である。

##### 唯一の不一致は GPU 非決定性であり、この修正とは無関係

260複合体のうち1件（`7cru__B1_P52292--7cru__A1_P61978`）だけが探索 pose 自体で
食い違う。ただし**1500行中2行**であり、positive 数（5）も最大 DockQ（0.6675）も同一で、
上位1500の境界で近接同点の pose が入れ替わったものである。

これが dedup 由来でありえないのはコード上明白である: `prov == 0` の行は
`docking_search()` と `docking_score_elec()` が生成し、重複除去はその**後**に
列挙側 `pos_*` のマスクとしてのみ作用する。原因は**旧キャッシュを GPU 0/3/6、
新キャッシュを GPU 1/2/4/5/6 で採掘したことによる cuFFT の非決定性**である。
`elec` の $2.4\times10^{-7}$ 程度の系統差（259/260）も同じ理由で、float32 の 1 ULP 相当である。

**これはこの修正が持ち込んだ問題ではなく、パイプラインの既存の性質**である
——同じ入力でも GPU が違えば探索の上位 $k$ は近接同点のところで入れ替わりうる。
以後の実験でこれを再現性の限界として記録する。

##### 決定レベルでの不変性（`scratchpad/score_invariance.py`）

丸め差が判断を動かすかを直接確かめるため、§5.14.12 の主要数値を出した
チェックポイント（`runs_full_m5/N220_seed0`）と baseline の両方で
新旧プールを採点した。探索由来 pose のみ、260複合体。

| 条件 | top-1 pose 一致 | 全順位一致 | AUC 一致（$<10^{-12}$） | max \|ΔAUC\| |
|---|---|---|---|---|
| baseline | **260/260** | 259/260 | **171/171**（計算可能な全件） | **0.000** |
| trained | **260/260** | 259/260 | **171/171** | **0.000** |

（AUC が計算できるのは探索 positive を持つ 171 複合体のみ。残り89件は positive ゼロで
未定義であり、その数も新旧で一致している。）

**結論**: success@1 を決める top-1 pose は全260複合体で一致し、AUC は計算可能な
全複合体で厳密に一致する。したがって §5.14.8 と §5.14.12 の
`--prov search` / `--loss-prov search` の数値は**この修正で変わらない**。
訂正が必要なのは all-provenance の positive 数（+10.1%）とそれに依存する
プール構成・coverage・reachability の記述だけである。

##### あわせて見つかった別の既存バグ（実害ゼロ）

`Params.NAMES = ("alpha", "rho", "iface", "log_clash")` に `rowcol` が無いため、
`train_params()` の best-checkpoint 巻き戻しループが低次元モード（`add`/`sym`）で
`rowcol` を復元していなかった。返るのは最終ステップの値である。

**実験1への影響を実測した**（`round0_trajectory.csv` の accepted フラグ）:

| モード | best ≠ last だった run |
|---|---|
| add (m0/m5 × 3 seed) | 0/6 |
| sym (m0/m5 × 3 seed) | 0/6 |
| full (m0/m5 × 3 seed) | 3/6（ただし `iface` は NAMES にあり正しく巻き戻る） |

低次元12本すべてで最終 validation チェックが accepted、すなわち best == last なので、
**§5.14.12 の数値は変わらない**。修正は入れたが、既報結果の訂正は不要である。

##### 修正

- `src/zdock/dataset.py` に `pose_identity(qi, cell) -> (N,4)` と
  `has_pose_identity(pose_key)` を追加。production の重複除去も `Feats.pose_key` も
  テストも同じ関数を通る。
- `has_pose_identity` は **rotation-index 列のみ**を見る。並進セルは原点をまたぐので
  負の $c_x/c_y/c_z$ は正常であり、全列を見ると有効な pose を identity 欠損と
  誤判定する（これは私が一度入れて codex に指摘された）。
- pool cache key に `POSE_KEY_SCHEMA`（`_pk2`）を付け、識別子を持たない旧キャッシュが
  同じキーで再利用されないようにする。
- Phase A / Phase B / Phase C の3箇所に preflight を置き、識別子欠損とプール欠損を
  **長時間の採掘の前に**検出する。

回帰テスト（`tests/test_scaling_streaming.py`）: 同一 cell の1944回転が全て distinct
かつ非負であること、負の並進セルが受理されること、sentinel と rank 違いが拒否されること。
全スイート 127 passed, 1 skipped。

#### 5.14.20 voxel テーブルの取り違え（2026-07-27、実行者の起動ミス）

Phase A′ の起動で `--grid-voxels` を指定し忘れ、既定の
`data/scaling/grid_voxels.json`（**3.0 Å 間隔**で計算されたもの）が使われた。
ボクセル数は間隔の3乗に反比例するので値は実際の約 1/16 になり、
`--max-grid-voxels 31250000` の上限が事実上機能しなかった。

**実測された帰結**

| | 誤テーブル（3.0 Å） | 正テーブル（1.2 Å） |
|---|---:|---:|
| オーバーサイズ除外 | **5件** | **164件** |
| 6fkf のボクセル数 | 4,062,609 | **61,501,356** |
| 7n6g のボクセル数 | 18,006,060 | **276,438,696** |

6100万〜2億7600万ボクセルの複合体が上限を通過して選択に入り、
採掘時に OOM して8件がスキップされた。`_adaptive_frame_chunk` の docstring が
「サイズ上限はまさにこの生存者バイアスを防ぐために存在する」と書いている、
その事態そのものである。

さらに重大なのは、**選択が実験1と15複合体ずれていた**ことである。
気づかずに進めていれば round-1 実験は実験1と別のデータセットで走り、
両者を並べて論じられなくなっていた。

**恒久対策。** `compute_grid_sizes.py` が出力へ `spacing` を記録し、
`run_pinder_scaling.py` が `--spacing` と照合して不一致なら停止する。
spacing を記録しない旧形式も拒否する（既定ファイルが 3.0 Å のものなので、
黙って読むと同じ事故が再現する）。実測で発火を確認した:

```
data/scaling/grid_voxels.json was computed at spacing 3.0 A but this run uses
1.2 A. Voxel counts scale as spacing^-3, so the size cutoff would be off by
(2.5)^3 = 16x and stop filtering. Run scripts/compute_grid_sizes.py --spacing 1.2.
```

正しいテーブルで再実行した結果、選択は実験1と **fit 220 / val 55 の ID 順まで完全一致**し、
追加15件の採掘は OOM ゼロで完了した（275/275、全件が pose identity を保持）。

#### 5.14.21 実験4の結果: hard-negative mining round 1 は効かない（2026-07-27）

##### 設定

§5.14.14 の設計どおり、契約A（hard-negative only、positive は round 0 に凍結）を
primary とし、**採掘の有無だけが違う2本**を同一の round-0 状態から分岐させた。

```
              round 0 の学習（1500 step、fit 220 / val 55）
                            │
                  round0_optstate.pt
                  ・パラメータ ・Adam 状態 ・minibatch 乱数列
                            │
             ┌──────────────┴──────────────┐
        【mine】hardneg                【continue】none
   学習後パラメータで再探索し           再探索せず、
   新しい negative のみ追加             プールは round 0 のまま
             │                              │
      さらに 1500 step               さらに 1500 step
             └──────────────┬──────────────┘
                    同じ236複合体でペア比較
```

full 144 + $\lambda_\mathrm{margin}=0.5$、`--freeze-psc --loss-prov search`、
seed 0。両arm は同じ step budget・停止規則・pool cap を共有する。

##### round-0 学習が実験1と完全一致（pose key 修正の最終確認）

再採掘したプールで round 0 を学習し直した結果:

| seed | val loss | ‖Δe‖ | steps | TEST DockQ@1 |
|---|---|---|---|---|
| 0 | 2.7240 = 2.7240 | 1.5594 = 1.5594 | 1500 | 0.4658 = 0.4658 |
| 1 | 2.7220 = 2.7220 | 1.5514 = 1.5514 | 1500 | 0.4660 = 0.4660 |
| 2 | 2.7268 = 2.7268 | 1.5717 = 1.5717 | 1500 | 0.4659 = 0.4659 |

（左が今回、右が実験1。すべて一致。）
列挙 positive が複合体あたり 147 → 183（+24%）と増えているにもかかわらず、
`--loss-prov search` の学習は同じ結果に到達した。勾配を出す複合体数も
**148/220 で一致**、IFACE coverage もゼロ成分 0/144・中央値 189/220 で一致。
§5.14.19 のテンソル監査を、学習パイプライン全体で裏付けたことになる。

##### 結果（seed 0、236複合体、`--prov search`）

| 条件 | success@1 | AUC | best DockQ@1 | first-hit % |
|---|---:|---:|---:|---:|
| baseline（公開パラメータ） | 69.49% | 0.8696 | 0.4457 | 1.098 |
| round 0（学習済み） | **77.54%** | 0.8933 | 0.4893 | 0.546 |
| round 1 **mine** | **77.54%** | 0.8935 | 0.4894 | 0.539 |
| round 1 **continue** | 77.12% | 0.8938 | 0.4875 | 0.532 |

**固定TESTプール上の主要対比: continue → mine**

これは §5.14.14 でsecondaryと事前規定したreranking機構評価である。
同節で科学的primaryとした全249件の **end-to-end success@1 は未評価**なので、
以下だけで実験4全体のprimaryがnullになったとはしない。

| 指標 | 変化 | 検定 |
|---|---|---|
| **success@1** | 77.12% → 77.54% | **1勝0敗、exact McNemar $p=1$**、bootstrap 95% CI $[+0.00, +1.27]$ pp |
| AUC | 0.8938 → 0.8935（**−0.0003**） | Wilcoxon $p=0.003$（符号一貫性であって効果量ではない） |
| best DockQ@1 | 0.4875 → 0.4894 | Wilcoxon $p=1$ |
| first-hit | 0.0053 → 0.0054 | Wilcoxon $p=0.14$ |

**動いた複合体は1件だけである。** AUC はむしろ mine のほうが低い。

**secondary: 分岐元からの変化**

| 対比 | success@1 | 勝/敗 | p |
|---|---|---|---|
| round 0 → round 1 mine | 77.54% → 77.54% | **0勝0敗** | 1 |
| round 0 → round 1 continue | 77.54% → 77.12% | 0勝1敗 | 1 |

**round 0 → round 1 mine は固定プールの success@1 が完全に不変**である。
最大1500 stepのbudget（実際は501 stepで早期停止）を追加して採掘しても、
この共通候補集合上で成功判定が変わった複合体はない。

##### candidate novelty の診断

round-1 採掘の内訳（220複合体、`round1_metrics.json` の `mining`）:

| 項目 | 値 |
|---|---:|
| 提案された negative | 333,475 |
| **うち round-0 プールとの重複** | **231,042（69.3%）** |
| 新規に追加された negative | 102,433 |
| cap 後に生き残った新規 | 102,433（全部） |
| **cap により追い出された old negative** | **0** |
| プールサイズ | 1,715 → 2,180（上限 4,000 に未到達） |

学習後パラメータの探索が返す negative の69%は、round-0探索が既に返していた。
ただし、これだけをnullの原因と断定してはならない。**102,433件（30.7%）は新規**で、
平均プールも1,715から2,180へ**27%増加**しており、「新情報が入らなかった」は
数としては誤りである。示されたのは、新規negativeを追加しても固定TEST順位が
改善しなかったことまでである。

原因を分けるには、新規negativeについてround-0 checkpointでのscore/rank、
positiveとのmargin violation率、basin softmax mass、lossおよびgradient寄与を
old negativeと比較する必要がある。このときlossはpool全体のscoreを複合体ごとに
標準化するため、raw scoreへmarginを当てず、**old+new結合poolを
`normalized_scores()`に通した値**で測る。最も直接的には同じround-0 checkpointで
old poolとold+new poolのloss、gradient norm、gradient cosineを複合体ごとに比較する。
新規でも既存例と特徴的に冗長、またはloss上
すでに容易なら、10万件追加されても学習信号はほとんど増えない。cap未到達で
old negativeの置換がゼロだったため、今回の介入は「hardest例への置換」ではなく
主としてadditive pool expansionだったことも限界である。

##### その測定（`scripts/measure_new_negative_leverage.py`）

**単位の訂正（codex 指摘）。** この診断を最初は**生スコア**に margin $=1$ を
適用して書いたが、それは学習が見ている量ではない。`mean_objective` は
`normalized_scores` を通す——プールを自身の平均で中心化し、自身の標準偏差
（生スコアでは $5\times10^2$〜$2\times10^3$）で割る。生スコアに margin $1.0$ を
当てるのは実質「最悪の positive を超える negative」を数えているだけで、別の量である。
さらに標準化は**与えられたプールごとに再計算される**ので、27%の pose を足せば
同じ pose の位置も変わる。以下は学習経路（`loss_view` の `prov==0` 部分集合、
round-0 プール＋新規 negative を結合してから標準化）を再現した測定である。

勾配を持つ148複合体、round-0 の学習済みパラメータ、**標準化スコア単位**:

| 項目 | 値 |
|---|---:|
| 148複合体に追加された新 negative | 67,278 |
| うち margin 帯域 $s > \min_i s(\text{pos}_i) - \lambda$ に入るもの | **52,723（78.4%）** |
| 同じ帯域内の old negative（round-1 プールで再標準化後） | 174,093 |
| **帯域に占める新規の割合** | **23.2%**（複合体ごと中央値 22.9%） |
| **新 negative が最難関（最高スコア）になった複合体** | **3/148** |
| **最難関 negative のスコア差（新 − 旧）** | **中央値 $-1.87$** |
| margin 閾値そのものの移動 | 中央値 $+0.037$ |

（参考: 生スコアで測った誤った版では 54.0% / 22.0% / 3-of-148 だった。
新規が最難関になった複合体数だけは単位に依らない。）

**新規 negative の 78% は損失が見る帯域に入っている。** 届いていないのではない。
プールも 27% 増えている。「新情報が入らなかった」という最初の説明は誤りである。

正しい機構は**帯域の上端**にある。`loss_margin_hard_negatives` は
$\max$ 側で駆動される hinge であり、勾配の大きさを決めるのは最も高くスコアする
negative である。新規 pose は同じ探索・同じ分布から引かれるので帯域は埋めるが、
**その上端より中央値で 1.87 標準偏差も下**にあり、上端を更新したのは
**148複合体のうち3件だけ**であった。帯域を埋めるだけの新規 pose が加える勾配は、
既存 negative が既に供給しているものと**冗長**である。

この区別は次の投資先を分ける。「新情報が届かない」なら探索深度 `ntop` を
増やせば解決しうるが、「分布は同じで上端が動かない」なら **`ntop` 増加は
同じ分布からの標本追加にすぎず期待が薄い**。探索そのものを変える介入
（PSC も学習する、回転グリッドを細かくする）か、negative ではなく
positive 側の欠損（勾配ゼロの72複合体）を扱う必要がある。

**なおこの機構説明は round-1 の null を「説明する」ものであって、
独立に検証された因果ではない。** 上端が動かないことと性能が動かないことの
同時観測であり、「上端を動かせば性能が上がる」は別に示す必要がある。

##### 訂正: 「上端が動かない」は機構ではなく、round 0 の成功の言い換えである

上を「機構」として書いたのは適切でなかった（ユーザ指摘: 「hard negative が
少なくなるのはそれはそうだよね？」）。**`loss_margin_hard_negatives` は
positive より上に来る negative を押し下げることを目的とする損失**である。
その学習後のパラメータで再探索すれば、返るのは**モデルが既に低く評価するよう
学習された pose** である。「新しい hard negative が見つからない」は
round 0 が成功したことの言い換えであって、独立した発見ではない。

**実測**（同一の round-0 プールを両パラメータで採点、標準化スコア単位、
勾配を持つ148複合体）:

| | margin 帯域内 negative | 複合体あたり | 最難関 negative | $\min(\text{pos}) - \max(\text{neg})$ |
|---|---:|---:|---:|---:|
| baseline $e_0$ | 191,725 | 1,295 | 5.214 | $-5.890$ |
| **学習後（round 0）** | **176,044** | **1,189** | **4.931** | $-5.318$ |

**学習は帯域内 negative を 8% 減らし、最難関を 5.21 → 4.93 へ押し下げている。**
margin 損失が設計どおり働いた結果である。

##### 蓄積式プールでは hinge の上端を張り直せない

（当初この見出しを「学習が成功すると自己無効化する」と書いたが、
下記のとおりそれは測定より強い。訂正した。）

さらに構造的な問題がある。`absorb()` はプールに**蓄積**し、今回 cap は
一度も効かなかった（`n_old_neg_evicted = 0`）。したがって old negative の
上端は残り続け、新規poseが「これまで見たどの pose よりも難しいもの」にならない限り
上端そのものは更新されない。round 0 が成功するほど、この上端更新は起きにくくなる。

ただし、**これは新規信号がゼロという意味ではない**。hingeは帯域内の全negativeに
非零勾配を出すため、3/148は「新規poseが最難関を更新した複合体」の頻度にすぎず、
新規poseの帯域内寄与やpool標準化の変化は別に残る。したがって実験4のnullを
完全に説明する因果証拠ではなく、少なくとも「蓄積式では上端を張り直せない」という
機構上の限界を示す測定と解釈する。

この読みは実務的な含意を持つ。蓄積式をやめてラウンドごとにプールを
入れ替える（あるいは cap を十分小さくして古い最難関を追い出す）設計なら、
上端を毎回「現在のモデルの最難関」に張り直せる。ただしそれは
**忘却を導入する**別の介入であり、本実験の契約Aとは別物として設計・検定する
必要がある。現時点では候補に挙げるにとどめる。

##### 標準的な hard-negative mining と何が違うか

物体検出などで mining loop が機能する典型的な条件と、今回の構成を並べる。

| | 典型的な mining | 本実験（契約A） |
|---|---|---|
| 候補空間 | 巨大な背景（事実上無限） | **有限**（Hopf 1944回転 × 並進格子） |
| 候補の選び方 | current model の上位 false positive を再抽出 | 同左（同じ scorer が探索を決める） |
| 古い negative | **保持しないことが多い** | **蓄積し、cap も効いていない** |
| positive | 更新されうる | **round 0 に凍結** |
| 探索を決めるパラメータ | 学習される | **PSC は凍結**（$\alpha=1,\rho=3.5$） |

「同じ scorer が候補を選ぶ」こと自体は mining の前提であって欠陥ではない。
本実験で上端が動かなかったのは、**有限な候補空間・蓄積・PSC 凍結**が
重なった結果である。

**`ntop` や `nside` の拡大は loop の改良ではなく探索介入である。**
72件中6件しか救済されなかったことを踏まえると、positive coverage を増やす
目的ではそちらのほうが有望である（codex の指摘）。

##### loop を続けるならどの設計変更か

単純な cap 縮小は**プールサイズ・重み・忘却を同時に変える**ので対照にならない。
筋がよいのは**同じプールサイズでの refresh 対照**である:

- positive quota と random negative は固定する
- negative の一部だけを毎ラウンド current-score 上位で置換する
- accumulate と refresh を**同じ compute・同じ pool size** で比較する

top-k / CVaR のような max 駆動でない loss は次の候補だが、
**候補分布そのものが変わらなければ効果は限定的**である。

**重複除去がなければこの結果は出なかった。** 231,042件を二重に格納すれば、
それらは損失で暗黙に2倍の重みを持ち、val loss の変化を「採掘効果」と
誤読するところだった（§5.14.15 の指摘1）。

##### matched-budget であって matched-compute ではない（実データで確認）

両arm とも `--patience 8` により上限1500の前で早期終了した:
**mine 501 step、continue 601 step**。§5.14.15 で用語を
matched-compute から matched-budget へ改めた判断が、実データで裏づけられた形である。
両者は同じ budget と同じ停止規則を共有するが、消費した step は同じではない。

##### 結論

**この契約・この設定の固定TESTプールでは、hard-negative mining round 1 の
検出可能な増分はなかった。** success@1 は1勝0敗（$p=1$）、AUCは微減し、
round 0からのsuccess判定は完全に不変だった。bootstrap CIの上端は+1.27 ppなので、
小さい正の効果との同等性まで証明した結果ではない。

**言明の範囲を限定する。** 示されたのは
「**positive を凍結し（契約A）、PSC を凍結し（$\alpha=1,\rho=3.5$）、
Hopf nside=3・探索深度top-1500で1 round回す限り**、seed 0の固定プール上で
matched-budget continuationを上回らなかった」であって、hard-negative mining一般が
無効という主張ではない。さらにend-to-end primaryは未測定である。

##### 限界

- **seed 0 のみ**。seed追加は同じ236複合体を独立標本に増やさず、optimizer感度を
  調べる。seed 1は採掘がほぼ完了しているため両armまで完了する価値があるが、
  seed 2よりend-to-end primaryを優先する。
- **契約Aに条件付き**。新 positive も取り込む契約B（on-policy）は未実施であり、
  round-0 探索が近native pose を返さない **72/220 複合体は契約A では永久に勾配ゼロ**である。
  この72件は次に直接検証すべき対象だが、性能限界の主因とはまだ証明されていない。
- **1 round のみ**。§5.14.14 の事前規定どおり round 2 へは進まない
  （改善した場合のみ進む規則だったため）。
- bound-bound、固定プール上の再ランキング、反復利用済み PINDER-S TEST。

##### 次の訓練実験への判断（結果レビュー）

72/220件のsearch-positive欠損は最大の構造的な非効率だが、性能限界の主因とはまだ
証明されていない。lossのmarginや温度を変えてもこの72件には勾配が出ないため、
次はこれを直接falsifyする。

1. **まずseed 0のmine/continueを全249件end-to-endで比較する。**
   これは事前規定primaryであり、seed 2の再採掘より優先する。
2. まず既存のround-1 `mined_raw` cacheを使い、72件のうち学習後探索で新たに
   search positiveを得た件数、positive数、DockQ、first-hit rankを測る。
   これは追加FFTなしで契約Bの救済可能性を判定できる。
3. まだ救済されない複合体だけについて、同じnside=3で`ntop`を段階的に増やし、
   新たにDockQ閾値を超える複合体数、first-hit rank、計算費用を測る。
   99%のreachable ceilingは「top-1500にsearch positiveが入る」ことを意味しないので、
   まずranking truncationを調べる。これで救済されない場合だけnside=4をscreenする。
4. 取得法を事前固定してpositiveを一度だけ生成し、その後は凍結する。
   既にactiveな148件のpositiveは変えず、救済された複合体だけを加えた
   **rescue-only対照**を、同じnegative pool・full144+m5・PSC固定で学習する。
   これにより「実効N増加」をon-policy driftから分離する。
5. rescue-onlyで改善した場合に限り契約Bを行い、§5.14.14のdrift診断と
   positive quota/balanced samplingを適用する。

優先度は **既存raw cacheの救済監査 → positive取得screen → rescue-only学習 → N scaling**。
N scalingは先に各Nでsearch-positiveを持つ実効Nも報告し、名目Nだけを横軸にしない。
loss variantはゼロ勾配を直さないため後回し。PSC同時学習は既にIFACE-onlyより悪く、
探索とrankingを再び交絡するので現時点では棄却する。nside=4の全面学習、round 2、
pool cap増加も、取得screenまたは今回の診断が必要性を示すまで行わない。

seedについては、seed 1はsunk costが大きいので完了してoptimizer頑健性を確認する。
seed 2を止めるなら「事前3-seed完遂」ではなく探索的go/no-go判断だったと明記する。
3-seed結果を論文上の確証として約束している場合だけseed 2も完遂する。

##### 救済監査の結果: 契約Bは却下される（2026-07-27）

上の項目2（既存 round-1 キャッシュによる救済可能性の判定、追加 FFT なし）を
実施した（`scripts/measure_rescue_potential.py`、seed 0）。

| 項目 | 値 |
|---|---:|
| 契約Aで勾配ゼロの複合体 | **72/220** |
| **学習後パラメータの探索が近native pose を見つけた数** | **6（8.3%）** |
| 実効 fit 集合（契約Bを実装した場合） | 148 → **154** |

救済されなかった66件の、探索が返した pose の **best DockQ**:

| | 中央値 | p90 | 最大 |
|---|---:|---:|---:|
| round 0 | 0.077 | 0.176 | 0.229 |
| round 1（学習後パラメータ） | 0.085 | 0.210 | 0.655 |

**閾値 0.23 に対して中央値 0.085 である。**

> **訂正（2026-07-27）。** 当初ここに「沈黙している複合体は**探索の到達範囲の
> 外**にいる」と書いたが、これは誤りである（codex 指摘）。測っているのは
> **top-1500 に入った pose の最良 DockQ** であって、到達可能性ではない。
> **nside=3 の reachable ceiling は 99.0%**（§5.10）であり、これらの複合体でも
> 近native pose は格子上に存在する。示されたのは「現在のスコア関数では
> 上位1500に入らない」——**ranking truncation** であって reach の欠如ではない。

**判断: 契約B（on-policy positive refresh）を却下する。**
実効 N は 148 → 154（+4%）にしかならず、この増分で success@1 の変化を
検出できる見込みはない（実験1の実測で、+3.39 pp の効果ですら 9勝1敗 $p=0.0215$
でようやく有意だった）。self-training の drift リスクを負う価値がない。

**72件の勾配ゼロは訓練契約の問題ではなく、候補取得（ranking truncation）の
問題である。** これは §5.14.21 の negative 側の診断と同じ方向を指す——
上端が動かないのも positive が見つからないのも、**探索が返す候補集合が
変わらない**ことに帰着する。

ただし `ntop` の拡大は**配備時の top-1 を直接は変えない**（深い順位から
positive を取得して学習に参加させる介入である）ため、本命ではなく
**安価な coverage 診断**として位置づける。

##### パラメータは動いたのか、そして round 0 は収束していたのか

「学習でパラメータは実際に動いているのか」を確かめるため、公開表 $e_0$ に対する
学習差分の大きさを測った。

| | $\|\Delta e\|$ | $\|e_0\|=8.669$ 比 | 自分の値の10%超動いた成分 | 50%超 |
|---|---:|---:|---:|---:|
| round 0 | 1.5594 | **18.0%** | **95/144** | 34/144 |
| round 1 mine | 1.5884 | 18.3% | 98/144 | 33/144 |
| round 1 continue | 1.6197 | 18.7% | 95/144 | 34/144 |

**round 0 では確かに動いている**（144成分中95個が自分の値の10%以上、34個が50%以上、
success@1 は +8.0 pp）。一方 **round 0 → round 1 の変化は $\|\Delta e\|$ で
+1.9% にすぎない**。

**round 0 は収束していなかった。** `round0_trajectory.csv` を読むと:

| | 検証チェック | accepted | 最後に改善した step | val loss |
|---|---:|---:|---|---|
| round 0 | 31 | 26 | **1500 / 1500** | 2.9653 → 2.7240 |
| round 1 mine | 11 | **2** | **101** / 501 | 2.7240 → 2.7210 |
| round 1 continue | 13 | **3** | **201** / 601 | 2.7240 → 2.7191 |

round 0 は plateau に達したのではなく、**1500 step の予算を使い切っただけ**である
（§5.14.12 の限界「全 run が 1500 step 上限に到達しており収束 plateau は未確認」を
実データが裏づけた）。ところが同じ最適化を round 1 で続けると、両arm とも
**100〜200 step で頭打ち**になる。さらに `continue` は val loss を
2.7240 → 2.7191 と下げたのに、**TEST success@1 は 77.54% → 77.12% と下がった**。

**これは round 1 の null に対する別の説明を示唆する。** 「採掘が情報を足さない」
のとは独立に、**round 0 の 1500 step 地点から先は validation が下がっても
TEST が改善しない領域**に入っている可能性がある。

> **訂正（2026-07-27）。** これを口頭で「過学習の兆候」と述べたが、**誤りである**
> （codex 指摘）。通常の過学習は train が改善し held-out validation が悪化する
> ことを要するが、ここでは **validation も改善している**。しかも val loss の
> 低下は 0.18%、TEST の低下は 236件中 **1件**で $p=1$ である。
> より自然な候補は次のとおりで、いずれも過学習ではない:
>
> - 平滑な損失と離散指標 success@1 の**不一致**（objective mismatch）
> - checkpoint noise（1複合体の反転は測定限界の内側）
> - validation と TEST の**分布シフト**
> - validation そのものへの**反復選択**
>
> 分離には TEST を使わず、複数の held-out fit fold で checkpoint 軌跡を測り、
> さらに探索 recall の高群・低群へ層別する。全 held-out で後半悪化なら過学習、
> 高recall群だけ TEST 同様に悪化なら分布シフト、損失だけ改善して top-1 が
> 不変なら objective mismatch である。

**なお §5.14.11 の「validation recall 65.8% vs TEST 94.8%」を自然な分布差と
決めつけてはならない。** 同じ prep cache・同じ除外リスト・同じ探索条件で
生成されているかをまず監査すべきである（codex 指摘）。原因未特定のまま
「交換可能でない」と書いたのは踏み込みすぎであった。

**round 1 の null に対する2つの説明は現データでは分離できていない。**

##### 次の実験候補（優先順、2026-07-27 改訂）

codex の効果量見積もり（下記「確率的な投資判断」）を受けて順序を改めた。
**当初 step budget 対照を最優先に置いたが、N scaling を本命に繰り上げる。**
理由は、active complex が 148 しかない以上、現在の制約は**モデル容量ではなく
推定分散**である可能性が高いこと（full 144 が additive 23 を 3.4 pp 上回る以上、
容量は現に効いている）。

1. **N scaling（本命）。** $N = 220 \to 500 \to 1000$。
   期待 $+1$〜$3$ pp、$1$ pp 以上の確率 50〜70%。
   **各 $N$ で search-positive を持つ実効 $N$ も併記**し、名目 $N$ だけを
   横軸にしない（148/220 = 67% が実効値である）。
2. **未使用または unbound cohort での end-to-end 確認。**
   反復利用済み PINDER-S TEST でさらに 0.5 pp を探すより、外部妥当性の確認の
   ほうが研究上の価値が高い。
3. **step budget 対照。** round 0 を 1500 と 3000 step で、early stopping を
   無効化し**固定 step checkpoint** を同一 validation 規則で記録する。
   round 1 の null が「採掘の無効」なのか「1500 step 以降に伸びしろがない」のかを
   分離する。採掘を含まず pool cache を再利用するので追加コストはゼロ。
4. **`ntop` / `nside` の小規模 coverage 診断。** 救済されなかった66件について
   `ntop` を段階的に増やし、閾値を超える複合体が増えるかを測る。
   **これは配備時の top-1 を直接変える介入ではない**（深い順位から positive を
   取得して学習に参加させるだけ）ので、安価な診断として位置づける。
5. **refresh 型 mining の seed 0 pilot。** 総 pool size・positive 数・
   random-negative 数を固定し、**hard-negative 部分だけ**を置換する。
   old-hard の一部は reservoir として残し、全置換による round 間 cycling を監視する。
   **単純な cap 縮小は pool size・忘却・class 比を同時に変えるので対照にならない。**
   validation で明瞭な改善がなければ終了。

**却下・後回し**（根拠つき）:

- **契約B（on-policy positive refresh）**: 救済監査で 72件中6件（8.3%）しか
  救えないと判明。実効 N は 148 → 154 にしかならない。
- **現行 accumulate 型の追加ラウンド**: $1$ pp 以上改善する確率 <5%。
- PSC 同時学習: §5.14.8 で IFACE-only より悪く、探索と ranking を再び交絡する。
- loss 改良（top-k / CVaR 等）: 期待 $0$〜$1$ pp、確率 10〜25%。
  候補分布自体が変わらなければ効果は限定的。
- nside=4 の**全面学習**: 期待 $0$〜$2$ pp、確率 25〜40%。項目4の診断が
  必要性を示すまで行わない。
- round 2、pool cap 増加。

##### 確率的な投資判断と研究の停止条件（2026-07-27、結果レビュー）

以下は新しい測定値ではなく、§5.14.21までの結果に基づく**主観的な事前見積もり**である。
統計的confidence intervalではない。

**mining loop。** 現行のaccumulate型をもう1 round回してsuccess@1が1 pp以上改善する
確率は5%未満と見る。refresh型には、current searchのtop-1500に残る約69%を維持しつつ、
古くなった約31%を新しい候補へ同数置換するという、現行実験が試していない介入が残る。
したがってrefresh型が1 pp以上改善する確率はおよそ20〜30%、2 pp以上は10%未満と見る。
これは小規模pilotを正当化するが、3 seedの本実験へ直行する強さではない。

refreshを試す場合は総pool size、positive、random-negative数を両armで固定し、
`stale negative -> current-score上位negative`の置換だけを変える。単なるcap縮小では
pool size、class比、忘却が同時に変わる。old-hardの一部をreservoirとして残し、
全置換によるround間cyclingも診断する。比較対象は同じstep budgetの固定pool継続である。

**重要な限定。** `loss_margin_hard_negatives`はmax一個だけでなく帯域内全negativeの
平均hingeである。したがって3/148は上端更新の希少性を示すが、refreshの効果を
論理的に否定しない。一方、mine armは新規negativeをすでに帯域の23.2%まで追加して
固定pool順位を改善しなかったため、refreshが大幅改善するとの期待も支持しない。

**continueの解釈。** val lossの2.7240から2.7191への低下は0.18%であり、
TEST success@1の低下は236件中1件だけ（$p=1$）である。これは通常の意味のoverfitを
示す証拠ではない。train loss低下とval loss上昇が観測されておらず、
smoothな学習lossと離散top-1の不一致、checkpoint noise、validation/TEST shiftの
いずれでも説明できる。特にrecall 65.8%対94.8%の原因を未同定のまま
「分布差」と確定してはならない。

分離には、まず同じcache schema・探索条件・除外規則でsplit別recall差を再監査する。
次にTESTを使わず、fit側interface clusterから複数のheld-out foldを作り、
各checkpointについてtrain loss、held-out loss、AUC、success@1の軌跡を測る。
さらにheld-outをsearch recallの低群・高群に層別する。全held-outでstep増加により
悪化すればoverfit、高recall群だけTESTと同じ悪化を示せばshift、lossだけ改善して
top-1が動かなければobjective/endpoint mismatchが主因である。最終確認は未使用cohortで
一度だけ行う。

**期待効果の順位（1 pp以上改善する主観確率）。**

1. `N=500/1000` scaling: 50〜70%。144成分に対してactive complexが148しかなく、
   fullがadditiveを3.4 pp上回ったため、容量より推定分散が制約である可能性が高い。
   名目Nとsearch-positiveを持つ実効Nを併記する。
2. nside=4 end-to-end: 25〜40%。候補集合を直接変えるが、nside=3のreachable ceilingが
   99%なので巨大な改善は期待しにくく、計算量増加も大きい。
3. `ntop`拡大 + 再学習: 15〜30%。同じ最終scoreでtop-1を選ぶ限り、`ntop`だけを
   増やしても配備時top-1は直接変わらず、価値は深い順位からpositiveを取得して
   訓練へ参加させる点にある。未救済66件のbest DockQ中央値0.085は大半が
   閾値直下でないことを示すため、まず段階的screenに留める。
4. loss variant: 10〜25%。現lossは公開値から+8 ppを実現しhard negativeも減らしている。
   148 complexのままtail weightingを変えると、改善より分散・overfitを増す可能性がある。

実行順は費用も考慮し、step-budget対照、`ntop`の安価なcoverage screen、N scaling、
nside=4 pilotとする。refresh miningは機構確認としてseed 0の小規模pilotまでに留め、
validationで明瞭な改善がなければ終了する。

**研究の落とし所。** 公開パラメータ69.5%から固定pool 77.5%への+8.0 pp
（21勝2敗）と、全249件end-to-endでの改善は、IFACEを微分可能に学習する主結果として
すでに十分に大きい。full 144がadditiveを上回る対照と、fixed-positive accumulate型
miningのnullも一つの整合した物語になる。ただしPINDER-S TESTは反復利用済みであり、
$p$値を最終confirmatory evidenceとは呼べない。

追加投資は「さらに0.5 ppを探す」より、(1) 未使用またはunbound cohortでのend-to-end
確認、(2) N scaling、(3) 探索coverageの小規模診断へ限定する。この三つのうち
N scalingがflatで、探索pilotも1 pp未満なら、モデル容量をさらにいじらず現結果を
まとめる。refresh loopは論文の成立条件ではなく、negative resultの機構補足である。

##### コマンド

```bash
# Phase A: round-0 学習（seed ごと）-> round0_ckpt.pt + round0_optstate.pt
uv run python scripts/run_pinder_scaling.py \
    --n-fit 220 --seed $s --rounds 0 \
    --freeze-psc --loss-prov search --alpha0 1.0 \
    --iface-mode full --lambda-margin 0.5 \
    --grid-voxels data/scaling/grid_voxels_1.2.json \
    --pool-cache data/scaling/pool_cache \
    --test-cache data/shards_pinder/test_pool_reachable.pt \
    --max-grid-voxels 31250000 \
    --exclude-bad-geometry data/scaling/excluded_bad_geometry.txt \
    --exclude-homodimer --out-dir data/scaling/runs_r1

# Phase B: round-1 採掘（5シャード、パラメータ指紋でキャッシュを分離）
uv run python scripts/run_pinder_scaling.py ... \
    --rounds 1 --mine-only --mine-from-ckpt <round0_ckpt.pt> --mine-shard $i/5

# Phase C: 両arm を同一 round-0 状態から分岐
uv run python scripts/run_pinder_scaling.py ... \
    --rounds 1 --resume-from <round0_optstate.pt> \
    --mining-contract {hardneg,none}

# 固定TESTプール評価と mine/continue contrast
uv run python scripts/compare_conditions.py --pool data/shards_pinder/test_pool_reachable.pt \
    --ckpt data/scaling/runs_r1/N220_seed0_{hardneg,none}/round1_ckpt.pt --prov search
uv run python scripts/r1_contrast.py
```

**hardware**: round-1 採掘は 220複合体を5シャードで約1時間半（1シャード44件、
実測 140〜180 秒/複合体、共有機・load average 20〜30）。学習は1本あたり約40分。
評価は CPU のみ。

#### 5.14.22 評価体系を PINDER に統一する（2026-07-28）

ここまで自前の指標（平均 success@K、閾値 DockQ ≥ 0.23 の2値）で報告してきたが、
**以後は PINDER の用語・分割・指標に揃える。** 「PINDER で評価した」と言うなら
PINDER に採点させるべきであり、独自指標は近似にとどまる。

##### PINDER の座標系と、本研究の位置

| 軸 | PINDER の選択肢 | 本研究 |
|---|---|---|
| subset | `pinder_xl` / **`pinder_s`** / `pinder_af2` | **`pinder_s`**（公式250件、現cache 249件） |
| **monomer** | **`holo`** / `apo` / `predicted` | **`holo` のみ** |
| 集計 | 全複合体の**中央値** | これまで平均 |
| CAPRI | 4段階（Incorrect / Acceptable / Medium / High） | これまで acceptable の2値 |
| 系統 | `Oracle` / `Max(Top 1)` / `Max(Top 5)` | success@K |

**`holo` は PINDER の3設定のうち最も易しい。** 結合型構造どうしをドッキングするので
構造変化を扱わない。実用上意味があるのは `apo`（非結合型）と `predicted`（AF2 構造）で、
そこでは側鎖・主鎖の再配置への頑健性が問われる。本研究の数値はすべて
**`pinder_s` / `holo` 列の結果**であり、他手法の `apo` / `predicted` の数値と
並べてはならない。これまで「bound-bound」と書いてきたものは PINDER 用語で `holo` である。

##### 指標の定義（`pinder.eval.dockq` の実装より）

各 decoy について `BiotiteDockQ` が **DockQ / iRMS / LRMS / Fnat / CAPRI** を計算する。
CAPRI は `Incorrect=0, Acceptable=1, Medium=2, High=3`。

リーダーボードは複合体ごとに oracle 集約（`DockQ`,`Fnat`,`CAPRI_rank` は max、
`LRMS`,`iRMS` は min）してから、**全複合体の中央値**を取る。hit rate は

- `CAPRI_rank ≥ 1`（acceptable）、`≥ 2`（medium）、`≥ 3`（high）の複合体割合
- `iRMS ≤ 2 Å`、`LRMS ≤ 5 Å`、`Fnat ≥ 0.8` の複合体割合

を `Oracle`（返した全 pose の最良）、`Max(Top 1)`、`Max(Top 5)` の3系統で報告する。

**本研究の success@1 / success@5 は `Max(Top 1)` / `Max(Top 5)` の
acceptable hit rate に対応する**が、(a) 閾値が自前 DockQ の 0.23 であって
`BiotiteDockQ` の CAPRI 判定ではない、(b) 平均であって中央値ではない、
という2点で厳密には別物である。

**`Oracle` は本研究の設定では情報を持たない。** PINDER の `Oracle` は
「提出した decoy 全部の中の最良」であって到達可能性の上限ではない。
上位5件しか提出しないので `Max(Top 5)` と同一の値になる。
探索が1500件中に近native pose を見つけられたかは、同じ実行が出す
`recall` と `best_dockq_out`、および回転格子上の上限 `ceiling_dockq`
（§5.10 の reachable ceiling、スコア関数に依存しない）から報告する。
1500件すべてを PDB 化すれば `Oracle` が意味を持つが、249 × 1500 = 37万ファイルになる。

##### 提出形式

```
eval_dir/{method_name}/{pinder_system_id}/{monomer}/models/model_{rank}.pdb
```

各 decoy は**鎖 R と L のみ**（それ以外は `ExpectedChainsNotFoundError`）。
rank はファイル名末尾の整数から読まれる。

##### 本研究でこれをどう満たすか

固定 TEST プールは特徴量しか保存しておらず pose 座標を持たないため、そこからは
書き出せない。`scripts/eval_search_test.py` に `--export-pdb-dir` を追加し、
**end-to-end 探索（学習後パラメータで実際に FFT 探索を回す）の上位 K pose** を
直接 PDB 化する。固定プール上の再ランキングではなく実際にドッキングした結果を
PINDER に採点させる形になり、codex が優先度2に挙げた
「全249件 end-to-end 評価」と同じ計算で済む。

decoy の原子は、prep cache が原子名・残基番号・鎖を保持しないため、
元の複合体 PDB を `parse_pdb_plain` と同一の規則で読み直して復元する。
両者の原子数が一致しない複合体は**書き出さずに報告する**（対応がずれた decoy を
採点させると別の原子を評価することになる）。

##### PDB export / 公式harness実装レビュー（2026-07-28、採点前）

対象は未コミットの`scripts/eval_search_test.py`と
`scripts/score_decoys_with_pinder.py`。新しいend-to-end探索や公式leaderboard値は
まだ生成していない。

**座標と原子順の実測監査。** 公式`pinder_s` 250 IDのうちprep cacheがある249件すべてで、
元complex PDBを`parse_pdb_plain(..., "R"/"L")`に通し、受容体重心を引いた座標を
cacheの`rec_xyz` / `native_lig`と全原子比較した。原子数不一致は0/249、
最大絶対座標差は$7.5\times10^{-5}$ Åであった。したがって現在のPDB版とcacheについては、
filtered atom行$i$とpose行$i$の対応、および受容体中心座標frameは一致する。
ただし原子数だけでは別版PDB・同数の順序入替を検出できないため、この座標照合を
export preflightへ入れる。

現writerでnative poseを1件書き、Biotite 1.7.1で直接parseした結果、1648原子、
chain集合`{R,L}`、座標範囲も有限であった。PDB列、chain位置、`TER` / `END`はparse可能。
現在のwriterはoccupancy/B-factor以降とelement列を保持する。全250件でZDOCK filterが
除いた68,418/1,065,271 ATOM（6.42%）は水素だけで、重原子欠落は0件であった。
PINDERのcontact計算は`heavy_only=True`、RMSDはbackbone定義なので、
水素を書かないことは公式指標との不整合ではない。

**production blocker 1: 現環境でPINDER harness自体が動かない。**
実測環境は`pinder 0.5.0 / fastpdb 1.3.3 / biotite 1.7.1`。
`MethodMetrics`が使うfastpdb readerはnative PDB読込時に
`AttributeError: attribute 'lines' of 'biotite.structure.PDBFile' objects is not writable`
で停止した。PINDER側の`safe_get_eval_metrics()`はsystem単位の例外を捕捉して`None`へ
変えるため、一部失敗はmissing penaltyへ化け、全件失敗は後段の空`concat`でしか
停止しない。互換依存をpinするか、
公式実装をbiotite backendで確実に動かし、既知native poseがDockQ=1 / Highになる
round-trip testを通すまで本採点を開始しない。

**production blocker 2: exportと採点がfail-open。**
`eval_search_test.py`の広い`except Exception`は探索だけでなくPDB exportの失敗も
`skipped`へ落とし、長時間の進捗表示を正常に見せる。一部失敗ならexit 0になり、
全件失敗なら最後の`rows[0]`で初めて別の例外になる。source欠損・atom mismatchも
`exported=0`を返すだけである。公式export時は探索結果生成とartifact書出しの例外を分け、
書出し失敗をfatalにし、終了時に要求IDごとにexactly K modelがあることをassertする。

さらに`MethodMetrics(..., allow_missing_systems=False)`は「全件必須でエラー」ではなく、
missing systemをDockQ/Fnat 0、RMSD 100のpenalty rowで補う。現在cacheにない
`7w93__A1_A0A140N873--7w93__A2_A0A140N873`は公式250件の一つなので、
現状のofficial denominatorでは1件のゼロ点になる。`score_decoys_with_pinder.py`の
`--allow-missing` helpにある「require full subset」は不正確である。採点前に
`get_expected_counts`と照合し、real decoy数、penalty数、欠損IDを別に表示する。
headlineで`--allow-missing`は使わない。

**production blocker 3: partial/stale artifact。**
writerは最終`model_K.pdb`へ直接書くため、途中例外で壊れたPDBが残る。既存method dirを
再利用すると、今回書かなかったsystemや以前の`model_6.pdb`以降もharnessが拾う。
一時ファイルからatomic renameし、method dirはcheckpoint/config fingerprintで新規作成、
または開始時に空であることを要求する。各systemのrank集合がexactly
`{1,...,K}`で、全modelが`{R,L}`だけを持つことを全ファイル検査する。

**production blocker 4: `holo`の誤ラベルがCLI上可能。**
このpipelineはbound complexしかdockしないのに`--monomer`で`apo/predicted`を選べる。
公式比較では誤ラベルが重大なので、export経路は`holo`へ固定し、他値を拒否する。
`--export-pdb-dir`はeval rootではなくmethod rootであることもhelpへ明記する。

**仕様確認。** `{method}/{id}/holo/models/model_{rank}.pdb`と`model_1`〜`model_5`は
PINDER 0.5.0のdirectory探索とrank regexに一致する。5件だけ提出する限り
OracleとMax(Top 5)は同一で、探索1500件のoracleとは呼ばない。連続指標はsystemごとの
oracle後に中央値だが、CAPRI hit rateは全systemの割合であり、「全指標を中央値集計」
という表現は正確でない。leaderboardは複数subset行を返しうるため、主表は明示的に
`Dataset=pinder_s, Monomer=holo`だけを選ぶ。

**評価設計の限定。** これは`pinder_s / holo`のrigid-body redockingであり、
3設定中最も易しい。`apo` / `predicted`で必要なconformational robustnessは測らない。
公式250件中1件を内部geometry floorで除外しているため、その1件をfailureとして
ゼロ点にするか、floorが過剰拒否でないかを再監査して実行するかを採点前に固定する。

##### geometry guardの再監査と、除外1件の復帰（2026-07-28）

**問い。** §5.12 で導入した `MIN_PLAUSIBLE_CONTACT_ANGSTROM = 2.0` は、
公式250件のうち `7w93__A1_A0A140N873--7w93__A2_A0A140N873` を拒否していた。
このguardはPINDERの収録判断を上書きしてよいものか。

**判断（利用者の指摘による）。** データセットの作成・収録判断はPINDER側の仕事であり、
本研究の仕事ではない。§5.12 の失敗はPINDERのデータ欠陥ではなく、
**どのファイルを読むかを我々が間違えた**（`test_set_pdbs/` の原点中心docking入力を
複合体として対にした）ことによる。`pdbs/{id}.pdb` は最初から正しかった。
したがってguardが検出すべきなのは*我々のframe取り違え*であって、参照構造の品質ではない。

**測定。** 誤った旧cache（`prep_cache_test_BROKEN_origin_centred`、250件）と
正しいcache（`prep_cache_test`、249件）が両方残っていたので、両者を分ける統計を実測した。

| 由来 | ligand重原子のうち2.0 Å未満に相手を持つ割合 | 最近接距離 |
| --- | --- | --- |
| 原点中心monomerを対にした旧入力 | 0.247 – 1.000（中央値 0.607） | 0.000 – 0.405 Å |
| `pdbs/{id}.pdb` の複合体 | **249件すべて 0.000** | 2.021 – 3.225 Å |
| うち `7w93`（旧guardが拒否） | 1 / 2323 原子 = 0.00043 | 1.892 Å |

frame崩壊はligandの1/4以上を受容体内部へ埋没させ、正しく組まれた複合体は1原子も
埋没させない。両者の間は3桁空いている。

**変更。** 判定を絶対距離から**埋没率**へ移した
（`MAX_FRAC_LIG_BELOW_CONTACT_FLOOR = 0.05`、`src/zdock/dataset.py`）。
0.05は観測された失敗の最小値0.247の1/5であり、正しい構造側には0.000しかない。
距離閾値を下げる案は採らない。それでは真に貫入した参照も通ってしまうし、
「1.892 Åの複合体は収録すべきでない」という判断自体が我々の越権である。
`7w93` を再prepし、公式250件すべてを提出する。単体テスト128件は変更後も全て通る。

##### 公式PINDER評価の結果（`pinder_s` / `holo`、250/250、2026-07-28）

`pinder.eval.dockq.MethodMetrics` による採点。baselineは**同一実装＋ZDOCK 3.0.2公開表**、
`trained_N220` は `data/scaling/runs_nfixed/N220_seed0/round0_ckpt.pt`。
どちらも学習後パラメータでFFT探索をやり直したend-to-end結果の上位5 poseを提出している。
再ランキングではない。母数は公式250件で、欠損は0件。

| 指標（`Max(Top 1)`） | published | trained_N220 | 差 |
| --- | --- | --- | --- |
| DockQ acceptable | 66.40 % | **74.80 %** | **+8.40 pp** |
| DockQ medium | 61.60 % | 69.20 % | +7.60 pp |
| DockQ high | 30.80 % | 32.40 % | +1.60 pp |
| iRMS ≤ 2 Å | 61.20 % | 68.00 % | +6.80 pp |
| LRMS ≤ 5 Å | 58.40 % | 65.20 % | +6.80 pp |
| Fnat ≥ 0.8 | 44.00 % | 46.80 % | +2.80 pp |

| 指標（`Max(Top 5)`） | published | trained_N220 | 差 |
| --- | --- | --- | --- |
| DockQ acceptable | 81.20 % | 86.00 % | +4.80 pp |
| DockQ medium | 76.40 % | 81.60 % | +5.20 pp |
| DockQ high | 36.80 % | 37.60 % | +0.80 pp |
| DockQ 中央値 Top 1 | 0.71 | 0.73 | +0.02 |
| DockQ 中央値 Max(Top 5) | 0.77 | 0.77 | ±0 |

`7w93` 復帰の寄与は事前の見積りどおりで、両条件とも rank 1 でヒットしたため
（DockQ 0.659、両条件同一）各条件が +0.40 pp 上がり、差の +8.40 pp は変わらなかった。
249件時点の値は acceptable 66.00 → 74.40 % であった。

`Oracle` 列は本設定では `Max(Top 5)` と同一であり、探索1500件の到達可能性ではない。
`pinder_xl` / `pinder_af2` 行は同じ250件を大きい母数に対して採点したもので、
提出していないsystemはゼロ点penaltyになっている。主張に使えるのは
`Dataset=pinder_s, Monomer=holo` 行だけである。

**この数値が何であって何でないか。** これは`holo`（結合型どうし）のrigid-body
redockingであり、PINDERの3設定で最も易しい。またPINDER-Sの250件は
損失設計・margin・パラメータ部分空間・mining可否といった設計判断に反復利用しており、
+8.40 ppは**探索的・記述的なpaired improvement**である。選択後バイアスのない
confirmatory効果量ではない（§5.14.23の訂正1）。baselineは本実装＋公開表であって
original ZDOCK binaryではないため、原著実装を上回った証拠でもない。

```bash
# 1) 除外1件の再prep（guard変更後）
PINDER_BASE_DIR=$PWD/external/pinder uv run python scripts/prep_pinder_cache.py \
    --ids-file <(echo 7w93__A1_A0A140N873--7w93__A2_A0A140N873) \
    --cache-dir data/scaling/prep_cache_test \
    --complex-dir external/pinder/pinder/2024-02/pdbs --gpus 0 --force

# 2) end-to-end探索 + PDB書き出し（両条件）
uv run python scripts/eval_search_test.py --label published \
    --export-pdb-dir data/pinder_eval/published --export-top-k 5 --monomer holo \
    --prep-cache data/scaling/prep_cache_test --out-dir data/scaling/eval_search_pinder
uv run python scripts/eval_search_test.py --label trained_N220 \
    --ckpt data/scaling/runs_nfixed/N220_seed0/round0_ckpt.pt \
    --export-pdb-dir data/pinder_eval/trained_N220 --export-top-k 5 --monomer holo \
    --prep-cache data/scaling/prep_cache_test --out-dir data/scaling/eval_search_pinder

# 3) 公式harnessで採点
PINDER_BASE_DIR=$PWD/external/pinder uv run python scripts/score_decoys_with_pinder.py \
    --eval-dir data/pinder_eval --out data/pinder_eval/leaderboard.csv --max-workers 16
```

#### 5.14.23 次実験の優先順位と停止規則（2026-07-28、実行前レビュー）

**問い。** §5.14.21 の mining null、N=220→500 の null、§5.14.22 の
PINDER公式holo評価を踏まえ、次のGPU投資を外部妥当性、探索診断、モデル容量、
miningのどこへ配分するか。本節は既報値の解釈と実行計画であり、新しい測定値はない。

**優先順位。**

1. **未使用outer cohortで、凍結済みrecipeを一度だけend-to-end評価する。**
   PINDER-S 250件は設計判断に反復利用済みなので、現在の+8.40 ppは探索的効果量である。
   interface clusterで学習系と分離した未使用cohortを事前固定し、公開表と学習表を
   同一探索・同一denominatorでpaired比較する。checkpoint、閾値、欠損処理は結果を見る前に
   凍結する。
2. **PINDER apo評価。** leaderboard上の本命であり、holoで学んだIFACE更新が
   conformational shiftに耐えるかを測る。ただし同じ250 systemなので、未使用標本による
   confirmatory testの代用にはしない。apoが存在する全対象ID、欠損、失敗を事前固定し、
   baselineと学習表を同じ入力でpaired評価する。
3. **`ntop`→`nside`の順の小規模診断。** 勾配ゼロだった66件を中心に、深い順位で
   near-nativeが回収されるかを測る。`ntop`は配備時top-1を直接変えず、nside=3の
   reachabilityは99%なので、全面nside=4へ直行しない。候補回収が実効active Nを
   十分増やすときだけ再学習する。
4. **N=1000を完了。** 既存cacheのsunk costを利用するが、N=220→500がsuccess@10で
   nullだった以上、本命とはしない。名目Nだけでなくsearch-positiveを持つactive N、
   split composition、success@1/5、公式CAPRIを報告する。
5. **refresh miningはseed 0のgo/no-go pilotだけ。** positive、random negative、
   総pool size、optimizer stepを固定し、stale hard negativeだけをcurrent model由来の
   hard negativeへ置換する。old-hard reservoirを残しcycling/forgettingを測る。
   validationのpaired endpointに事前規定した明瞭な改善がなければ終了する。
6. **独立したstep-budget対照は実施しない。** 既にmineとno-mining continuationを
   同じround-0状態から比較して増分がnullである。exact step matchingはrefresh pilotの
   内部対照としてのみ行う。現行accumulate miningの解釈を精緻化するためだけにGPUを
   使わない。

**miningの残る最強形。** 毎epoch同じtop-1500を再取得するだけでは69.3%重複を
再現する可能性が高い。試すなら、蓄積ではなく同一サイズで置換し、current scorerとは
独立に広く作った候補reservoir（深い`ntop`、必要なら小規模nside=4）からcurrent-score
上位false positiveを選ぶ。これでもvalidationが動かなければmining lineを終了する。

**容量仮説の反証可能な試験。** N scalingのflatnessだけでは「144パラメータだから容量が
限界」とは言えない。まず同一候補集合・同一splitで、現行線形scoreを包含する
非線形reranker（入力は同じPSC/ELEC/144 IFACE特徴）と比較し、線形結合の制約を診断する。
次に改善した場合だけ、距離shellまたは局所環境で条件づけた複数IFACE表など、
現行scoreをパラメータ tyingで厳密に包含するFFT互換scoreを実装し、未使用cohortを
end-to-end paired評価する。現行144が十分に最適化され、rich modelだけが再現性のある
改善を示したときに限り、ZDOCK functional formの容量をbinding constraintと結論する。
同一特徴の非線形modelもnullなら容量制約は示されず、特徴表現またはデータが候補として残る。

**+8.40 ppの主要な過大評価リスク。** 実装baselineとのpaired差そのものより、
反復利用した同じPINDER-S 250件への研究者側のtest adaptation/winner's curseである。
凍結recipeの未使用outer cohort評価がこれを検出する。公式harnessについては別途、
expected ID、exactly K、missing penalty、atom-coordinate round trip、空のfingerprinted
artifact directoryをfail-closedで検査する。Highの増分が小さいため、acceptableへの
遷移が少数の閾値近傍例へ集中していないかもsystem別CAPRI遷移で報告する。

**計算と停止条件。** 7 GPUなら250-system search条件は並列化し、baselineと学習表の
各end-to-end比較を同じwaveで行う。固定pose上のreranker診断はsearchより安価である。
FFT互換rich scoreは実装と追加相関が主費用で、距離/環境channel数にほぼ比例するため、
rerankerが明確に勝つ前には実装しない。未使用cohortでgainが再現せず、apoでも改善せず、
`ntop`診断がactive Nを実質的に増やさない場合は、N=1000完了後に追加モデル探索を止める。

**追加の必須監査。** same-UniProt homodimerの対称解をnegative扱いする既知問題について、
対称性対応評価または除外感度をheadlineと併記する。また本結果のbaselineは
「本PyTorch実装＋公開表」であり、original ZDOCK binaryとの同一性を示さないため、
原著実装を上回る主張には別のparity checkが必要である。

##### §5.14.23の争点再検討と訂正（2026-07-28）

**状況更新。** geometry guardは最近接距離の絶対閾値から、
`MAX_FRAC_LIG_BELOW_CONTACT_FLOOR=0.05`へ変更した。誤って原点中心monomerを重ねた
旧入力では2.0 Å未満に埋没するligand重原子割合が24.7〜100%（中央値60.7%）だったのに対し、
正しいcomplex PDBは249件で0%、`7w93`だけ2323原子中1原子だった。したがって
`7w93`を含む公式250件を評価し、内部guardでPINDERのcurationを上書きしない。

PINDER公式harnessによる`pinder_s / holo` 250/250の確定値は、公開表baselineから
N220 seed 0学習表へ、Max(Top 1) acceptable 66.40→74.80%（+8.40 pp）、
medium 61.60→69.20%、high 30.80→32.40%、Max(Top 5) acceptable 81.20→86.00%、
DockQ median Top 1 0.71→0.73、iRMS≤2 Å 61.20→68.00%、
LRMS≤5 Å 58.40→65.20%、Fnat≥0.8 44.00→46.80%だった。

**訂正1: 「未使用outer cohort」は同格の第二PINDER-Sを意味しない。**
そのような公式cohortは存在しない。したがって「PINDER-Sと同品質のconfirmatory set」を
最優先とした読みは撤回する。現在の+8.40 ppは、
**反復的な設計判断に使った公式PINDER-S/holo上の探索的・記述的なpaired improvement**
として主張できるが、未見PINDER-S母集団に対する選択後バイアスのない効果推定や、
confirmatoryな$p$値とは呼ばない。

独立transport確認を実施する場合は、候補(ii)を次のように具体化する。
PINDER 2024-02 indexで`pinder_xl & ~pinder_s & holo_R & holo_L`は1705 systemで、
すべて`split=test`、かつN220 fit+valおよびPINDER-Sとの`cluster_id`重複は実測0だった。
この1705件を固定salt付きSHA-256で順序づけ、先頭250件を**結果を見る前に**凍結する。
これは第二PINDER-Sではなく、selection stringencyの異なる未使用
`PINDER-XL-minus-S/holo`へのtransport testである。

事前登録するのはexact ID listとhash、N220 seed 0 checkpoint、公開表baseline、
Hopf nside=3 / spacing 1.2 Å / top-5、全250件denominator、失敗・欠損はゼロ点、
primaryをMax(Top 1) acceptableのpaired差とexact McNemar、secondaryを
medium/high/Max(Top 5/連続中央値)、結果後のrecipe変更なし、とする。
このフル探索費用を払わないならouter確認は計画から削除し、存在を仮定しない。

**訂正2: 任意の非線形rerankerを容量診断の第一歩にはしない。**
top-1500保持pose上の非線形modelはFFT retrievalを変えず、勝っても配備可能性を示さず、
負けても現行FFT-compatible線形式の十分性を示さない。§5.14.23のこの提案は撤回する。

代わりに、現行radial contact kernelを
$K_0(d)=K_{\rm near}(d)+K_{\rm far}(d)$と厳密に分割し、

$$S_{\rm IFACE}=\sum_{ij}e^{\rm near}_{ij}T^{\rm near}_{ij}
                 +\sum_{ij}e^{\rm far}_{ij}T^{\rm far}_{ij}$$

とするFFT-compatible nested extensionを第一候補にする。
$e^{\rm near}_{ij}=e^{\rm far}_{ij}=e_{ij}$をtieすれば現行scoreをbitwiseまたは
規定tol内で再現することを、feature scoreとFFT searchの両方でtestする。
まず固定pose上でこの**同じ配備可能関数族**のuntied residualがvalidation改善するかをscreenし、
改善時だけ追加FFT channelを実装してend-to-end比較する。実装費以外に、
この順序を任意の非線形rerankerより後へ置く理由はない。

ただし追加144自由度をactive N=148で判断するとdata不足と容量不足が交絡する。
shell residualへ階層priorを置き、可能ならN=1000でもnull/untiedを比較する。
rich modelのvalidation選択にPINDER-Sを使わず、最終PINDER-S値は探索的と明記する。

**明確化3: apoの順位は主に外的妥当性によるが、interaction contrastも定義できる。**
PINDER-Sで両方のapo monomerがあるのは93/250件である。この同じ93件について
baseline/trainedをholoとapoの両方で走らせ、

$$G_{\rm holo}=M({\rm trained,holo})-M({\rm baseline,holo}),\qquad
G_{\rm apo}=M({\rm trained,apo})-M({\rm baseline,apo})$$
$$\Delta_{\rm interaction}=G_{\rm apo}-G_{\rm holo}$$

をsystem-paired bootstrapまたはclustered binary modelで評価する。
apoの絶対低下はbaselineにも入るため、$\Delta_{\rm interaction}$は
「apoが全手法に難しい」ことと、学習表のgainがapoで特異的に減衰することを分ける。
さらにapo→holo interface RMSD/PINDER difficultyに対するgainの傾きと、
内部top-1500 recall対Max(Top 1)を併記し、候補回収の失敗とranking transferを分ける。

$G_{\rm apo}<G_{\rm holo}$だけでbound側鎖packingへの過学習を確定はできないが、
負のinteractionが構造変位とともに強まればその仮説を支持する。
apoは同じsystemなので独立confirmationではなく、順位2の根拠はまず実用的な外的妥当性、
次にこのmodel×monomer interaction診断である。

**上記のindex由来の数値を独立に再測した（2026-07-28）。** `pinder_xl & ~pinder_s` は
1705件、`split` は全件 `test`、`holo_R & holo_L` も1705件で全件揃う。
`cluster_id` の重複は、N220 seed 0 の `fit_ids + val_ids` 計275件（275 cluster）に対して
`xl \ s` と0件、`pinder_s` とも0件。`pinder_s` と `xl \ s` の間も0件。
PINDER-Sで `apo_R & apo_L` が両方あるのは 93 / 250 件（`predicted` は250件全て）。
以上はcodexの報告と一致する。

#### 5.14.24 +8.40 ppの内訳と、`ntop` 診断の決着（2026-07-28、GPU不要）

**問い。** §5.14.23 が挙げた検査のうち、追加GPUを使わずに既存の出力だけで
答えられるものを先に片付ける。(1) `Max(Top 1)` の +8.40 pp は
DockQ 0.23 の閾値をわずかに跨いだ少数例の集積か。(2) same-UniProt homodimerの
対称解をnegative扱いしている既知の制限が結果を作っていないか。
(3) 深い `--ntop` に伸びしろはあるか。

##### CAPRI遷移（`scripts/capri_transitions.py`、`pinder_s` / `holo` / `Max(Top 1)`）

`MethodMetrics` のper-decoy frameをPINDERと同じ規則（提出decoyのmax）で集約した。
`scripts/score_decoys_with_pinder.py` に `--metrics-out` を追加し、
leaderboardの隣にper-decoy CSVを残すようにした（集約値からは「どの系が動いたか」が
復元できないため）。

| クラス | published | trained_N220 | 獲得 | 喪失 |
| --- | --- | --- | --- | --- |
| acceptable | 66.40 % | 74.80 % | 22 | 1 |
| medium | 61.60 % | 69.20 % | 20 | 1 |
| high | 30.80 % | 32.40 % | 6 | 2 |

**閾値アーティファクトではない。** 新たにacceptableになった22件のうち、
DockQが 0.23 + 0.02 以内にあるものは **0件**。22件のDockQは
min 0.391 / median 0.675 / max 0.907 で、遷移の大半は
`Incorrect 0.005–0.066 → Medium/High 0.5–0.9` である。すなわち top-1 pose が
別のbasinへ移っており、境界をかすめて越えたのではない。
唯一の大きな後退は `2a8f__A1_P15569--2a8f__B1_P15569`（High 0.890 → Incorrect 0.062）で、
これはhomodimerである。

##### homodimer cohort

| cohort | n | published | trained_N220 | 差 |
| --- | --- | --- | --- | --- |
| same-UniProt homodimer | 71 | 54.93 % | 63.38 % | +8.45 pp |
| それ以外 | 179 | 70.95 % | 79.33 % | +8.38 pp |

**cohort差は無い。** 対称解をnegativeとして扱っている制限は実在するが、
+8.40 pp をこの制限が作っているわけではない。homodimerの絶対水準は
16 pp 低く、これは制限そのものの影響として別に残る。

##### ranking truncation（`scripts/rank_truncation_diag.py`）

既存の per-complex CSV（`first_hit_rank`、`recall`、`ceiling_dockq`）だけで測れる。

| | published | trained_N220 |
| --- | --- | --- |
| 回転格子で DockQ≥0.23 に到達可能 | 249 / 250 (99.6 %) | 249 / 250 |
| 上位1500に near-native が入った | 237 (94.8 %) | **242 (96.8 %)** |
| 到達可能だが上位1500に入らなかった | **13** | **8** |
| 取得済みだが rank 1 ではない | 72 | 58 |

first-hit rank の分布は published で `≤1: 165, ≤5: 200, ≤10: 207, ≤100: 228,
≤1500: 237`、trained で `≤1: 184, ≤5: 213, ≤10: 219, ≤100: 237, ≤1500: 242`。

**`ntop` を深くして救えるのは最大 13 / 250 = 5.2 %（学習後は 8 = 3.2 %）で、
しかも救ったうえで1位に来なければ `Max(Top 1)` は動かない。** 残りはすべて
ranking の問題である。§5.14.23 が優先度3に置いた `ntop`→`nside` 診断は、
追加のGPU実験を要さずここで決着し、優先度から落とす。
副次的な観察として、学習は**上位1500の保持率そのもの**も上げている
（237→242）。探索の枝刈りがスコアに依存するため、良いスコアは
「近native poseを1500件の中に残す」段階でも効いている。

**限界。** 以上はすべて `pinder_s` / `holo`、N=220 seed 0 の1 checkpointに対する
記述である。seed分散は§5.14.25で測る。

#### 5.14.25 実行中: 公式指標でのN scalingとapo（2026-07-28 23:27 投入）

**問い。** 公式PINDER指標での +8.40 pp は1 seed 1点であり、seed分散が無い。
また N=220 と N=500 の比較は固定TESTプール上の自前DockQで行ったもので
（§5.14.19、success@10 で null）、評価体系をPINDERへ移した後には測っていない。

**投入した条件（追加学習なし、既存checkpointの end-to-end 探索のみ）。**
`N220_seed1`, `N220_seed2`, `N500_seed0`, `N500_seed1`, `N500_seed2`
を GPU 0/3/6 に3分割で流し、`data/pinder_eval/trained_<cond>/` へ公式提出形式で
書き出してから harness で一括採点する（`scripts/run_official_nscaling.sh`）。
29 s/複合体、1条件あたり約41分。

続けて `scripts/overnight_queue.sh` が、(A) apo 93件の両条件、
(C) N=1000 採掘の残りシャード 1/6・2/6・4/6・5/6 を順に流す。
シャード0と3は採掘済み。`--mine-only` はシャード完走時にしかキャッシュを書かないため、
中断した4本は先頭からやり直しになる（208複合体 × 約150 s ≒ 8.7 h/シャード、
3 GPUで2波）。**朝までには終わらない。**

`apo` は新スクリプト `scripts/eval_search_apo.py` で行う。apo monomerは別々に解かれた
構造なので**共通frameが無く、内部DockQ・recall・rotation ceilingは定義できない**。
このスクリプトは探索とdecoy書き出しだけを行い、採点はすべてPINDERに任せる。
`check_geometry` は意図的に無効化する（apo入力は定義上「共通frameに無い2鎖」であり、
guardが検出すべき対象そのものではない）。1件で経路を確認済み:
`3k1i__D1_O25709--3k1i__A1_O25448` の apo decoyは
iRMS 9.18 Å / LRMS 23.1 Å / Fnat 0.0 / DockQ 0.048、`native_contacts` 54 が正しく解決され、
penalty行ではなく実測行として採点された（この系はbaseline・apoでは外れている）。

##### 結果1: 公式指標でのseed分散とN scaling（2026-07-29 04:30完了）

`pinder_s` / `holo`、公式harness、250/250。`Max(Top 1)` acceptable。

| 条件 | acceptable | medium | high | Max(Top 5) acc | 該当数 /250 |
| --- | --- | --- | --- | --- | --- |
| published | 66.4 % | 61.6 % | 30.8 % | 81.2 % | 166 |
| N=220 seed 0 | **74.8 %** | 69.2 % | 32.4 % | 86.0 % | 187 |
| N=220 seed 1 | **74.8 %** | 69.2 % | 32.8 % | 85.6 % | 187 |
| N=220 seed 2 | **74.8 %** | 69.2 % | 32.4 % | 85.6 % | 187 |
| N=500 seed 0 | 73.2 % | 67.6 % | 32.4 % | 86.0 % | 183 |
| N=500 seed 1 | 72.0 % | 67.2 % | 32.0 % | 86.0 % | 180 |
| N=500 seed 2 | 72.8 % | 67.2 % | 32.4 % | 86.0 % | 182 |

**seed分散は N=220 で厳密にゼロである。** 3 seedは acceptable 判定が
**1複合体も違わない**（seed 0 vs 1、0 vs 2、1 vs 2 いずれも相違0件）。
したがって +8.40 pp は seed noise ではない。published との対比は
exact McNemar で 22勝1敗、$p = 5.7\times10^{-6}$。
N=500 では seed 間で 1〜3 複合体が入れ替わるので、分散が無いのは N=220 固有の性質である。

**N=500 は N=220 より一貫して悪い。** seed対ごとの paired exact McNemar は
(4勝0敗, $p=0.125$)、(7勝0敗, $p=0.016$)、(5勝0敗, $p=0.063$)。
3 seed の多数決でも 187 対 182（5勝0敗, $p=0.063$）。
**16件の不一致すべてが N=220 側の勝ち**で、N=500 が勝った複合体は一つも無い。
個々の $p$ は 0.05 前後だが、方向は完全に一貫している。

これは §5.14.19 の固定プール上の結果（success@10 で null、AUCは N=500 が上）と
矛盾しない。公式のtop-1指標では、データを増やしても改善せず、むしろわずかに悪い。
**交絡の明示**: N=500 の fit集合は N=220 の上位集合ではないので、
「件数」と「どの複合体が入るか」は分離できていない。

##### 結果1b: N=1000 まで伸ばしても平坦（2026-07-29 23:45完了）

採掘6シャード（§5.14.25 の C）が 11:50 に揃い、3 seed を学習（各57.7〜64.3分、
`--min-steps 6818` = 1500 × 1000/220）、同じ公式手続きで採点した。

| N | acceptable @1（3 seed） | 平均 ± SD | 多数決の該当数 |
| --- | --- | --- | --- |
| 220 | 74.8 / 74.8 / 74.8 % | **74.80 ± 0.00** | 187 / 250 |
| 500 | 73.2 / 72.0 / 72.8 % | 72.67 ± 0.61 | 182 / 250 |
| 1000 | 72.8 / 72.8 / 73.2 % | 72.93 ± 0.23 | 182 / 250 |

paired exact McNemar（seed対ごと、左が勝ち/右が勝ち）:

| 対比 | seed 0 | seed 1 | seed 2 | 多数決 |
| --- | --- | --- | --- | --- |
| N=220 vs N=500 | 4-0 (p=0.125) | 7-0 (p=0.016) | 5-0 (p=0.063) | 5-0 (p=0.063) |
| N=220 vs N=1000 | 5-0 (p=0.063) | 5-0 (p=0.063) | 4-0 (p=0.125) | 5-0 (p=0.063) |
| N=500 vs N=1000 | 2-1 (p=1.0) | 1-3 (p=0.63) | 1-2 (p=1.0) | 1-1 (p=1.0) |

**6つのseed対すべてで N=220 が勝ち越し、負けた複合体は一つも無い**（計28件の不一致が
全部 N=220 側）。N=500 と N=1000 の間は完全に引き分け。曲線は
「220 → 500 で下がり、そこから平坦」である。N=1000 の seed 分散も小さい
（seed間の相違 0/1/1 複合体）。

**訂正（2026-07-30）: N=1000 は停止規則が他の2条件と違う。私の起動ミスである。**

当初この節には「`--min-steps` を N に比例させたので N=1000 は N=220 の 4.5 倍の
ステップを踏んでおり、長く最適化した効果とデータ量の効果が分離できない」と書いた。
**これは事実と逆であった。** `round0_metrics.json` の `steps_run` を確認すると:

| 条件 | steps_run | steps_budget | patience | epoch_passes | val_every |
| --- | --- | --- | --- | --- | --- |
| N=220 seed 0/1/2 | 1500 / 1500 / 1500 | 1500（完走） | 100000 | 1 | 50 |
| N=500 seed 0/1/2 | 3409 / 3409 / 3409 | 3409（完走） | 100000 | 1 | 50 |
| **N=1000 seed 0/1/2** | **1701 / 1837 / 1973** | 6818（**予算の1/4で停止**） | **8** | **100** | **68** |

N=1000 を投入したとき `--patience` と `--epoch-passes` を渡さなかったため、
既定値（`patience=8`, `epoch_passes=100`）が適用された。N=220 と N=500 は
`patience=100000` で early stopping が実質無効だったのに対し、
**N=1000 だけが「validation損失が8回連続で改善しなければ打ち切る」規則で学習している**。
`val_every` も 50 対 68 と違う（budget から導出されるため）。
これは §5.14.20 の voxel テーブル取り違えと同種の、実行者の起動ミスである。

したがって **§5.14.25 の N=1000 の行は N=220 / N=500 と対等な比較ではない**。
72.93 % という値そのものは公式手続きで測った実測値だが、
「データを増やすと悪くなる」の N=1000 側の根拠としては使えない。
N=500 が N=220 に負ける部分（両者は停止規則が一致している）は影響を受けない。

再実行を `--patience 100000 --epoch-passes 1 --min-steps 6818` で行う。
その結果が出るまで、N scaling の結論は
**「N=220 → N=500 で悪化する。N=1000 は未確定」**である。

##### 結果2: apo への転移（2026-07-29 05:35完了、93/93 docked、skip 0）

`apo_R` と `apo_L` が両方ある **93件**。holo側も同じ93件に制限して比較する
（250件の holo率と93件の apo率を並べると、apo構造の有無という選択と交絡する）。

| monomer | published | trained_N220 | gain | 勝/敗 |
| --- | --- | --- | --- | --- |
| holo（93件に限定） | 49.46 % | 59.14 % | **+9.68 pp** | 10勝1敗 |
| apo | 22.58 % | 26.88 % | **+4.30 pp** | 4勝0敗 |

$$\Delta_\text{int}=G_\text{apo}-G_\text{holo}=-5.38\ \text{pp},\quad
\text{95\% CI}\ [-11.83,\ +1.08],\quad P(\Delta_\text{int}<0)=0.934$$

（複合体を再標本化した paired bootstrap、10,000回。`scripts/apo_holo_interaction.py`）

**読み。** 学習した表は apo でも改善する（+4.30 pp、4勝0敗、負けなし）。
利得は holo の半分以下に減衰しているように見えるが、**この減衰は統計的に確立していない**
（CIが0を跨ぐ、不一致件数が holo 11件 / apo 4件と少ない）。
すなわち「bound側鎖のpackingへの過学習」も「apoは誰にとっても難しいだけ」も、
このデータでは棄却できない。判定には93件では検出力が足りず、
`pinder_xl` のapo（342件）まで広げる必要がある。

**apoの絶対水準は holo の半分以下である**（22.58 % 対 49.46 %、DockQ中央値 0.03 対 …）。
またこの93件は holo でも平均より難しい（49.46 % 対 全250件の 66.4 %）。

`trained_N220_seed1/2` と `trained_N500_*` の apo 行が 0.00 なのは、
apo を published と trained_N220 の2条件でしか走らせていないためで、
PINDER の欠損penalty（DockQ 0）で埋まった行である。結果ではない。

#### 5.14.26 margin項は hard negative 項ではなく広域正則化だった（2026-07-28）

**問い（利用者提起）。** 損失としてrank lossに再考の余地はあるか。

**まず事実誤認の訂正（codexによる、検算済み）。** 「`loss_basin` は正例集合を
一様に持ち上げるので top-1 に無関心」という私の読みは**誤り**である。

$$L_\text{basin}=-\log\frac{\sum_P e^{s/T}}{\sum_{P\cup N} e^{s/T}}
=\log\Bigl(1+\tfrac{N}{P}\Bigr)
=\operatorname{softplus}\bigl[\mathrm{LSE}(s_N/T)-\mathrm{LSE}(s_P/T)\bigr]$$

すなわち basin は既に「softmax重み付きの最良正例 vs softmax重み付きの最良負例」の
比較であり、$T=0.5$ では LSE は best positive が支配する。基本設計は
`Max(Top 1)` と整合している。

##### 測定（`scripts/rank_loss_headroom.py`、固定TESTプール、search由来pose、N220 seed 0学習後、236複合体）

スコアは複合体ごとに中心化し自身のSDで割る（学習時の `normalized_scores` と同じ。
`margin = 1.0` はこの単位）。

| | 値 |
| --- | --- |
| top-1 が正例 | 183 (77.5 %) |
| top-1 が負例 | 53 (22.5 %) |
| 外した53件の (best negative − best positive) | p10 0.36 / p25 0.63 / **median 1.29** / p75 2.48 / p90 4.06 SD |
| うち 1.0 SD 以内 | 20件（外れの37.7 %） |
| best positive を上回る負例の個数 | median 3 / p75 14 / p90 76 |
| **(best positive − worst positive)** | p25 4.49 / **median 7.10** / p75 9.79 / p90 12.52 SD |

`loss_margin_hard_negatives` のアンカーは `min(positive)` であり、それは
best positive の中央値 **7.10 SD 下**にある。`Max(Top 1)` が要求するのは
`max_pos > max_neg` だけなので、この項はおよそ 8 SD 分だけ過剰に厳しい条件を課す。

##### hinge が実際に触っている負例の数

| hinge が有効な負例の割合（複合体あたり負例は中央値1494件） | `min_pos` アンカー（現行） | `max_pos` アンカー |
| --- | --- | --- |
| p25 | 34.47 % | 0.00 % |
| median | **74.95 %** | **0.00 %** |
| p75 | 97.52 % | 0.13 % |

有効件数の比は中央値 **740倍**。`hinge.mean()` は全負例で割るため、現行の項は
約1120件の負例に $1/1494$ ずつ勾配を配っている。**名前に反して hard negative 項では
なく、負例全体を押し下げる広域正則化として働いている。** 勝敗を決める上位数件
（best positive を上回るのは中央値3件）の勾配は約1000倍に希釈される。

**これは §5.14.21 の mining null の機構説明にもなる。** 損失が既に負例の3/4を
押し下げている以上、新しく採掘した負例を追加しても損失の景色はほとんど変わらない。
§5.14.21 では「新規負例はbandに入るが最難関より1.87 SD下」と記述したが、
より根本的には**bandが広すぎた**ということである。

##### 次の実験（codexとの議論、2026-07-28）

`max_pos` へのhard切り替えは採らない。中央値で有効負例が0 %になり、53件の失敗以外で
勾配が消えるうえ、勝者1 poseにしか正例勾配が流れずargmaxの切替とDockQノイズで
高分散になる。代わりに soft top-tail:

$$p_\tau=\tau_p\log\!\sum_{i\in P}e^{s_i/\tau_p},\quad
n_{\tau,k}=\tau_n\log\operatorname{meanexp}\bigl(\mathrm{TopK}_{j\in N}(s_j)/\tau_n\bigr),\quad
L_\text{top}=\operatorname{softplus}\bigl((m+n_{\tau,k}-p_\tau)/\tau_h\bigr)$$

**3条件のablation**（同一fit・同一初期値・同一minibatch列、3 seed）:
(1) 現行 basin + min-anchor hinge、(2) basin のみ（min-anchor の機構ablation）、
(3) basin + soft top-tail。

一次指標は凍結した開発集合での end-to-end `Max(Top 1)` acceptable。機構指標として
top-1500 acceptable recall、recall条件付きTop 1、Medium/High、Top 1 DockQ を併記し、
$P(\text{Top1 hit})=P(\text{retrieved})\cdot P(\text{Top1}\mid\text{retrieved})$ に
分解して、min-anchor が持つかもしれない「正例basin全体を持ち上げて上位1500への保持を
広げる」副次効果（§5.14.24 で観測した 237→242）を検証する。
固定プールは勾配診断と高速screeningのみに使い、選択の一次根拠にしない。
seedは独立標本として検定せず、複合体を再標本化して3 seed平均のpaired差をbootstrapする。

`loss_listnet_dockq`（graded listwise、実装済みだが未使用）は Top 1 acceptable とは
直接整合しないが、正例ゼロの72件に勾配を与えられる唯一の候補なので、
subthreshold DockQ に分散がある複合体限定の補助損失としてのみ検討する。
ただし現実装は正例ゼロ時に一様targetでスコアを平坦化するため、そのままでは使えない。

**容量診断との順序。** 損失を先に直す。「容量がbinding」は損失に条件付けられた問いで
あり、目的関数が配備指標とずれたまま shell 拡張すると追加容量を誤った分離に使う。
最終的には（現行/新損失）×（144/shell拡張）の2×2が、objective不足とcapacity不足を
最も明確に分離する。

##### 実装（2026-07-29、未実行）

`zdock.train.loss_top_tail` と `run_pinder_scaling.py --loss-neg
{minanchor,none,toptail}` を追加した。既定は `minanchor`、すなわち
2026-07-28 までの全結果を出したレシピのままで、明示的に指定しない限り挙動は変わらない。
`--toptail-k`（既定32）、`--tau-pos` / `--tau-neg`（既定0.5）、`--tau-hinge`（既定1.0）。
`resume_identity()` のキーにも加えたので、損失を変えたまま resume することはできない。

**実装中に見つけた設計上の誤り。** 正例側を `tau*logsumexp` にすると、それは真のmaxを
最大 $\tau\log|P|$ 上回る。$\tau=0.5$、正例200件なら 2.65 SD の下駄になり、
**目的関数が「top-1を当てたこと」ではなく「near-native basinが大きいこと」を報酬にする**。
両側とも mean-exp（$\tau\log\operatorname{mean}\exp$、値域 $[\text{mean},\max]$）に
変更した。単体テスト6件を追加し、うち1件はこの性質そのもの
（正例を2件から61件へ増やしても損失が0.05以内）を検査する。
既存テストは `tests/test_scaling_streaming.py` と `tests/test_phase7_train.py` の
43件が通る。

**この節の最大の限界（codexの指摘）。** この損失仮説は **PINDER-S TEST の53失敗例を
見て作られた**。以後TESTで選択しなくても、変更後のPINDER-S値は未使用標本での確認では
ない。未使用 `pinder_train` から fit と interface-cluster disjoint な開発集合を確保し、
そこで設計を凍結したうえで、独立confirmationは XL-minus-S 等で行う必要がある。
現行の55件validationは Top 1 差の解像度が不足している。

#### 5.14.27 データが効かない理由は「量」ではなく「誰も勾配を出していない」（2026-07-29）

**問い（利用者提起）。** データ量を活かせていないのなら、損失と採掘の両方を
作り直す必要があるのではないか。

**測定**（`scripts/gradient_share.py`、採掘プールのみを使うのでGPU不要。
学習後パラメータでプールを採点し、複合体ごとの目的関数の IFACE 勾配ノルムを取る）。

| | N=220（282複合体） | N=1000（1250複合体） |
| --- | --- | --- |
| 正例が1つも無い（勾配が恒等的に0） | 100 (35.5 %) | **461 (36.9 %)** |
| 勾配ノルムが最大の1 %未満 | 102 | 477 |
| 勾配の50 %を出す複合体 | 50 (17.7 %) | 215 (17.2 %) |
| 勾配の80 %を出す複合体 | 97 (34.4 %) | 417 (33.4 %) |

**勾配の集中度はスケール不変である。** コーパスを 282 → 1250 に増やしても、
死んでいる複合体の割合（約37 %）も、勾配の8割を出す複合体の割合（約34 %）も変わらない。
増えるのは同じ分布の同じ比率のコピーである。

**訂正（2026-08-01、利用者の指摘）。** 当初ここに
「これが『データ量がレバーでない』ことの直接の測定である」と書いた。**論理が飛んでいる。**
比率が不変でも実数は増える。勾配を出す複合体は **182 → 789 と 4.3 倍**になっており、
「比率が同じ」から「足しても信号が増えない」は導けない。原文は研究史として残す。

この測定が正しく示すのは**構成の不変性**だけである。すなわちコーパスを増やしても
「情報量の多い複合体が相対的に増える」ことはなく、同じ分布のコピーが増える。
死んでいる37 %が `--loss-prov search` の産物であること（下記）も、
データ量ではなく損失設定の問題を指している。

「データ量はレバーでない」の根拠は別の3つである。

1. **学習曲線**（下記）。これは**勾配を出す複合体だけ**を数えて引いており、
   400件を超えると効果が消える。効くデータの実数が増えても頭打ちになる、
   という利用者の反論に直接答える形になっている。
2. **seed分散がゼロ**（§5.14.25）。N=220 の3 seedが250件中1件も判定が違わない。
   推定に揺らぎが残っていないので、データを足して減らすべき分散がない。
3. **N scaling の実測**（§5.14.25）。220 / 500 / 1000 を公式指標で比較し、
   6組のseed対すべてで220が勝ち越し、負けた複合体はゼロ。推論ではなく観測である。

##### 死んでいる461件は `--loss-prov search` が作っている

`--prov all` にすると**正例ゼロの複合体は0件**になる。全1250件が列挙由来の
near-native pose を持っている。つまりデータが無いのではなく、
**現行レシピが意図的に3分の1を損失から締め出している**。

§5.14.2 は `--loss-prov search` が決定的だと測定した。その理由が今回わかった。

| 正例が探索で見つかっている789件 | search のみ | 全pose |
| --- | --- | --- |
| min(positive)（hingeのアンカー） | median **+0.14** | median **−8.23** |
| max(positive)（top-1の決定者） | median +1.16 | median +1.17 |

**列挙poseを入れると min アンカーが中央値 8.32 SD 沈む**（p90 10.90 SD）のに対し、
max は 95.9 % の複合体で**まったく変わらない**。すなわち `--loss-prov all` が
悪かったのは列挙poseそのものではなく、**min アンカーがそれを掴むから**である可能性が高い。
現行 hinge は「全negativeを、決定者より8 SD下のposeよりさらに1 SD下へ」と要求することになる。

soft-max アンカー（§5.14.26 の `--loss-neg toptail`）なら、低スコアの列挙poseは
指数的に抑圧されて無視される。**アンカーを直すことは、同時に締め出していた3分の1を
取り戻す前提条件でもある。**

461件について、最良の列挙positiveは **461件すべてで最良negativeより下**
（中央値 −2.00 SD、p10 −3.82）。したがってこれらは飽和した無害なデータではなく、
すべて「まだ解けていない」勾配を出す複合体である。

##### 採掘についての含意

**hard-negative 採掘はこの461件を原理的に助けられない。** 足りないのは negative ではなく
positive だからである。§5.14.21 の mining null は、(a) 損失が既に negative の75 %を
押し下げている（§5.14.26）ことに加えて、(b) 最も助けが要る3分の1に対しては
採掘している対象そのものが的外れ、という二重の理由で説明できる。
この461件に必要なのは、列挙positiveを損失へ入れるか（上記）、
探索がそれらを取得できるだけの回転格子の細かさである。

##### 新しい損失は「より多くのデータ」を使わない（重要な但し書き）

同じ測定を `--loss-neg toptail` で行うと、勾配はむしろ**より集中する**
（勾配80 %を出すのは 33.4 % → 29.8 %、最大の1 %未満は 477 → 590件）。
top-1 に合わせるとは失敗している複合体に集中することなので、これは設計どおりだが、
**「損失を直せばデータが効くようになる」わけではない**。
量を効かせたいなら、閾値0.23より下に勾配を与える graded relevance
（`loss_listnet_dockq` の系統）が別途必要になる。

##### 次の実験（改訂）

§5.14.26 の3条件ではなく、**2×2** にする。

| | `--loss-prov search` | `--loss-prov all` |
| --- | --- | --- |
| `--loss-neg minanchor` | 現行（既知） | §5.14.2 で悪かった条件 |
| `--loss-neg toptail` | アンカー単独の効果 | アンカー修正が3分の1を取り戻すか |

事前予測: `toptail × all` が最も良い。理由は、アンカーの修正が
`--loss-prov all` の既知の失敗機構（min アンカーが列挙poseを掴む）を取り除き、
かつ 461 件の未解決複合体を損失へ戻すからである。この予測が外れた場合、
「列挙positiveは探索が返さないpose を持ち上げるので end-to-end では無意味」という
§5.14.2 の元の解釈が支持される。`--loss-neg none × prov search` も機構対照として残す。

すべて3 seed、選択は凍結した validation のみ、TEST は最後に一度だけ。
`loss_neg` と `loss_prov` は `resume_identity()` のキーに入っているので、
条件を跨いだ resume は拒否される。

#### 5.14.29 144パラメータの天井を線形計画で直接計算する（2026-07-30）

**問い。** 残り25 %の失敗は「訓練の仕方が下手」なのか「144個では原理的に足りない」のか。
損失を作り替える前に、その賭けの天井を知りたい。

**構造上の観察。** 現行レシピは `--freeze-psc` で $\alpha=1.0$, $\rho=3.5$ が固定され
（全実行の `round0_metrics.json` で実測）、学習されるのは144次元の $e$ だけである。
`score.py:491` は `IFACE_SIGN * e.view(12,12).T`、`IFACE_PAIR_OFFSET = 0.0`、
`IFACE_SIGN = -1.0` なので、キャッシュ済み特徴からのスコアは $e$ について**アフィン**:

$$s_p(e)=c_p+\langle A_p,e\rangle,\quad
c_p=\alpha\,\mathrm{sc}_0-\mathbf w\cdot\mathrm{sc}_{1:4}+\beta\,\mathrm{elec},\quad
A_p=-\operatorname{vec}(T_p^\top)$$

実装で確認したところ再構成誤差は **0**（`scripts/capacity_ceiling.py` の起動時チェック）。
したがって「正例 $p$ が全negativeに勝つ」は $e$ の線形不等式系であり、
**SGDを使わずに線形計画で解ける**。

**codexによる論法の修正（2026-07-30）。** 当初私は「個別LPが高ければ損失が取り逃している」
と読んだが、これは誤りである。個別LPは複合体ごとに*専用の* $e_c$ を許すので、
共有テーブルの達成可能値がそこまで届く保証はない（複合体Aが上げろと要求する成分を
複合体Bが下げろと要求しうる）。**低い値は決定的、高い値は決定的でない。**
また上限として成立させるには (a) 全正例を列挙する、(b) $e$ を有界領域に制限する、
(c) 厳密不等号を事前規定の $\epsilon$ で置き換える、の3条件が要る。すべて実装した。

##### 結果（N=1000 の validation プール、探索由来pose のみ、$\epsilon=1$ raw単位）

母数250件のうち、プールに acceptable pose があるのは **152件**（retrieval 60.8 %）。

| | 勝てた数 / 152 | reachable比 |
| --- | --- | --- |
| 学習済み表（SGD、fitで学習しvalで採点） | 65 | 42.8 % |
| **共有表LPの大域最適、fitで解きvalで採点**（半径∞） | **71** | **46.7 %** |
| 同、半径 1.0 / 0.316 | 69 / 67 | 45.4 % / 44.1 % |
| 共有表LP、valで解きvalで採点（in-sample） | 86 | 56.6 % |
| 個別上限（複合体ごとに専用の表）、半径 0.316 / 1.0 / ∞ | 136 / 152 / 152 | 89.5 % / 100 % / 100 % |

半径は $\lVert e-e_0\rVert_\infty\le r$、中心は公開表。学習済み表は
$\lVert\Delta e\rVert_2=1.370$、$\lVert\Delta e\rVert_\infty=0.316$ 動いている。

##### 解釈

1. **容量は制約ではない。** 個別には reachable の 89.5 %（学習済みと同じ移動幅の範囲内）
   から 100 %（半径1.0以上）が勝てる。144個の数字の表現力が壁ではない。
2. **損失と最適化の取り分は小さい。** 同じfitデータで凸代理の**大域最適**を取っても、
   validation では 42.8 % → 46.7 %。**6複合体、約4 pp** である。
   §5.14.26–28 で作っている損失の改良が取れる上限は、この見積りではその程度になる。
3. **支配的な制約は汎化である。** 同一のLPが、当てはめた集合そのものでは 56.6 %、
   別集合では 46.7 %。**約10 pp が汎化で失われている。**

**含意（未検証の予想）。** 汎化が壁であるなら、関数形の拡張（§5.14.23 で議論した
2シェル分割は144→288パラメータ）は**悪化させる方向**に働きうる。
これは §5.14.11 の実験1（full 144 が additive 23 を +3.39 pp 上回る）と緊張関係にあり、
「パラメータを増やすと良くなる領域」と「増やすと汎化が落ちる領域」の境界が
144の周辺にある可能性がある。同じLPを add(23) / sym(12) 部分空間で解けば、
訓練を回さずにこの境界を測れる。

##### 訂正: 上の「解釈」は誤りである（codexの指摘、2026-07-30）

上の3項目、および私が立てた「100 % → 56.6 % → 46.7 % → 42.8 %」という予算分解は
**成立しない**。原文は研究史として残す。

1. **56.6 % は「共有表の限界」ではない。** これは固定アンカー（$e_0$ の下で最高スコアの
   正例）＋総ヒンジ最小化という**1つの凸代理**が達成した値にすぎず、真の共有表最大値は
   $56.6\,\%\le H^*_\text{shared,val}\le 100\,\%$ のどこかである。
   したがって「43 pp の誤特定」は推定値ではなく**その最大可能値**である。
2. **`misspecification` は誤用。** 個別解は複合体IDとの交互作用を許す $144\times C$
   パラメータのモデルであり、共有表と同一の関数族ではない。正しい呼称は
   個別上限が *per-complex oracle separability*、個別と共有の差が
   *parameter-sharing / cross-complex heterogeneity gap*。
   `misspecification` は、独立集合で共有表の**真の**最大値も低いと示して初めて使える。
3. **「損失の取り分は最大4 pp」も誤り。** $46.7-42.8$ は、このLPレシピが示した
   **最低限**の達成可能改善である。別のアンカーや0/1直接最適化が 46.7 % を超えうる。
4. **学習曲線の射程を過大に述べた。** 曲線は正例を持つ637件だけで引いており、
   §5.14.28 の shaping 損失が狙う461件を除外している。よって
   「データ量がレバーでないことが損失非依存に確立した」とは言えない。
   400→637 の差は実質1複合体で、3反復では飽和の証拠として弱い
   （interface-cluster blocked K-fold が要る）。
5. **10 pp を「汎化」と呼んだのも誤り。** 比べていたのは
   $H_V(e_V)-H_V(e_F)$、すなわち「validationのラベルへ直接適応する利益」であって、
   通常の汎化ギャップ $H_F(e_F)-H_V(e_F)$ ではない。
6. **実装の欠陥。** `shared_table()` は制約生成が `max_rounds` に達しても
   `added > 0` のまま成功として返していた。大域最適である保証が無かった。
   収束フラグを返すよう修正し、本節の全結果で `added == 0` に収束したことを確認した。
7. 半径 0.316 → ∞ の改善は約4複合体であり、しかも LP の $L_\infty$ 箱制約と
   SGD の $L_2$ prior は別物である。「公開表への prior は効いていない」とは言えない。

##### cross-swap（2×2、正しい汎化ギャップ、2026-07-30）

codexの要求に従い、4つのテーブルを fit 637件と val 152件の**両方**で採点した。

| テーブル | $H_\text{fit}$ | $H_\text{val}$ | $H_\text{fit}-H_\text{val}$ |
| --- | --- | --- | --- |
| 公開表 $e_0$ | 230/637 = 36.1 % | 59/152 = 38.8 % | −2.7 pp |
| **SGD（fitで学習）** | **277/637 = 43.5 %** | 65/152 = 42.8 % | **+0.7 pp** |
| LP（fitで解く、収束確認済み） | **344/637 = 54.0 %** | 71/152 = 46.7 % | **+7.3 pp** |
| LP（valで解く、収束確認済み） | 256/637 = 40.2 % | 86/152 = 56.6 % | −16.4 pp |

**読み。**

1. **我々の学習は過学習していない。汎化ギャップは +0.7 pp しかない。**
   代わりに**過少適合している**: 自分が学習した637件ですら 43.5 % で、
   同じデータに対しLPは 54.0 % を達成する。**訓練データ上で 10.5 pp を取り逃している。**
   これは損失と最適化の問題であって、データ量でも汎化でもない。
2. 真の汎化ギャップは LP 側にあり **+7.3 pp**。訓練データ上の 10.5 pp のうち
   転移するのは約 4 pp である。残りは正則化で取り返せる可能性がある部分。
3. **fit と val は、固定のテーブルから見れば難易度がほぼ同じ**（公開表で 36.1 % 対
   38.8 %）。両者とも `pinder_train` からの決定的な選択だが、大きな分布ずれは無い。

したがって §5.14.26–28 の損失の作り替えは、**測定可能な過少適合**を対象にしている点で
正当化される。ただし取り分の見積りは「LPレシピが示した下界として約4 pp」であり、
それ以上でも以下でもあり得る。

##### 限界

- LPは0/1のカウントではなく**ヒンジ代理**を最小化している。複合体ごとに目標正例を
  1つ固定してもいる（現在のスコアが最も高い正例＝持ち上げるのが最も安いもの）。
  どちらも共有表の達成可能値を**過小評価する**方向で、真の最適はこれより高い。
  正確な値は二値変数 $z_{cp}$ を持つ混合整数計画が要る。
- 固定プール上の再ランキングであり、配備系はスコア依存の枝刈りを含む探索である
  （学習で上位1500への保持が 237→242 件に増えたことが実測されている）。
  $e$ を変えれば候補集合自体が変わるので、この天井は探索を含む系の天井ではない。
- 正例の定義は DockQ ≥ 0.23 の二値。
- validation プールは retrieval 60.8 % で、TEST（95–97 %）とは難易度の水準が違う。
  絶対値を公式指標と並べてはならない。
- TEST プールでは実行していない。設計判断に使うと汚染になるため。

#### 5.14.30 checkpoint選択のバグと、凸最適化で解いたテーブル（2026-07-31）

**損失ablationの結果（§5.14.26–28、N=1000、4 arm × 3 seed、全12本完走）。**
固定プールのvalidation top-1（250件、うちプールにacceptable poseがあるのは152件）:

| arm | seed 0 / 1 / 2 | 平均 |
| --- | --- | --- |
| 公開表 | — | 23.60 % |
| `matched`（現行レシピ） | 26.00 / 25.20 / 25.60 | 25.60 % |
| `toptail` | 26.40 / 25.20 / 25.20 | 25.60 % |
| `shape` | 25.60 / 25.60 / 25.20 | 25.47 % |
| `both` | 26.40 / 25.60 / 26.80 | **26.27 %** |

差は250件中1.7件で seed 分散（SD 0.6 pp）に埋もれる。**実質 null。**
codexの指摘: `shape` が固定プールで null なのは予想どおりである。正例ゼロの98件を
**固定プール上で正解にすることは原理的に不可能**で、shaping の狙いは再探索時の
候補取得改善だからである。固定プールで測ったこと自体が設計ミスであった。

##### バグ: checkpoint選択がpriorを含んでいた

`_val_loss` は `mean_objective` を呼び、それは `total + lambda_prior * prior` を返す
（`run_pinder_scaling.py:742`）。prior は validation 複合体に**一切依存しない**ので、
選択基準が「公開表にどれだけ近いか」を測っていたことになる。

大きさが問題である。学習済み表は $\lVert\Delta e\rVert_2=1.37$ で prior は 0.188。
一方、下記の凸最適化が到達する位置は $6.61$ で prior は **4.37**、
実測の `val_loss` 約 **2.95**（データ項全体）より大きい。
**予測が良いテーブルほど選択基準で不利になる**構造だった。

修正: `mean_objective(..., with_prior=False)` を追加し、`_val_loss` の既定を
データ項のみに変更した。`--val-loss objective` で 2026-07-30 までの全実行を
厳密に再現できる。`resume_identity()` にもキーを追加。単体テストを1件追加。

##### 凸最適化で直接解く

スコアは144次元 $e$ についてアフィンなので（§5.14.29）、学習は凸問題として直接解ける:

$$\min_e\ \frac{1}{|F|}\sum_c\max\Bigl(0,\ \max_n\bigl[\epsilon+s_n(e)-s_{p(c)}(e)\bigr]\Bigr)
+\frac{\lambda}{2}\lVert e-e_0\rVert^2$$

$p(c)$ は $e_0$ の下で最高スコアの正例（持ち上げるのが最も安いもの）。
L-BFGS で解く。**学習率・ステップ予算・早期終了・checkpoint巻き戻しが一切介在しない。**
$\lambda$ は fit 内の interface-cluster blocked 5-fold CV で選び、
固定validationは選択後に**一度だけ**触る（`scripts/qp_path.py`）。

| $\lambda$ | blocked CV top-1 | $\lVert\Delta e\rVert_2$ |
| --- | --- | --- |
| 1e-4 | 49.61 % | 6.595 |
| **3e-4（CVで選択）** | **49.76 %** | 6.608 |
| 1e-2 | 49.76 % | 6.562 |
| 1e-1 | 49.61 % | 5.895 |
| 1 | 49.14 % | 4.304 |

**$\lambda$ path は平坦である**（4桁動かして 0.6 pp）。正則化の強さはレバーではない。
$\lambda=1$（$\lVert\Delta e\rVert=4.3$）でも CV 49.14 % で SGD の 43.49 % を大きく上回るので、
**prior の強さだけが原因ではなく、目的関数の形が効いている**。
QPは「アンカー正例 対 最大違反negative」の**生スコア**ヒンジ、
SGDは「basin ＋ 全1494 negativeの**平均**ヒンジ ＋ 複合体ごとSD正規化」である。

固定validationでの一度きりの確認（152件）:
公開表 59 (38.82 %) → SGD 65 (42.76 %) → **QP 70 (46.05 %)**。

##### 公式PINDER評価（`pinder_s` / `holo`、250/250）

QPの表をそのまま配備し（144次元ベクトルを差し替えるだけ）、end-to-end探索をやり直した。

| 手法 | acceptable @1 | medium @1 | high @1 | acc @5 | DockQ中央値 @1 |
| --- | --- | --- | --- | --- | --- |
| published | 66.40 % | 61.60 % | 30.80 % | 81.20 % | 0.71 |
| trained_N220（従来最良） | 74.80 % | 69.20 % | 32.40 % | 86.00 % | 0.73 |
| **trained_QP** | **76.80 %** | **72.40 %** | **38.00 %** | **87.60 %** | **0.76** |

paired exact McNemar:

| 対比 | acceptable | medium | high |
| --- | --- | --- | --- |
| QP vs published | 42勝16敗, $p=8.6\times10^{-4}$ | 40勝13敗, $p=2.7\times10^{-4}$ | 22勝4敗, $p=5.3\times10^{-4}$ |
| QP vs trained_N220 | 23勝18敗, **$p=0.53$** | 22勝14敗, $p=0.24$ | **16勝2敗, $p=1.3\times10^{-3}$** |

**従来最良に対する acceptable の +2.00 pp は有意ではない**（23勝18敗）。
有意なのは **high の +5.60 pp**（16勝2敗）であり、DockQ中央値も 0.73 → 0.76 に上がる。
すなわちQP表の効果は「当たる件数を増やす」より「**上位poseの質を上げる**」方向である。
これまで high はほとんど動かなかった（公開表→trained_N220 で +1.60 pp）ので、
質の面では初めての明確な改善である。

またQP表は公開表に対して**16件を落としている**（trained_N220 は1件しか落とさなかった）。
$\lVert\Delta e\rVert$ が 1.37 から 6.61 へ 4.8 倍動いた結果であり、
入れ替わりの大きさとして記録しておく。

内部指標（自前DockQ、TEST 250件）: 公開表 succ@1 66.0 % → trained_N220 73.6 %
→ N1000_seed0 72.4 % → **QP 75.6 %**。`recall`（上位1500への保持）だけは
96.8 % → 96.0 % とわずかに下がる。順位づけの改善が保持の低下を上回っている。

##### 限界

- QPは複合体ごとに**アンカー正例を1つ固定**し、0/1ではなくヒンジを最小化している。
  したがってこの表は「達成可能な一例」であって最適ではない。
- 採掘ループ（round > 0）や `--loss-shape` のような非凸項とは、この形のままでは組めない。
- `pinder_s` は設計判断に反復使用しており、この +10.40 pp も探索的な値である。
  QP表自体は fit 集合のみで解き $\lambda$ も fit 内 CV で選んだが、
  「凸最適化を試す」という判断自体が過去のTEST観察に基づいている。
- 落とした16件の内訳（どの系が、なぜ）は未分析。

#### 5.14.31 品質の天井は回転格子にあった（2026-08-01）

**問い（利用者提起）。** パラメータを 12×12 より増やせば性能は上がるのか。
データ量が効くようになるのか。そして `--mine-ntop 1500` という数字に根拠はあるのか。

##### まず自前DockQの系統誤差（他の結論にも影響する）

同一の1位poseについて、自前DockQと公式 `BiotiteDockQ` を250件で比較した。

| | 自前 | 公式 |
| --- | --- | --- |
| DockQ 中央値 | 0.592 | 0.757 |
| acceptable (≥0.23) | 75.6 % | 76.8 % |
| medium (≥0.49) | 63.2 % | 74.4 % |
| **high (≥0.80)** | **6.0 %** | **39.6 %** |

Pearson 相関 **0.9923** と形はよく一致するが、**系統的に 0.117 低い**。
閾値0.23付近では影響がない（75.6 % 対 76.8 %）が、
**高品質域では全く違う数字になる**。したがって自前DockQで品質（medium/high、
平均DockQ）を論じた箇所はすべて無効である。acceptable 閾値付近の主張は影響を受けない。

##### 回転格子の上限（公式指標）

探索は Hopf nside=3 の **1944通りの向き**しか試さない。正解の向きは格子点に乗らず、
最も近い格子点まで**中央値 9.5°**（最大14.4°）ある。
その残差が品質の上限を作る。

「正解に最も近い格子回転＋正解に最も合う並進」で作ったposeを250件分書き出し、
公式harnessに採点させた（`scripts/export_rotation_ceiling.py`）。

| 公式指標 | 格子の上限 | 実測（QP表） | 残り |
| --- | --- | --- | --- |
| acceptable | 100.00 % | 76.80 % | 23.2 pp |
| medium | 99.60 % | 72.40 % | 27.2 pp |
| **high** | **39.60 %** | **38.00 %** | **1.6 pp** |
| DockQ 中央値 | 0.78 | 0.76 | 0.02 |

**high は格子で頭打ちである。** 上限39.6 %に対し38.0 %、すなわち**到達可能分の96 %を
既に取っている**。公開表30.8 % → 学習32.4 % → 凸最適化38.0 % と high が動きにくかった
理由がこれで説明される。**スコア関数をどう改良しても high はこれ以上上がらない。**
一方 acceptable と medium には20 pp以上残っており、こちらは純粋に順位づけの問題である。

**§5.14.24 の訂正。** そこで「`ntop`/`nside` の診断は不要」と結論したが、
あれは acceptable 閾値での到達可能性（100 %）の話であった。
**品質については格子が支配的**であることを見落としていた。

##### 角度と品質の関係（公式指標）

正解poseを、ランダムな軸まわりに指定角度だけ回して書き出し、採点した
（`scripts/angle_quality_curve.py`。0°で DockQ 1.000 となり検算が通る）。

| ずれ | 0° | 2° | 4° | 6° | 8° | 10° | 12° |
| --- | --- | --- | --- | --- | --- | --- | --- |
| acceptable | 100 | 100 | 100 | 100 | 100 | 100 | 99.6 |
| medium | 100 | 100 | 100 | 99.6 | 99.6 | 98.4 | 96.0 |
| **high** | **100** | **100** | **98.8** | **81.2** | **48.4** | **18.0** | **4.8** |
| DockQ中央値 | 1.000 | 0.970 | 0.930 | 0.870 | 0.800 | 0.740 | 0.680 |

**high は 4°〜8° で崖のように落ち、90 %を割るのは約 5.0°**（95 %は約4.4°）。
acceptable は12°でも99.6 %で、角度にほぼ無関係である。

##### 格子を細かくする場合の費用

| nside | 回転数 | 最近接角の中央値 | 上の曲線からの予測high | 探索コスト |
| --- | --- | --- | --- | --- |
| **3（現行）** | 1,944 | 9.7° | 33.7 % | 1.0× |
| 4 | 4,608 | 7.2° | 61.3 % | 2.4× |
| 5 | 9,000 | 5.8° | 79.7 % | 4.6× |
| **6** | **15,552** | **4.9°** | **89.9 %** | **8.0×** |
| 7 | 24,696 | 4.1° | 95.0 % | 12.7× |
| 8 | 36,864 | 3.6° | 97.6 % | 19.0× |

回転数は角度の3乗で効く。**high ≥ 90 % には nside=6、探索8倍**が必要。
この予測モデルは nside=3 で 33.7 % と出すが実測の上限は 39.6 % なので、
約6 pp 控えめである（ランダム軸を仮定しているため）。補正しても結論は変わらない。

##### 正解付近のスコア地形

格子を細かくする代わりに、返ってきたposeを**格子の外へ動かして磨く**（局所精密化）道がある。
ただしスコアの極大が正解から遠ければ、磨くほど悪化する。そこで先に地形を測った
（`scripts/score_landscape_near_native.py`。正解の周りに12°・2.5 Å以内で400点を撒き、
最高スコアの点がどこにあるかを見る）。

| | 学習後の表（QP） | 公開表 |
| --- | --- | --- |
| 正解poseが最高スコア | **0 / 55** | 0 / 51 |
| 最高スコア点の回転ずれ | **中央値 1.0°**（p25 0.6、p75 2.6） | 中央値 1.4° |
| 同 並進ずれ | **0.84 Å** | 1.02 Å |
| そこでのDockQ（自前、正解=1.0） | **0.890** | 0.849 |
| 正解より悪化 | 55/55 | 51/51 |

**スコアの極大は正解にない**（全件で例外なし）。しかし**ずれは 1.0° と小さく、
high が要求する5°の内側**にある。したがって9.5°の格子点から登れば品質は上がるはずである。

**副産物: 学習はスコアの地形も改善していた。** 極大が正解へ近づき（1.4°→1.0°、
1.02→0.84 Å）、そこでのDockQが上がり（0.849→0.890）、
「正解から離れて得られるスコア」が半減した（287→164）。
これまで順位づけの改善しか測っていなかった。

##### 局所精密化の実装と結果

`scripts/refine_poses.py`。既に書き出した提出poseを入力にするので**探索のやり直しは不要**。
6パラメータ（回転3＋並進3）のパターン探索で、各反復に12方向の候補を作り
**1回のバッチで採点**する。勾配法を採らない理由は、微分可能なのは
`scatter_mode="trilinear"` の IFACE と ELEC だけで、
**clash罰則を含む PSC 項は非微分の近傍経路のまま**だからである。
勾配で磨くと衝突が見えない目的関数を最適化することになる。

60複合体、自前DockQ（公式より0.117低い点に注意）:

| K | 平均DockQ（前→後） | 改善/不変/悪化 | acceptable | medium | high |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.476 → 0.553 | 35 / 18 / **7** | 71.7 → 71.7 | 60.0 → 65.0 | 6.7 → 36.7 |
| 3 | 0.476 → 0.570 | 40 / 9 / 11 | 71.7 → 71.7 | 60.0 → 68.3 | 6.7 → **40.0** |
| **5** | 0.476 → **0.578** | 40 / 6 / **14** | 71.7 → **73.3** | 60.0 → 68.3 | 6.7 → **40.0** |

poseは回転中央値 2.0〜4.1°、並進 0.50〜0.63 Å 動く。**格子誤差9.5°の半分弱を回収**する。
**acceptable はほぼ不変で high だけが大きく上がる**、という予測どおりの形である。

K を増やすと平均は伸びるが**悪化する複合体も倍増**する（7→14件）。
磨いてから並べ替えると、スコアの極大が正解から1.0°ずれている分の判断誤りが入るためである。
K=5 を採るのは、提出が上位5件なので5件すべてを磨けば `Max(Top 5)` も改善するからで、
K=3 との差は medium/high では無い。

**実行者の誤り2件。** (a) K比較は入力poseが `trained_QP`（アンカー1回、
$\lVert\Delta e\rVert=6.608$）、精密化スコアが `runs_convex`（アンカー反復、4.928）と
**別の表が混在**していた。250件の本番は `trained_QP` で揃えて回した。
K の相対比較は同一条件同士なので結論は保たれると考えるが、追検証していない。
(b) `pkill -f` が自分のツールラッパーを巻き込み、実行中のジョブを2度落とした。
このプロジェクトで既出の失敗を繰り返した。以後PID指定にする。

##### パラメータ数を増やす方向は効かない

`scripts/capacity_series.py`。すべてパラメータについてアフィンなので同一の凸ソルバで解け、
学習率も初期値も介在しない。$\lambda$ は各族ごとに blocked CV で選ぶ。
validation 152件（プールに正例がある複合体）での成績。

| モデル | パラメータ数 | n=50 | n=100 | n=200 | n=400 | n=637 |
| --- | --- | --- | --- | --- | --- | --- |
| 公開表 | 0 | — | — | — | — | 38.8 % |
| psc（$\alpha$/clash/$\beta$のみ） | 5 | 39.7 | 37.7 | 39.9 | 38.4 | 40.1 |
| **sym** | **12** | **48.9** | **48.7** | 47.8 | 48.7 | **50.0** |
| add | 23 | 47.6 | 46.5 | 47.4 | 48.5 | 48.0 |
| full | 144 | 35.7 | 41.7 | 42.8 | 45.6 | 46.1 |
| full+ | 149 | 34.9 | 39.5 | 43.9 | 45.8 | 46.1 |
| bucket2（大きさで表を2つ） | 288 | 34.9 | 36.8 | 41.4 | 44.7 | 47.4 |
| bucket4 | 576 | 40.6 | 41.7 | 42.1 | 46.5 | 46.1 |

1. **どのサイズでも12次元の `sym` が最良**で、144次元より +3.9 pp 上。
   288/576 に増やしても届かない。
2. **曲線の形が分かれる。** 小さいモデルは n=50 で既に飽和、
   大きいモデルは n=637 でもまだ上を向いている（bucket2 は 34.9→47.4）。
   **「パラメータを増やせばデータが効くようになる」は正しいが、637件では追いつかない。**
3. **実験1（§5.14.11）との緊張。** そこでは full(144) が additive(23) を +3.39 pp
   上回った（p=0.0215）。条件が違う（SGD＋旧損失＋prior、TESTのend-to-end success@1）。
   どちらが正しいかは `sym` を凸最適化で解いて end-to-end 評価すれば決まる。未実施。

**ただしこれは「原子タイプを増やす」実験ではない**（利用者の指摘）。
`bucket` は複合体を層別して**同じ特徴に別の重み**を与えるもので、
原子タイプ細分は**いま同一視している接触を区別する**ものである。前者の失敗は後者を否定しない。
むしろ §5.14.29 で個別上限が89.5〜100 %だったことは「144個の接触回数に情報は足りていて、
足りないのはどの重みを使うかの知識」を意味するので、
**接触を細かく分けて1つの表で通用するようにする**方向には根拠がある。未実施。

##### `--mine-ntop 1500` に根拠はなかった

報告書初期の記述に由来が残っていた。DB5.5時代は**ランダム回転を1500通り**取り、
並進が回転あたり1つだったので候補がちょうど1500個、
すなわち `ntop=1500` は「全部残す」の意味だった。
その後 Hopf nside=3（**1944回転**）に変えたのに 1500 が持ち越され、
**1944個中444個（22.8 %）を理由なく捨てている**。

実測でも全複合体で探索由来が厳密に1500、列挙由来が216（8回転×27並進セル）だった。

配備時の答えは変わらない（学習後の表で探索し直し、上位5件を提出するため）。
変わるのは**損失が見る負例の集合**であり、`hinge.mean()` が負例数で割る以上
**`ntop` は損失の隠れたハイパーパラメータ**である。
`--mine-ntop 1944` で採掘し直し中（キャッシュキーに `ntop` が入るので既存プールは残る）。

##### 局所精密化の公式結果（250件、2026-08-02）

K=5、`trained_QP` の提出poseを同じ表で磨いて並べ替え直した。探索は一切やり直していない。
所要 24.3分。poseは回転中央値 2.3°、並進 0.50 Å 動いた。

| 公式指標 | acc@1 | med@1 | **high@1** | acc@5 | med@5 | high@5 | DockQ中央値 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| published | 66.40 | 61.60 | 30.80 | 81.20 | 76.40 | 36.80 | 0.71 |
| trained_N220 | 74.80 | 69.20 | 32.40 | 86.00 | 81.60 | 37.60 | 0.73 |
| trained_QP | 76.80 | 72.40 | 38.00 | 87.60 | 83.60 | 42.40 | 0.76 |
| **trained_QP_refined** | **80.40** | **78.40** | **68.40** | 87.60 | 84.40 | **70.80** | **0.91** |
| （回転格子の上限） | 100.00 | 99.60 | *39.60* | — | — | — | 0.78 |

**high 38.00 → 68.40 %。格子の上限39.60 %を明確に突破した。**
精密化は格子の外へ出る操作なので、上限そのものが適用されなくなる。
DockQ中央値も 0.76 → 0.91。

paired exact McNemar:

| 比較 | acceptable | medium | high |
| --- | --- | --- | --- |
| 精密化 vs trained_QP | 15勝6敗, $p=0.078$ | 20勝5敗, $p=0.0041$ | **78勝2敗, $p=5.4\times10^{-21}$** |
| 精密化 vs published | 49勝14敗, $p=1.1\times10^{-5}$ | 53勝11敗, $p=1.0\times10^{-7}$ | 97勝3敗, $p=2.6\times10^{-25}$ |

**acceptable だけは有意でない**（15勝6敗）。これは予測どおりで、
acceptable は角度に鈍感（12°ずれても99.6 %）だからである。
精密化が効くのは品質であって、当たり外れではない。

24分の後処理で、再学習も探索の変更も無しに得られた。
本研究で最大の一手であり、費用対効果は他のどの施策より2桁良い。

**帰属が未分離である（重要な限界）。** 上表の `published`（66.40 / 61.60 / 30.80）は
**精密化なし**、`trained_QP_refined`（80.40 / 78.40 / 68.40）は**学習と精密化の両方**を
含む。したがってこの差 +14.0 / +16.8 / **+37.6 pp** は2つの効果の合計であり、
どちらがどれだけ寄与したかは分けていない。精密化は汎用の後処理なので、
**公開表のposeに同じ精密化をかければ、それだけでも大きく上がるはずである。**

分離には $2\times2$（公開表／学習表 × 精密化なし／あり）が要る。
§5.14.31 の地形測定は交互作用の存在を示唆する — スコアの極大が公開表では
正解から1.4°・1.02 Å（そこでのDockQ 0.849）、学習後は1.0°・0.84 Å（0.890）なので、
**学習した表の方が精密化の着地点が良い**はずである。ただしこれは地形からの推論であって
end-to-endの測定ではない。`published_refined` を実行中（2026-08-02）。

現時点で言えるのは「**学習済みの表に精密化を足すと 38.0 → 68.4 %**」（この比較は
表が同一なので精密化単独の効果、78勝2敗、$p=5.4\times10^{-21}$）までであり、
公開表からの +37.6 pp を学習の功績として述べてはならない。

##### `--mine-ntop 1944` は原理的には正しいが、性能には効かない（null）

切り捨てをやめて採掘し直し、同じ手続きで凸最適化を解いた。
`ntop=1944` のプールで両方の表を採点する（母数を揃えるため）。

| 表 | 250件中 | 到達可能157件中 | 最良正例の順位中央値 |
| --- | --- | --- | --- |
| 公開表 | 23.60 % | 37.58 % | 34 |
| **`ntop=1500` で学習** | **28.00 %** | **44.59 %** | 14 |
| `ntop=1944` で学習 | 27.20 % | 43.31 % | 12 |

**捨てていた444個には確かに正例が入っていた。** 到達可能な複合体が
fit で 637 → **655**、validation で 152 → **157** と約3 %増える。
根拠の無い切り捨てが実際に複合体を損失から隠していた。

**しかしそれで学習しても改善しない**（70件 → 68件）。
増えた負例444個を押し下げる負担が、増えた正例の利得を上回ったか、
新たに到達可能になった複合体がどのみち解けないかである。
**欠陥としては実在し直すべきだが、性能のレバーではない。**

##### PINDER論文との比較（Table 6、`pinder_s` / holo / Top-1、250件）

| 手法 | Acceptable | Medium | High |
| --- | --- | --- | --- |
| HDOCK | 94.4 | 94.0 | **93.6** |
| FRODOCK | 89.6 | 87.6 | 84.8 |
| PatchDock | 53.6 | 52.4 | 43.6 |
| **本研究（凸最適化＋精密化）** | **80.4** | **78.4** | **68.4** |
| 本研究（凸最適化のみ） | 76.8 | 72.4 | 38.0 |
| 本研究（公開表） | 66.4 | 61.6 | 30.8 |

apo（93件）では HDOCK 44.09 / 40.86 / 33.33 に対し本研究 26.88 / 19.35 / 4.30。

**この形は本節の測定で説明でき、実際に検証された。** High の差は
**ほぼ全部が回転格子の有無**であった（我々の格子上限が39.6 %）。HDOCKはFFT探索後に
局所精密化を持つのでこの天井が無い。**同じ精密化を入れたところ High は 38.0 → 68.4 %**
となり、PatchDock（43.6）を大きく超えてFRODOCK（84.8）に近づいた。
仮説と処方と検証が一致した数少ない例である。

残る差は、Acceptable の 14.0 pp（80.4 対 94.4）で、これは順位づけの問題である。
**探索の recall が 96.0〜96.8 %** なので、探索を変えずに再ランキングだけで
HDOCK水準に届きうる範囲にある。ただし §5.14.29–30 の測定では、
144次元線形・加法モデルの枠内で残っているのは数 pp であり、
そこから先は特徴量（原子タイプ細分・距離シェル）か、
加法性を捨てた再ランキング段が要る。

#### 5.14.32 精密化と学習の分離、そして certificate は小規模でしか成立しない（2026-08-02）

##### $2\times2$: 学習と精密化の効果を分離する

§5.14.31 では「公開表（精密化なし）対 学習表（精密化あり）」を比較しており、
2つの効果が混ざっていた。対照 `published_refined`（公開表のposeを公開表で精密化）を
走らせて分離した。公式 `pinder_s`/holo/Max(Top 1)、250件。

| | acceptable | medium | high | DockQ中央値 |
| --- | --- | --- | --- | --- |
| published | 66.40 | 61.60 | 30.80 | 0.71 |
| **published_refined** | **74.40** | **72.40** | **60.40** | **0.88** |
| trained_QP | 76.80 | 72.40 | 38.00 | 0.76 |
| trained_QP_refined | 80.40 | 78.40 | 68.40 | 0.91 |

| | 精密化の効果（公開表 / 学習後） | 学習の効果（精密化なし / あり） | 交互作用 |
| --- | --- | --- | --- |
| acceptable | +8.0 / +3.6 pp | +10.4 / +6.0 pp | **−4.4 pp** |
| medium | +10.8 / +6.0 | +10.8 / +6.0 | −4.8 |
| **high** | **+29.6 / +30.4** | +7.2 / +8.0 | +0.8 |

1. **high は精密化がほぼ全部**である。学習の4倍近く、しかも交互作用 +0.8 pp と
   ほぼ加法的。§5.14.31 で「学習の功績として述べてはならない」と書いた懸念は正しかった。
2. **acceptable では学習の方が大きい**が、交互作用が **−4.4 pp** と負である。
   両者は**同じ複合体を部分的に直している**。
3. **`published_refined`（学習なし・精密化のみ）が 74.40 / 72.40 / 60.40。**
   学習した表に精密化なし（76.80 / 72.40 / 38.00）を high で22 pp 上回り、
   acceptable でも2.4 pp差まで迫る。**今回の伸びの大半は学習なしでも得られた。**
   本研究の学習が寄与しているのは、精密化の上に **+6.0〜8.0 pp** である。

##### exact Top-1 MILP: n=12 では成立し、n=50 では破綻する

`scripts/exact_top1_milp.py`。codexの定式化（$y_c$ が複合体の解決、$z_{cp}$ が
狙う正例、big-M は箱制約から**制約ごとに計算**、負例は切除平面で遅延追加）。

| 部分集合 | ヒンジ解 | MILPの実解 | 上界 | gap | ラウンド | 所要 |
| --- | --- | --- | --- | --- | --- | --- |
| n=12 | 11 | **12** | 12 | **0** | 25 | 296 s |
| n=50 #0 | **29** | 20 | 49 | 29 | 17 | 5163 s |
| n=50 #1 | **25** | 17 | 50 | 33 | 30 | 9107 s |

**n=12 では certificate が成立した。** 単一の共有表で12件すべてを解けることが確認でき、
ヒンジ解は11件だった。すなわちこの規模では**容量はまったく制約でなく、
代理関数が1件（8.3 pp）を取り逃していた**。

**n=50 では破綻した。** MILPが返した解はヒンジ解より**悪く**（20 < 29、17 < 25）、
上界は $n$ そのもので自明である。certificate は「29以上50以下」としか言えず無情報。
原因は (a) big-M 緩和が弱くLP緩和が常に $n$ を返す、(b) 切除平面が30ラウンドでも
収束しない、(c) MILP solve に300秒の制限を課しているため途中の悪い解を返す、である。

##### n=25 でも certificate は成立しない（各2.5時間）

| 部分集合 | ヒンジ解 | MILPの実解 | 上界 | ラウンド |
| --- | --- | --- | --- | --- |
| n=25 #0 | **16** | 14 | 25（自明） | 30 |
| n=25 #1 | 11 | **14** | 25（自明） | 30 |

上界は両方とも $n$ そのもので自明、MILPの実解はヒンジ解に勝ったり負けたりする。
**n=12 と n=25 の間で壊れる**ことが確定した。

##### 「MILPの解はそのままパラメータになる」— しかし汎化しない（利用者提起）

MILPが返す incumbent は単なる下界ではなく**テーブルそのもの**であり、
しかも代理関数ではなく**配備される指標（Top-1の的中数）を直接最大化した**解である。
これはcodexの案A（Exact Top-1 Risk Minimization）そのものなので、
「certificateであると同時に学習器なのではないか」という問いが立つ。

n=12 の部分集合を3つ解き、それぞれの incumbent を **152件の固定validationで採点**した。

| 部分集合 | 学習した12件 ヒンジ / MILP | val 公開表 | val ヒンジ | **val MILP** |
| --- | --- | --- | --- | --- |
| 0 | 11 / **12** | 38.8 % | **42.1 %** | **28.9 %** |
| 1 | 9 / **10** | 38.8 % | **41.4 %** | **30.9 %** |
| 2 | 8 / **9** | 38.8 % | 32.2 % | 33.6 % |

**MILPは学習した12件では毎回ヒンジに勝つ**（12>11、10>9、9>8）。
指標を直接最大化しているので当然である。
**しかしvalidationでは3つ中2つで出発点の公開表（38.8 %）より悪い**（28.9 %、30.9 %）。
ヒンジ解（42.1 %、41.4 %）には大きく劣る。

原因は明らかで、12件の二値結果に144パラメータを合わせており、しかも
**マージンも正則化も無い**。的中数だけを最大化するので「スコア差 0.001 で勝った」
複合体を大量に作り、新しいデータでそれが反転する。
ヒンジ解が汎化するのは、マージン $\epsilon=1$ と正則化 $\lambda$ を持つからである。

**結論: 案Aは certificate であって学習器ではない。**

| 用途 | 判定 |
| --- | --- |
| 容量の証明（真の最大値を知る） | $n\le12$ でのみ成立 |
| **パラメータを求める学習器** | **公開表より悪い表を返す** |

ただし救いはある。codexの元の提案は「**的中数を最大化し、同数なら norm 最小化**」という
辞書式目的であった。本実装は第1段だけで、第2段（同点解の中からマージン最大・
norm最小を選ぶ）を省いている。これを入れれば汎化する可能性がある。
すなわち「配備指標を直接最適化すればよい」は誤りで、
**指標を最適化しつつマージンを稼ぐ**必要がある、という機械学習の基本原則の再確認である。

**含意。** codexが提案した「certifiable learning framework」という主張は、
現状のツール（scipy/HiGHS、big-M定式化）では **$n\approx12$〜25 までしか成立しない**。
250複合体のベンチマークについて証明付きの主張はできない。
打開策の候補は、(1) ヒンジ解のアンカーに $z$ を固定して二値変数を
$\sum_c\lvert P_c\rvert+n$ から $n$ へ減らす（有効な下界は得られるが上界を失う）、
(2) warm start と indicator制約に対応したソルバ（Gurobi等）へ移る、
(3) big-M をスコア差の実際の範囲でより厳しくする。

実装上の知見として、切除平面の初期集合を**公開表の下だけ**で張ると n=12 でも
10ラウンド以上を「12件中5件しか解けない緩和」に費やす。
**ヒンジ解の下での上位負例も初期集合に入れる**と大幅に速くなる。

#### 5.14.33 上位50件の精密化と Oracle@50: 残る差の所在が確定した（2026-08-04）

**問い。** HDOCK（94.4 / 94.0 / 93.6）との差は、候補が足りないのか、
精密化が足りないのか、選択が足りないのか。codexの設計に従い、
**上位50件を精密化して Oracle@50 を測る**ことでこの3つを分離する。
事前に判定基準を固定した（Oracle acc < 92 % なら候補不足 → multi-head / nside4 へ、
acc は十分だが high < 94 % なら精密化エネルギーの学習へ、両方十分なら純粋に順位づけ）。

##### 手続き

学習表（凸最適化、`runs_loss/qp`）で探索し直して**上位50 poseを書き出し**、
同じ表で全50件を6自由度パターン探索で精密化して並べ替え、
**生の50件と精密化後の50件の両方**を公式harnessに採点させた。
250件 × 50 pose × 2条件 = 25,000 decoy。

##### 結果（公式 `pinder_s` / holo、250件）

| | acceptable | medium | high |
| --- | --- | --- | --- |
| **精密化なし** Top-1 | 76.80 | 72.40 | 38.00 |
| 同 Max(Top 5) | 87.60 | 83.60 | 42.40 |
| 同 **Oracle@50** | **94.80** | 90.00 | 44.80 |
| **精密化あり** Top-1 | **84.00** | **82.40** | **73.60** |
| 同 Max(Top 5) | **92.00** | 88.40 | 78.40 |
| 同 **Oracle@50** | **94.80** | 91.60 | **79.60** |
| （参考）HDOCK Top-1 | 94.4 | 94.0 | 93.6 |

##### 判定: クラスごとに原因が違う

| | 生Oracle | 精密化後Oracle | Top-1 | 診断 |
| --- | --- | --- | --- | --- |
| acceptable | 94.80 | 94.80（**不変**） | 84.00 | **順位づけ**（−10.8 pp） |
| medium | 90.00 | 91.60（+1.6） | 82.40 | **順位づけ**（−9.2 pp） |
| high | **44.80** | **79.60（+34.8）** | 73.60 | **精密化が効いている**（順位づけは−6.0 pp） |

1. **候補不足ではない。** 生の Oracle@50 の acceptable が **94.80 %** で、
   **HDOCK の 94.4 % を上回る**。事前基準（< 92 %）は発動せず、
   **multi-head FFT と nside=4 は不要**と判定された。
   codexの予測（multi-head の効果は +0.3〜1.0 pp）とも整合する。
2. **acceptable/medium は純粋に順位づけの問題。** 精密化しても Oracle が
   94.80 % のまま動かないのは当然で、精密化は pose の質を上げるだけで
   「正解に近い basin が候補にあるか」は変えない。
   候補には94.80 %で正解があるのに Top-1 は 84.00 %、**10.8 pp を選び間違えている**。
3. **high は精密化の問題。** 生の Oracle が 44.80 % と低いのは探索の失敗ではなく
   **回転格子の刻み（9.5°ずれ）**であり、精密化が格子の外へ出て +34.8 pp 回復した。
   ただし精密化後でも 79.60 % で、HDOCK の 93.6 % には届かない。
   **完璧に選んでも 79.60 % 止まり**なので、high には精密化そのものの改良が要る。

##### 新しい最良値

上位5件だけ精密化していた §5.14.31 から、上位50件へ広げた効果:

| | 上位5 | **上位50** | 差 |
| --- | --- | --- | --- |
| acceptable | 80.40 | **84.00** | +3.6 pp |
| medium | 78.40 | **82.40** | +4.0 pp |
| high | 68.40 | **73.60** | +5.2 pp |
| Max(Top 5) | 87.60 | **92.00** | +4.4 pp |

codexの予測は acc +5〜8 pp で、実測 +3.6 pp はやや下振れした。
精密化の内部統計では、上位5件のとき悪化率 9.2 %（23/250）だったのが
上位50件では 4.8 % に下がっている。候補が多いぶん並べ替えで拾い直せている。

##### 次にやること（優先順位が数字で決まった）

| 施策 | 対象 | 取れる上限 |
| --- | --- | --- |
| **reranker**（収束安定性、physics特徴） | acceptable / medium | **+10.8 / +9.2 pp** |
| **精密化エネルギーの学習** | high | Oracle 79.60 → 93.6 が目標 |
| multi-head FFT / nside=4 | — | **不要と判定** |

**acceptable は順位づけだけで HDOCK に並ぶ**（84.00 → 上限94.80、HDOCK 94.4）。

##### 実装上の不具合（実行者）

(a) 最初の起動でシャード指定を忘れ、3ジョブが同じ250件を重複処理していた。
`refine_poses.py` に `--shard i/n` を追加した。
(b) 50 pose すべての DockQ を一括計算しており、大きな複合体で
$(50, N_\text{lig}, N_\text{rec})$ の距離行列が 40 GB になって OOM した。
報告に使うのは「元の1位」と「精密化後の1位」の2件だけなので、そこだけ計算するよう修正。
(c) `kill` のパターンが自分のツールラッパーを巻き込む問題が再発した
（ラッパーのコマンドライン文字列にスクリプト名が含まれるため）。PIDを2段階で取得する方式にした。

#### 5.14.34 Q1: 精密化後の分布で学習し直しても順位づけは直らない（2026-08-05）

**問い（利用者提起）。** rerankerを作らず、スコア関数のパラメータ学習だけで
§5.14.33 の順位づけ損失（acceptable −10.8 pp）を回収できないか。

**着眼点。** 現行の 84.00 % は、**精密化前のpose（格子上）で学習した表**で
**精密化後のpose（格子の外）**を並べ替えた結果である。表は一度も、実際に
順位づけの対象となる分布を見ていない。関数形を変えずに**当てはめるデータの分布だけ**を
合わせれば、rerankerなしで回収できる可能性がある。

**手続き。** 訓練400 + validation 250 = 650複合体について、
学習表で探索し直して上位50 poseを書き出し（4.6時間）、
全50件を精密化し（4時間）、精密化後のposeの特徴量
$(S_\text{PSC}, T, S_\text{ELEC})$ を再計算してプール化した
（`scripts/build_refined_pool.py`、探索が無いので0.6〜1.4秒/複合体）。
その上で訓練400件（正例あり215件）に凸最適化を当てはめ、
validation 250件（正例あり123件）で採点した。**すべて精密化後のposeで統一**している。

##### 結果

| 表 | 学習に使ったposeの分布 | validation 123件中 |
| --- | --- | --- |
| 公開表 | — | 75 = 60.98 % |
| 現行（`runs_loss/qp`） | 精密化**前**（格子上） | 81 = 65.85 % |
| **精密化後のプールで学習** | **精密化後（格子の外）** | **82 = 66.67 %** |

**+1複合体、+0.82 pp。実質ゼロである。**

$\lambda$ path（CVで $\lambda=1$ を選択）:

| $\lambda$ | CV top-1 | fit top-1 | $\lVert\Delta e\rVert_2$ |
| --- | --- | --- | --- |
| 1e-3 | 62.33 % | 185/215 (86.0 %) | 9.24 |
| 1e-2 | 63.26 % | 186/215 (86.5 %) | 9.10 |
| 1e-1 | 65.12 % | 185/215 | 8.04 |
| **1** | **68.37 %** | 171/215 (79.5 %) | 5.23 |
| 10 | 60.00 % | 138/215 | 1.62 |

現行の表は同じfit集合で 160/215 = 74.42 % なので、
**新しい表は訓練データに遥かによく当てはまる（最大86.5 %）が転移しない**。
§5.14.29–30 で繰り返し見てきた形と同じである。

##### 結論

**順位づけの10.8 ppはパラメータ学習では回収できない。**
関数形が同じである限り、当てはめるデータを変えても
「1つの表を全複合体で共有することによる衝突」（個別上限 89.5〜100 % 対
共有表 in-sample 56.6 %、§5.14.29）は解けない。

したがって **reranker が必要である**という証拠が揃った。必要なのは
(a) 加法性を捨てる（非線形モデル）か、
(b) 現在のスコアが一切見ていない**新しい特徴量**を足すか、である。
codexの設計では (b) が先で、**「共通12摂動で元クラスタへ戻る割合＝basin安定性」**が
最有力候補である。これは学習を必要とせず幾何だけで計算できる。

##### 副次的な観察

訓練コーパスはTESTより遥かに難しい。同じ精密化を当てても、

| | 訓練650件 | TEST 250件 |
| --- | --- | --- |
| acceptable（精密化前 → 後） | 33.2 % → 36.9 % | 70.2 % → 79.8 % |
| high | 5.4 % → 19.7 % | 3.6 % → 48.8 % |

validation 250件のうち精密化後プールに正例があるのは123件（49 %）で、
TESTの Oracle@50 94.8 % とは水準がまったく違う。
**この2つを並べてはならない**（§5.14.25 で指摘した validation の代表性の問題が、
精密化後も解消していない）。

#### 5.14.35 basin安定性は信号を持つが、スコアに何も足さない（2026-08-06）

**問い。** §5.14.34 で「順位づけの10.8 ppはパラメータ学習では回収できない」と分かった。
codexの設計に従い、**現在のスコアが一切見ていない情報**を作って reranker を組めるか。

**仮説。** 本物のエネルギー谷にいるposeは、そこから叩かれても戻ってくる。
偶然のスコアの盛り上がりに乗っているposeは戻らない。すなわち

- `basin_size` 上位50件のうち何個が同じ終点へ収束したか
- `stability` 終点を12方向へ叩いたとき何個が戻るか

##### 摂動の大きさは較正が必要だった（codexの指定値では判別が逆向き）

codexは ±4°/±1 Å を指定したが、実測すると**判別力が逆向き**であった。

| 摂動 | stability 平均（acceptable / 他） | 全体AUC | 複合体内AUC |
| --- | --- | --- | --- |
| **±4° / ±1 Å**（指定値） | 0.939 / 1.000 | **0.364（逆向き）** | 0.463 |
| ±8° / ±2 Å | 0.910 / 0.872 | 0.627 | 0.463 |
| **±16° / ±4 Å**（採用） | 0.769 / 0.534 | 0.835 | **0.777** |
| ±24° / ±6 Å | 0.526 / 0.223 | 0.880 | 0.762 |

**すべてのbasinが±4°より広い**ので、何もかも戻ってきて分離しない。
そのまま走らせていれば「信号なし」という誤った結論を出していた。

**もう一つの設計上の問題。** `basin_size` は機能しない。
50候補が45〜50個のクラスタに分かれ、中央値が1である。
探索のNMSが候補を離して保持しているので**同じ終点に集まりようがない**。
codexは「top-50の本数はFFTのサンプリング密度を測るだけ」と警告したが、
実際には**NMSの分離幅を測っていた**。

##### パイロット（訓練300 + validation 100 = 397複合体、19,850候補、11時間）

複合体内AUC（両クラスが揃う187複合体）:

| 特徴 | AUC |
| --- | --- |
| **score1**（精密化後のスコア） | **0.849 ± 0.217** |
| gain | 0.776 |
| **stability** | **0.738 ± 0.237** |
| score0 | 0.727 |
| rot_deg | 0.723 |
| shift_A | 0.694 |
| basin_size | 0.644 |
| dispersion | 0.639 |

**stability は codex のゲート（AUC ≥ 0.60）を通過する。**
しかし線形に足しても何も起きない:

| $w$ | 0 | 25 | 50 | 100 | 150 | 200 | 300 | 500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\text{score1}+w\cdot\text{stability}$ | 37.3 | 37.3 | 37.0 | 37.3 | 36.8 | 37.3 | 37.3 | 37.3 % |

##### 本ゲート（学習した組み合わせ、cluster-blocked 5-fold CV）

| モデル | top-1 acceptable | score1単独との差 |
| --- | --- | --- |
| **score1 のみ** | **37.28 %** | — |
| 線形（全特徴） | 35.26 % | **−2.02 pp** |
| 線形（basin特徴を除く） | 33.00 % | −4.28 pp |
| GBDT（深さ3、全特徴） | 36.78 % | −0.50 pp |
| GBDT（basin特徴を除く） | 37.03 % | −0.25 pp |

**学習した組み合わせはすべて最終スコア単独に負ける。** codexのゲート（+2 pp）は
不合格であり、しかも「効果なし」ではなく**負**である。

**解釈。** stability は単独では 0.738 の判別力を持つのに、スコアと組み合わせると
何も足さない。**情報が完全に重複している。** 学習を挟むと過学習側に振れて悪化する。

##### 限界

- **訓練コーパスの複合体**であり top-1 は 37 %（TESTは84 %）。難易度帯が違う。
- stability を測れたのは上位10クラスタ内の 4,345 候補のみ（全19,850中）。
- 目的関数が $P(\text{acceptable})$ の**分類**であり、codexが指定した
  **LambdaRank（CAPRI順位）ではない**。ただし分類で −2 pp という結果からして、
  順位学習に変えて +2 pp に転じる見込みは薄い。
- 397複合体が397クラスタなので blocked CV は通常のCVと同じになっている。

##### 判定

codexの停止規則が発動する。**reranker の拡大を止め、幾何Oracle診断へ移る**
（DockQそのものを目的関数にして top-50 を精密化し、
幾何的に到達可能な high Oracle を測る。90 %以上なら精密化エネルギーの学習へ、
88 %未満なら multi-scale / multi-start を先に直す）。

#### 5.14.36 sym(12) は full(144) を +7.24 pp 上回る（固定プール、2026-08-06）

§5.14.31 の `capacity_series` で「12次元が144次元より良い」と出ていたが、
一度も配備形式で当てはめておらず、end-to-end 評価もしていなかった。
`scripts/fit_subspace.py` で同じレシピ（アンカー付きヒンジ、$\lambda$ は
interface-cluster blocked CV）により部分空間を当てはめ、
`run_pinder_scaling` 形式のcheckpointを書き出した。

| 固定validation 152件 | |
| --- | --- |
| 公開表 | 59 = 38.82 % |
| 配備中の full-144（凸最適化） | 65 = 42.76 % |
| **sym(12)** | **76 = 50.00 %** |

**12個の自由度が144個を +7.24 pp 上回る。** $\lVert\Delta e\rVert$ も 5.19 と
full-144 の 6.61 より小さい。**より少ないパラメータで、より小さく動いて、より良い。**

**実装バグの記録。** 最初の実装は $e=B\theta$ を当てはめて $e_0+B\theta$ を保存していた。
**学習中に公開表の寄与が消えていた**ので、当てはめたものと保存したものが別物になる。
この状態では sym が 36.18 %（公開表より悪い）と出ており、
そのままなら「sym は駄目」という誤った結論になっていた。
正しくは $e=e_0+B\theta$ で、公開表を原点として部分空間を張る。

**未確定。** これは**固定プール上の再ランキング**の値である。
sym(12) は探索にも使われるので、表を変えると**探索が返す候補そのものが変わる**。
end-to-end 評価を実行中（250件、上位50件書き出し → 精密化 → 公式採点）。

---

### 5.14.37 理論的背景の解説文書（実験ではない）

§5.14 までの経緯を、理論的背景から数式込みで説明した日本語文書を
`docs/theory_ja.tex` に置いた。新しい測定は含まず、既出の数値の再掲である。

構成: 問題設定（SE(3)、ZDOCK スコアの3項、FFT 全探索が要求する加法性の命題）
→ 評価指標（DockQ / CAPRI / Max(Top 1) / Oracle）
→ アフィン性 $s_p(e) = c_p + \langle A_p, e\rangle$
→ 診断 I（アンカー位置と事前分布による到達不能性、§5.14.11〜13）
→ 診断 II（凸代理問題と LP による容量上限、§5.14.20〜21, §5.14.31）
→ 診断 III（回転格子が作る品質の天井、§5.14.27〜29）
→ 誤差の 4 成分分解（§5.14.33）
→ negative results（§5.14.22〜26, §5.14.34〜36）
→ 結果表と未解決問題。

コンパイル（本機に LuaTeX が無いため XeTeX 系を使う。9ページ、警告なし）:

```bash
tectonic docs/theory_ja.tex        # docs/theory_ja.pdf が出る
```

`ltjsarticle` + `lualatex` に戻す手順は .tex 冒頭のコメントに記した。
PDF は生成物なので `.gitignore` に入れてある。

> **重要な訂正（2026-07-26）。この節と §7 の以下の記述は §5.12〜§5.14 で
> 無効化された。** TEST セットの参照構造が 250/250 で立体的に不可能だった
> （PINDER の `test_set_pdbs/` モノマーが各々原点中心の「ドッキング入力」で
> 複合体の座標系を持たない）ため、**§5.3〜§5.11 の TEST 由来の数値はすべて無効**
> である。修復後の実測は:
>
> | | success@1（249複合体、end-to-end） |
> |---|---:|
> | 既定パラメータ（学習なし） | **65.9%** |
> | IFACE のみ学習（PSC 凍結） | **73.5%**（21勝2敗、p=6.6e−05） |
>
> すなわち「baseline 0%」「deleaked test で 1〜4%」「mining の gain は 0 ポイント」
> という以下の記述はいずれも破綻した参照構造の産物である。
> 現在の正しい要約は §5.13（真のベースライン）と §5.14（学習結果）にある。
> 以下は研究史として残す。

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

> **注意（2026-07-26）**: 上のコマンド群は §5.1〜§5.5 のもので、
> **参照構造が破綻していた時期のパイプライン**である（§5.12）。
> `uv run pytest -q` の「95 passed」も陳腐化しており、現在は **116 passed** である。

### 8.1 現行の再現手順（§5.13〜§5.14）

```bash
export PINDER_BASE_DIR=$PWD/external/pinder HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/home/yasu/tmp/ddock-tmp

# 0) 参照構造の健全性を必ず先に検査する（§5.12 の教訓）
uv run python scripts/check_prep_cache.py \
    --cache-dir data/scaling/prep_cache --ids-file data/scaling/master_ids.txt

# 1) TEST 構造は複合体ファイルの鎖 R/L から作る（test_set_pdbs は原点中心で不可）
uv run python scripts/prep_pinder_cache.py \
    --ids-file data/pinder_test_ids.txt --cache-dir data/scaling/prep_cache_test \
    --complex-dir external/pinder/pinder/2024-02/pdbs \
    --manifest-dir data/scaling/prep_manifest_test_v2 \
    --merged-manifest data/scaling/prep_manifest_test_v2.jsonl --gpus cpu --device cpu

# 2) 1.2 A の voxel 表（generate_grid ではなく grid_shape を使う）
uv run python scripts/compute_grid_sizes.py --spacing 1.2 \
    --out data/scaling/grid_voxels_1.2.json

# 3) 固定 TEST プール（3 GPU 分割 → マージ）
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i uv run python scripts/build_test_pool.py \
    --prep-cache data/scaling/prep_cache_test --shard $i/3 \
    --grid-voxels data/scaling/grid_voxels_test_1.2.json --max-grid-voxels 31250000 \
    --out data/shards_pinder/test_pool_shard$i.pt & done; wait

# 4) 学習プールの採掘（3 GPU 分割、round 0 は seed 非依存なのでキャッシュ共有）
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i uv run python scripts/run_pinder_scaling.py \
    --n-fit 220 --rounds 0 --seed 0 --alpha0 1.0 --mine-shard $i/3 --mine-only \
    --grid-voxels data/scaling/grid_voxels_1.2.json --max-grid-voxels 31250000 \
    --test-cache data/shards_pinder/test_pool_reachable.pt \
    --out-dir data/scaling/runs_ifonly & done; wait

# 5) 学習（キャッシュから数分）。PSC を凍結し IFACE だけを学習するのが最良
for sd in 0 1 2; do uv run python scripts/run_pinder_scaling.py \
    --n-fit 220 --rounds 0 --seed $sd --alpha0 1.0 --loss-prov search --freeze-psc \
    --grid-voxels data/scaling/grid_voxels_1.2.json --max-grid-voxels 31250000 \
    --test-cache data/shards_pinder/test_pool_reachable.pt \
    --out-dir data/scaling/runs_ifonly; done

# 6) 評価。固定プールの paired 検定 + 学習後パラメータでの end-to-end 再探索
uv run python scripts/compare_conditions.py \
    --pool data/shards_pinder/test_pool_reachable.pt \
    --ckpt data/scaling/runs_ifonly/N220_seed0/round0_ckpt.pt --prov search
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i uv run python scripts/eval_search_test.py \
    --shard $i/3 --ckpt data/scaling/runs_ifonly/N220_seed0/round0_ckpt.pt \
    --out data/scaling/eval_search/if_s$i.json & done; wait
```
