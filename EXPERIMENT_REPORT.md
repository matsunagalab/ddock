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
- **過学習していない**: train-set の top-1 RMSD 70.5% と test の 71.0% がほぼ同一。
  学習した SC 重みと 12×12 IFACE 行列が複合体間で汎化している。

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
1. **split がランダム複合体分割で interface 非クラスタリング**。抗体系など類似
   interface が train/test に跨る leakage が残り、71% は楽観的な可能性がある。次は
   interface/配列クラスタリングでの deleaked split（または PINDER deleaked split）。
2. **hard-negative mining ループ**（現行 param で FFT を再生成して高スコア非 native を
   採掘）を回すと、探索そのものの品質も上げられる。
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
```

生成物（`data/`, `external/`, `logs/`, `*.h5`）は `.gitignore` 済み。上記スクリプトで
再現できる。
